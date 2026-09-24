"""
recover_descriptions_bulk.py — recover the 26,992 books Hardcover catalogues but
does not describe, using the 2017 UCSD Goodreads dump.

WHY THIS IS THE HIGHEST-LEVERAGE BOOKS WORK
The books tab's recurring complaint is "this book is missing". The corpus is
74,074 rows against 120,098 harvested: 26,992 were dropped at the description
filter, not because they are obscure but because Hardcover's catalogue record
was bare. They are already fetched. They are sitting in raw.jsonl.

The targeted version of this recovered 12 of 19 on the HARDEST possible sample
-- obscure Indian non-fiction that Hardcover, Open Library, Google Books and
Wikipedia had all failed on. On a general population the rate should be better.

AUTHOR VERIFICATION IS NOT OPTIONAL
Title-only matching against 2.36M books returned David Baldacci's Zero Day and
Catherine Coulter's Eleventh Hour for S. Hussain Zaidi. Generic titles collide
constantly at that scale. Every match here requires normalised title AND
normalised first author. A book that matches on title alone is NOT recovered.

WHY TWO FILES
goodreads_books.json.gz has author IDs, not names. goodreads_book_authors.json.gz
maps id -> name. The author map loads into RAM (~800K entries, small); the books
dump is streamed and never held.

WHY THE WANTED SET IS BUILT FIRST
Streaming 2.36M records while holding every description would need gigabytes.
Building the ~27K keys we want first means the stream keeps only what matches,
bounded by the size of the gap rather than the size of the dump.

STRATIFIED REPORTING
The keyword pilot was wrong because it sampled the head of a power-law
distribution. This reports hit rate by users_count stratum, so a great rate on
popular books cannot hide a collapse in the tail -- which is where the missing
books people complain about actually live.

LICENCE: the dump is academic use only (Wan & McAuley, RecSys'18). Recovered
rows must carry source="hardcover+goodreads_dump_2017" so the mixed provenance
is visible in the parquet. Embedding derived vectors is fine; republishing the
text is not.

STALENESS: these descriptions are from 2017. For a 2019 book there will be no
match at all, which is a coverage limit, not a bug.

USAGE
    python notebooks\\pipeline\\recover_descriptions_bulk.py --stage match
    python notebooks\\pipeline\\recover_descriptions_bulk.py --stage match --limit 2000
    python notebooks\\pipeline\\recover_descriptions_bulk.py --stage report
"""
import argparse
import gzip
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

RAW_DIR = Path("data/raw/books/hardcover_v2")
BOOKS_DIR = Path("data/raw/books")
HC_RAW = RAW_DIR / "raw.jsonl"
GR_BOOKS = BOOKS_DIR / "goodreads_books.json.gz"
GR_AUTHORS = BOOKS_DIR / "goodreads_book_authors.json.gz"
OUT = RAW_DIR / "gr_desc_bulk.json"

MIN_DESC_CHARS = 100      # the same bar the harvest applied
STRATA = [(0, 25), (25, 100), (100, 500), (500, 2000), (2000, 10 ** 9)]


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def _core(title):
    """Drop parentheticals and subtitle, then leading article.

    Goodreads carries series parentheticals ('(The Lord of the Rings, #1)') that
    Hardcover does not, so they have to go or nothing matches. Splitting on ':'
    is safe HERE because the author is also part of the key -- the collapse that
    would be dangerous for a work-id is harmless for a lookup.
    """
    t = re.sub(r"\([^)]*\)", " ", str(title))
    t = re.split(r":| - ", t)[0]
    return re.sub(r"^(the|a|an) ", "", _norm(t)).strip()


def author_key(v):
    """Order-independent author key.

    Goodreads stores some names as 'Hammett, Dashiell' and others as
    'Dashiell Hammett'; Hardcover is consistently natural order. Splitting on
    the comma and taking the head gives 'hammett' on one side and
    'dashiell hammett' on the other, and The Maltese Falcon -- the book this
    whole line of work started from -- silently fails to match.

    Sorting the tokens makes both orders collapse to the same key, and it is
    still strict enough to keep S. Hussain Zaidi apart from David Baldacci,
    which is the false positive that made author verification mandatory.
    """
    toks = _norm(v).split()
    return " ".join(sorted(toks)) if toks else ""


def load_author_map():
    if not GR_AUTHORS.exists():
        sys.exit(f"{GR_AUTHORS} not found")
    amap = {}
    with gzip.open(GR_AUTHORS, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                a = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid, name = a.get("author_id"), a.get("name")
            if aid and name:
                amap[str(aid)] = name
    print(f"  author map: {len(amap):,} authors")
    return amap


def load_wanted(limit=None):
    """Hardcover rows with NO usable description -> {(core, author): [rows]}."""
    if not HC_RAW.exists():
        sys.exit(f"{HC_RAW} not found")
    wanted = defaultdict(list)
    n = miss = 0
    with HC_RAW.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            if len((b.get("description") or "").strip()) >= MIN_DESC_CHARS:
                continue
            miss += 1
            title = b.get("title") or ""
            author = ""
            for c in (b.get("contributions") or []):
                a = c.get("author") if isinstance(c, dict) else None
                if isinstance(a, dict) and a.get("name"):
                    author = a["name"]
                    break
            ct, ca = _core(title), author_key(author)
            if not ct or not ca:
                continue
            wanted[(ct, ca)].append({
                "id": b.get("id"), "title": title, "author": author,
                "users_count": b.get("users_count") or 0,
            })
            if limit and miss >= limit:
                break
    print(f"  hardcover raw: {n:,} rows | {miss:,} without a description")
    print(f"  keyed for lookup: {sum(len(v) for v in wanted.values()):,} "
          f"across {len(wanted):,} title+author keys")
    print("  (rows with no author name are unkeyable and excluded -- author "
          "verification is the whole point)")
    return wanted


def stage_match(limit=None):
    print("building the wanted set ...")
    wanted = load_wanted(limit)
    if not wanted:
        sys.exit("nothing to recover")
    print("\nloading the author map ...")
    amap = load_author_map()

    print(f"\nstreaming {GR_BOOKS.name} (2.36M records, several minutes) ...")
    if not GR_BOOKS.exists():
        sys.exit(f"{GR_BOOKS} not found")

    found = {}
    seen = 0
    with gzip.open(GR_BOOKS, "rt", encoding="utf-8") as fh:
        for line in fh:
            seen += 1
            if seen % 250_000 == 0:
                print(f"  {seen:,} scanned | {len(found):,} matched")
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                continue
            desc = (g.get("description") or "").strip()
            if len(desc) < MIN_DESC_CHARS:
                continue
            ct = _core(g.get("title") or "")
            if not ct:
                continue
            auths = g.get("authors") or []
            aid = None
            for a in auths:
                if isinstance(a, dict) and a.get("author_id"):
                    aid = str(a["author_id"])
                    break
            if not aid:
                continue
            ca = author_key(amap.get(aid, ""))
            if not ca:
                continue
            key = (ct, ca)
            if key not in wanted:
                continue
            # several Goodreads editions can match one key; keep the longest
            prev = found.get(key)
            if prev is None or len(desc) > len(prev["desc"]):
                found[key] = {"desc": desc, "gr_title": g.get("title") or "",
                              "gr_book_id": g.get("book_id")}

    print(f"  {seen:,} scanned | {len(found):,} keys matched")

    out = {}
    for key, rows in wanted.items():
        hit = found.get(key)
        if not hit:
            continue
        for r in rows:
            if r["id"] is None:
                continue
            out[str(r["id"])] = {
                "title": r["title"], "author": r["author"],
                "users_count": r["users_count"],
                "desc": hit["desc"], "gr_title": hit["gr_title"],
                "gr_book_id": hit["gr_book_id"],
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}  ({len(out):,} books recovered)")
    report(wanted, out)


def report(wanted=None, out=None):
    if out is None:
        if not OUT.exists():
            sys.exit(f"{OUT} not found -- run --stage match first")
        out = json.loads(OUT.read_text(encoding="utf-8"))
    if wanted is None:
        wanted = load_wanted()

    total = sum(len(v) for v in wanted.values())
    print("\n" + "=" * 70)
    print("RECOVERY RATE")
    print("=" * 70)
    print(f"  wanted {total:,} | recovered {len(out):,} "
          f"({len(out)/max(total,1):.1%})")

    # Stratified, because a great rate on the head can hide a dead tail -- and
    # the tail is where "this book is missing" comes from.
    want_by, got_by = Counter(), Counter()
    for rows in wanted.values():
        for r in rows:
            for lo, hi in STRATA:
                if lo <= (r["users_count"] or 0) < hi:
                    want_by[(lo, hi)] += 1
                    break
    for v in out.values():
        for lo, hi in STRATA:
            if lo <= (v.get("users_count") or 0) < hi:
                got_by[(lo, hi)] += 1
                break

    print(f"\n  {'users_count':<18}{'wanted':>9}{'got':>9}{'rate':>8}")
    print("  " + "-" * 44)
    for lo, hi in STRATA:
        w = want_by[(lo, hi)]
        g = got_by[(lo, hi)]
        label = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        print(f"  {label:<18}{w:>9,}{g:>9,}{g/max(w,1):>7.0%}")

    lens = [len(v["desc"]) for v in out.values()]
    if lens:
        import statistics
        print(f"\n  description length: median {int(statistics.median(lens))}, "
              f"min {min(lens)}, max {max(lens)}")

    print("\n  READ THESE -- a publisher blurb about the AUTHOR embeds worse than")
    print("  no description at all, and the hit rate cannot tell you which it is:")
    for i, (bid, v) in enumerate(list(out.items())[:8]):
        print(f"\n    {v['title'][:56]}  ({v['author'][:28]})")
        print(f"      gr: {v['gr_title'][:56]}")
        print(f"      {v['desc'][:150].replace(chr(10), ' ')}...")

    print("\n" + "=" * 70)
    print("NEXT")
    print("=" * 70)
    print("  Spot-check the samples above BY EYE before merging any of it.")
    print("  Then merge as APPEND-ONLY rows carrying")
    print("  source='hardcover+goodreads_dump_2017', embed from the first new")
    print("  index onward, and re-run eval_goodbooks against 0.016.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["match", "report"], default="match")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the wanted set (pilot); the dump scan costs the same")
    args = ap.parse_args()
    if args.stage == "match":
        stage_match(args.limit)
    else:
        report()


if __name__ == "__main__":
    main()
