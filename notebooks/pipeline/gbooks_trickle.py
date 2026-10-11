"""
gbooks_trickle.py -- fetch Google Books blurbs for the books Hardcover has no
blurb for, a day's quota at a time, so the set is ready for the November update.

WHY
raw.jsonl holds ~27K books Hardcover catalogues but does not describe. They
were dropped at the description filter, so they can never be recommended.
check_gbooks.py (Sept 12) got real blurbs for 3 of 3 books that Hardcover and
Open Library had both failed on. The only limit is the free quota, ~1,000
requests a day -- so this runs once a day and resumes where it stopped.

WHAT IT DOES
  first run   builds gbooks_targets.jsonl: every raw.jsonl book with no usable
              description, an author name, and not already in the corpus,
              most-read first (so if the quota runs out, the books people
              actually read are done).
  every run   looks up the next --max targets, appends the RAW Google answers
              to gbooks_raw.jsonl, and stops cleanly at the daily limit.
  --report    progress, and hit rate by popularity band.

WHAT IT DOES NOT DO
Choose a blurb or touch the corpus. Choosing happens at merge time in
November, with the rules check_gbooks.py showed are needed: English only,
author must match, plot summary over literary commentary ("is a seminal work
in..."), no film entries. Keeping raw answers means those rules can be changed
later without spending quota again.

SAFE TO INTERRUPT. Every answer is written as it arrives; a re-run skips
anything already in gbooks_raw.jsonl.

USAGE (repo root)
    $env:GOOGLE_BOOKS_KEY = "your_key"
    python notebooks\\pipeline\\gbooks_trickle.py            # one day's batch
    python notebooks\\pipeline\\gbooks_trickle.py --report
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import requests

RAW_DIR = Path("data/raw/books/hardcover_v2")
HC_RAW = RAW_DIR / "raw.jsonl"
CORPUS = Path("data/processed/books_v2_hc3.parquet")
TARGETS = RAW_DIR / "gbooks_targets.jsonl"
OUT = RAW_DIR / "gbooks_raw.jsonl"

API = "https://www.googleapis.com/books/v1/volumes"
MIN_DESC_CHARS = 100          # same bar the harvest used
STRATA = [(0, 25), (25, 100), (100, 500), (500, 2000), (2000, 10 ** 9)]


def first_author(b):
    for c in (b.get("contributions") or []):
        a = c.get("author") if isinstance(c, dict) else None
        if isinstance(a, dict) and a.get("name"):
            return a["name"]
    return ""


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def _core(title):
    """Title without series parentheticals, subtitle or leading article --
    the same key recover_descriptions_bulk.py matches on."""
    t = re.sub(r"\([^)]*\)", " ", str(title))
    t = re.split(r":| - ", t)[0]
    return re.sub(r"^(the|a|an) ", "", _norm(t)).strip()


def _akey(name):
    toks = _norm(name).split()
    return " ".join(sorted(toks))


def book_key(title, author):
    ct, ca = _core(title), _akey(author)
    return (ct, ca) if ct and ca else None


def build_targets():
    """Every raw.jsonl book that is NOT already servable, deduplicated.

    Three cross-references, so quota is only spent on books that are really
    missing:
      1. same Hardcover id already in the corpus
      2. same title + author already in the corpus under ANOTHER Hardcover
         record (Hardcover keeps several records per book -- editions,
         duplicates -- so one copy can be live with a blurb while a bare
         copy sits in raw.jsonl)
      3. the same title + author queued twice -> keep the most-read copy
    """
    if not HC_RAW.exists():
        sys.exit(f"{HC_RAW} not found -- run from the repo root")
    have_ids, have_keys = set(), set()
    if CORPUS.exists():
        import pandas as pd
        c = pd.read_parquet(CORPUS, columns=["book_id", "title", "primary_author"])
        have_ids = set(c["book_id"].astype(str))
        have_keys = {k for k in (book_key(t, a) for t, a in
                                 zip(c["title"], c["primary_author"])) if k}
        print(f"corpus: {len(have_ids):,} books, {len(have_keys):,} title+author keys")
    else:
        print(f"WARNING: {CORPUS} not found -- cannot cross-reference; every "
              f"bare book will be queued")

    skip = Counter()
    best = {}                       # key -> row (most-read copy wins)
    with HC_RAW.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                continue
            bid = str(b.get("id"))
            if bid in have_ids:
                skip["already in corpus (same id)"] += 1
                continue
            if len((b.get("description") or "").strip()) >= MIN_DESC_CHARS:
                skip["has a Hardcover blurb"] += 1
                continue
            title, author = (b.get("title") or "").strip(), first_author(b)
            k = book_key(title, author)
            if not k:
                skip["no title or author"] += 1
                continue
            if k in have_keys:
                skip["already in corpus (other record)"] += 1
                continue
            row = {"id": bid, "title": title, "author": author,
                   "users_count": int(b.get("users_count") or 0)}
            if k in best:
                skip["duplicate record in queue"] += 1
                if row["users_count"] <= best[k]["users_count"]:
                    continue
            best[k] = row

    rows = sorted(best.values(), key=lambda r: -r["users_count"])
    with TARGETS.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("skipped:")
    for why, n in skip.most_common():
        print(f"  {n:>8,}  {why}")
    print(f"built {TARGETS}: {len(rows):,} books to look up (most-read first)")
    done = {r["id"] for r in load_jsonl(OUT)}
    wasted = len(done - {r["id"] for r in rows})
    if wasted:
        print(f"  ({wasted:,} already-fetched books are no longer targets -- "
              f"kept in {OUT.name}, just not counted)")


def load_jsonl(path):
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass      # a line cut off by Ctrl+C; it will be redone
    return out


def _clean_title(title):
    """Search form of a title: no series brackets, no subtitle. Keeps case."""
    t = re.sub(r"\([^)]*\)", " ", str(title))
    t = re.split(r":| - ", t)[0]
    return re.sub(r"\s+", " ", t).strip()


def _surname(author):
    toks = [w for w in re.split(r"[\s.]+", str(author)) if len(w) > 1]
    return toks[-1] if toks else str(author)


def query_variants(title, author):
    """Strictest first. The v1 query used the full title and full author in
    quotes, and returned 0 volumes for The Silver Chair / C. S. Lewis -- an
    exact-phrase author match is brittle ("C. S. Lewis" vs "C.S. Lewis"), so
    every variant matches on SURNAME, and only falls back to looser forms when
    the stricter one comes back empty."""
    # The quoted form intitle:"..." returned 0 volumes for The Silver Chair
    # in the Oct 10 probe (it worked on Sept 12) -- Google's exact-phrase title
    # search is unreliable, so it is not used. Unquoted title + surname found
    # it first try; the plain-text form is the fallback.
    # Oct 10 first real batch: 399 books took 657 calls -- the intitle form
    # missed ~65% of the time and the plain-text form found them. So plain
    # text goes first (one call for most books) and intitle is the fallback.
    # Plain text can return other books (The Chronicles of Narnia for The
    # Silver Chair), so hits are counted only when the AUTHOR matches.
    ct, sn = _clean_title(title), _surname(author)
    return [f'{ct} {author}',
            f'intitle:{ct} inauthor:{sn}']


RETRY_WAITS = (15, 30, 60, 120)     # seconds; ~4 minutes total before giving up


def call(q, key, retry=True):
    """One search. 10 results, not 5: same quota cost, and the plain-text query
    needs the room because the right edition is not always in the top five.

    500/502/503/504 are Google being temporarily unavailable (Oct 11: five 503s
    in a row stopped the whole run). Those are retried after a wait instead of
    counted as errors. A 429 is NOT retried -- that is the daily quota."""
    r = requests.get(API, params={"q": q, "key": key, "maxResults": 10},
                     timeout=20)
    if not retry:
        return r
    for wait in RETRY_WAITS:
        if r.status_code not in (500, 502, 503, 504):
            return r
        print(f"  HTTP {r.status_code} (Google busy) -- waiting {wait}s and retrying")
        time.sleep(wait)
        r = requests.get(API, params={"q": q, "key": key, "maxResults": 10},
                         timeout=20)
    return r


# A book Google certainly has (found on Sept 12 and Oct 10). If THIS comes back empty, the
# API is returning silent empty results (throttling without a 429), and every
# "0 volumes" in that window is fake.
CANARY = ("The Maltese Falcon", "Dashiell Hammett")


def canary_ok(key):
    try:
        r = call(query_variants(*CANARY)[0], key)
    except requests.RequestException:
        return False, "network error"
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    n = len(r.json().get("items") or [])
    return n > 0, f"{n} volumes"


def probe(title, author):
    """Print what each query form returns for one book. ~4 calls."""
    key = os.environ.get("GOOGLE_BOOKS_KEY")
    print(f"key: {'set, ends ...' + key[-6:] if key else 'NOT SET'}\n")
    tries = [(q, key) for q in query_variants(title, author)]
    tries.insert(0, (f'intitle:"{title}" inauthor:"{author}"', key))   # the v1 form
    for q, k in tries:
        try:
            r = call(q, k)
        except requests.RequestException as e:
            print(f"{q}\n  network error {e}\n")
            continue
        body = r.json() if r.status_code == 200 else {}
        items = body.get("items") or []
        print(f"{q}\n  HTTP {r.status_code} | totalItems {body.get('totalItems')} | "
              f"{len(items)} returned")
        if r.status_code != 200:
            print(f"  {r.text[:300]}")
        for it in items[:3]:
            vi = it.get("volumeInfo", {})
            print(f"    {str(vi.get('title'))[:45]:47s} {str((vi.get('authors') or ['?'])[0])[:20]:22s}"
                  f" {vi.get('language')} {len(vi.get('description') or '')} chars")
        print()
        time.sleep(1)


def is_hit(row):
    """An English blurb of 100+ chars from a volume whose author list contains
    the target's surname -- so a plain-text search that returned a different
    book does not count."""
    sn = _norm(_surname(row.get("author", "")))
    for i in row.get("items") or []:
        if i.get("language") != "en":
            continue
        if len(i.get("description") or "") < MIN_DESC_CHARS:
            continue
        if sn and any(sn in _norm(a) for a in (i.get("authors") or [])):
            return True
    return False


def slim(item):
    vi = item.get("volumeInfo", {})
    return {"gid": item.get("id"), "title": vi.get("title"),
            "subtitle": vi.get("subtitle"), "authors": vi.get("authors"),
            "language": vi.get("language"),
            "publishedDate": vi.get("publishedDate"),
            "categories": vi.get("categories"),
            "pageCount": vi.get("pageCount"),
            "description": vi.get("description")}


def run(max_n, pause):
    key = os.environ.get("GOOGLE_BOOKS_KEY")
    if not key:
        sys.exit('GOOGLE_BOOKS_KEY not set. Run: $env:GOOGLE_BOOKS_KEY = "your_key"')
    if not TARGETS.exists():
        build_targets()
    ok, why = canary_ok(key)
    if not ok:
        sys.exit(f"CANARY FAILED ({why}): Google is not returning results for a "
                 f"book it certainly has. Not spending quota. Try later, or run "
                 f"--probe to see the raw answers.")
    targets = load_jsonl(TARGETS)
    tids = {t["id"] for t in targets}
    # only v2 rows count as done: v1 rows (full-title exact-phrase query)
    # returned 0 volumes for books Google certainly has, so they are redone
    done = {r["id"] for r in load_jsonl(OUT) if r.get("v") == 2} & tids
    todo = [t for t in targets if t["id"] not in done]
    print(f"{len(done):,} done | {len(todo):,} left | this budget: {max_n} calls")

    calls = n = hits = errs = 0
    empties = []          # held back until the next canary passes
    stop = False
    with OUT.open("a", encoding="utf-8") as fh:
        for t in todo:
            if calls >= max_n or stop:
                break
            items, used = [], []
            for q in query_variants(t["title"], t["author"]):
                if calls >= max_n:
                    break
                try:
                    r = call(q, key)
                except requests.RequestException as e:
                    errs += 1
                    print(f"  network error ({type(e).__name__})")
                    if errs >= 5:
                        stop = True
                    time.sleep(5)
                    break
                calls += 1
                used.append(q)
                if r.status_code in (429, 403):
                    print(f"\nHTTP {r.status_code} -- daily quota probably used up.")
                    print(f"  {r.text[:300]}")
                    stop = True
                    break
                if r.status_code in (500, 502, 503, 504):
                    print(f"\nGoogle still unavailable after ~4 minutes of retries "
                          f"(HTTP {r.status_code}). Stopping; nothing is lost -- "
                          f"run again later today.")
                    stop = True
                    break
                if r.status_code != 200:
                    errs += 1
                    print(f"  HTTP {r.status_code}: {r.text[:120].strip()}")
                    if errs >= 5:
                        stop = True
                    break
                got = [slim(i) for i in (r.json().get("items") or [])]
                seen = {i.get("gid") for i in items}
                items += [i for i in got if i.get("gid") not in seen]
                time.sleep(pause)
                # QUALITY OVER QUOTA: stop only when there is a GOOD candidate
                # (English, 100+ chars, right author). A result that is merely
                # non-empty -- other editions, other books -- earns the next
                # search, and candidates from both are kept for the merge.
                if is_hit({"author": t["author"], "items": items}):
                    break
            if stop and not is_hit({"author": t["author"], "items": items}):
                break     # budget hit mid-book: redo it fully next time
            if (len(used) < len(query_variants(t["title"], t["author"]))
                    and not is_hit({"author": t["author"], "items": items})):
                continue      # ran out of budget mid-book; redo it next time
            row = {**t, "v": 2, "queries": used, "items": items}
            if items:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                hits += is_hit(row)
            else:
                empties.append(row)
            n += 1
            if n % 50 == 0:
                ok, why = canary_ok(key)
                calls += 1
                if not ok:
                    print(f"\nCANARY FAILED after {n} books ({why}). Discarding "
                          f"{len(empties)} 'no result' answers from this window as "
                          f"untrustworthy and stopping.")
                    empties = []
                    break
                for e in empties:
                    fh.write(json.dumps(e, ensure_ascii=False) + "\n")
                fh.flush()
                empties = []
                print(f"  {n} books | {calls} calls | {hits} with an English blurb")
        ok, _ = canary_ok(key) if empties else (True, "")
        if ok:
            for e in empties:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\nthis run: {n} books, {calls} calls, {hits} with an English blurb "
          f"({hits / max(n, 1):.0%})")
    left = len(todo) - n
    print(f"left {left:,} (~{-(-left // max(max_n, 1))}+ more days)")


def report():
    allrows = load_jsonl(OUT)
    rows = [r for r in allrows if r.get("v") == 2]
    targets = load_jsonl(TARGETS)
    if len(allrows) > len(rows):
        print(f"({len(allrows) - len(rows):,} old v1 lookups ignored -- they are redone)")
    if not rows:
        sys.exit("no v2 lookups yet")
    print(f"{len(rows):,} of {len(targets):,} looked up\n")
    tot, hit = Counter(), Counter()
    for r in rows:
        u = r.get("users_count", 0)
        band = next(f"{lo}-{hi if hi < 10**9 else '+'}" for lo, hi in STRATA
                    if lo <= u < hi)
        tot[band] += 1
        hit[band] += is_hit(r)
    print(f"  {'readers':<12}{'looked up':>10}{'English blurb':>16}")
    for lo, hi in STRATA:
        b = f"{lo}-{hi if hi < 10**9 else '+'}"
        if tot[b]:
            print(f"  {b:<12}{tot[b]:>10,}{hit[b]:>10,} ({hit[b] / tot[b]:.0%})")
    q = [len(r.get("queries") or []) for r in rows]
    if q:
        print(f"\n  calls per book: {sum(q) / len(q):.2f} "
              f"(1.0 = every book found on the first search)")
    allh = sum(hit.values())
    print(f"\n  overall: {allh:,} / {len(rows):,} ({allh / len(rows):.0%}) have a candidate")
    print("  'candidate' = an English result with 100+ characters whose author")
    print("  matches the book's author (surname).")
    print("  Whether it is the RIGHT book and a real plot blurb is decided at")
    print("  merge time -- this number is an upper bound, not the final count.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=950,
                    help="lookups this run (default 950, under the ~1,000/day quota)")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between calls")
    ap.add_argument("--probe", nargs=2, metavar=("TITLE", "AUTHOR"),
                    help="show what each query form returns for one book (~4 calls)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rebuild-targets", action="store_true",
                    help="rebuild the target list (keeps everything already fetched)")
    args = ap.parse_args()
    if args.probe:
        probe(*args.probe)
        return
    if args.rebuild_targets:
        build_targets()
        return
    if args.report:
        report()
    else:
        run(args.max, args.pause)


if __name__ == "__main__":
    main()
