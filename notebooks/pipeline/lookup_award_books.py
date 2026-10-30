"""
lookup_award_books.py -- find the award books that are not in the catalogue
yet: Hardcover first (ratings, tags, cover, blurb), Google Books for the blurb
when Hardcover has none or does not have the book at all.

    python notebooks\\pipeline\\lookup_award_books.py --stage hardcover
    python notebooks\\pipeline\\lookup_award_books.py --stage gbooks      (once a day)
    python notebooks\\pipeline\\lookup_award_books.py --stage report

INPUT   data/raw/books/awards/award_targets.jsonl   (build_award_list.py --stage match)

STAGES
  hardcover  For each target: exact-title search, then the author's books
             (cached per author -- many award authors have several books on
             the list), matched with the same title/author rules as
             audit_gbooks.py. No daily cap; ~1 request a second. Resumable.
  gbooks     For targets with no usable Hardcover blurb: Google Books, same
             queries as gbooks_trickle.py, stopping at --max-calls (default
             950, the free daily quota). Run it once a day until "left 0".
             Do not run it on a day the trickle still runs -- they share the
             quota. Resumable.
  report     Picks one blurb per book (Hardcover if usable, else the cleaned
             Google one, chosen by audit_gbooks.py's rules), prints what came
             out by award group, and writes award_lookup.jsonl -- the rows the
             November merge adds, with source "award_books".

OUTPUT (data/raw/books/awards/)
  award_hc.jsonl          one row per target: Hardcover match or none
  award_hc_authors.jsonl  cached author lookups
  award_gbooks_raw.jsonl  raw Google answers (slim), same shape as gbooks_raw
  award_lookup.jsonl      merge-ready rows (report stage)

NOTE FOR THE MERGE: app.py exempts source == "targeted_authors" from the
10-rating floor (recommend_books and recommend_books_from_watching) and
src/space_ranker.py's _servable_mask does the same. Award books need the same
exemption -- most have few Hardcover readers -- so widen those three checks to
{"targeted_authors", "award_books"} in the same change that adds the rows.

Keys: $env:HARDCOVER_TOKEN (as the other Hardcover scripts) and
$env:GOOGLE_BOOKS_KEY (as gbooks_trickle.py).
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_gbooks as ag  # noqa: E402  (same folder: title/author/blurb rules)

AW = Path("data/raw/books/awards")
TARGETS = AW / "award_targets.jsonl"
HC_OUT = AW / "award_hc.jsonl"
HC_AUTH = AW / "award_hc_authors.jsonl"
GB_OUT = AW / "award_gbooks_raw.jsonl"
LOOKUP = AW / "award_lookup.jsonl"
CORPUS = Path("data/processed/books_v2_hc3.parquet")

HC_API = "https://api.hardcover.app/v1/graphql"
GB_API = "https://www.googleapis.com/books/v1/volumes"
MIN_CHARS = ag.MIN_CHARS

BOOK_FIELDS = """
        id
        slug
        title
        subtitle
        description
        release_date
        pages
        rating
        ratings_count
        users_count
        cached_tags
        cached_image
        contributions { author { name } }"""
TITLE_QUERY = """
query ByTitle($title: String!) {
  books(where: {title: {_eq: $title}}, order_by: {users_count: desc}, limit: 10) {%s
  }
}""" % BOOK_FIELDS
AUTHOR_QUERY = """
query ByAuthor($name: String!) {
  authors(where: {name: {_eq: $name}}, limit: 3) {
    name
    contributions(limit: 200) {
      book {%s
      }
    }
  }
}""" % BOOK_FIELDS


# ---------------------------------------------------------------- shared
def key_of(t):
    return t.get("wikidata") or f"{ag.norm(t['title'])}|{ag.norm(t['author'])}"


def load_jsonl(path):
    return ag.load_jsonl(path)


def append(path, row):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def targets():
    if not TARGETS.exists():
        sys.exit(f"{TARGETS} not found -- run build_award_list.py --stage match first "
                 "(from the repo root)")
    return load_jsonl(TARGETS)


def corpus_ids():
    if not CORPUS.exists():
        return set()
    import pandas as pd
    ids = pd.read_parquet(CORPUS, columns=["book_id"])["book_id"].astype(str)
    return {re.sub(r"^hc_", "", i) for i in ids}


def corpus_by_surname():
    """surname -> [catalogue titles]. For the looser duplicate check: the award
    list's own check compares the title before any colon plus the full author
    key, so "Babel" (R.F. Kuang, on Wikidata) missed "Babel, or The Necessity
    of Violence" by "R.F. Kuang" already in the catalogue."""
    if not CORPUS.exists():
        return {}
    import pandas as pd
    c = pd.read_parquet(CORPUS, columns=["title", "primary_author"])
    out = defaultdict(list)
    for t, a in zip(c["title"].astype(str), c["primary_author"].astype(str)):
        sn = ag.surname(a)
        if sn:
            out[sn].append(t)
    return out


def in_catalogue(by_sn, titles, author):
    """Same surname + any title matching by the audit rules, or one title
    starting the other ("Babel" / "Babel, or The Necessity of Violence")."""
    have = by_sn.get(ag.surname(author), [])
    for t in titles:
        if not t:
            continue
        tf, _, _ = ag.title_parts(t)
        for ct in have:
            cf, _, _ = ag.title_parts(ct)
            if ag.title_match(t, {"title": ct}) or ag.title_match(ct, {"title": t}):
                return ct
            short, long_ = sorted((tf, cf), key=len)
            if len(short.split()) >= 1 and len(short) >= 4 and long_.startswith(short + " "):
                if re.match(r"(or|and|a|an|the)\b", long_[len(short):].strip()) or "," in ct:
                    return ct
    return None


def credited_author(book, target_author):
    """Hardcover lists every contributor, narrators and translators included,
    and not always the writer first ('Song of Time' came back under its
    audiobook narrator). Take the one whose surname matches the award record."""
    sn = ag.surname(target_author)
    for name in hc_authors(book):
        if sn and sn in ag.norm(name):
            return name
    return target_author


def hc_authors(book):
    return [c["author"]["name"] for c in (book.get("contributions") or [])
            if isinstance(c, dict) and isinstance(c.get("author"), dict) and c["author"].get("name")]


def hc_match(t, books):
    """Best Hardcover book for target t: author surname on the record, title
    matches by audit_gbooks rules; most readers wins among several."""
    sn = ag.surname(t["author"])
    ok = []
    for b in books:
        names = " ".join(ag.norm(a) for a in hc_authors(b))
        if sn and sn not in names:
            continue
        if ag.title_match(t["title"], {"title": b.get("title"), "subtitle": b.get("subtitle")}):
            ok.append(b)
    ok.sort(key=lambda b: -(b.get("users_count") or 0))
    return ok[0] if ok else None


def clean_blurb(desc):
    """Hardcover or Google text -> (cleaned text, None) or (None, why)."""
    if not desc:
        return None, "none"
    t = ag.strip_reviews(desc)
    if len(t) < MIN_CHARS:
        return None, "too short"
    why = ag.junk_reason(t)
    return (None, why) if why else (t, None)


# ---------------------------------------------------------------- hardcover
def hc_headers():
    tok = os.environ.get("HARDCOVER_TOKEN")
    if not tok:
        sys.exit("Set HARDCOVER_TOKEN first: $env:HARDCOVER_TOKEN = 'your_token'")
    return {"authorization": tok, "Content-Type": "application/json"}


class HardcoverDown(Exception):
    """Every retry failed. The book is NOT recorded, so a network outage cannot
    turn into thousands of false 'not on Hardcover' rows."""


def hc_query(query, variables, pause):
    """Network errors: 5 tries over ~75s. Rate limits (HTTP 429): Hardcover's
    Retry-After if it sends one, else 1, 2, 4, 8 minutes -- the Oct 21 run hit a
    limit after ~3,700 books that 30-150s waits did not outlast."""
    net = limited = 0
    while True:
        try:
            r = requests.post(HC_API, headers=hc_headers(),
                              json={"query": query, "variables": variables}, timeout=60)
        except requests.RequestException as e:
            net += 1
            if net > 5:
                raise HardcoverDown("network")
            wait = 5 * net
            print(f"    network error ({type(e).__name__}), retry in {wait}s")
            time.sleep(wait)
            continue
        time.sleep(pause)
        if r.status_code == 429:
            limited += 1
            if limited > 4:
                raise HardcoverDown("rate limit")
            try:
                wait = int(r.headers.get("Retry-After") or 0)
            except ValueError:
                wait = 0
            wait = max(wait, 60 * 2 ** (limited - 1))
            print(f"    Hardcover rate limit, waiting {wait // 60} min {wait % 60}s")
            time.sleep(wait)
            continue
        if r.status_code in (401, 403):
            sys.exit(f"Hardcover refused the token (HTTP {r.status_code}). Check HARDCOVER_TOKEN.")
        if r.status_code >= 500:
            net += 1
            if net > 5:
                raise HardcoverDown("server errors")
            time.sleep(10 * net)
            continue
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            print("    GraphQL errors:", json.dumps(data["errors"])[:300])
            return None
        return data.get("data") or {}


def drop_unreliable_none():
    """--retry-none: forget every 'not on Hardcover' row and every empty author
    lookup, so they are asked again. For rows written during an outage by the
    first version of this script, which recorded failed requests as misses."""
    rows = load_jsonl(HC_OUT)
    keep = [r for r in rows if r["status"] == "found"]
    HC_OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep),
                      encoding="utf-8")
    auth = load_jsonl(HC_AUTH)
    akeep = [a for a in auth if a["books"]]
    HC_AUTH.write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in akeep),
                       encoding="utf-8")
    print(f"--retry-none: {len(rows) - len(keep):,} 'none' rows and "
          f"{len(auth) - len(akeep):,} empty author lookups will be asked again")


def stage_hardcover(pause, limit, retry_none=False):
    if retry_none:
        drop_unreliable_none()
    try:
        _stage_hardcover(pause, limit)
    except HardcoverDown as e:
        if str(e) == "rate limit":
            print("\nHardcover kept rate-limiting for ~15 min. Stopped cleanly; nothing was "
                  "recorded for the book in progress.\nWait an hour, then run the same "
                  "command with --pause 2 -- it resumes where it stopped.")
        else:
            print(f"\nHardcover is not answering ({e}). Stopped cleanly; nothing was recorded "
                  "for the book in progress.\nCheck https://hardcover.app opens in a browser, "
                  "then run the same command again -- it resumes.")


def _stage_hardcover(pause, limit):
    tg = targets()
    done = {r["key"] for r in load_jsonl(HC_OUT)}
    acache = {r["name"]: r["books"] for r in load_jsonl(HC_AUTH)}
    todo = [t for t in tg if key_of(t) not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(tg):,} targets, {len(done):,} already looked up, {len(todo):,} to go "
          f"(~{len(todo) * 1.6 * pause / 60:.0f} min)")
    found = Counter()
    for n, t in enumerate(todo, 1):
        hit, via = None, ""
        titles = list(dict.fromkeys([t["title"], re.split(r":| - ", t["title"])[0].strip()]))
        for title in titles:
            d = hc_query(TITLE_QUERY, {"title": title}, pause)
            if d is None:
                continue
            hit = hc_match(t, d.get("books") or [])
            if hit:
                via = "title"
                break
        if not hit and t["author"]:
            name = t["author"]
            if name not in acache:
                books = []
                # "J. M. Coetzee" on Wikidata can be "J.M. Coetzee" on Hardcover
                for v in dict.fromkeys([name, re.sub(r"\.\s+(?=[A-Z]\.)", ".", name),
                                        re.sub(r"\.(?=[A-Z])", ". ", name)]):
                    d = hc_query(AUTHOR_QUERY, {"name": v}, pause)
                    for a in (d or {}).get("authors") or []:
                        books += [c["book"] for c in a.get("contributions") or [] if c.get("book")]
                    if books:
                        break
                acache[name] = books
                append(HC_AUTH, {"name": name, "books": books})
            hit = hc_match(t, acache[name])
            via = "author" if hit else ""
        append(HC_OUT, {"key": key_of(t), "title": t["title"], "author": t["author"],
                        "status": "found" if hit else "none", "via": via, "book": hit})
        found[via or "none"] += 1
        if n % 50 == 0 or n == len(todo):
            print(f"  {n:,}/{len(todo):,}  found by title {found['title']:,}, "
                  f"by author {found['author']:,}, not on Hardcover {found['none']:,}")
    print("\nnext: --stage gbooks (once the gbooks trickle has finished), then --stage report")


# ---------------------------------------------------------------- gbooks
def gb_variants(title, author):
    # same order as gbooks_trickle.py after the Oct 10 probe: plain text first,
    # then title + surname; the quoted exact-phrase form is unreliable
    ct, sn = ag.main_title(title) or title, ag.surname(author)
    return list(dict.fromkeys([f"{title} {author}", f"intitle:{ct} inauthor:{sn}"]))


def slim(item):
    vi = item.get("volumeInfo") or {}
    return {"gid": item.get("id"), "title": vi.get("title"), "subtitle": vi.get("subtitle"),
            "authors": vi.get("authors") or [], "language": vi.get("language"),
            "publishedDate": vi.get("publishedDate"), "categories": vi.get("categories") or [],
            "pageCount": vi.get("pageCount"), "description": vi.get("description") or ""}


class QuotaDone(Exception):
    pass


def gb_call(q, key):
    for attempt in range(4):
        try:
            r = requests.get(GB_API, params={"q": q, "key": key, "maxResults": 10}, timeout=20)
        except requests.RequestException:
            time.sleep(10)
            continue
        if r.status_code == 200:
            return [slim(i) for i in r.json().get("items") or []]
        if r.status_code == 429 or (r.status_code == 403 and "quota" in r.text.lower()):
            raise QuotaDone()
        if r.status_code in (500, 502, 503):
            print(f"  HTTP {r.status_code} (Google busy) -- waiting 15s and retrying")
            time.sleep(15)
            continue
        r.raise_for_status()
    return None      # gave up: not recorded, so it is retried next run


def needs_google(tg):
    """Targets whose Hardcover match is missing or has no usable blurb."""
    hc = {r["key"]: r for r in load_jsonl(HC_OUT)}
    out = []
    for t in tg:
        r = hc.get(key_of(t))
        if r is None:
            continue                     # Hardcover stage has not reached it yet
        if r["status"] == "found" and clean_blurb((r["book"] or {}).get("description"))[0]:
            continue
        out.append(t)
    return out


def stage_gbooks(max_calls, pause):
    key = os.environ.get("GOOGLE_BOOKS_KEY")
    if not key:
        sys.exit("GOOGLE_BOOKS_KEY not set: $env:GOOGLE_BOOKS_KEY = 'your_key'")
    tg = targets()
    want = needs_google(tg)
    done = {r["key"] for r in load_jsonl(GB_OUT)}
    todo = [t for t in want if key_of(t) not in done]
    print(f"{len(want):,} books need Google, {len(want) - len(todo):,} done, {len(todo):,} to go")
    if not todo:
        print("nothing left for Google -- run --stage report")
        return
    # canary, as in the trickle: a book Google certainly has. Empty = silent throttling.
    try:
        canary = gb_call("The Maltese Falcon Dashiell Hammett", key)
    except QuotaDone:
        sys.exit("Google says today's quota is used up -- run again tomorrow.")
    calls = 1
    if not canary:
        sys.exit("Canary search came back empty: Google is returning silent empties. Try later.")
    books = hits = 0
    try:
        for t in todo:
            variants = gb_variants(t["title"], t["author"])
            if calls + len(variants) > max_calls:
                break                     # not enough calls left for this book today
            used, items = [], []
            for q in variants:
                res = gb_call(q, key)
                calls += 1
                used.append(q)
                time.sleep(pause)
                if res is None:
                    items = None
                    break
                items += res
                if ag.classify({"title": t["title"], "author": t["author"], "items": items})[0] == "usable":
                    break
            if items is None:
                continue                  # Google kept failing: not recorded, retried next run
            append(GB_OUT, {"key": key_of(t), "title": t["title"], "author": t["author"],
                            "queries": used, "items": items})
            books += 1
            hits += ag.classify({"title": t["title"], "author": t["author"], "items": items})[0] == "usable"
            if books % 50 == 0:
                print(f"  {books:,} books | {calls:,} calls | {hits:,} usable blurbs")
    except QuotaDone:
        print("  Google quota reached for today")
    left = len(todo) - books
    print(f"\nthis run: {books:,} books, {calls:,} calls, {hits:,} usable "
          f"({ag.pct(hits, books)})\nleft {left:,}"
          + (f" (~{-(-left * 13 // 10 // max_calls)} more day(s), at ~1.3 calls a book)"
             if left else " -- run --stage report"))


# ---------------------------------------------------------------- report
def year_of(b, t):
    m = re.match(r"\d{4}", str((b or {}).get("release_date") or ""))
    return int(m.group()) if m else (int(t["year"]) if str(t.get("year") or "").isdigit() else None)


def stage_report():
    tg = targets()
    hc = {r["key"]: r for r in load_jsonl(HC_OUT)}
    gb = {r["key"]: r for r in load_jsonl(GB_OUT)}
    have = corpus_ids()
    by_sn = corpus_by_surname()
    if not hc:
        sys.exit("run --stage hardcover first")

    rows, outcome, hc_why, dups = [], {}, Counter(), []
    for t in tg:
        k = key_of(t)
        h = hc.get(k)
        if h is None:
            outcome[k] = "not looked up yet"
            continue
        book = h["book"] if h["status"] == "found" else None
        if book and str(book.get("id")) in have:
            outcome[k] = "already in catalogue (other title)"
            continue
        dup = in_catalogue(by_sn, [t["title"], (book or {}).get("title")],
                           credited_author(book, t["author"]) if book else t["author"])
        if dup:
            outcome[k] = "already in catalogue (other title)"
            dups.append((t["title"], dup))
            continue
        blurb, src = None, None
        if book:
            blurb, why = clean_blurb(book.get("description"))
            if not blurb:
                hc_why[why] += 1
            src = "hardcover" if blurb else None
        gitem = None
        if not blurb and k in gb:
            o, gitem, _ = ag.classify({"title": t["title"], "author": t["author"],
                                       "items": gb[k]["items"]})
            if o == "usable":
                blurb, src = gitem["description"], "google"
        if not blurb:
            outcome[k] = ("waiting for Google" if k not in gb else
                          "on Hardcover, no blurb" if book else "not found anywhere")
            continue
        outcome[k] = ("Hardcover + its blurb" if src == "hardcover" else
                      "Hardcover + Google blurb" if book else "Google only")
        rows.append({
            "book_id": str(book["id"]) if book else f"gb_{gitem['gid']}",
            "source": "award_books",
            "title": (book or {}).get("title") or t["title"],
            "subtitle": (book or {}).get("subtitle") or (gitem or {}).get("subtitle"),
            "primary_author": credited_author(book, t["author"]) if book else t["author"],
            "description": blurb, "description_source": src,
            "publication_year": year_of(book, t),
            "num_pages": (book or {}).get("pages") or (gitem or {}).get("pageCount"),
            "average_rating": (book or {}).get("rating"),
            "ratings_count": (book or {}).get("ratings_count") or 0,
            "users_count": (book or {}).get("users_count") or 0,
            "cached_tags": (book or {}).get("cached_tags"),
            "cached_image": (book or {}).get("cached_image"),
            "slug": (book or {}).get("slug"),
            "google_categories": (gitem or {}).get("categories") or [],
            "awards": t["awards"], "award_groups": t["groups"], "award_won": t["won"],
            "wikidata": t.get("wikidata") or "", "original_language": t.get("language") or "",
        })

    with LOOKUP.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    order = ["Hardcover + its blurb", "Hardcover + Google blurb", "Google only",
             "on Hardcover, no blurb", "not found anywhere", "already in catalogue (other title)",
             "waiting for Google", "not looked up yet"]
    c = Counter(outcome.values())
    n = len(tg)
    print(f"{n:,} award books not in the catalogue\n")
    for o in order:
        if c[o]:
            print(f"  {o:<38}{c[o]:7,}  {ag.pct(c[o], n):>4}")
    if hc_why:
        print("  (found on Hardcover but its blurb unusable: "
              + ", ".join(f"{k} {v:,}" for k, v in hc_why.most_common()) + ")")
    if dups:
        print(f"  ({len(dups):,} caught by the looser title check, e.g. "
              + "; ".join(f"{a} = {b}" for a, b in dups[:4]) + ")")
    print(f"\n  ready to add: {len(rows):,}  ({ag.pct(len(rows), n)})  -> {LOOKUP}")

    per = defaultdict(lambda: [0, 0, 0])
    for t in tg:
        k = key_of(t)
        ok = outcome.get(k, "").startswith(("Hardcover +", "Google only"))
        for g in t["groups"]:
            per[g][0] += 1
            per[g][1] += ok
        if ok and t["won"]:
            for g in t["groups"]:
                per[g][2] += 1
    print(f"\n  {'group':<14}{'queued':>8}{'ready':>8}{'winners':>9}")
    for g, (q, r, w) in sorted(per.items(), key=lambda x: -x[1][1]):
        print(f"  {g:<14}{q:>8,}{r:>8,}{w:>9,}  {ag.pct(r, q)}")

    if rows:
        src = Counter(r["description_source"] for r in rows)
        rc = sorted(r["ratings_count"] for r in rows)
        print(f"\n  blurbs: Hardcover {src['hardcover']:,}, Google {src['google']:,}")
        print(f"  Hardcover ratings: median {rc[len(rc) // 2]}, "
              f"{sum(1 for x in rc if x < 10):,} under 10 (need the award_books floor "
              "exemption, see the note at the top)")
        print("\n  sample:")
        for r in rows[:15]:
            print(f"    {str(r['publication_year'] or '----')}  {r['title'][:40]:42s} "
                  f"{r['primary_author'][:22]:24s} {r['description_source']:<9} {r['awards'][0][:34]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["hardcover", "gbooks", "report"], required=True)
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--limit", type=int, default=0, help="hardcover: stop after this many")
    ap.add_argument("--retry-none", action="store_true",
                    help="hardcover: ask again about every book recorded as not found")
    ap.add_argument("--max-calls", type=int, default=950, help="gbooks: daily call budget")
    args = ap.parse_args()
    AW.mkdir(parents=True, exist_ok=True)
    if args.stage == "hardcover":
        stage_hardcover(args.pause, args.limit, args.retry_none)
    elif args.stage == "gbooks":
        stage_gbooks(args.max_calls, args.pause)
    else:
        stage_report()


if __name__ == "__main__":
    main()
