"""
probe_description_sources.py — can Google Books or Open Library supply the
descriptions Hardcover does not have?

WHY
fetch_hardcover_targeted_v2.py was built on a hypothesis: that Hardcover HAS a
described record for these books, just not the most-read one, so inverting the
edition selection would recover them. Running it falsified that for 19 of 27
works. The Maltese Falcon -- the example the whole theory was built on -- has
three records and NONE carries a description. Same for Shikhandi, My Gita,
Myth = Mithya, 7 Secrets of Shiva and 13 of 16 S. Hussain Zaidi titles.

So the gap is not edition selection. It is SOURCE COVERAGE, and no amount of
harvesting Hardcover fixes it. The question becomes whether a different source
has what Hardcover lacks.

Goodreads is not an option: developer keys stopped being issued in December
2020 and the API was retired.

PILOT, NOT A FETCHER. This measures hit rate on the known-missing titles and
prints what it finds so the descriptions can be judged by eye. Building a full
enrichment path before knowing the hit rate is the mistake the keyword pilot
already taught -- and that pilot was wrong because it sampled the head of a
power-law distribution, so this deliberately probes the TAIL: the exact titles
that failed.

USAGE
    python notebooks\\pipeline\\probe_description_sources.py
    python notebooks\\pipeline\\probe_description_sources.py --source google
    python notebooks\\pipeline\\probe_description_sources.py --limit 5

Optional: $env:GOOGLE_BOOKS_KEY = "..."  (raises the rate limit; not required)
"""
import argparse
import json
import os
import time
from pathlib import Path

import requests

REJECTS = Path("data/raw/books/hardcover_v2/targeted_rejects.jsonl")
OUT = Path("data/raw/books/hardcover_v2/description_probe.jsonl")

MIN_DESC_CHARS = 100      # same bar as the harvest: below this there is
                          # nothing worth embedding
# Google Books' unauthenticated quota is tight and returns 429 rather than
# throttling. The first run hit 429 on 5/5 calls, which the summary then
# counted as "no description found" -- a rate limit reported as a data finding.
# Slower by default, with a retry, and 429 now reported as UNKNOWN.
SLEEP = 2.0
GB_RETRIES = 3

GB_URL = "https://www.googleapis.com/books/v1/volumes"
OL_SEARCH = "https://openlibrary.org/search.json"
OL_WORK = "https://openlibrary.org{key}.json"
OL_EDITIONS = "https://openlibrary.org{key}/editions.json"
WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"

# Wikimedia REQUIRES a descriptive User-Agent and returns 403 to clients that
# send a generic one. The first run 403'd on all 19 titles, which the summary
# reported as "0% hit rate" -- an infrastructure failure presented as a data
# finding, the same shape as the Google 429s an hour earlier. Both times the
# per-title note was the only thing that gave it away.
UA = {"User-Agent": "recommendation-engine-probe/1.0 "
                    "(https://github.com/Arya-Sardesai/recommendation-engine; "
                    "research use) python-requests"}


def google_books(title, author):
    """Best described volume for a title+author."""
    params = {"q": f'intitle:"{title}" inauthor:"{author}"', "maxResults": 5}
    key = os.environ.get("GOOGLE_BOOKS_KEY")
    if key:
        params["key"] = key
    for attempt in range(GB_RETRIES):
        try:
            r = requests.get(GB_URL, params=params, timeout=20)
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
        if r.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"      rate limited; waiting {wait}s")
            time.sleep(wait)
            continue
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        items = r.json().get("items") or []
        break
    else:
        # RATE LIMITED, not absent. Must not be counted as a miss.
        return None, "RATE_LIMITED"

    best = ""
    for it in items:
        d = (it.get("volumeInfo", {}).get("description") or "").strip()
        if len(d) > len(best):
            best = d
    if not best:
        return None, "no description in any volume"
    return best, f"{len(items)} volumes"


def open_library(title, author):
    try:
        r = requests.get(OL_SEARCH,
                         params={"title": title, "author": author, "limit": 3},
                         timeout=20)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        docs = r.json().get("docs") or []
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if not docs:
        return None, "no search hit"

    for d in docs[:3]:
        wk = d.get("key")
        if not wk:
            continue
        try:
            w = requests.get(OL_WORK.format(key=wk), timeout=20)
            if w.status_code != 200:
                continue
            desc = w.json().get("description")
        except Exception:
            continue
        if isinstance(desc, dict):
            desc = desc.get("value", "")
        desc = (desc or "").strip()
        if desc:
            return desc, f"{len(docs)} works"
        time.sleep(SLEEP)
    return None, f"{len(docs)} works, none described"


def open_library_deep(title, author):
    """Work description, THEN editions, THEN first_sentence.

    The first version of this probe only read the work record and reported
    "14 works, none described". Open Library stores descriptions on EDITIONS
    too, and a work with an empty description field can still have a described
    edition underneath it -- the same edition-selection shape that the
    Hardcover harvest got wrong, repeated here by me.
    """
    try:
        r = requests.get(OL_SEARCH,
                         params={"title": title, "author": author, "limit": 3},
                         timeout=30)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        docs = r.json().get("docs") or []
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if not docs:
        return None, "no search hit"

    for d in docs[:3]:
        wk = d.get("key")
        if not wk:
            continue
        # 1. work-level description
        try:
            w = requests.get(OL_WORK.format(key=wk), timeout=30).json()
            desc = w.get("description")
            if isinstance(desc, dict):
                desc = desc.get("value", "")
            if (desc or "").strip():
                return desc.strip(), "work"
        except Exception:
            pass
        time.sleep(SLEEP)
        # 2. any described edition
        try:
            eds = requests.get(OL_EDITIONS.format(key=wk),
                               params={"limit": 20}, timeout=30).json()
            for e in eds.get("entries", []) or []:
                ed = e.get("description")
                if isinstance(ed, dict):
                    ed = ed.get("value", "")
                if (ed or "").strip():
                    return ed.strip(), "edition"
        except Exception:
            pass
        time.sleep(SLEEP)
        # 3. first_sentence is not a synopsis but records that the text exists
        fs = d.get("first_sentence")
        if isinstance(fs, list) and fs:
            return str(fs[0]).strip(), "first_sentence (NOT a synopsis)"
    return None, f"{len(docs)} works, nothing described"


def wikipedia(title, author):
    """Wikipedia article intro, if the book has one.

    Different coverage profile from the book catalogues: notability is
    editorial rather than commercial, so a well-known Indian author's work can
    have a full article while its catalogue record is bare. REGISTER DIFFERS --
    an encyclopedia intro is longer and more analytical than a jacket synopsis,
    so this is a candidate source, not a drop-in replacement.
    """
    try:
        r = requests.get(WIKI_SEARCH, headers=UA, params={
            "action": "query", "format": "json", "list": "search",
            "srsearch": f"{title} {author} book", "srlimit": 3}, timeout=20)
        if r.status_code == 403:
            return None, "RATE_LIMITED"      # blocked, not absent
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        hits = r.json().get("query", {}).get("search", []) or []
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if not hits:
        return None, "no article"

    for h in hits[:2]:
        try:
            e = requests.get(WIKI_SEARCH, headers=UA, params={
                "action": "query", "format": "json", "prop": "extracts",
                "exintro": 1, "explaintext": 1,
                "titles": h["title"]}, timeout=20).json()
            pages = e.get("query", {}).get("pages", {})
            for _, pg in pages.items():
                txt = (pg.get("extract") or "").strip()
                if len(txt) >= MIN_DESC_CHARS:
                    return txt, f"article: {h['title'][:40]}"
        except Exception:
            continue
        time.sleep(SLEEP)
    return None, f"{len(hits)} articles, no usable intro"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",
                    choices=["google", "openlibrary", "wikipedia", "all"],
                    default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rejects", default=str(REJECTS))
    args = ap.parse_args()

    path = Path(args.rejects)
    if not path.exists():
        raise SystemExit(f"{path} not found — run fetch_hardcover_targeted_v2 "
                         f"--stage fetch first")

    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows = [r for r in rows if r.get("reason") == "no_described_record"]
    if args.limit:
        rows = rows[:args.limit]
    print(f"probing {len(rows)} titles Hardcover could not describe\n")

    sources = (["google", "openlibrary", "wikipedia"] if args.source == "all"
               else [args.source])
    found = {s: 0 for s in sources}
    unresolved = {s: 0 for s in sources}   # rate-limited: unknown, not absent
    results = []

    for i, r in enumerate(rows, 1):
        title = str(r.get("title") or "")
        author = str(r.get("author") or "")
        print(f"[{i}/{len(rows)}] {title[:52]}")
        rec = {"title": title, "author": author}
        for s in sources:
            fn = {"google": google_books,
                  "openlibrary": open_library_deep,
                  "wikipedia": wikipedia}[s]
            desc, note = fn(title, author)
            unknown = note == "RATE_LIMITED"
            ok = bool(desc) and len(desc) >= MIN_DESC_CHARS
            if ok:
                found[s] += 1
            if unknown:
                unresolved[s] += 1
            rec[s] = {"chars": len(desc) if desc else 0, "note": note,
                      "text": desc[:400] if desc else ""}
            mark = "OK " if ok else ("SHORT" if desc else ("?? " if unknown else "-- "))
            print(f"    {s:12s} {mark} {len(desc) if desc else 0:>5} chars  ({note})")
            if ok:
                print(f"      {desc[:150].replace(chr(10), ' ')}...")
            time.sleep(SLEEP)
        results.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for rec in results:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n" + "=" * 66)
    print("HIT RATE on titles Hardcover could not describe")
    print("=" * 66)
    n = max(len(rows), 1)
    for s in sources:
        tested = len(rows) - unresolved[s]
        rate = f"{found[s]/tested:.0%}" if tested else "n/a"
        print(f"  {s:12s} {found[s]:>3}/{tested:<3} ({rate}) with >= "
              f"{MIN_DESC_CHARS} chars", end="")
        print(f"   [{unresolved[s]} rate-limited, NOT counted]"
              if unresolved[s] else "")
    if len(sources) > 1:
        either = sum(1 for rec in results
                     if any(rec[s]["chars"] >= MIN_DESC_CHARS for s in sources))
        print(f"  {'either':12s} {either:>3}/{len(rows)}  ({either/n:.0%})")
    print(f"\n  full text -> {OUT}")
    print("\n" + "=" * 66)
    print("READ THIS AS")
    print("=" * 66)
    print("  READ THE TEXT, not just the hit rate. A publisher blurb is not")
    print("  the same kind of object as a Hardcover description: it is")
    print("  marketing copy, often about the AUTHOR or the series rather than")
    print("  the book, and it embeds differently. A high hit rate with")
    print("  promotional text is worse than a low one with real synopses,")
    print("  because it silently degrades the vectors for exactly the titles")
    print("  the corpus is thinnest on.")
    print("  Mixing sources also makes description length and register")
    print("  inconsistent across the corpus, which the embedding sees.")


if __name__ == "__main__":
    main()
