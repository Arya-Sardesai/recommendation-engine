"""
build_award_list.py -- collect award winners and shortlists from Wikidata.

WHY
Books that won or were shortlisted for a serious prize already passed a
quality bar set by judges, not by reader counts. They fill exactly the gaps
the catalogue has (literary, translated, crime, real SFF, non-fiction) and
never appeared in the Hardcover harvest's popularity ranking. Added books are
treated like the hand-picked author sweep: exempt from the 10-rating serving
floor because they were chosen deliberately.

WHY WIKIDATA, NOT WIKIPEDIA TABLES
Every award page on Wikipedia lays its tables out differently. Wikidata
records the same facts in one shape: work --award received (P166)--> award
and work --nominated for (P1411)--> award. One query per award, no parsing.
Coverage of NOMINEES varies by award; the report shows counts per award so a
thin one is visible, and --extra CSVs can fill gaps by hand.

STAGES (run from the repo root)
  1. resolve  -- look up each award name in awards.json and record its Wikidata
                 id. Prints the match and its description: READ THEM. A wrong
                 match ("Hugo Award" the ceremony vs "Hugo Award for Best
                 Novel") silently fills the list with junk. Fix a bad one by
                 putting "qid": "Q..." on that award in awards.json and rerun.
  2. fetch    -- one query per award: every work that won or was nominated,
                 with author, year and language.
  3. match    -- merge duplicates (a book shortlisted for three prizes is one
                 row), drop books already in the catalogue (same title+author
                 rule as gbooks_trickle.py), write the queue and a report.

    python notebooks\\pipeline\\build_award_list.py --stage resolve
    python notebooks\\pipeline\\build_award_list.py --stage fetch
    python notebooks\\pipeline\\build_award_list.py --stage match
    python notebooks\\pipeline\\build_award_list.py --stage match --extra data\\raw\\books\\awards\\nyt_100.csv

--extra takes CSVs with columns title,author,source (e.g. a canon list typed
or pasted by hand). The first run writes awards.json with the default list --
edit it freely (add, remove, set "kind": "author" for prizes given to people).

OUTPUT
  data/raw/books/awards/award_books.jsonl  every book found, with its awards
  data/raw/books/awards/award_targets.jsonl  only books NOT in the catalogue --
      the lookup queue (Hardcover first for real ratings, then Google Books)

No API key needed. Wikidata asks for a descriptive User-Agent and polite
pacing; both are built in.
"""
import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import requests

OUT_DIR = Path("data/raw/books/awards")
CONFIG = OUT_DIR / "awards.json"
RESOLVED = OUT_DIR / "awards_resolved.json"
RAW = OUT_DIR / "award_raw.jsonl"
BOOKS = OUT_DIR / "award_books.jsonl"
TARGETS = OUT_DIR / "award_targets.jsonl"
CORPUS = Path("data/processed/books_v2_hc3.parquet")

SEARCH_API = "https://www.wikidata.org/w/api.php"
SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "rec-engine-awards/1.0 "
                         "(https://github.com/Arya-Sardesai/recommendation-engine)",
           "Accept": "application/sparql-results+json"}

# group -> award names as Wikidata labels them. kind "work" = the prize is
# given to a book; kind "author" = given to a person (their notable works are
# taken instead). Romance and YA/children's prizes are left out on purpose.
DEFAULT_AWARDS = {
    "literary": [
        "Booker Prize", "Pulitzer Prize for Fiction",
        "National Book Award for Fiction",
        "National Book Critics Circle Award for Fiction",
        "Women's Prize for Fiction", "Costa Book Awards",
        "Scotiabank Giller Prize", "Miles Franklin Award",
        "International Dublin Literary Award",
        "James Tait Black Memorial Prize", "Rathbones Folio Prize",
        "PEN/Faulkner Award for Fiction", "Kirkus Prize"],
    "translated": [
        "International Booker Prize", "Prix Goncourt", "Strega Prize",
        "German Book Prize", "Akutagawa Prize", "Naoki Prize",
        "International Prize for Arabic Fiction", "Rómulo Gallegos Prize",
        "National Book Award for Translated Literature",
        "Warwick Prize for Women in Translation"],
    "south_asia": [
        "JCB Prize for Literature", "DSC Prize for South Asian Literature",
        "Crossword Book Award", "Sahitya Akademi Award"],
    "sff": [
        "Hugo Award for Best Novel", "Nebula Award for Best Novel",
        "Arthur C. Clarke Award",
        "Locus Award for Best Science Fiction Novel",
        "Locus Award for Best Fantasy Novel",
        "World Fantasy Award for Best Novel", "Philip K. Dick Award"],
    "crime": [
        "Edgar Award for Best Novel", "Gold Dagger", "Anthony Award",
        "Agatha Award", "Macavity Award"],
    "horror": ["Bram Stoker Award for Best Novel", "Shirley Jackson Award"],
    "historical": ["Walter Scott Prize"],
    "comics": ["Eisner Award"],
    "nonfiction": [
        "Pulitzer Prize for General Nonfiction", "Pulitzer Prize for History",
        "Pulitzer Prize for Biography or Autobiography",
        "National Book Award for Nonfiction", "Baillie Gifford Prize",
        "Orwell Prize", "Royal Society Science Books Prize",
        "Wolfson History Prize",
        "Financial Times and McKinsey Business Book of the Year Award"],
}
AUTHOR_AWARDS = {"Nobel Prize in Literature"}   # given to people


def write_default_config():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = []
    for group, names in DEFAULT_AWARDS.items():
        for n in names:
            cfg.append({"name": n, "group": group, "kind": "work"})
    for n in AUTHOR_AWARDS:
        cfg.append({"name": n, "group": "literary", "kind": "author"})
    CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {CONFIG} ({len(cfg)} awards) -- edit it to add or remove prizes")
    return cfg


def load_config():
    if not CONFIG.exists():
        return write_default_config()
    return json.loads(CONFIG.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- resolve
def search_entity(name):
    r = requests.get(SEARCH_API, params={
        "action": "wbsearchentities", "search": name, "language": "en",
        "type": "item", "limit": 3, "format": "json"},
        headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json().get("search") or []


def stage_resolve():
    cfg = load_config()
    out = []
    print(f"{'award':<52} {'qid':<11} description")
    for a in cfg:
        if a.get("qid"):
            out.append({**a, "label": a["name"], "description": "(set by hand)"})
            print(f"{a['name'][:51]:<52} {a['qid']:<11} (set by hand)")
            continue
        try:
            hits = search_entity(a["name"])
        except requests.RequestException as e:
            print(f"{a['name'][:51]:<52} {'ERROR':<11} {type(e).__name__}")
            continue
        if not hits:
            print(f"{a['name'][:51]:<52} {'NONE':<11} no match -- set a qid by hand")
            continue
        h = hits[0]
        out.append({**a, "qid": h["id"], "label": h.get("label", ""),
                    "description": h.get("description", "")})
        flag = "" if h.get("label", "").lower() == a["name"].lower() else "  <-- CHECK"
        print(f"{a['name'][:51]:<52} {h['id']:<11} {h.get('label','')[:30]} | "
              f"{h.get('description','')[:50]}{flag}")
        time.sleep(0.5)
    RESOLVED.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {RESOLVED} ({len(out)} of {len(cfg)} resolved)")
    print("Rows marked CHECK matched a different label: confirm the description")
    print("is the BOOK prize. If not, put the right \"qid\" on that award in")
    print("awards.json and run --stage resolve again.")


# ---------------------------------------------------------------- fetch
WORK_QUERY = """
SELECT ?work ?workLabel ?auth ?authLabel ?date ?langLabel ?kind WHERE {
  { ?work p:P166 ?st . ?st ps:P166 wd:%(qid)s . BIND("won" AS ?kind) }
  UNION
  { ?work p:P1411 ?st . ?st ps:P1411 wd:%(qid)s . BIND("nominated" AS ?kind) }
  UNION
  { ?person p:P166 ?st . ?st ps:P166 wd:%(qid)s ; pq:P1686 ?work . BIND("won" AS ?kind) }
  UNION
  { ?person p:P1411 ?st . ?st ps:P1411 wd:%(qid)s ; pq:P1686 ?work . BIND("nominated" AS ?kind) }
  FILTER NOT EXISTS { ?work wdt:P31 wd:Q5 }
  OPTIONAL { ?work wdt:P50 ?writer . }
  BIND(COALESCE(?writer, ?person) AS ?auth)
  OPTIONAL { ?work wdt:P577 ?date . }
  OPTIONAL { ?work wdt:P407 ?lang . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,mul,fr,de,es,it,pt,nl,sv,ja". }
}"""
# v1 matched the prize only on the BOOK. Wikidata often records it on the
# AUTHOR instead ("Hilary Mantel -- received Booker Prize -- for work Wolf
# Hall", the pq:P1686 qualifier). v1 got the person back as the "work", with
# no author, and dropped 8,415 of 15,391 rows -- most of crime and non-fiction.
# The two person patterns take the work from that qualifier and the author
# from the person; FILTER drops a person who came back as the work. The label
# fallback languages stop authors with no English name printing as "Q268755".

# Prize given to a PERSON: take their notable works (P800).
AUTHOR_QUERY = """
SELECT ?work ?workLabel ?auth ?authLabel ?date ?langLabel ?kind WHERE {
  ?auth p:P166/ps:P166 wd:%(qid)s .
  ?auth wdt:P800 ?work .
  OPTIONAL { ?work wdt:P577 ?date . }
  OPTIONAL { ?work wdt:P407 ?lang . }
  BIND("laureate" AS ?kind)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,mul,fr,de,es,it,pt,nl,sv,ja". }
}"""


def sparql(query):
    """Retries server errors AND dropped connections. Oct 12: the Locus SF
    query (a large result) failed with 'Response ended prematurely', a cut
    connection rather than an HTTP error, and v1 did not retry those."""
    last = None
    for wait in (5, 15, 45, 90):
        try:
            r = requests.get(SPARQL, params={"query": query, "format": "json"},
                             headers=HEADERS, timeout=180)
        except requests.RequestException as e:
            last = e
            print(f"    connection dropped ({type(e).__name__}), waiting {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 200:
            try:
                return r.json()["results"]["bindings"]
            except ValueError as e:          # body cut off mid-JSON
                last = e
                print(f"    truncated response, waiting {wait}s")
                time.sleep(wait)
                continue
        if r.status_code in (429, 500, 502, 503, 504):
            print(f"    HTTP {r.status_code}, waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError(f"Wikidata query service unavailable -- try later ({last})")


def val(b, k):
    return (b.get(k) or {}).get("value", "")


def stage_fetch(only=None):
    """only=None: fetch every award and rewrite award_raw.jsonl.
    only=[names]: refetch just those awards, keeping every other award's rows
    -- for retrying one that failed without redoing the rest."""
    if not RESOLVED.exists():
        sys.exit("run --stage resolve first")
    awards = json.loads(RESOLVED.read_text(encoding="utf-8"))
    keep = []
    if only:
        wanted = {n.lower() for n in only}
        unknown = wanted - {a["name"].lower() for a in awards}
        if unknown:
            sys.exit(f"not in awards_resolved.json: {sorted(unknown)}")
        awards = [a for a in awards if a["name"].lower() in wanted]
        if RAW.exists():
            keep = [l.rstrip("\n") + "\n" for l in RAW.open(encoding="utf-8")
                    if l.strip() and json.loads(l)["award"].lower() not in wanted]
    n_total, failed = 0, []
    with RAW.open("w", encoding="utf-8") as fh:
        fh.writelines(keep)
        for a in awards:
            q = (AUTHOR_QUERY if a.get("kind") == "author" else WORK_QUERY) % {"qid": a["qid"]}
            try:
                rows = sparql(q)
            except Exception as e:
                print(f"  {a['name'][:50]:<52} FAILED ({e})")
                failed.append(a["name"])
                continue
            kept = 0
            for b in rows:
                title = val(b, "workLabel")
                # an unlabelled item comes back as its own id ("Q12345")
                if not title or re.fullmatch(r"Q\d+", title):
                    continue
                author = val(b, "authLabel")
                if re.fullmatch(r"Q\d+", author):
                    author = ""        # no name in any fallback language
                fh.write(json.dumps({
                    "wikidata": val(b, "work").rsplit("/", 1)[-1],
                    "title": title, "author": author,
                    "year": val(b, "date")[:4], "language": val(b, "langLabel"),
                    "award": a["name"], "group": a["group"], "kind": val(b, "kind"),
                }, ensure_ascii=False) + "\n")
                kept += 1
            n_total += kept
            print(f"  {a['name'][:50]:<52} {kept:>5} rows")
            time.sleep(1.0)     # Wikidata asks for polite pacing
    print(f"\nwrote {RAW} ({n_total:,} new rows"
          + (f", {len(keep):,} kept from other awards" if only else "") + ")")
    if failed:
        quoted = " ".join(f'--only "{n}"' for n in failed)
        print(f"FAILED: {len(failed)} -- retry just those with:\n"
              f"  python notebooks\\pipeline\\build_award_list.py --stage fetch {quoted}")
    else:
        print("run --stage match next")


# ---------------------------------------------------------------- match
def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def _core(title):
    t = re.sub(r"\([^)]*\)", " ", str(title))
    t = re.split(r":| - ", t)[0]
    return re.sub(r"^(the|a|an) ", "", _norm(t)).strip()


def book_key(title, author):
    ct = _core(title)
    toks = _norm(author).split()
    ca = " ".join(sorted(toks))
    return (ct, ca) if ct and ca else None


def stage_match(extra):
    rows = []
    if RAW.exists():
        rows += [json.loads(l) for l in RAW.open(encoding="utf-8") if l.strip()]
    for path in extra:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append({"wikidata": "", "title": r.get("title", ""),
                             "author": r.get("author", ""), "year": r.get("year", ""),
                             "language": "", "award": r.get("source") or Path(path).stem,
                             "group": r.get("group") or "list", "kind": "listed"})
    if not rows:
        sys.exit("nothing to match -- run --stage fetch (or pass --extra)")

    books, no_author, sahitya_dropped = {}, 0, 0
    for r in rows:
        # Sahitya Akademi covers 24 Indian languages; most winners have no
        # English edition and would only waste lookups. Keep its English ones.
        if r["award"] == "Sahitya Akademi Award" and r["language"] not in ("English",):
            sahitya_dropped += 1
            continue
        k = book_key(r["title"], r["author"])
        if not k:
            no_author += 1
            continue
        b = books.setdefault(k, {"title": r["title"], "author": r["author"],
                                 "year": r["year"], "language": r["language"],
                                 "wikidata": r["wikidata"], "awards": [],
                                 "groups": set(), "won": False})
        b["awards"].append(f"{r['award']} ({r['kind']})")
        b["groups"].add(r["group"])
        b["won"] |= r["kind"] in ("won", "laureate")

    # One book, several authors (Abaddon's Gate = Daniel Abraham + Ty Franck,
    # writing as James S. A. Corey) came out as one row per author. Rows that
    # share a Wikidata id are the same book: keep the first author, merge the
    # awards. Rows with no id (Wikipedia shortlists) keep the title+author key.
    by_qid, merged = {}, 0
    for k in list(books):
        q = books[k]["wikidata"]
        if not q:
            continue
        if q in by_qid:
            keep = books[by_qid[q]]
            for aw in books[k]["awards"]:
                if aw not in keep["awards"]:
                    keep["awards"].append(aw)
            keep["groups"] |= books[k]["groups"]
            keep["won"] |= books[k]["won"]
            del books[k]
            merged += 1
        else:
            by_qid[q] = k

    have = set()
    if CORPUS.exists():
        import pandas as pd
        c = pd.read_parquet(CORPUS, columns=["title", "primary_author"])
        have = {k for k in (book_key(t, a) for t, a in
                            zip(c["title"], c["primary_author"])) if k}
    else:
        print(f"WARNING: {CORPUS} not found -- cannot check against the catalogue")

    with BOOKS.open("w", encoding="utf-8") as fa, TARGETS.open("w", encoding="utf-8") as ft:
        new = []
        for k, b in books.items():
            b["groups"] = sorted(b["groups"])
            b["in_catalogue"] = k in have
            fa.write(json.dumps(b, ensure_ascii=False) + "\n")
            if not b["in_catalogue"]:
                new.append(b)
        # winners first, English-language before translated (an English
        # edition is likelier to exist), then books on several lists, then newest
        new.sort(key=lambda b: (not b["won"], b["language"] not in ("English", ""),
                                -len(b["awards"]), -(int(b["year"] or 0))))
        for b in new:
            ft.write(json.dumps(b, ensure_ascii=False) + "\n")

    if merged:
        print(f"merged {merged:,} extra-author rows of the same book (shared Wikidata id)")
    print(f"{len(rows):,} award rows -> {len(books):,} distinct books "
          f"({no_author:,} rows had no author and were dropped; "
          f"{sahitya_dropped:,} non-English Sahitya Akademi rows skipped)")
    inc = sum(b["in_catalogue"] for b in books.values())
    print(f"  already in the catalogue: {inc:,}")
    print(f"  NEW (lookup queue):       {len(new):,}  -> {TARGETS}\n")

    per = defaultdict(lambda: [0, 0])
    for b in books.values():
        for g in b["groups"]:
            per[g][0] += 1
            per[g][1] += not b["in_catalogue"]
    print(f"  {'group':<14}{'books':>8}{'new':>8}")
    for g, (t, n) in sorted(per.items(), key=lambda x: -x[1][0]):
        print(f"  {g:<14}{t:>8,}{n:>8,}")

    langs = Counter(b["language"] or "(not recorded)" for b in new)
    print("\n  original language of NEW books:",
          ", ".join(f"{l} {n}" for l, n in langs.most_common(6)))
    print("\n  sample of NEW books:")
    for b in new[:20]:
        print(f"    {b['year'] or '----'}  {b['title'][:45]:47s} {b['author'][:24]:26s}"
              f" {b['awards'][0][:40]}")
    print("\nTranslated books are kept: the lookup searches the English title")
    print("and the merge keeps English blurbs only.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["resolve", "fetch", "match"], required=True)
    ap.add_argument("--only", action="append", default=[],
                    help="fetch: refetch only this award (exact name), keeping the "
                         "rest; repeatable")
    ap.add_argument("--extra", action="append", default=[],
                    help="CSV with title,author[,year,source] columns; repeatable")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "resolve":
        stage_resolve()
    elif args.stage == "fetch":
        stage_fetch(args.only or None)
    if args.stage == "match":
        stage_match(args.extra)


if __name__ == "__main__":
    main()
