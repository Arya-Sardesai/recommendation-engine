"""
build_screen_awards.py -- film and TV awards from Wikidata, matched by TMDB id.

WHAT IT MAKES
    data/processed/film_awards.parquet   one row per (film, award category, year)
    data/processed/tv_awards.parquet     one row per (series, award category, year)

    columns: tmdb_id, org, category, year, result (won|nominated),
             tier (major|minor), award_qid

Read by app.py for two things:
  * the trophy badge -- every row, major and minor, summarised in the tooltip
    ("Oscars: 4 wins, 11 nominations . Best Picture (won, 1998)")
  * the ranking bonus -- the single best MAJOR row only: +0.04 won,
    +0.02 nominated, never summed (Titanic does not get 11 x 0.04)

WHY WIKIDATA, AND WHY THIS IS EASIER THAN BOOKS
Wikidata stores TMDB ids on film items (P4947) and series items (P4983), the
same ids the catalogue uses. No title matching: a hit is the right row or no
row at all. Books needed title + author guessing; this does not.

THE TWO SHAPES AN AWARD TAKES IN WIKIDATA
  1. On the WORK:   <film>   award received (P166) / nominated for (P1411)
                    <Academy Award for Best Picture>
  2. On a PERSON, with the work as a qualifier:
                    <actor>  award received <Academy Award for Best Actor>
                             for work (P1686) <film>
Acting, directing and writing awards are almost always shape 2. Querying only
shape 1 would miss most of the Oscars and nearly all the acting Emmys, so
every award is queried both ways.

For TV, Emmy writing and directing awards often point at an EPISODE, not the
series. The query follows "part of the series" (P179) from the episode up to
the series that carries the TMDB id.

STAGES (run from the repo root)
    python notebooks\\pipeline\\build_screen_awards.py --stage check  [--endpoint qlever]
        confirms the Wikidata property ids this script relies on. Stops if any
        label is not what it should be.
    python notebooks\\pipeline\\build_screen_awards.py --stage resolve
        award names -> Wikidata ids. Writes data\\raw\\screen_awards\\awards.json,
        which you can edit (drop an award, fix a qid, change a tier).
        READ THE OUTPUT: anything marked CHECK matched a different label.
    python notebooks\\pipeline\\build_screen_awards.py --stage probe --only "Golden Bear"
        for an award that came back NONE or with 0 rows: lists what the name
        could mean in Wikidata, with how many award statements use each.
        Then --stage probe --set "Golden Bear=Q..." saves the right one.
    python notebooks\\pipeline\\build_screen_awards.py --stage fetch
        one query per award per kind. Raw rows to award_rows.jsonl. Resumable:
        awards already fetched are skipped (--refetch to redo).
    python notebooks\\pipeline\\build_screen_awards.py --stage build
        joins to the catalogue, writes the two parquets, prints a report with
        sample rows. Read the samples before trusting the counts.

Wikidata only -- free, no keys, nothing to do with the Google Books quota.
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "screen_awards"
PROCESSED = REPO / "data" / "processed"
APP_PY = REPO.parent / "hf-space" / "app.py"
AWARDS_JSON = RAW / "awards.json"
ROWS_JSONL = RAW / "award_rows.jsonl"

ENDPOINTS = {
    # Wikidata's own query service. Fastest when healthy; during an outage it
    # rate-limits to 1 request a minute and times out heavy queries.
    "wdqs": "https://query.wikidata.org/sparql",
    # QLever: an independent public copy of Wikidata (University of Freiburg),
    # refreshed regularly. Use it when WDQS is down: --endpoint qlever.
    "qlever": "https://qlever.cs.uni-freiburg.de/api/wikidata",
}
ENDPOINT = "wdqs"
PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""
API = "https://www.wikidata.org/w/api.php"
UA = "recommendation-engine-awards/1.0 (personal portfolio project; contact via GitHub Arya-Sardesai)"

# The properties this script depends on, with the label each must have. The
# check stage refuses to continue if Wikidata disagrees, so a wrong id here
# fails loudly instead of silently returning nothing.
PROPS = {
    "P166": "award received",
    "P1411": "nominated for",
    "P1686": "for work",
    "P585": "point in time",
    "P179": "part of the series",
    "P4947": "TMDB movie ID",
    "P4983": "TMDB TV series ID",
}
TMDB_PROP = {"film": "P4947", "tv": "P4983"}

# ---------------------------------------------------------------------------
# The award list. tier "major" can earn the ranking bonus; "minor" is badge
# only. Labels are Wikidata's English labels as best known; the resolve stage
# checks each one and marks a mismatch CHECK. Edit awards.json, not this list,
# after the first resolve.
# ---------------------------------------------------------------------------
def _a(name, org, category, tier, search=None, desc=None):
    """search: what to type into Wikidata search when the full name finds
    nothing. desc: a word the item's description must contain, to pick the
    right one among same-named items."""
    d = {"name": name, "org": org, "category": category, "tier": tier}
    if search:
        d["search"] = search
    if desc:
        d["desc"] = desc
    return d


DEFAULT_AWARDS = {
    "film": [
        # Oscars -- major
        _a("Academy Award for Best Picture", "Oscar", "Best Picture", "major"),
        _a("Academy Award for Best Director", "Oscar", "Best Director", "major"),
        _a("Academy Award for Best Actor", "Oscar", "Best Actor", "major"),
        _a("Academy Award for Best Actress", "Oscar", "Best Actress", "major"),
        _a("Academy Award for Best Supporting Actor", "Oscar", "Best Supporting Actor", "major"),
        _a("Academy Award for Best Supporting Actress", "Oscar", "Best Supporting Actress", "major"),
        _a("Academy Award for Best Original Screenplay", "Oscar", "Best Original Screenplay", "major"),
        _a("Academy Award for Best Adapted Screenplay", "Oscar", "Best Adapted Screenplay", "major"),
        _a("Academy Award for Best International Feature Film", "Oscar", "Best International Feature", "major"),
        _a("Academy Award for Best Animated Feature", "Oscar", "Best Animated Feature", "major"),
        _a("Academy Award for Best Documentary Feature Film", "Oscar", "Best Documentary Feature", "major"),
        # Oscars -- minor (badge only)
        _a("Academy Award for Best Cinematography", "Oscar", "Best Cinematography", "minor"),
        _a("Academy Award for Best Film Editing", "Oscar", "Best Film Editing", "minor"),
        _a("Academy Award for Best Original Score", "Oscar", "Best Original Score", "minor"),
        _a("Academy Award for Best Original Song", "Oscar", "Best Original Song", "minor"),
        _a("Academy Award for Best Production Design", "Oscar", "Best Production Design", "minor"),
        _a("Academy Award for Best Costume Design", "Oscar", "Best Costume Design", "minor"),
        _a("Academy Award for Best Makeup and Hairstyling", "Oscar", "Best Makeup and Hairstyling", "minor"),
        _a("Academy Award for Best Sound", "Oscar", "Best Sound", "minor"),
        _a("Academy Award for Best Visual Effects", "Oscar", "Best Visual Effects", "minor"),
        # Golden Globes
        _a("Golden Globe Award for Best Motion Picture – Drama", "Golden Globe", "Best Motion Picture – Drama", "major"),
        _a("Golden Globe Award for Best Motion Picture – Musical or Comedy", "Golden Globe", "Best Motion Picture – Musical or Comedy", "major"),
        _a("Golden Globe Award for Best Director", "Golden Globe", "Best Director", "major"),
        # BAFTA
        _a("BAFTA Award for Best Film", "BAFTA", "Best Film", "major"),
        _a("BAFTA Award for Best Film Not in the English Language", "BAFTA", "Best Film Not in the English Language", "major"),
        # Festivals (top prizes only)
        _a("Palme d'Or", "Cannes", "Palme d'Or", "major"),
        _a("Grand Prix (Cannes Film Festival)", "Cannes", "Grand Prix", "major",
           search="Grand Prix", desc="Cannes"),
        _a("Golden Lion", "Venice", "Golden Lion", "major"),
        _a("Golden Bear", "Berlin", "Golden Bear", "major"),
        # India
        _a("National Film Award for Best Feature Film", "National Film Award", "Best Feature Film", "major"),
        _a("Filmfare Award for Best Film", "Filmfare", "Best Film", "major"),
    ],
    "tv": [
        # Primetime Emmys -- major
        _a("Primetime Emmy Award for Outstanding Drama Series", "Emmy", "Outstanding Drama Series", "major"),
        _a("Primetime Emmy Award for Outstanding Comedy Series", "Emmy", "Outstanding Comedy Series", "major"),
        _a("Primetime Emmy Award for Outstanding Limited or Anthology Series", "Emmy", "Outstanding Limited or Anthology Series", "major"),
        # The limited-series category has been renamed several times; older
        # winners may sit on a separate Wikidata item. Duplicates collapse in
        # the build step, so listing both costs nothing.
        _a("Primetime Emmy Award for Outstanding Miniseries", "Emmy", "Outstanding Limited or Anthology Series", "major"),
        _a("Primetime Emmy Award for Outstanding Lead Actor in a Drama Series", "Emmy", "Lead Actor in a Drama Series", "major"),
        _a("Primetime Emmy Award for Outstanding Lead Actress in a Drama Series", "Emmy", "Lead Actress in a Drama Series", "major"),
        _a("Primetime Emmy Award for Outstanding Lead Actor in a Comedy Series", "Emmy", "Lead Actor in a Comedy Series", "major"),
        _a("Primetime Emmy Award for Outstanding Lead Actress in a Comedy Series", "Emmy", "Lead Actress in a Comedy Series", "major"),
        _a("Primetime Emmy Award for Outstanding Writing for a Drama Series", "Emmy", "Writing for a Drama Series", "major"),
        _a("Primetime Emmy Award for Outstanding Writing for a Comedy Series", "Emmy", "Writing for a Comedy Series", "major"),
        _a("Primetime Emmy Award for Outstanding Directing for a Drama Series", "Emmy", "Directing for a Drama Series", "major"),
        _a("Primetime Emmy Award for Outstanding Directing for a Comedy Series", "Emmy", "Directing for a Comedy Series", "major"),
        # Emmys -- minor
        _a("Primetime Emmy Award for Outstanding Supporting Actor in a Drama Series", "Emmy", "Supporting Actor in a Drama Series", "minor"),
        _a("Primetime Emmy Award for Outstanding Supporting Actress in a Drama Series", "Emmy", "Supporting Actress in a Drama Series", "minor"),
        _a("Primetime Emmy Award for Outstanding Supporting Actor in a Comedy Series", "Emmy", "Supporting Actor in a Comedy Series", "minor"),
        _a("Primetime Emmy Award for Outstanding Supporting Actress in a Comedy Series", "Emmy", "Supporting Actress in a Comedy Series", "minor"),
        _a("Primetime Emmy Award for Outstanding Animated Program", "Emmy", "Outstanding Animated Program", "minor"),
        # Golden Globes
        _a("Golden Globe Award for Best Television Series – Drama", "Golden Globe", "Best Television Series – Drama", "major"),
        _a("Golden Globe Award for Best Television Series – Musical or Comedy", "Golden Globe", "Best Television Series – Musical or Comedy", "major"),
        _a("Golden Globe Award for Best Limited or Anthology Series or Television Film", "Golden Globe", "Best Limited or Anthology Series", "major"),
        # Others
        _a("British Academy Television Award for Best Drama Series", "BAFTA TV", "Best Drama Series", "major"),
        _a("International Emmy Award for Best Drama Series", "International Emmy", "Best Drama Series", "major"),
        _a("Peabody Award", "Peabody", "Peabody Award", "major"),
    ],
}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
# Minimum gap between requests to one host. Starts at 1s; a 429 that names a
# per-minute limit (WDQS during an outage: "1 req / min") raises it to 61s for
# the rest of the run, so the script stops hammering a limit it cannot beat.
_GAP = {}
_LAST = {}


def _host(url):
    return urllib.parse.urlparse(url).netloc


def _pace(url):
    h = _host(url)
    wait = _GAP.get(h, 1.0) - (time.time() - _LAST.get(h, 0.0))
    if wait > 0:
        time.sleep(wait)
    _LAST[h] = time.time()


def _get(url, params, timeout=120, tries=6, accept="application/json"):
    q = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(tries):
        _pace(url)
        try:
            req = urllib.request.Request(q, headers={"User-Agent": UA, "Accept": accept})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            code = getattr(e, "code", None)
            if attempt == tries - 1:
                raise
            if code == 429:
                body = ""
                try:
                    body = e.read().decode("utf-8", "replace")
                except Exception:
                    pass
                retry_after = None
                try:
                    retry_after = int(e.headers.get("Retry-After", ""))
                except (TypeError, ValueError, AttributeError):
                    pass
                if "/ min" in body or "per minute" in body:
                    if _GAP.get(_host(url), 1.0) < 61:
                        print(f"    {_host(url)} is limited to 1 request a minute "
                              f"(Wikidata outage). Slowing down; or rerun with "
                              f"--endpoint qlever.")
                    _GAP[_host(url)] = 61.0
                wait = max(retry_after or 0, 61)
            elif code in (502, 503, 504):
                if attempt >= 2:          # three timeouts on one query: give up on it
                    raise
                wait = 15 * (attempt + 1)
            else:
                wait = 5 * (attempt + 1)
            print(f"    retry in {wait}s ({type(e).__name__} {code or ''})")
            time.sleep(wait)


def sparql(query):
    url = ENDPOINTS[ENDPOINT]
    return _get(url, {"query": PREFIXES + query},
                accept="application/sparql-results+json")["results"]["bindings"]


def _val(b, k):
    v = b.get(k)
    return v["value"] if v else None


def _qid(uri):
    return uri.rsplit("/", 1)[-1] if uri else None


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_check():
    ids = " ".join(f"wd:{p}" for p in PROPS)
    rows = sparql(f"""
        SELECT ?p ?label WHERE {{
          VALUES ?p {{ {ids} }}
          ?p rdfs:label ?label . FILTER(LANG(?label) = "en")
        }}""")
    got = {_qid(_val(b, "p")): _val(b, "label") for b in rows}
    bad = 0
    for p, want in PROPS.items():
        have = got.get(p)
        ok = have is not None and have.lower() == want.lower()
        bad += not ok
        print(f"  {p:<7} {want:<22} {'ok' if ok else f'MISMATCH: Wikidata says {have!r}'}")
    if bad:
        sys.exit("\nA property id is wrong. Paste this output and stop here.")
    print("\nAll property ids confirmed. Run --stage resolve next.")


def _search(name, limit=5):
    data = _get(API, {"action": "wbsearchentities", "search": name, "language": "en",
                      "type": "item", "limit": limit, "format": "json"})
    return [(h["id"], h.get("label", ""), h.get("description", ""))
            for h in data.get("search", [])]


def _norm_label(s):
    return re.sub(r"[\s–—-]+", " ", s.lower()).strip()


def stage_resolve():
    RAW.mkdir(parents=True, exist_ok=True)
    existing = (json.loads(AWARDS_JSON.read_text(encoding="utf-8"))
                if AWARDS_JSON.exists() else {})
    keep = {(k, a["name"]): a for k, lst in existing.items() for a in lst if a.get("qid")}
    out, checks = {}, 0
    for kind, lst in DEFAULT_AWARDS.items():
        out[kind] = []
        print(f"\n{kind.upper()}")
        for a in lst:
            a = dict(a)
            prev = keep.get((kind, a["name"]))
            if prev:                         # keep a qid you fixed by hand
                a["qid"], a["matched"] = prev["qid"], prev.get("matched", "")
                print(f"  kept   {a['qid']:<11} {a['name']}")
                out[kind].append(a)
                continue
            hits = _search(a["name"])
            exact = [h for h in hits if _norm_label(h[1]) == _norm_label(a["name"])]
            if not exact and a.get("search"):
                hits = _search(a["search"], limit=30)
                if a.get("desc"):
                    hits = [h for h in hits if a["desc"].lower() in h[2].lower()]
                exact = [h for h in hits if _norm_label(h[1]) == _norm_label(a["search"])]
            pick = exact[0] if exact else (hits[0] if hits else None)
            if pick is None:
                a["qid"], a["matched"] = None, ""
                print(f"  NONE              {a['name']}   <- no result; set qid by hand or drop")
                checks += 1
            else:
                a["qid"], a["matched"] = pick[0], pick[1]
                mark = "ok    " if exact else "CHECK "
                checks += not exact
                print(f"  {mark} {pick[0]:<11} {a['name'][:58]:<60}"
                      + ("" if exact else f"\n           matched: {pick[1]} | {pick[2][:70]}"))
            out[kind].append(a)
    AWARDS_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {AWARDS_JSON}")
    if checks:
        print(f"{checks} award(s) need a look. A CHECK matched a different label: if it is")
        print("the right award anyway, leave it; if not, put the right qid in awards.json")
        print("(or delete the entry). A wrong match fills the badge with nonsense.")
    print("Run --stage fetch next.")


def _usage_count(qid):
    """How many award / nomination statements point at this item. The real
    award category has hundreds; a trophy object or a disambiguation has ~0."""
    rows = sparql(f"""
        SELECT (COUNT(?st) AS ?n) WHERE {{
          {{ ?st ps:P166 wd:{qid} }} UNION {{ ?st ps:P1411 wd:{qid} }}
        }}""")
    try:
        return int(_val(rows[0], "n")) if rows else 0
    except (TypeError, ValueError):
        return 0


def stage_probe(names):
    """For an award that resolved wrongly or not at all: list every Wikidata
    item the name (and its search alias) could mean, with how many award
    statements actually use it. The right one is the one with the count."""
    if not names:
        sys.exit('probe needs --only "<award name>" (one or more)')
    known = {a["name"]: a for lst in DEFAULT_AWARDS.values() for a in lst}
    for name in names:
        a = known.get(name, {"name": name})
        terms = [name] + ([a["search"]] if a.get("search") else [])
        seen, cands = set(), []
        for t in terms:
            for qid, label, desc in _search(t, limit=30):
                if qid not in seen:
                    seen.add(qid)
                    cands.append((qid, label, desc))
        if a.get("desc"):
            pref = [c for c in cands if a["desc"].lower() in c[2].lower()]
            cands = pref + [c for c in cands if c not in pref]
        print(f"\n{name}: {len(cands)} candidates, counting award statements on each")
        scored = []
        for qid, label, desc in cands[:20]:
            n = _usage_count(qid)
            scored.append((n, qid, label, desc))
        for n, qid, label, desc in sorted(scored, reverse=True):
            if n or len(scored) <= 8:
                print(f"  {n:>6}  {qid:<11} {label[:40]:<42} {desc[:60]}")
        # The count alone is NOT the answer: "Grand Prix" also finds the Prix
        # de Rome (1,076 statements) ahead of the Cannes Grand Prix (103). When
        # the award has a required word (desc), only candidates whose label or
        # description contains it can be suggested.
        pool = [x for x in scored if x[0]]
        if a.get("desc"):
            w = a["desc"].lower()
            pool = [x for x in pool if w in x[2].lower() or w in x[3].lower()]
        best = max(pool) if pool else None
        if best:
            print(f'  -> most likely {best[1]} ({best[2]}). Read its description above,')
            print(f'     then save it with:  --stage probe --set "{name}={best[1]}"')
        elif a.get("desc"):
            print(f'  -> nothing mentioning "{a["desc"]}" is used as an award. Set it by')
            print("     hand if one of the rows above is right, or drop the award.")
        else:
            print("  -> nothing in Wikidata uses any of these as an award. Drop it.")


def stage_set(pairs):
    """--set "Golden Bear=Q12345": write a qid into awards.json by hand and
    clear the award from the zero-row list so the next fetch retries it."""
    awards = json.loads(AWARDS_JSON.read_text(encoding="utf-8"))
    zero_path = RAW / "zero_row_awards.json"
    zeros = set(map(tuple, json.loads(zero_path.read_text()))) if zero_path.exists() else set()
    for pair in pairs:
        name, _, qid = pair.partition("=")
        name, qid = name.strip(), qid.strip()
        if not re.fullmatch(r"Q\d+", qid):
            sys.exit(f"{qid!r} is not a Wikidata id (expected Q followed by digits)")
        hit = 0
        for kind, lst in awards.items():
            for a in lst:
                if a["name"] == name:
                    a["qid"], a["matched"] = qid, "(set by hand)"
                    zeros.discard((kind, name))
                    hit += 1
        if not hit:
            sys.exit(f"no award named {name!r} in awards.json")
        print(f"  set {name} = {qid}")
    AWARDS_JSON.write_text(json.dumps(awards, indent=2, ensure_ascii=False), encoding="utf-8")
    zero_path.write_text(json.dumps(sorted(zeros)), encoding="utf-8")
    names = " ".join(f'"{p.partition("=")[0].strip()}"' for p in pairs)
    print(f"\nNow fetch just those: --stage fetch --endpoint qlever --only {names}")


def _award_query(qid, kind, result):
    """One award, one result kind (won | nominated), both shapes: on the work
    itself, and on a person 'for work'. For TV, a for-work that is an episode
    or season climbs P179 to the series.

    Patterns are written most-selective first (the award's statements, then
    their subject), and on WDQS the optimizer is told to keep that order. Left
    to choose, it can start from every award statement in Wikidata and time out
    -- which is what the 504s on the first run looked like."""
    pr = "P166" if result == "won" else "P1411"
    tmdb = TMDB_PROP[kind]
    hint = 'hint:Query hint:optimizer "None" .' if ENDPOINT == "wdqs" else ""
    climb = "?target wdt:P179* ?work ." if kind == "tv" else "BIND(?target AS ?work)"
    return f"""
    SELECT DISTINCT ?work ?tmdb ?year ?shape WHERE {{
      {hint}
      {{
        ?st ps:{pr} wd:{qid} .
        ?target p:{pr} ?st .
        BIND("work" AS ?shape)
      }} UNION {{
        ?st ps:{pr} wd:{qid} .
        ?st pq:P1686 ?target .
        BIND("person" AS ?shape)
      }}
      {climb}
      ?work wdt:{tmdb} ?tmdb .
      OPTIONAL {{ ?st pq:P585 ?t . BIND(YEAR(?t) AS ?year) }}
    }}"""


def stage_fetch(refetch=False, only=None):
    if not AWARDS_JSON.exists():
        sys.exit("run --stage resolve first")
    awards = json.loads(AWARDS_JSON.read_text(encoding="utf-8"))
    done = set()
    kept = []
    if ROWS_JSONL.exists() and not refetch:
        for line in ROWS_JSONL.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if only and r["name"] in only:
                continue
            kept.append(r)
            done.add((r["kind"], r["name"]))
        # an award that legitimately returned zero rows leaves no trace in the
        # rows file, so remember those separately
    zero_path = RAW / "zero_row_awards.json"
    zeros = set(map(tuple, json.loads(zero_path.read_text()))) if zero_path.exists() and not refetch else set()
    done |= {z for z in zeros if not (only and z[1] in only)}

    new_rows, n_failed, stop = [], 0, False
    for kind, lst in awards.items():
        for a in lst:
            if not a.get("qid") or (kind, a["name"]) in done:
                continue
            if only and a["name"] not in only:
                continue
            res, failed = [], None
            for result in ("won", "nominated"):
                try:
                    for b in sparql(_award_query(a["qid"], kind, result)):
                        b["result"] = {"value": result}
                        res.append(b)
                except Exception as e:
                    failed = f"{type(e).__name__}: {e}"
                    break
            if failed:
                n_failed += 1
                print(f"  FAILED {a['name']}: {failed}  (rerun fetch to retry)")
                if n_failed >= 3:
                    print("\n  Three awards failed. The endpoint is struggling, so stopping")
                    print("  here rather than burning time. Everything fetched so far is")
                    print("  saved. Rerun later, or with --endpoint qlever.")
                    stop = True
                    break
                continue
            rows = []
            for b in res:
                try:
                    tid = int(_val(b, "tmdb"))
                except (TypeError, ValueError):
                    continue
                y = _val(b, "year")
                rows.append({"kind": kind, "name": a["name"], "qid": a["qid"],
                             "org": a["org"], "category": a["category"], "tier": a["tier"],
                             "tmdb_id": tid, "work_qid": _qid(_val(b, "work")),
                             "result": _val(b, "result"), "shape": _val(b, "shape"),
                             "year": int(float(y)) if y else None})
            shapes = Counter(r["shape"] for r in rows)
            print(f"  {kind:<4} {a['name'][:62]:<64} {len(rows):>5} rows"
                  f"  (work {shapes.get('work', 0)}, person {shapes.get('person', 0)})")
            if not rows:
                zeros.add((kind, a["name"]))
            new_rows += rows
        if stop:
            break

    RAW.mkdir(parents=True, exist_ok=True)
    with ROWS_JSONL.open("w", encoding="utf-8") as f:
        for r in kept + new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    zero_path.write_text(json.dumps(sorted(zeros)), encoding="utf-8")
    print(f"\nwrote {ROWS_JSONL} ({len(new_rows):,} new rows, {len(kept):,} kept)")
    if n_failed:
        print(f"{n_failed} award(s) failed and were NOT recorded; rerun fetch to get them.")
    if zeros:
        print(f"{len(zeros)} award(s) returned 0 rows: {sorted(n for _, n in zeros)}")
        print("Usually a wrong qid. Fix it in awards.json and rerun with --only \"<name>\".")
    print("Run --stage build next.")


def _catalogue(kind):
    """The corpus app.py serves, via the same constants the Space reads."""
    sys.path.insert(0, str(REPO / "tests"))
    from ranker_loader import resolve_artifacts
    art = resolve_artifacts(APP_PY, PROCESSED, kind=("movie" if kind == "film" else "tv"),
                            verbose=False)
    df = pd.read_parquet(art["CORPUS"])
    id_col, title_col = ("id", "title") if kind == "film" else ("tmdb_id", "name")
    df = df.assign(_tid=pd.to_numeric(df[id_col], errors="coerce"))
    return df, title_col


def stage_build():
    if not ROWS_JSONL.exists():
        sys.exit("run --stage fetch first")
    raw = pd.DataFrame([json.loads(l) for l in
                        ROWS_JSONL.read_text(encoding="utf-8").splitlines()])
    for kind, out_name in (("film", "film_awards.parquet"), ("tv", "tv_awards.parquet")):
        sub = raw[raw["kind"] == kind]
        if sub.empty:
            print(f"\n{kind}: no rows")
            continue
        # One row per (title, award, year). A win is usually also recorded as
        # a nomination, and one film can be nominated twice in a category
        # (two actors): collapse to the best result.
        sub = sub.assign(_rank=(sub["result"] == "won").astype(int),
                         year=sub["year"].astype("Int64"))
        sub = (sub.sort_values("_rank", ascending=False)
                  .drop_duplicates(["tmdb_id", "qid", "year"])
                  .drop(columns="_rank"))

        df, title_col = _catalogue(kind)
        in_cat = set(df["_tid"].dropna().astype(int))
        title_of = dict(zip(df["_tid"].dropna().astype(int), df.loc[df["_tid"].notna(), title_col]))
        hit = sub[sub["tmdb_id"].isin(in_cat)].copy()

        out = hit[["tmdb_id", "org", "category", "year", "result", "tier", "qid"]].rename(
            columns={"qid": "award_qid"}).reset_index(drop=True)
        out.to_parquet(PROCESSED / out_name, index=False)

        titles = out["tmdb_id"].nunique()
        major = out[out["tier"] == "major"]
        print(f"\n=== {kind.upper()} " + "=" * 60)
        print(f"  {len(sub):,} award rows from Wikidata -> {len(out):,} on titles in the "
              f"catalogue ({titles:,} distinct titles)")
        print(f"  bonus-eligible (major): {major['tmdb_id'].nunique():,} titles "
              f"({(major['result'] == 'won').groupby(major['tmdb_id']).any().sum():,} with a major win)")
        missing = sub[~sub["tmdb_id"].isin(in_cat)]["tmdb_id"].nunique()
        print(f"  award titles NOT in the catalogue: {missing:,} (not added; badge needs a row)")
        print(f"\n  {'org':<20}{'category':<44}{'rows':>6}{'won':>6}")
        g = out.groupby(["org", "category"])
        for (org, cat), grp in sorted(g, key=lambda x: (x[0][0], -len(x[1]))):
            print(f"  {org:<20}{cat[:42]:<44}{len(grp):>6}{(grp['result'] == 'won').sum():>6}")

        # The checks that caught every past label bug were READING EXAMPLES.
        print("\n  most-awarded titles (should look like the obvious ones):")
        top = out.groupby("tmdb_id").size().sort_values(ascending=False).head(8)
        for tid, n in top.items():
            w = ((out["tmdb_id"] == tid) & (out["result"] == "won")).sum()
            print(f"    {str(title_of.get(tid, tid))[:40]:<42} {n:>3} rows, {w} won")
        print("\n  random sample (read these):")
        for _, r in out.sample(min(12, len(out)), random_state=7).iterrows():
            award = f"{r.org}: {r.category}"
            print(f"    {str(title_of.get(r.tmdb_id, r.tmdb_id))[:34]:<36}"
                  f"{award[:46]:<48}{r.result:<10}{r.year if pd.notna(r.year) else ''}")
        fan = out.groupby("tmdb_id").size()
        if (fan > 60).any():
            print(f"\n  WARNING: {int((fan > 60).sum())} title(s) carry 60+ award rows. Real record "
                  "holders top out around 30-40; more is a broken join.")
        print(f"\n  wrote {PROCESSED / out_name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["check", "resolve", "probe", "fetch", "build"])
    ap.add_argument("--refetch", action="store_true", help="fetch: redo every award")
    ap.add_argument("--only", nargs="+", default=None,
                    help="fetch: (re)fetch just these award names")
    ap.add_argument("--set", nargs="+", default=None, metavar='"NAME=QID"',
                    help='probe: save a qid by hand, e.g. --set "Golden Bear=Q12345"')
    ap.add_argument("--endpoint", choices=sorted(ENDPOINTS), default="wdqs",
                    help="SPARQL endpoint for check and fetch. qlever = an "
                         "independent copy of Wikidata, for when WDQS is down")
    args = ap.parse_args()
    global ENDPOINT
    ENDPOINT = args.endpoint
    if args.stage == "check":
        stage_check()
    elif args.stage == "resolve":
        stage_resolve()
    elif args.stage == "probe":
        if args.set:
            stage_set(args.set)
        else:
            stage_probe(args.only)
    elif args.stage == "fetch":
        stage_fetch(refetch=args.refetch, only=args.only)
    else:
        stage_build()


if __name__ == "__main__":
    main()
