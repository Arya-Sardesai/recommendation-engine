"""
space_ranker.py -- the local agents call the Space's own rankers.

WHY THIS EXISTS
The agents used to carry their own ranking code and their own hard-coded file
names, and both drifted. By October the books agent was still reading the 2017
Goodreads corpus, the movies agent paired a 121,303-row parquet with
109,921-row embeddings (new films were invisible or crashed FAISS), and none
of the three had the quality floor, the dislike penalty or the tag layer.

This module removes the copy instead of refreshing it:

  * load_ranker() execs every top-level definition in hf-space/app.py
    (tests/ranker_loader.py, the same loader the evals use), so
    recommend_books / recommend_movies / recommend_tv ARE the Space's.
  * The Space's own loaders (load_books_corpus, load_books_tags, ...) are
    reused with _dl() pointed at data/processed instead of the Hub, so the
    agent reads exactly the files the B_/M_/T_ constants name.
  * The arguments each Space tab passes to its ranker are read out of app.py's
    call site (see _space_call), so if the Books tab starts passing
    tag_idf=1.0, the agent passes it too on its next start.

Change app.py and the agent follows. Nothing in here has to be kept in sync by
hand.

Paths: app.py is expected at ../hf-space/app.py next to this repo; set
SPACE_APP_PY to override. Data is read from data/processed; set
REC_PROCESSED_DIR to override.
"""
import ast
import difflib
import inspect
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
PROCESSED_DIR = Path(os.environ.get("REC_PROCESSED_DIR", str(REPO_ROOT / "data" / "processed")))
APP_PY = Path(os.environ.get("SPACE_APP_PY", str(REPO_ROOT.parent / "hf-space" / "app.py")))
DEFAULT_PROFILE = Path.home() / "Downloads" / "my-recommendation-profile.json"

_TESTS = str(REPO_ROOT / "tests")
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)
from ranker_loader import load_ranker, resolve_artifacts  # noqa: E402


# What each kind is called inside app.py, and the keys its ratings use.
_SPEC = {
    "book": {
        "corpus": "load_books_corpus", "index": "load_books_index",
        "tags": "load_books_tags", "search_index": "build_book_search_index",
        "search": "search_books", "rank": "recommend_books",
        "id_col": "book_id", "title_col": "title", "year_col": "publication_year",
        "people_col": "primary_author", "pop_col": "ratings_count",
        "rating_id": "book_id", "people_key": "matched_author",
        "ratings_key": "ratings", "skipped_key": "skipped",
    },
    "movie": {
        "corpus": "load_movies_corpus", "index": "load_movies_index",
        "tags": "load_movies_tags", "search_index": "build_movie_search_index",
        "search": "search_movies", "rank": "recommend_movies",
        "id_col": "id", "title_col": "title", "year_col": "release_year",
        "people_col": "director", "pop_col": "vote_count",
        "rating_id": "movie_id", "people_key": "matched_director",
        "ratings_key": "movie_ratings", "skipped_key": "movie_skipped",
    },
    "tv": {
        "corpus": "load_tv_corpus", "index": "load_tv_index",
        "tags": "load_tv_tags", "search_index": "build_tv_search_index",
        "search": "search_tv", "rank": "recommend_tv",
        "id_col": "tmdb_id", "title_col": "name", "year_col": "start_year",
        "people_col": "created_by", "pop_col": "vote_count",
        "rating_id": "tv_id", "people_key": "matched_creator",
        "ratings_key": "tv_ratings", "skipped_key": "tv_skipped",
    },
}

# Call-site arguments that are expressions in app.py (session state, loaded
# frames) rather than literals. The agent supplies its own equivalent.
_DYNAMIC_ARGS = {"tags_df", "skip_ids", "forms"}

_NS = None
_BUNDLES = {}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def space_ns():
    """The Space's app.py, exec'd headlessly. Loaded once per process."""
    global _NS
    if _NS is None:
        if not APP_PY.exists():
            raise FileNotFoundError(
                f"Space app.py not found at {APP_PY}. Set SPACE_APP_PY to the "
                f"path of hf-space/app.py.")
        need = {s[k] for s in _SPEC.values()
                for k in ("corpus", "index", "tags", "search_index", "search", "rank")}
        need |= {"IndexVectors", "load_books_forms", "_migrate_book_ratings",
                 "_drop_stale_indices", "_normalize", "lang_label", "network_label",
                 "clean_genres", "_book_genre_allow", "_creator_names"}
        ns = load_ranker(APP_PY, require=tuple(sorted(need)), verbose=False)
        # The Space downloads from the Hub; locally the same filenames sit in
        # data/processed. Every loader calls _dl(), so this one line is the
        # whole redirect.
        ns["_dl"] = lambda fname: str(PROCESSED_DIR / fname)
        _NS = ns
    return _NS


def _space_call(kind):
    """The keyword arguments the Space's main tab passes to its ranker.

    Returns (literal_kwargs, keyword_names, problems). Literal values (n=20,
    tag_idf=1.0, ...) are copied as-is. Expressions the agent cannot copy are
    fine when they are in _DYNAMIC_ARGS and reported as problems otherwise.
    The genre-tab pool call (flat_ranking=True) is skipped: that is a
    different list.
    """
    name = _SPEC[kind]["rank"]
    tree = ast.parse(APP_PY.read_text(encoding="utf-8"))
    found = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == name
                and "flat_ranking" not in {k.arg for k in node.keywords}):
            found = node
            break
    if found is None:
        return {}, set(), [f"no call to {name}() found in app.py"]
    literal, keywords, problems = {}, set(), []
    for k in found.keywords:
        if k.arg is None:
            problems.append("**kwargs at the call site")
            continue
        keywords.add(k.arg)
        try:
            literal[k.arg] = ast.literal_eval(k.value)
        except (ValueError, SyntaxError, TypeError):
            if k.arg not in _DYNAMIC_ARGS:
                problems.append(f"{k.arg}=<expression the agent cannot copy>")
    return literal, keywords, problems


def _defaults(fn):
    return {k: p.default for k, p in inspect.signature(fn).parameters.items()
            if p.default is not inspect.Parameter.empty}


def _num(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _servable_mask(kind, df, ns):
    """Rows the Space's ranker could ever return, from its own defaults.

    Popularity floor + Bayesian-shrunk quality floor (+ runtime floor for
    films), read from the ranker's signature so a changed default carries
    over. Profile-dependent filters (kids, maturity, rated authors) are not
    here; they need a profile. Search tools use this so they never surface a
    title the website would refuse to recommend.
    """
    if kind == "book":
        d = _defaults(ns["recommend_books"])
        rc = _num(df, "ratings_count").fillna(0)
        ok = rc >= d["min_ratings"]
        if "source" in df.columns:
            ok |= df["source"].astype(str).eq("targeted_authors")
        if "average_rating" in df.columns:
            ar = _num(df, "average_rating")
            c = ar[rc >= d["min_ratings"]].mean()
            c = float(c) if np.isfinite(c) else 3.9
            m = d["shrink_ratings"]
            adj = (rc * ar + m * c) / (rc + m)
            ok &= ~(ar.notna() & (adj < d["min_avg_rating"]))
        return ok.to_numpy()

    d = _defaults(ns[_SPEC[kind]["rank"]])
    vc = _num(df, "vote_count").fillna(0)
    va = _num(df, "vote_average")
    c = float(va[vc >= d["min_votes"]].mean())
    m = d["shrink_votes"]
    adj = (vc * va + m * c) / (vc + m)
    ok = (vc >= d["min_votes"]) & va.notna() & (adj >= d["min_vote_average"])
    if kind == "movie" and d.get("min_runtime") and "runtime" in df.columns:
        rt = _num(df, "runtime")
        ok &= ~(rt.notna() & (rt > 0) & (rt < d["min_runtime"]))
    return ok.to_numpy()


def load(kind):
    """Everything the agent needs for one kind, loaded the Space's way.

    Refuses to load when parquet rows, .npy rows and index.ntotal disagree:
    the old movies agent ran for two months on misaligned vectors without a
    single error.
    """
    if kind in _BUNDLES:
        return _BUNDLES[kind]
    if kind not in _SPEC:
        raise ValueError(f"kind must be one of {list(_SPEC)}")
    spec, ns = _SPEC[kind], space_ns()

    files = resolve_artifacts(APP_PY, PROCESSED_DIR, kind=kind, verbose=False)
    missing = [str(files.get(k, k)) for k in ("CORPUS", "FAISS")
               if k not in files or not files[k].exists()]
    if missing:
        raise FileNotFoundError(
            f"{kind}: app.py serves files that are not in {PROCESSED_DIR}: "
            f"{missing}. Download them from the data repo.")

    df = ns[spec["corpus"]]()
    index = ns[spec["index"]]()
    tags = ns[spec["tags"]]()                      # None if missing, as on the Space
    forms = ns["load_books_forms"]() if kind == "book" else {}

    rows = {"parquet": len(df), "faiss": int(index.ntotal)}
    emb = files.get("EMB")
    if emb is not None and emb.exists():
        try:
            rows["npy"] = int(np.load(emb, mmap_mode="r").shape[0])
        except Exception:
            pass
    if len(set(rows.values())) != 1:
        raise RuntimeError(
            f"{kind}: row counts disagree {rows}. Parquet rows must equal .npy "
            f"rows and index.ntotal; refusing to rank against misaligned vectors.")

    literal, keywords, problems = _space_call(kind)
    b = {
        "kind": kind, "spec": spec, "ns": ns, "df": df, "index": index,
        "emb": ns["IndexVectors"](index), "tags": tags, "forms": forms or {},
        "files": {k: p.name for k, p in files.items()}, "rows": rows,
        "call_literal": literal, "call_keywords": keywords,
        "call_problems": problems,
    }
    b["search_idx"] = ns[spec["search_index"]](df)
    b["pos_by_id"] = {str(v): i for i, v in enumerate(df[spec["id_col"]])}
    b["servable"] = _servable_mask(kind, df, ns)
    b["title_norms"] = frozenset(t for t in (norm_title(x) for x in df[spec["title_col"]]) if t)
    _BUNDLES[kind] = b
    return b


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------
def read_profile(raw, kind):
    """(ratings, skipped) for one kind from the website's exported JSON.

    Also accepts a bare list (the old my_matched_ratings.json format).
    """
    data = json.loads(raw)
    if isinstance(data, list):
        return data, []
    d = data.get("data") if isinstance(data, dict) else None
    if not isinstance(d, dict):
        raise ValueError("not a profile exported from the website")
    spec = _SPEC[kind]
    return list(d.get(spec["ratings_key"]) or []), list(d.get(spec["skipped_key"]) or [])


def normalize_ratings(bundle, ratings):
    """Put ratings in the exact shape the Space ranker reads.

    Books go through the Space's own _migrate_book_ratings (re-matches stale
    rows by title/author). Films and TV are re-pointed by id first, since ids
    are stable and row positions are not, then bounds-checked. Old TV ratings
    stored under 'tmdb_id' are renamed to 'tv_id'.
    Returns (ratings, note).
    """
    kind, spec, df, ns = bundle["kind"], bundle["spec"], bundle["df"], bundle["ns"]
    items = []
    for r in ratings or []:
        if not isinstance(r, dict) or r.get("rating") is None:
            continue
        r = dict(r)
        if kind == "tv" and "tv_id" not in r and "tmdb_id" in r:
            r["tv_id"] = r.pop("tmdb_id")
        try:
            r["corpus_index"] = int(r["corpus_index"])
        except (KeyError, TypeError, ValueError):
            r["corpus_index"] = None
        if kind != "book":
            rid = r.get(spec["rating_id"])
            pos = bundle["pos_by_id"].get(str(rid)) if rid is not None else None
            if pos is not None:
                r["corpus_index"] = pos
        items.append(r)

    notes = []
    if kind == "book":
        items, mig, drop, merged = ns["_migrate_book_ratings"](items, df)
        if mig:
            notes.append(f"{mig} re-matched to the current catalogue")
        if merged:
            notes.append(f"{merged} duplicates merged")
        if drop:
            notes.append(f"{drop} not in the current catalogue")
    else:
        items, drop = ns["_drop_stale_indices"](items, len(df))
        if drop:
            notes.append(f"{drop} not in the current catalogue")

    out, seen = [], set()
    for r in items:
        row = df.iloc[int(r["corpus_index"])]
        rid = row[spec["id_col"]]
        if str(rid) in seen:
            continue
        seen.add(str(rid))
        r[spec["rating_id"]] = rid if kind == "book" else int(rid)
        if not r.get("matched_title"):
            r["matched_title"] = str(row[spec["title_col"]])
        pk, pc = spec["people_key"], spec["people_col"]
        if not r.get(pk) and pc in df.columns and isinstance(row[pc], str):
            r[pk] = row[pc]
        r["rating"] = float(r["rating"])
        out.append(r)
    return out, "; ".join(notes)


def catalogue_rating(bundle, pos, rating):
    """A rating dict for corpus row `pos`, shaped like the Space's own."""
    spec, df = bundle["spec"], bundle["df"]
    row = df.iloc[int(pos)]
    rid = row[spec["id_col"]]
    r = {spec["rating_id"]: rid if bundle["kind"] == "book" else int(rid),
         "matched_title": str(row[spec["title_col"]]),
         "corpus_index": int(pos), "rating": float(rating)}
    pc = spec["people_col"]
    if pc in df.columns and isinstance(row[pc], str):
        r[spec["people_key"]] = row[pc]
    return r


def resolve_title(bundle, title, ratings):
    """Corpus position for a title: the user's ratings first (exact, then
    contains), then the Space's own search. None if nothing matches."""
    t = (title or "").strip().lower()
    if not t:
        return None
    for exact in (True, False):
        for r in ratings:
            mt = str(r.get("matched_title", "")).lower()
            if mt and (t == mt if exact else (t in mt or mt in t)):
                return int(r["corpus_index"])
    hits = bundle["ns"][bundle["spec"]["search"]](title, bundle["df"], bundle["search_idx"], limit=1)
    return int(hits.index[0]) if len(hits) else None


def excluded_positions(bundle, ratings, skipped=None):
    """Rated rows plus 'not interested' rows."""
    out = {int(r["corpus_index"]) for r in ratings if r.get("corpus_index") is not None}
    for s in skipped or []:
        p = bundle["pos_by_id"].get(str(s))
        if p is not None:
            out.add(p)
    return out


def rated_people(bundle, ratings):
    """Lower-cased authors / directors / creators the user has rated."""
    key, out = bundle["spec"]["people_key"], set()
    for r in ratings:
        v = r.get(key)
        if not isinstance(v, str) or not v.strip():
            continue
        if bundle["kind"] == "tv":
            out |= bundle["ns"]["_creator_names"](v)
        else:
            out.add(v.strip().lower())
    return out


def shares_people(bundle, pos, people):
    """True if row `pos` is by someone in `people` (the Space's exclusion rule)."""
    if not people:
        return False
    col, df = bundle["spec"]["people_col"], bundle["df"]
    if col not in df.columns:
        return False
    v = df.iloc[int(pos)][col]
    if not isinstance(v, str) or not v:
        return False
    if bundle["kind"] == "tv":
        return bool(bundle["ns"]["_creator_names"](v) & people)
    return v.strip().lower() in people


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def rank(bundle, ratings, skipped=None, n=None, anchor_positions=None):
    """The Space's ranker, called the way the Space's tab calls it.

    With no anchors this returns exactly the website's list for the profile.

    anchor_positions steers the same ranker toward particular titles. Liked
    ratings that are NOT anchors are passed at the neutral score for this call
    only: a neutral rating still counts as read and still excludes its
    author, but stops pulling the list toward itself. Dislikes are untouched.
    An anchor the user never rated is added as a 4-star rating.

    Returns (recs DataFrame, taste-signature Series or None).
    """
    ns, spec = bundle["ns"], bundle["spec"]
    fn = ns[spec["rank"]]
    ratings = [dict(r) for r in ratings]

    if anchor_positions:
        d = _defaults(fn)
        like, neutral = float(d.get("like_threshold", 3.5)), float(d.get("neutral", 3.0))
        anchors = {int(p) for p in anchor_positions}
        have = set()
        for r in ratings:
            ci = int(r["corpus_index"])
            if ci in anchors:
                r["rating"] = max(float(r["rating"]), 4.0)
                have.add(ci)
            elif float(r["rating"]) >= like:
                r["rating"] = neutral
        ratings += [catalogue_rating(bundle, p, 4.0) for p in sorted(anchors - have)]

    kwargs = dict(bundle["call_literal"])
    dynamic = {"tags_df": bundle["tags"], "skip_ids": list(skipped or []),
               "forms": bundle["forms"]}
    for k in bundle["call_keywords"]:
        if k in dynamic:
            kwargs[k] = dynamic[k]
    if n is not None:
        kwargs["n"] = int(n)
    return fn(ratings, bundle["df"], bundle["emb"], bundle["index"], **kwargs)


# ---------------------------------------------------------------------------
# Tags (shared by the three agents' search_by_tags)
# ---------------------------------------------------------------------------
def _tagkey(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _tag_vocab(bundle):
    if "tag_vocab" not in bundle:
        tags = bundle["tags"]
        cols = list(tags.columns) if tags is not None else []
        bundle["tag_vocab"] = {_tagkey(c): c for c in cols}
        counts = (tags.values > 0).sum(axis=0) if cols else []
        bundle["tag_popular"] = [c for _, c in sorted(zip(counts, cols), reverse=True)]
    return bundle["tag_vocab"]


def common_tags(bundle, k=40):
    _tag_vocab(bundle)
    return [str(c) for c in bundle["tag_popular"][:k]]


def resolve_tags(bundle, names):
    """Map loose tag names ('Slow Burn', 'slow_burn') to real columns.
    Returns (columns, unknown_names, {unknown: close matches})."""
    vocab = _tag_vocab(bundle)
    ok, bad = [], []
    for n in names or []:
        c = vocab.get(_tagkey(n))
        if c is not None and c not in ok:
            ok.append(c)
        elif c is None:
            bad.append(n)
    keys = list(vocab)
    sugg = {n: [str(vocab[k]) for k in difflib.get_close_matches(_tagkey(n), keys, n=3, cutoff=0.6)]
            for n in bad}
    return ok, bad, sugg


def _tag_floor(bundle, col):
    """25th percentile of the tag's non-zero scores: below it the tag is a
    faint trace (a romance subplot), at or above it the item really is that."""
    cache = bundle.setdefault("tag_floor", {})
    if col not in cache:
        v = bundle["tags"][col].to_numpy()
        nz = v[v > 0]
        cache[col] = float(np.percentile(nz, 25)) if len(nz) else float("inf")
    return cache[col]


def avoid_reason(bundle, pos, avoid_names, avoid_cols):
    """Why row `pos` should be dropped for an 'avoid' request, or None.

    Two signals, either is enough: the item carries one of the tags strongly
    (see _tag_floor), or one of its genres IS the avoided word ('Romance').
    Genres catch untagged items; the tag floor keeps a novel with a minor
    love story from being thrown out for 'no romance'.
    """
    row = bundle["df"].iloc[int(pos)]
    keys = {_tagkey(n) for n in avoid_names or []}
    if bundle["kind"] == "book":
        genres = as_list(row.get("genres"))
    else:
        genres = split_genres(row.get("genres"))
    for g in genres:
        if _tagkey(g) in keys:
            return f"genre {g}"
    tags = bundle["tags"]
    if tags is not None and avoid_cols:
        rid = str(row[bundle["spec"]["id_col"]])
        if rid in tags.index:
            trow = tags.loc[rid]
            for c in avoid_cols:
                if trow[c] > 0 and trow[c] >= _tag_floor(bundle, c):
                    return f"tag {c}"
    return None


def tag_search(bundle, pos_tags, neg_tags, skip_positions, keep=None, limit=15):
    """Rows scoring high on pos_tags and low on neg_tags, servable only.
    Ties (tag scores are coarse) are broken by log popularity.
    Returns [(position, score, top_tags)]."""
    tags, df = bundle["tags"], bundle["df"]
    score = tags[pos_tags].sum(axis=1) if pos_tags else pd.Series(0.0, index=tags.index)
    if neg_tags:
        score = score - tags[neg_tags].sum(axis=1)
    score = score[score > 0]
    ids = score.index.to_numpy()
    vals = score.to_numpy(dtype="float64")
    pos = np.array([bundle["pos_by_id"].get(str(i), -1) for i in ids], dtype=np.int64)
    m = pos >= 0
    ids, vals, pos = ids[m], vals[m], pos[m]
    m = bundle["servable"][pos]
    ids, vals, pos = ids[m], vals[m], pos[m]
    pop = _num(df, bundle["spec"]["pop_col"]).fillna(0).to_numpy()
    order = np.argsort(-(vals + np.log1p(pop[pos]) * 1e-3), kind="stable")
    out = []
    for o in order:
        p = int(pos[o])
        if p in skip_positions or (keep is not None and not keep(p)):
            continue
        top = tags.loc[ids[o]].nlargest(5)
        out.append((p, float(vals[o]), [str(t) for t, v in top.items() if v > 0]))
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def norm_title(s):
    """Lower-case alphanumerics only: the key the grounding check compares on."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def text(v):
    return v if isinstance(v, str) else ""


def as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    if isinstance(v, float) and v != v:
        return []
    return [v]


def split_genres(v):
    if isinstance(v, str):
        return [g.strip() for g in v.split(",") if g.strip()]
    return [str(g).strip() for g in as_list(v) if str(g).strip()]


def book_genres(bundle, genres, limit=3):
    """Book genres cleaned exactly as the Space's cards clean them."""
    if "genre_allow" not in bundle:
        bundle["genre_allow"] = bundle["ns"]["_book_genre_allow"](bundle["df"])
    allow, counts = bundle["genre_allow"]
    return bundle["ns"]["clean_genres"](as_list(genres), allow, counts, limit=limit)


def year_of(v):
    try:
        if v is None or (isinstance(v, float) and v != v):
            return None
        y = int(v)
    except (TypeError, ValueError):
        return None
    return y if 1000 <= y <= 2100 else None


def year_ok(v, lo=None, hi=None):
    if lo is None and hi is None:
        return True
    y = year_of(v)
    if y is None:
        return False
    return (lo is None or y >= int(lo)) and (hi is None or y <= int(hi))


def has_value(v):
    if v is None:
        return False
    if isinstance(v, float) and v != v:
        return False
    return bool(v)


def short(s, n=220):
    if not isinstance(s, str):
        return ""
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n - 1].rsplit(" ", 1)[0] + "…"


def clean(v):
    """JSON-safe copy: numpy scalars to Python, NaN to None."""
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in v]
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, float):
        return None if v != v else round(v, 4)
    return v
