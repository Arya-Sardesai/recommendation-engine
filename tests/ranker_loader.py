"""
ranker_loader.py — one way to pull the deployed ranker into a test harness.

WHY THIS EXISTS
eval_loo.py and eval_movielens.py each carried their own copy of an
AST loader that whitelisted names by hand:

    wanted_assign = {"LANG_NAMES", "_TITLE_STOPWORDS"}
    wanted_funcs  = {"lang_label", "_title_tokens", "_is_title_dup",
                     "recommend_movies"}

Every new module-level constant in app.py breaks that silently-ish: adding
MOVIE_MIN_RUNTIME and MOVIE_KIDS_FILTER as defaulted parameters made both
harnesses die with `NameError: name 'MOVIE_MIN_RUNTIME' is not defined` at
compile time, in two places, for one change.

Taking EVERY top-level FunctionDef / AsyncFunctionDef / ClassDef / Assign
tracks app.py automatically. Module-level UI code is skipped because it is not
a definition, and definitions that need Streamlit at import time fail
individually and are recorded rather than swallowed.

THE SECOND BUG THIS FIXES
Both harnesses also hardcoded:

    CORPUS_PATH = PROCESSED / "movies.parquet"

which is no longer what the app serves. An eval run against the pre-enrichment
parquet has no cert_tier column, so the kids-content filter is inert by design
and the harness measures a ranker that is NOT the deployed one -- while
printing numbers that look completely normal. resolve_artifacts() reads the
B_/M_/T_ constants out of app.py instead, so the harness follows the app.

USAGE
    from ranker_loader import load_ranker, resolve_artifacts
    ns = load_ranker(APP_PY)
    paths = resolve_artifacts(APP_PY, PROCESSED, kind="movie")
"""
import ast
import collections
import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


class _StubST:
    """Streamlit stand-in so app.py's top-level defs exec headlessly.

    Cache decorators appear both bare (@st.cache_resource) and called
    (@st.cache_data(show_spinner=False)), so both forms must work.
    """
    def _cache(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return lambda f: f
    cache_data = cache_resource = property(lambda self: self._cache)

    def __getattr__(self, name):
        return lambda *a, **k: None


def load_ranker(app_path, require=("recommend_movies",), verbose=True):
    """Exec every top-level definition from app.py and return the namespace."""
    app_path = Path(app_path)
    if not app_path.exists():
        sys.exit(f"{app_path} not found — check the path")

    try:
        import faiss
    except ImportError:
        faiss = None

    src = app_path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    ns = {
        "np": np, "pd": pd, "faiss": faiss, "st": _StubST(),
        "json": json, "re": re, "unicodedata": unicodedata,
        "Path": Path, "sys": sys,
        "SequenceMatcher": difflib.SequenceMatcher,
        "Counter": collections.Counter,
        "defaultdict": collections.defaultdict,
        "hf_hub_download": None,
    }

    parts, ok, failures = [], 0, {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Assign)):
            seg = ast.get_source_segment(src, node)
            if seg:
                parts.append(seg)

    for seg in parts:
        try:
            exec(seg, ns)
            ok += 1
        except Exception as e:
            failures[seg.strip().splitlines()[0][:60]] = f"{type(e).__name__}: {e}"

    missing = [n for n in require if n not in ns]
    if missing:
        print(f"\ncould not load {missing} from {app_path}. Exec failures:")
        for head, err in list(failures.items())[:15]:
            print(f"   {head:62s} {err}")
        sys.exit(1)

    if verbose:
        print(f"  loaded {ok}/{len(parts)} top-level definitions from app.py")
    return ns


_PREFIX = {"movie": "M_", "book": "B_", "tv": "T_"}


def resolve_artifacts(app_path, processed_dir, kind="movie", verbose=True):
    """Read the artifact filenames app.py actually loads.

    Returns a dict of {suffix: Path}, e.g. {"CORPUS": .../movies_v2_meta.parquet}.
    Hardcoding these in a harness is how an eval ends up measuring a corpus the
    app stopped serving.
    """
    src = Path(app_path).read_text(encoding="utf-8")
    pre = _PREFIX[kind]
    out = {}
    pat = re.compile(
        rf'^\s*{pre}([A-Z0-9_]+)\s*=\s*["\']([^"\']+\.(?:parquet|npy|index))["\']',
        re.MULTILINE)
    for m in pat.finditer(src):
        out[m.group(1)] = Path(processed_dir) / m.group(2)
    if verbose:
        print(f"  artifacts from app.py ({pre}*):")
        for k, v in out.items():
            mark = "" if v.exists() else "   <- MISSING LOCALLY"
            print(f"    {pre + k:12s} {v.name}{mark}")
    return out
