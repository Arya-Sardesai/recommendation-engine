"""
Upload data artifacts to the Hugging Face dataset repo.
The deployed Space downloads these files at startup.

v2: added the movie artifacts alongside the books artifacts so the unified
two-tab Space can serve both modalities.
v3: added the TV artifacts for the third tab (no tv_tags - TV ships tagless in v1).
v4: path fix for scripts/ location (parents[1], not parent) + optional CLI
    args to upload a subset:  python scripts/upload_data.py movies.parquet tv_corpus.parquet
    No args = upload everything in FILES (original behavior).
v5: added the v2 tag-layer artifacts (movie_tags_v2, tv_tags, book_tags_knn).
    IMPORTANT BEHAVIOUR CHANGE: bare `python scripts/upload_data.py` with no
    args now requires --all. The no-arg default used to re-upload every file
    in FILES, which overwrites LIVE artifacts the Space pulls unpinned at
    startup (app.py has no revision= pin). Named subsets are the safe default.
v6: LIVE is now DERIVED from the deployed app.py's B_/M_/T_ constants instead
    of hand-maintained here. The hardcoded list had drifted: it still named
    movies.parquet-era embeddings and did not know the books v2 Hardcover
    artifacts were live, so the "this will change production" prompt was
    firing on the wrong files in both directions - warning about inert ones
    and staying silent on real ones. A stale safety prompt is worse than none,
    because it trains you to hit y.
    Also adds movies_v2_meta.parquet (runtime + certification + cert_tier).
"""
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

USERNAME = "Arya2305"
DATASET_REPO = f"{USERNAME}/recommendation-engine-data"

REPO_ROOT = Path(__file__).resolve().parents[1]   # scripts/ -> repo root
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SPACE_APP = REPO_ROOT.parent / "hf-space" / "app.py"

# Fallback only, used when app.py cannot be read. Treated as stale by
# definition - the derived set below is the one to trust.
LIVE_FALLBACK = {
    "books_v2_hc.parquet",
    "book_embeddings_bgem3_v2_hc_notitle.npy",
    "book_faiss_bgem3_v2_hc_notitle.index",
    "book_tags_v2_hc.parquet",
    "movies.parquet",
    "movie_tags.parquet",
    "tv_corpus.parquet",
}


def derive_live():
    """Read the artifact filenames the deployed app.py actually loads.

    app.py holds them in module-level string constants (B_CORPUS, M_CORPUS,
    M_FAISS, M_TAGS, T_EMB, ...). Anything named there is served on the next
    cold start; anything not named is inert no matter what it overwrites.
    Parsing the source keeps this in step with whatever is deployed, which a
    hand-edited set here provably did not.
    """
    if not SPACE_APP.exists():
        print(f"NOTE: {SPACE_APP} not found - falling back to the hardcoded")
        print("      LIVE set, which may be out of date. Verify by hand.")
        return set(LIVE_FALLBACK), False

    src = SPACE_APP.read_text(encoding="utf-8")
    live = set()
    # B_CORPUS = "books_v2_hc.parquet"   /   M_FAISS = 'movie_faiss_....index'
    pattern = re.compile(
        r'^\s*(?:B_|M_|T_)[A-Z0-9_]*\s*=\s*["\']([^"\']+\.(?:parquet|npy|index))["\']',
        re.MULTILINE)
    for m in pattern.finditer(src):
        live.add(m.group(1))
    if not live:
        print("NOTE: no B_/M_/T_ artifact constants matched in app.py -")
        print("      falling back to the hardcoded LIVE set.")
        return set(LIVE_FALLBACK), False
    return live, True


FILES = [
    # --- books ---
    # "books_v1.parquet",
    # "embeddings_minilm_v1.npy",
    # "faiss_minilm_v1.index",
    # "my_matched_ratings.json",
    # "book_tags.parquet",
    # "book_tags_knn.parquet",              # kNN-propagated, 99.3% coverage
    # "embeddings_bgem3.npy",               # books BGE-M3
    # "faiss_bgem3.index",
    # "embeddings_bgem3_fp16.npy",
    # --- books v2: Hardcover corpus (current) ---
    "books_v2_hc.parquet",
    "book_embeddings_bgem3_v2_hc_notitle.npy",
    "book_faiss_bgem3_v2_hc_notitle.index",
    "book_tags_v2_hc.parquet",
    # --- books v2b: targeted backfill (NOT built yet - fetch_hardcover_targeted_v2) ---
    # "books_v2_hc2.parquet",
    # --- movies ---
    # "movies.parquet",
    # "movie_embeddings_minilm_v1.npy",
    # "movie_faiss_minilm_v1.index",
    # "movie_tags.parquet",                 # genome-only, superseded by v2
    # "movie_tags_v2.parquet",              # genome + keyword union, 40% coverage
    # "movie_embeddings_bgem3_v2_kw.npy",   # movies BGE-M3
    # "movie_faiss_bgem3_v2_kw.index",
    # --- movies: runtime + certification enrichment (columns only, rows unmoved) ---
    # Embeddings and FAISS are UNCHANGED by this file: enrich_movies_metadata.py
    # asserts row order and refuses to write if it shifted. Upload this alone,
    # confirm it is serving, then flip M_CORPUS.
    "movies_v2_meta.parquet",
    # --- tv ---
    # "tv_corpus.parquet",
    # "tv_tags.parquet",                    # keyword-derived, 49.9% coverage
    # "tv_embeddings_minilm_v1.npy",
    # "tv_faiss_minilm_v1.index",
    # "tv_embeddings_bgem3_v2_notitle.npy", # tv BGE-M3
    # "tv_faiss_bgem3_v2_notitle.index",
]

LIVE, derived = derive_live()

args = sys.argv[1:]

if not args:
    print("Refusing to upload everything without --all.")
    print("The Space pulls these unpinned at startup, so a blanket re-upload")
    print("changes production on the next cold start.\n")
    print("Name the files you mean, e.g.:")
    print("  python scripts/upload_data.py movies_v2_meta.parquet")
    print(f"\nLIVE set {'derived from ' + str(SPACE_APP) if derived else 'HARDCODED (stale)'}:")
    for f in sorted(LIVE):
        print(f"  {f}")
    print("\nKnown files:")
    for f in FILES:
        print(f"  {'[LIVE] ' if f in LIVE else '       '}{f}")
    sys.exit(1)

if args == ["--all"]:
    requested = FILES
else:
    requested = args
    unknown = [f for f in requested if f not in FILES]
    if unknown:
        print(f"WARNING: not in the known FILES list: {unknown}")
        print("Uploading them anyway if they exist in data/processed/.")

touching_live = [f for f in requested if f in LIVE]
if touching_live:
    print("These files are LIVE on the Space and will change production:")
    for f in touching_live:
        print(f"  {f}")
    if not derived:
        print("  (LIVE set is the hardcoded fallback - app.py was unreadable)")
    if input("Continue? [y/N] ").strip().lower() != "y":
        sys.exit("Aborted.")
else:
    inert = ", ".join(requested)
    print(f"None of these are named in app.py, so they are inert on upload: {inert}")
    print("They change nothing until the corresponding constant is flipped.")

api = HfApi()

# create the dataset repo (private=False so the Space can read it without auth)
print(f"Dataset repo: {DATASET_REPO}")
create_repo(DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)

uploaded, missing = 0, 0
for fname in requested:
    local_path = PROCESSED_DIR / fname
    if not local_path.exists():
        print(f"  MISSING: {fname} - skipping")
        missing += 1
        continue
    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"  Uploading {fname} ({size_mb:.1f} MB)...")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=fname,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print("    done")
    uploaded += 1

print(f"\n{uploaded} uploaded, {missing} missing.")
if missing == len(requested):
    print("EVERYTHING was missing - check PROCESSED_DIR path resolution!")
print(f"Repo: https://huggingface.co/datasets/{DATASET_REPO}")
if uploaded and not touching_live:
    print("\nInert upload complete. Flip the app.py constant only after")
    print("probing a known-changed value - the HF CDN etag is not instant.")
