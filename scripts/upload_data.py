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
"""
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

USERNAME = "Arya2305"
DATASET_REPO = f"{USERNAME}/recommendation-engine-data"

REPO_ROOT = Path(__file__).resolve().parents[1]   # scripts/ -> repo root
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Files the deployed Space loads by name (app.py: B_/M_/T_ constants).
# Overwriting one of these changes production on the next cold start.
LIVE = {
    "books_v1.parquet",
    "embeddings_bgem3.npy",
    "faiss_bgem3.index",
    "book_tags.parquet",
    "movies.parquet",
    "movie_embeddings_bgem3_v2_kw.npy",
    "movie_faiss_bgem3_v2_kw.index",
    "movie_tags.parquet",
    "tv_corpus.parquet",
    "tv_embeddings_bgem3_v2_notitle.npy",
    "tv_faiss_bgem3_v2_notitle.index",
}

FILES = [
    # --- books ---
    # "books_v1.parquet",
    # "embeddings_minilm_v1.npy",
    # "faiss_minilm_v1.index",
    # "my_matched_ratings.json",
    # "book_tags.parquet",
    # "book_tags_knn.parquet",              # NEW: kNN-propagated, 99.3% coverage
    # "embeddings_bgem3.npy",               # books BGE-M3
    # "faiss_bgem3.index",
    # "embeddings_bgem3_fp16.npy",
    # --- books v2: Hardcover corpus (current) ---
    "books_v2_hc.parquet",
    "book_embeddings_bgem3_v2_hc_notitle.npy",
    "book_faiss_bgem3_v2_hc_notitle.index",
    "book_tags_v2_hc.parquet",
    # --- movies ---
    # "movies.parquet",
    # "movie_embeddings_minilm_v1.npy",
    # "movie_faiss_minilm_v1.index",
    # "movie_tags.parquet",                 # genome-only, superseded by v2
    # "movie_tags_v2.parquet",              # NEW: genome + keyword union, 40% coverage
    # "movie_embeddings_bgem3_v2_kw.npy",   # movies BGE-M3
    # "movie_faiss_bgem3_v2_kw.index",
    # --- tv ---
    # "tv_corpus.parquet",
    # "tv_tags.parquet",                    # NEW: keyword-derived, 49.9% coverage
    # "tv_embeddings_minilm_v1.npy",
    # "tv_faiss_minilm_v1.index",
    # "tv_embeddings_bgem3_v2_notitle.npy", # tv BGE-M3
    # "tv_faiss_bgem3_v2_notitle.index",
]

args = sys.argv[1:]

if not args:
    print("Refusing to upload everything without --all.")
    print("The Space pulls these unpinned at startup, so a blanket re-upload")
    print("changes production on the next cold start.\n")
    print("Name the files you mean, e.g.:")
    print("  python scripts/upload_data.py movie_tags_v2.parquet tv_tags.parquet")
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
    if input("Continue? [y/N] ").strip().lower() != "y":
        sys.exit("Aborted.")

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