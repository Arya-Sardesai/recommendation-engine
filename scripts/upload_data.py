"""
Upload data artifacts to the Hugging Face dataset repo.
The deployed Space downloads these files at startup.

v2: added the movie artifacts alongside the books artifacts so the unified
two-tab Space can serve both modalities.
v3: added the TV artifacts for the third tab (no tv_tags - TV ships tagless in v1).
v4: path fix for scripts/ location (parents[1], not parent) + optional CLI
    args to upload a subset:  python scripts/upload_data.py movies.parquet tv_corpus.parquet
    No args = upload everything in FILES (original behavior).
"""
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

USERNAME = "Arya2305"
DATASET_REPO = f"{USERNAME}/recommendation-engine-data"

REPO_ROOT = Path(__file__).resolve().parents[1]   # scripts/ -> repo root
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

FILES = [
    # --- books ---
    "books_v1.parquet",
    "embeddings_minilm_v1.npy",
    "faiss_minilm_v1.index",
    "my_matched_ratings.json",
    "book_tags.parquet",
    "embeddings_bgem3.npy",            # books BGE-M3
    "faiss_bgem3.index",
    "embeddings_bgem3_fp16.npy",
    # --- movies ---
    "movies.parquet",
    "movie_embeddings_minilm_v1.npy",
    "movie_faiss_minilm_v1.index",
    "movie_tags.parquet",
    "movie_embeddings_bgem3_v2_kw.npy",   # movies BGE-M3
    "movie_faiss_bgem3_v2_kw.index",
    # --- tv (v1: no tag file, ships tagless) ---
    "tv_corpus.parquet",
    "tv_embeddings_minilm_v1.npy",
    "tv_faiss_minilm_v1.index",
    "tv_embeddings_bgem3_v2_notitle.npy", # tv BGE-M3
    "tv_faiss_bgem3_v2_notitle.index",
]

# Optional CLI subset: python scripts/upload_data.py file1 file2 ...
if len(sys.argv) > 1:
    requested = sys.argv[1:]
    unknown = [f for f in requested if f not in FILES]
    if unknown:
        print(f"WARNING: not in the known FILES list: {unknown}")
        print("Uploading them anyway if they exist in data/processed/.")
    FILES = requested

api = HfApi()

# create the dataset repo (private=False so the Space can read it without auth)
print(f"Dataset repo: {DATASET_REPO}")
create_repo(DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)

uploaded, missing = 0, 0
for fname in FILES:
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
if missing == len(FILES):
    print("EVERYTHING was missing - check PROCESSED_DIR path resolution!")
print(f"Repo: https://huggingface.co/datasets/{DATASET_REPO}")