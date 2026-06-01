"""
One-time script: upload the v0 data artifacts to a Hugging Face dataset repo.
Run once. The deployed Space downloads these files at startup.
"""
from pathlib import Path
from huggingface_hub import HfApi, create_repo

USERNAME = "Arya2305"
DATASET_REPO = f"{USERNAME}/recommendation-engine-data"

REPO_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

FILES = [
    "books_v1.parquet",
    "embeddings_minilm_v1.npy",
    "faiss_minilm_v1.index",
    "my_matched_ratings.json",
]

api = HfApi()

# create the dataset repo (private=False so the Space can read it without auth)
print(f"Creating dataset repo: {DATASET_REPO}")
create_repo(DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)

# upload each file
for fname in FILES:
    local_path = PROCESSED_DIR / fname
    if not local_path.exists():
        print(f"  MISSING: {fname} — skipping")
        continue
    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"  Uploading {fname} ({size_mb:.1f} MB)...")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=fname,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"    done")

print(f"\nAll files uploaded to: https://huggingface.co/datasets/{DATASET_REPO}")