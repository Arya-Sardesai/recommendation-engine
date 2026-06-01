"""
Test the recommendation agent against the real v1 corpus + your ratings.
Run from repo root.  $env:ANTHROPIC_API_KEY = "sk-ant-..."
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss

import sys
sys.path.insert(0, "src")
from agent import run_agent

PROCESSED_DIR = Path("data/processed")

print("Loading corpus...")
df = pd.read_parquet(PROCESSED_DIR / "books_v1.parquet").reset_index(drop=True)
embeddings = np.load(PROCESSED_DIR / "embeddings_minilm_v1.npy").astype("float32")
index = faiss.read_index(str(PROCESSED_DIR / "faiss_minilm_v1.index"))
with open(PROCESSED_DIR / "my_matched_ratings.json") as f:
    ratings = json.load(f)

state = {"df": df, "embeddings": embeddings, "index": index, "ratings": ratings}
print(f"Loaded {len(df):,} books, {len(ratings)} ratings\n")

# the showpiece query
queries = [
    "Recommend me something like my noir / crime books, but a bit lighter in tone.",
    "I want something recent, published after 2020, similar to the literary fiction I've enjoyed.",
    "What should I read for a long flight? Something immersive like my favorites.",
]

for q in queries:
    print("=" * 70)
    print(f"USER: {q}\n")
    final, _ = run_agent(q, state, verbose=True)
    print(f"\nAGENT:\n{final}\n")