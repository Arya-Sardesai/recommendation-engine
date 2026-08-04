"""
notebooks/pipeline/identify_scifi_candidates.py
Layer 1 keyword detection for sci-fi backfill candidates from tagged books.
"""
import pandas as pd
import re
from pathlib import Path

PROCESSED = Path("data/processed")

master = pd.read_parquet(PROCESSED / "book_tags.parquet")
tagged_ids = set(master["book_id"].astype(str))
print(f"Books already tagged: {len(tagged_ids):,}")

corpus = pd.read_parquet(PROCESSED / "books_v1.parquet")
corpus["book_id"] = corpus["book_id"].astype(str)
tagged_books = corpus[corpus["book_id"].isin(tagged_ids)].copy()

tagged_books["genres_str"] = tagged_books["genres"].apply(
    lambda g: ", ".join(g).lower() if hasattr(g, "__iter__") and not isinstance(g, str) else ""
)
tagged_books["searchable"] = (
    tagged_books["title"].fillna("").str.lower() + " " +
    tagged_books["description"].fillna("").str.lower() + " " +
    tagged_books["genres_str"]
)

scifi_patterns = [
    r"science[- ]fiction", r"\bsci[- ]?fi\b", r"speculative fiction",
    r"\bspace\b", r"\bspaceship\b", r"\bstarship\b", r"interstellar", r"\bgalax",
    r"\bnebula\b", r"\borbit", r"\bcosmos\b",
    r"\balien\b", r"\baliens\b", r"extraterrestrial", r"first contact",
    r"dystopia", r"post[- ]apocalyp", r"apocalyp", r"totalitarian future",
    r"surveillance state", r"climate collapse", r"resource war",
    r"cyberpunk", r"cybernet", r"\bandroid\b", r"\brobot\b",
    r"artificial intelligence", r"sentient machine", r"\bcyborg",
    r"virtual reality", r"\bsimulation\b", r"digital consciousness",
    r"upload(?:ed)? mind", r"neural implant",
    r"time[- ]travel", r"time[- ]loop", r"time displacement",
    r"climate change", r"climate crisis", r"climate fiction",
    r"rising sea", r"ecological collapse", r"climate refugee",
    r"\bmars\b", r"\bvenus\b", r"\bcolony\b", r"colonization",
    r"future[- ]earth", r"far[- ]future", r"near[- ]future", r"22nd century",
    r"genetic engineering", r"\bclone\b", r"bioengineered", r"genetically modified",
    r"alternate (?:history|reality|timeline)", r"parallel universe", r"multiverse",
    r"after the (?:collapse|fall|plague|virus)",
    r"megacorp", r"corporate dystopia",
]
pattern = re.compile("|".join(scifi_patterns), re.IGNORECASE)
tagged_books["candidate"] = tagged_books["searchable"].str.contains(pattern, na=False)

candidates = tagged_books[tagged_books["candidate"]].copy()
print(f"Sci-fi candidates: {len(candidates):,}")

export = candidates[["book_id", "title", "primary_author", "publication_year",
                     "genres_str", "description"]].copy()
export["description"] = export["description"].str.slice(0, 500)
export = export.rename(columns={"primary_author": "author"})

OUTPUT = PROCESSED / "scifi_candidates_for_tagging.csv"
export.to_csv(OUTPUT, index=False)
print(f"Exported to: {OUTPUT}")