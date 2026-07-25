"""
check_knn_tags.py  --  eyeball propagated book tags on books you actually know.

The holdout eval says precision 0.541 / top3-recovery 0.623. This is the
qualitative counterpart: do the inherited tags LOOK right on famous books?

Run from REPO ROOT:
    python notebooks/pipeline/check_knn_tags.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
PROC = ROOT / "data" / "processed"

N_SHOW = 20
MIN_RATINGS = 50_000

knn = pd.read_parquet(PROC / "book_tags_knn.parquet")
llm = pd.read_parquet(PROC / "book_tags.parquet")
books = pd.read_parquet(PROC / "books_v1.parquet",
                        columns=["book_id", "title", "ratings_count"])

print(f"kNN: {len(knn):,} rows / {knn.book_id.nunique():,} books "
      f"({len(knn)/knn.book_id.nunique():.2f} tags per book)")
print(f"LLM: {len(llm):,} rows / {llm.book_id.nunique():,} books")
print(f"score range: {knn.score.min():.3f} - {knn.score.max():.3f} "
      f"| median {knn.score.median():.3f}\n")

m = knn.merge(books, on="book_id")
pop = (m[m.ratings_count >= MIN_RATINGS]
       .sort_values(["ratings_count", "score"], ascending=[False, False]))

print("=" * 78)
print(f"MOST-READ PROPAGATED BOOKS (>= {MIN_RATINGS:,} ratings)")
print("=" * 78)
for i, (bid, g) in enumerate(pop.groupby("book_id", sort=False)):
    if i >= N_SHOW:
        break
    title = str(g.title.iloc[0])[:52]
    tags = ", ".join(f"{r.tag}({r.score:.2f})"
                     for r in g.sort_values("score", ascending=False).itertuples())
    print(f"\n  {title}")
    print(f"    {tags}")

print("\n" + "=" * 78)
print("TAG FREQUENCY: propagated vs LLM (share of each corpus)")
print("=" * 78)
a = knn.tag.value_counts(normalize=True)
b = llm.tag.value_counts(normalize=True)
cmp = pd.DataFrame({"knn": a, "llm": b}).fillna(0)
cmp["ratio"] = cmp.knn / cmp.llm.replace(0, 1e-9)
cmp = cmp.sort_values("knn", ascending=False)
print(f"{'tag':<26}{'knn%':>8}{'llm%':>8}{'ratio':>8}")
for t, r in cmp.head(15).iterrows():
    print(f"{t:<26}{r.knn*100:>7.2f}%{r.llm*100:>7.2f}%{r.ratio:>8.2f}")
print("  ...")
for t, r in cmp.tail(10).iterrows():
    print(f"{t:<26}{r.knn*100:>7.2f}%{r.llm*100:>7.2f}%{r.ratio:>8.2f}")
print("\nratio >> 1 means propagation OVER-produces that tag relative to the LLM;")
print("ratio ~0 means the tag effectively died in propagation.")
