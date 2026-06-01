# Recommendation Engine

A content-based book recommender that learns your taste from explicit ratings and recommends books across your distinct interests — with an explanation for *why* each book is suggested.

**Live demo:** https://huggingface.co/spaces/Arya2305/recommendation-engine

![status](https://img.shields.io/badge/status-v1-blue) ![python](https://img.shields.io/badge/python-3.11-green)

---

## What it does

Rate a few books you've read (or load a sample profile), and the engine recommends books you're likely to enjoy. Unlike a single "average taste" recommender, it preserves your *distinct* interests — so if you like both literary fiction and trail memoirs, you get recommendations for each, not a blurred average of the two. Every recommendation tells you which of your rated books it came from.

It is **content-based** (it recommends based on what books are *about*, via their descriptions) rather than collaborative (it does not rely on "people like you also read..."). This means it works from your ratings alone, with no need for a large user base.

---

## How it works

**1. Corpus.** ~419K English-language books with descriptions, ratings, and metadata, built from two sources:
- **Goodreads** (UCSD academic dataset) — 409K books, filtered from a 2.36M raw catalog (English, description ≥ 100 chars, ≥ 10 ratings) and deduplicated by work.
- **Hardcover.app API** — ~9.7K popular books released 2018–2025, added to cover the recency gap left by the Goodreads snapshot (which ends in 2017).

**2. Embeddings.** Each book's description is encoded into a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence-transformer (GPU-accelerated). Books with similar themes, tone, and subject matter land near each other in this vector space.

**3. Similarity search.** A FAISS index (inner-product over normalized vectors = cosine similarity) enables fast nearest-neighbor lookup across the full corpus.

**4. Recommendation.** Your rated books (≥ 3.5 stars) become "anchors." For each anchor, the engine finds the nearest unread books, then **round-robins** across anchors so every one of your distinct tastes is represented — not just the ones in dense regions of the corpus. Results are filtered for popularity (to avoid obscure noise), deduplicated by work, and exclude authors you've already rated (to surface new authors rather than more of the same).

**5. Explanation.** Each recommendation is tagged with the rated book that triggered it ("because you liked *X*").

---

## Architecture

The system separates heavy offline work from the lightweight interactive app — the same split used by production recommender systems:

- **Offline pipeline** (local, GPU): ingest → clean → dedup → embed → build FAISS index → produce data artifacts.
- **Data hosting:** artifacts (corpus, embeddings, index) live in a Hugging Face dataset repo.
- **Interactive app** (Streamlit, Dockerized, deployed on Hugging Face Spaces): downloads the artifacts on startup and serves recommendations. New ratings reuse precomputed embeddings, so the app needs no GPU or embedding model at runtime.

```
Goodreads + Hardcover  ->  filter/dedup/embed  ->  Delta artifacts (HF dataset)
                                                          |
                                                          v
                                    Streamlit app (Docker, HF Spaces)
                                    search -> rate -> recommend -> explain
```

---

## Results

**Adding recent books measurably improved coverage.** Using a fixed set of 28 rated books as a test profile:

| | v0 (Goodreads only) | v1 (+ Hardcover) |
|---|---|---|
| Corpus size | 409K | 419K |
| Date coverage | up to 2017 | up to 2025 |
| Test books matched | 21 / 28 | 23 / 28 |
| Recent books in recommendations | 0 | ~half of top 20 |

Books like *Yellowface* (2023), *The Poppy War* (2018), and *The Eyes Are the Best Part* (2024) — impossible to match on the 2017 corpus — now match and drive relevant recommendations.

**An honest finding on cross-source popularity:** Goodreads and Hardcover measure popularity on very different scales (Goodreads ratings counts run to the millions; Hardcover's user counts are far smaller). I reconciled them with a scale factor and tested values from 15–60 — and found the recommendation mix was largely **robust to the choice**, because recent books surface primarily on semantic match rather than the popularity prior. The cross-source reconciliation mattered less than expected.

---

## Tech stack

- **Python 3.11**, pandas, NumPy
- **sentence-transformers** (`all-MiniLM-L6-v2`) for embeddings, GPU-accelerated via PyTorch/CUDA
- **FAISS** for similarity search
- **rapidfuzz** for matching free-text book titles to corpus entries
- **Streamlit** for the UI
- **Docker** + **Hugging Face Spaces** for deployment; **Hugging Face Hub** for data hosting
- **Hardcover GraphQL API** for recent-book ingestion

---

## Known limitations

- **Content-based only.** No collaborative filtering, so it can't surface "readers like you also enjoyed" signals.
- **Single embedding model.** Only MiniLM; no comparison against larger models yet.
- **Author exclusion is binary.** Currently excludes all already-rated authors; a smarter version would down-weight rather than exclude (you often *do* want more by an author you love).
- **Round-robin diversity is an approximation.** A principled approach (e.g. Maximal Marginal Relevance) is planned.
- **Popularity reconciliation is heuristic.** The cross-source scale factor is a pragmatic fix, not a principled normalization.

---

## Roadmap

- **v2 — cross-modal.** Extend beyond books to films and TV (e.g. via TMDB), so taste can transfer across media: rate noir books, get noir film recommendations. Embeddings of plot summaries and blurbs share a semantic space, making this geometrically natural.
- **Better models.** MMR for diversity; author down-weighting instead of exclusion; multi-embedding comparison.
- **Agent layer.** A conversational interface that elicits taste through dialogue and handles compound requests ("something for a long flight, like my noir reads but lighter") via multi-step reasoning.
- **Spark pipeline.** Port the offline feature pipeline to PySpark for genuinely large-scale processing.

---

## Repository layout

```
notebooks/        data exploration, ingestion, embedding, evaluation scripts
src/app/          the Streamlit application
data/             raw + processed data (gitignored)
```

*Data files are not committed (too large); they are hosted on Hugging Face and downloaded by the app at runtime.*