# Multi-Modal Recommendation Engine

A content-based recommender that works across **books, films, and TV** from a single taste profile — rate a few things in any medium and get ranked, explained recommendations back. Built as a portfolio artifact and a personal Letterboxd-style app.

**Live demo:** [huggingface.co/spaces/Arya2305/recommendation-engine](https://huggingface.co/spaces/Arya2305/recommendation-engine)

---

## Overview

Three corpora, one embedding space, one ranking engine:

| Corpus | Titles | Source |
|--------|-------:|--------|
| Books  | 432,715 | Goodreads bulk dataset + Hardcover GraphQL API |
| Films  | 109,921 | TMDB (Kaggle dump) + IMDb non-commercial credits |
| TV     | 23,464  | TMDB TV dump (Kaggle) |

The system is **content-based**: recommendations come from semantic similarity between a user's rated items and candidate items, not from other users' behavior. That's a deliberate v1 scope — the evaluation section below quantifies exactly what that choice costs and why it was made.

## How it works

```
rated items ──► BGE-M3 embeddings ──► FAISS neighbor retrieval ──► signed whole-profile ranker ──► explained recommendations
```

**Embeddings.** All items are encoded with [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (568M params, 1024-d, multilingual). Multilingual matters here — the corpora include non-English film and book metadata, and BGE-M3 measurably beat a MiniLM baseline on that content. Vectors are precomputed offline on a laptop GPU and loaded at inference, so **no model runs at serving time**.

One debugging note baked into the pipeline: **title tokens were dropped from the embedding text.** Leaving them in caused title-string leakage — items were being matched on shared words in their names rather than on actual content — which was a root cause of early bad recommendations.

**Retrieval.** A FAISS `IndexFlatIP` (inner-product / cosine) index over the normalized vectors. At serving time the index is memory-reconstructed rather than holding a second copy of the raw vectors, saving ~2.3 GB of RAM on the free CPU Space.

**Ranking.** This is where most of the design lives — see below.

**Serving.** Streamlit app on Hugging Face Spaces (Docker), pulling precomputed artifacts from a companion Datasets repo at boot. Per-medium tabs, watchlists, star-rating feedback, and localStorage persistence so a taste profile survives a page reload.

There is also an experimental **chat/agent interface** (Claude Haiku via the Anthropic API) over the same corpus and embeddings, with simpler scoring — a natural-language front door rather than a replacement for the ranker.

## The ranking engine

Given a whole profile of ratings, the ranker produces a single ordered list. The pieces, and what each one is *for*:

- **Signed whole-profile scoring.** Every rated item contributes, weighted by `rating − 3`, so likes pull candidates in and dislikes push them away. It's not a "more like your favorite" nearest-neighbor lookup; it scores against your whole taste at once.
- **Diversity.** Greedy farthest-point selection so the list isn't ten near-duplicates of one anchor.
- **Cross-pollination with attribution.** Candidates can be surfaced by more than one thing you liked; the ranker tracks *why* a title showed up and renders a "because you liked X" line (and a second one when a strong secondary anchor exists).
- **Negative penalties.** Dislikes actively suppress similar candidates. (The ablations show this genuinely helps — removing it roughly halved recall.)
- **Quality floor + Bayesian-shrunk quality prior.** A vote-average floor and a shrinkage prior that pulls thinly-voted items toward the corpus mean, so a single 5-star item with three votes doesn't outrank a broadly-loved one.
- **Sequel suppression.** A title-token penalty that deliberately demotes obvious sequels — the point is to recommend things you *couldn't* have found yourself.

Per-corpus tuning differs (books have no vote data so they skip the quality floor and lean on content tags; films use genome-derived tags; TV is untagged in v1), and a `flat_ranking` mode disables the diversity machinery for clean offline evaluation.

## Evaluation

The evaluation is the part I'm proudest of, because it includes a result where the system **loses** — and shows that the loss is the design working as intended.

**On my own ratings** (leave-one-out over 70 hand-entered ratings), the tuned engine clearly beats the prior baseline:

| Metric | Old | New | Δ |
|--------|----:|----:|---:|
| hit@20 | 0.264 | 0.340 | **+29%** |
| MRR    | — | — | **+39%** |

That's roughly 17× the popularity floor.

**On MovieLens (ml-25m, 300 held-out users, 80/20 split), the story flips:**

| recall@20 | popularity | old engine | new engine |
|-----------|----:|----:|----:|
|  | **0.089** | 0.064 | 0.032 |

The tuned engine loses to a plain popularity baseline at recall@20. Rather than tune that away, I ablated to find out *where* the recall went:

| Ablation (recall@20) | Result |
|----------------------|-------:|
| diversity off (flat) | 0.046 |
| sequel penalty off   | 0.041 |
| negatives off        | 0.021 |
| quality floor off    | 0.024 |
| all off (≈ old)      | 0.057 |

Two hypotheses I expected to confirm were instead **falsified** — turning off negatives *and* turning off the quality floor both hurt, so both are earning their place. And the recall the engine "loses" is mostly spent on purpose: diversity and sequel-suppression together cost roughly 40% of recall@20 by refusing the obvious recommendations. When you widen the window to **recall@50, the tuned engine crosses back ahead** (0.095–0.097 vs. 0.0885) — the good picks are there, just past the sequels a popularity baseline happily returns.

**The honest conclusion:** a content-only model can't beat popularity for mainstream MovieLens raters, and that's the measured, quantified case for adding collaborative filtering (see roadmap). The one-line version: *beat the baseline on my own data, lost to popularity on MovieLens, and ablated to show the loss was the price of deliberate design.*

## Tech stack

Python 3.11 · SentenceTransformers (BGE-M3) · FAISS · Streamlit · Anthropic SDK (Claude Haiku) · Hugging Face Spaces & Datasets · Docker · pandas / numpy / pyarrow. Embeddings computed offline on an RTX 4060; inference runs CPU-only.

## Limitations

Stated plainly, because a portfolio piece should be honest about its edges:

- **No collaborative filtering yet.** Purely content-based — it can't learn "people who liked X also liked Y."
- **Single-user scale.** Built and evaluated for one taste profile at a time; there's no multi-user backend.
- **No A/B testing.** Evaluation is offline (leave-one-out + MovieLens), not live.
- **Content-only ceiling.** As the evaluation shows, this caps performance against popularity on mainstream raters.

## Roadmap

- **v2** — refreshed data layer (fresh TMDB pull + keyword tagging for films and TV, optional embedding upgrade), then a web-app rebuild (React + FastAPI + Postgres + auth) as individually shippable milestones.
- **v3** — a collaborative-filtering hybrid (matrix factorization → Spark ALS) targeting the measured popularity recall@20 gap, plus a cross-modal supervisor agent.

## Local development

The live demo is the easiest way to try it. To run the serving app locally you'll need the precomputed embedding and index artifacts from the companion Datasets repo (they're too large to version in git); with those in place it's a standard Streamlit app:

```bash
python -m streamlit run app.py
```

## Data & acknowledgements

Built on public datasets: Goodreads and the Hardcover API (books); TMDB and IMDb non-commercial data (films); TMDB (TV); and MovieLens ml-25m (ratings and genome tags used for evaluation and film tagging). Each source retains its own license and terms — this project uses them for a non-commercial personal/portfolio tool. Embedding model: BAAI/bge-m3.

## License

MIT — see [LICENSE](LICENSE).