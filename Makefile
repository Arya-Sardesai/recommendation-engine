# Recommendation-engine data pipeline.
# Run from the repo root. On Windows, use WSL, Git Bash with make installed,
# or the Docker container (see Dockerfile) — or run the underlying python
# commands directly; every target is a single documented command.
#
# Raw data is NOT versioned in git. Before corpus targets will run you need:
#   data/raw/movies/TMDB_movie_dataset_v11.csv   (Kaggle TMDB dump)
#   data/raw/archive (1).zip                     (Kaggle TMDB TV dump)
#   data/raw/movies/ml-25m/                      (MovieLens 25M, for tags + eval)
#   data/raw/hardcover_recent.jsonl              (built by fetch-books)
#
# Stages with external requirements:
#   fetch-books   needs HARDCOVER_TOKEN in env
#   tag-books     needs ANTHROPIC_API_KEY in env and SPENDS API CREDITS
#
# Embedding targets run on CPU if no GPU is present — correct but slow
# (movies ~110K rows: minutes on GPU, hours on CPU).

PY := python

.PHONY: help \
        books-corpus books-embed books-rematch books tag-books \
        movies-corpus movies-tags movies-embed movies \
        tv-corpus tv-embed tv \
        eval-neighbors smoke eval-loo eval-movielens \
        corpora

help:
	@echo "Per-corpus pipelines:"
	@echo "  make books    = merge Hardcover into corpus -> rescale -> embed -> rematch ratings"
	@echo "  make movies   = build corpus -> map MovieLens genome tags -> embed"
	@echo "  make tv       = build corpus -> embed"
	@echo "  make corpora  = all three"
	@echo ""
	@echo "Individual stages: books-corpus books-embed books-rematch tag-books"
	@echo "                   movies-corpus movies-tags movies-embed"
	@echo "                   tv-corpus tv-embed"
	@echo ""
	@echo "Evaluation:        smoke eval-neighbors eval-loo eval-movielens"

# ---------------- books ----------------
# fetch-books is intentionally NOT part of `make books`: it hits the Hardcover
# API and appends to the raw JSONL. Run it explicitly when refreshing data.
fetch-books:
	$(PY) notebooks/fetch_hardcover.py

books-corpus:
	$(PY) notebooks/pipeline/merge_hardcover.py
	$(PY) notebooks/pipeline/rescale_v1.py

books-embed:
	$(PY) notebooks/pipeline/embed_v1.py

books-rematch:
	$(PY) notebooks/pipeline/rematch_v1.py

books: books-corpus books-embed books-rematch

# LLM tagging (Anthropic Batch API). Costs real money; requires ANTHROPIC_API_KEY.
# MODE (pilot vs full) is set inside the script — pilot first, always.
tag-books:
	$(PY) notebooks/pipeline/tag_data.py

# ---------------- movies ----------------
movies-corpus:
	$(PY) notebooks/pipeline/build_movies_corpus.py

movies-tags:
	$(PY) notebooks/pipeline/build_movie_tags.py

movies-embed:
	$(PY) notebooks/pipeline/embed_movies.py

movies: movies-corpus movies-tags movies-embed

# ---------------- tv ----------------
tv-corpus:
	$(PY) notebooks/pipeline/build_tv_corpus.py

tv-embed:
	$(PY) notebooks/pipeline/embed_tv.py

tv: tv-corpus tv-embed

# ---------------- everything ----------------
corpora: books movies tv

# ---------------- evaluation ----------------
# smoke extracts the ranker from the sibling hf-space repo's app.py — the
# sibling checkout must exist at ../hf-space. Run before ANY ranking change.
smoke:
	$(PY) tests/smoke_test.py
	$(PY) tests/smoke_test_books_tv.py

# side-by-side embedding A/B on fixed seed titles (corpus set inside script)
eval-neighbors:
	$(PY) notebooks/pipeline/eval_neighbors.py

# leave-one-out on my own ratings
eval-loo:
	$(PY) tests/eval_loo.py

# MovieLens offline eval (needs data/raw/movies/ml-25m/ at the documented path)
eval-movielens:
	$(PY) tests/eval_movielens.py
