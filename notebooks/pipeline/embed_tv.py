"""
embed_tv.py  (v2 title-drop + TMDB keywords + Lever B multilingual model)

Embeds the TV corpus and builds a FAISS IndexFlatIP, mirroring embed_movies.py
(normalize_embeddings=True, float32, IndexFlatIP, row-position alignment, .npy +
.index out).

WHAT CHANGED THIS ROUND
-----------------------
TV HAS KEYWORDS NOW. The asaniczka dump had none, so v1 and the first v2 pass
embedded overview + genres only -- which is why TV was the weakest modality.
The TMDB bulk enrichment added `keywords_tmdb` at **77.7% coverage** (higher
than movies' 66.7%), and the vocabulary is unusually content-shaped:

    anime 11.82% | romance 9.54% | based on manga 5.53% |
    based on novel or book 5.31% | miniseries 5.15% | sitcom 3.53% |
    slice of life | shounen | ecchi | seinen | mecha | telenovela |
    boys' love (bl) | bromance

Movies' most frequent keyword is "woman director" (crew metadata). TV's is
"anime". This tag layer is genuinely better than the movie one.

NEW VARIANT = "v2_kw_notitle" (name dropped, overview leads, keywords + genres).
The old "v2_notitle" is KEPT and still runnable so the two are comparable and
rollback is a one-line change. Artifact names version off the variant, so the
old TV artifacts are not overwritten.

Format tags (miniseries / sitcom / telenovela) are deliberately KEPT. For TV,
format is content: a 6-episode miniseries and a 9-season sitcom are different
products, and users' tastes split on exactly that axis.

MODEL = BAAI/bge-m3, the multilingual model that won the movies pilot (fixed
Oldboy, improved Parasite/Bollywood cases). TV is ~57% non-English, so it
benefits at least as much as movies.

NOTE ON THE STALE `embed_text` COLUMN
-------------------------------------
tv_corpus.parquet carries a stored `embed_text` column that the TMDB merge did
NOT refresh, while ~15% of overviews DID change. That column is stale. This
script never reads it -- text is rebuilt from overview/genres/keywords at
runtime -- but grep the rest of the repo before trusting it anywhere else.

8GB-CARD NOTES: identical to embed_movies.py -- batch 32, max_seq_len 512, fp16.
If the norm check isn't ~1.0, set USE_FP16=False. If you OOM, drop batch to 16.

Artifact names auto-version off MODEL_NAME + variant:
  tv_embeddings_bgem3_v2_kw_notitle.npy / tv_faiss_bgem3_v2_kw_notitle.index

Run from REPO ROOT:
    python notebooks/pipeline/embed_tv.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import torch

from kw_common import KW_STOP, to_terms as _to_terms

ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
CORPUS = PROCESSED_DIR / "tv_corpus.parquet"

# ---- experiment knobs --------------------------------------------------------
MODEL_NAME = "BAAI/bge-m3"          # multilingual; won the movies pilot
EMBED_VARIANT = "v2_kw_notitle"     # overview | rich_v1 | v2_notitle | v2_kw_notitle
KW_COL = "keywords_tmdb"            # HARD REQUIREMENT for the _kw variant
KW_MAX = 20                         # p90 is 12, so this rarely binds
ENCODE_BATCH = 32                   # BGE-M3 on 8GB; drop to 16 if you OOM
MAX_SEQ_LEN = 512
USE_FP16 = True                     # flip off if the norm check drifts from 1.0
SHARD = 20_000

# Measured coverage on the enriched corpus: 77.7% of series have >=1 keyword.
KW_COVERAGE_FLOOR = 0.65

_MODEL_TAG = MODEL_NAME.split("/")[-1].replace("-", "").lower()[:12]
OUT_EMB = PROCESSED_DIR / f"tv_embeddings_{_MODEL_TAG}_{EMBED_VARIANT}.npy"
OUT_FAISS = PROCESSED_DIR / f"tv_faiss_{_MODEL_TAG}_{EMBED_VARIANT}.index"


def _clean(s) -> str:
    s = "" if s is None else str(s).strip()
    return "" if s.lower() == "nan" else s


def build_embed_text(df: pd.DataFrame) -> list[str]:
    """Per-series text fed to the encoder, per EMBED_VARIANT.

    TV column names: name / start_year / genres / overview / keywords_tmdb
    (genres is a comma-joined string, same as movies; keywords_tmdb is a list).
    """
    if EMBED_VARIANT == "overview":
        return df["overview"].fillna("").astype(str).tolist()

    if EMBED_VARIANT == "rich_v1":
        name = df["name"].fillna("").astype(str)
        year = df["start_year"].astype("Int64").astype(str).replace("<NA>", "")
        genres = df["genres"].fillna("").astype(str)
        overview = df["overview"].fillna("").astype(str)
        texts = []
        for n, y, g, o in zip(name, year, genres, overview):
            head = f"{n} ({y})." if y else f"{n}."
            mid = f" {g}." if g and g.lower() != "nan" else ""
            tail = f" {o}" if o and o.lower() != "nan" else ""
            texts.append((head + mid + tail).strip())
        return texts

    if EMBED_VARIANT not in ("v2_notitle", "v2_kw_notitle"):
        raise SystemExit(f"Unknown EMBED_VARIANT: {EMBED_VARIANT!r}")

    want_kw = EMBED_VARIANT == "v2_kw_notitle"
    if want_kw and KW_COL not in df.columns:
        raise SystemExit(
            f"ERROR: required keyword column {KW_COL!r} not in corpus.\n"
            f"  Present columns: {list(df.columns)}\n"
            f"  Run the TMDB merge first, or set EMBED_VARIANT='v2_notitle'."
        )

    # ---- v2: name dropped, overview leads, keywords + genres carry theme ----
    overview = df["overview"].tolist()
    genres = df["genres"].tolist() if "genres" in df.columns else [None] * len(df)
    keywords = df[KW_COL].tolist() if want_kw else [None] * len(df)

    texts, n_with_kw = [], 0
    for ov, g, kwv in zip(overview, genres, keywords):
        parts = []
        ov = _clean(ov)
        if ov:
            parts.append(ov)
        if want_kw:
            kw = _to_terms(kwv, max_n=KW_MAX, stop=KW_STOP)
            if kw:
                parts.append(f"Themes: {', '.join(kw)}.")
                n_with_kw += 1
        gens = [t.strip() for t in _clean(g).split(",") if t.strip()]
        if gens:
            parts.append(f"Genres: {', '.join(gens)}.")
        texts.append(" ".join(parts).strip())

    if want_kw:
        rate = n_with_kw / max(len(df), 1)
        print(f"Keyword coverage in embed text: {n_with_kw:,}/{len(df):,} ({rate:.1%}) "
              f"from column {KW_COL!r}")
        if rate < KW_COVERAGE_FLOOR:
            raise SystemExit(
                f"ERROR: keyword coverage {rate:.1%} is below the "
                f"{KW_COVERAGE_FLOOR:.0%} floor.\n  Wrong column, or KW_STOP is "
                f"eating real signal. Investigate before burning GPU time."
            )
    return texts


def main():
    if not CORPUS.exists():
        raise SystemExit(f"ERROR: {CORPUS} not found. Build the corpus first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding on: {device}")
    print(f"Model: {MODEL_NAME}  |  Variant: {EMBED_VARIANT}")
    print(f"batch={ENCODE_BATCH}  max_seq_len={MAX_SEQ_LEN}  fp16={USE_FP16 and device=='cuda'}")

    df = pd.read_parquet(CORPUS)
    print(f"Corpus rows: {len(df):,}")

    texts = build_embed_text(df)
    print(f"Empty embed texts: {sum(1 for t in texts if not t):,}")
    for k in (0, len(texts) // 2):
        print(f"Sample [{k}]:\n  {texts[k][:240]!r}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LEN
    if USE_FP16 and device == "cuda":
        model.half()

    embs = []
    for start in range(0, len(texts), SHARD):
        shard = texts[start:start + SHARD]
        emb = model.encode(
            shard, batch_size=ENCODE_BATCH, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype("float32")
        embs.append(emb)
        print(f"  shard {start:>7,}-{start + len(shard):>7,}  ({emb.shape[0]:,} rows)")
        if device == "cuda":
            torch.cuda.empty_cache()

    combined_emb = np.vstack(embs).astype("float32")
    print(f"\nEmbeddings: {combined_emb.shape}  dtype={combined_emb.dtype}")
    assert combined_emb.shape[0] == len(df), "row count mismatch emb vs corpus!"
    norms = np.linalg.norm(combined_emb[:1000], axis=1)
    print(f"norm check (first 1k): min={norms.min():.4f} max={norms.max():.4f}")
    if not (0.99 <= norms.min() and norms.max() <= 1.01):
        print("  WARNING: norms drifted from 1.0 -- set USE_FP16=False and re-run.")

    index = faiss.IndexFlatIP(combined_emb.shape[1])
    index.add(combined_emb)
    print(f"FAISS index: {index.ntotal:,} vectors, dim {combined_emb.shape[1]}")

    np.save(OUT_EMB, combined_emb)
    faiss.write_index(index, str(OUT_FAISS))
    print(f"\nWrote:\n  {OUT_EMB}\n  {OUT_FAISS}")

    D, I = index.search(combined_emb[:1], 4)
    print("\n--- sanity: nearest neighbours of series 0 ---")
    print(f"query: {df.iloc[0]['name']}")
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        print(f"  {rank}: [{score:.3f}] {df.iloc[idx]['name']}")


if __name__ == "__main__":
    main()
