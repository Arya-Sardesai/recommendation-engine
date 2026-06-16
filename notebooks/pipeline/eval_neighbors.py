"""
eval_neighbors.py  --  embedding A/B harness for movies / TV / books

Flip CORPUS_KEY below to switch corpus. Each config puts the MiniLM v1 build and
the new BGE-M3 build side by side on a fixed seed set, querying the RAW FAISS
index (no tag re-rank, no penalties) so you see EMBEDDING quality directly.

Readouts per seed, per build:
  1. top-K neighbours (title, year, lang, score)   <- the eyeball core
  2. ILD = 1 - mean pairwise cosine of the K neighbours  (low = near-dup cluster)
  3. novelty = median popularity percentile of the K neighbours (diagnostic)
  4. tag-overlap@K  -- movies only (needs movie_tags.parquet); n/a elsewhere.
     (Known weak: floored near 0 by mismatched candidate pools. Eyeball + ILD
     carry the verdict; this is corroboration at best.)

WHAT TO LOOK FOR
  movies : title-leak gone; Oldboy=Asian revenge (not exploitation); Om Shanti Om
           still partial (known limit).
  tv     : Doctor Who must NOT return "Doctors"; Barry must NOT return "The Flash";
           non-EN shows (Squid Game/Dark/Sacred Games) cluster by theme/language.
  books  : REGRESSION check, not a win check -- genre coherence as good or better
           than MiniLM. Books never had title leak and are ~all English, so a wash
           is an acceptable result (and an argument to keep books on MiniLM).

Run from REPO ROOT:
    python notebooks/pipeline/eval_neighbors.py
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import faiss

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "data" / "processed"

CORPUS_KEY = "books"     # movies | tv | books
K = 10

CONFIGS = {
    "movies": dict(
        corpus="movies.parquet",
        tags="movie_tags.parquet",
        year_col="release_year", lang_col="original_language",
        builds={
            "v1_rich":  ("movie_embeddings_minilm_v1.npy", "movie_faiss_minilm_v1.index"),
            "v2_bgem3": ("movie_embeddings_bgem3_v2_kw.npy", "movie_faiss_bgem3_v2_kw.index"),
        },
        seeds=["The Grand Budapest Hotel", "Om Shanti Om", "Blade Runner 2049",
               "Interstellar", "Parasite", "Oldboy", "Dilwale Dulhania Le Jayenge"],
    ),
    "tv": dict(
        corpus="tv_corpus.parquet",
        tags=None,
        year_col="start_year", lang_col="original_language",
        builds={
            "v1_minilm": ("tv_embeddings_minilm_v1.npy", "tv_faiss_minilm_v1.index"),
            "v2_bgem3":  ("tv_embeddings_bgem3_v2_notitle.npy", "tv_faiss_bgem3_v2_notitle.index"),
        },
        seeds=["Game of Thrones", "Breaking Bad", "Doctor Who", "Barry",
               "Squid Game", "Dark", "Sacred Games", "The Office"],
    ),
    "books": dict(
        corpus="books_v1.parquet",
        tags=None,
        year_col=None, lang_col=None,
        builds={
            "v1_minilm": ("embeddings_minilm_v1.npy", "faiss_minilm_v1.index"),
            "v2_bgem3":  ("embeddings_bgem3.npy", "faiss_bgem3.index"),
        },
        seeds=["The Hobbit", "Pride and Prejudice", "Dune", "Gone Girl",
               "The Hunger Games", "Sapiens: A Brief History of Humankind"],
    ),
}

POP_COLS = ["vote_count", "ratings_count", "rating_count", "num_ratings", "popularity"]


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def _title_col(df):
    for c in ("title", "name"):
        if c in df.columns:
            return c
    return df.columns[0]


def _pop_col(df):
    for c in POP_COLS:
        if c in df.columns:
            return c
    return None


def find_seed_row(df, title, tcol, pcol):
    norm = df[tcol].map(_normalize)
    q = _normalize(title)
    hits = np.where(norm.values == q)[0]
    if not len(hits):
        hits = df.index[norm.str.contains(re.escape(q), na=False)].values
    if not len(hits):
        return None
    if pcol is not None:
        return int(df.iloc[hits][pcol].astype("float64").idxmax())
    return int(hits[0])


def neighbours(index, emb, row, k):
    D, I = index.search(emb[row:row + 1], k + 1)
    out = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx == row:
            continue
        out.append((int(idx), float(score)))
        if len(out) == k:
            break
    return out


def intra_list_diversity(emb, idxs):
    if len(idxs) < 2:
        return float("nan")
    V = emb[idxs]
    sims = V @ V.T
    iu = np.triu_indices(len(idxs), k=1)
    return float(1.0 - sims[iu].mean())


def novelty_pct(df, idxs, pcol):
    if pcol is None:
        return float("nan")
    pct = df[pcol].astype("float64").rank(pct=True).values
    return float(np.median(pct[idxs]))


def load_tag_matrix(path):
    if path is None or not Path(path).exists():
        return None
    tdf = pd.read_parquet(path)
    tdf["movie_id"] = tdf["movie_id"].astype(str)
    return tdf.pivot_table(index="movie_id", columns="tag", values="score", fill_value=0.0)


def tag_overlap_at_k(df, tagm, seed_row, emb_idxs, k):
    if tagm is None or "id" not in df.columns:
        return None, "no tags"
    ids = df["id"].astype(str)
    seed_id = ids.iloc[seed_row]
    if seed_id not in tagm.index:
        return None, "seed not in genome"
    seed_vec = tagm.loc[seed_id].values.astype("float32")
    sn = np.linalg.norm(seed_vec)
    if sn == 0:
        return None, "seed tag vec empty"
    M = tagm.values.astype("float32")
    Mn = np.linalg.norm(M, axis=1)
    ok = Mn > 0
    sims = np.full(len(tagm), -1.0)
    sims[ok] = (M[ok] @ seed_vec) / (Mn[ok] * sn)
    order = np.argsort(-sims)
    genome_ids = [tagm.index[i] for i in order if tagm.index[i] != seed_id][:k]
    emb_ids = [ids.iloc[i] for i in emb_idxs]
    emb_in_genome = [m for m in emb_ids if m in tagm.index]
    if not emb_in_genome:
        return None, "no emb-neighbours in genome"
    return len(set(genome_ids) & set(emb_ids)) / k, f"{len(emb_in_genome)}/{k} nbrs tagged"


def _label(df, i, tcol, ycol, lcol):
    r = df.iloc[i]
    bits = []
    if ycol and ycol in df.columns:
        bits.append(str(r.get(ycol, "?")))
    if lcol and lcol in df.columns:
        bits.append(str(r.get(lcol, "?")))
    suffix = f" ({', '.join(bits)})" if bits else ""
    return f"{r[tcol]}{suffix}"


def run():
    cfg = CONFIGS[CORPUS_KEY]
    df = pd.read_parquet(PROCESSED / cfg["corpus"]).reset_index(drop=True)
    tcol, pcol = _title_col(df), _pop_col(df)
    tagm = load_tag_matrix(PROCESSED / cfg["tags"] if cfg["tags"] else None)
    print(f"[{CORPUS_KEY}] {len(df):,} rows | title={tcol} | pop={pcol} | "
          f"tags={'yes' if tagm is not None else 'no'}\n")

    loaded = {}
    for label, (ef, ixf) in cfg["builds"].items():
        ep, ip = PROCESSED / ef, PROCESSED / ixf
        if not ep.exists() or not ip.exists():
            print(f"[skip] {label}: artifacts not found ({ef})")
            continue
        emb = np.load(ep).astype("float32")
        idx = faiss.read_index(str(ip))
        assert emb.shape[0] == len(df) == idx.ntotal, f"{label}: alignment mismatch"
        loaded[label] = (emb, idx)
    if not loaded:
        raise SystemExit("No builds loaded -- check paths in CONFIGS.")

    for seed in cfg["seeds"]:
        row = find_seed_row(df, seed, tcol, pcol)
        print("=" * 74)
        if row is None:
            print(f"SEED {seed!r} -> NOT FOUND")
            continue
        print(f"SEED {seed!r} -> row {row}: {_label(df, row, tcol, cfg['year_col'], cfg['lang_col'])}")
        for label, (emb, idx) in loaded.items():
            nb = neighbours(idx, emb, row, K)
            idxs = [i for i, _ in nb]
            ild = intra_list_diversity(emb, idxs)
            nov = novelty_pct(df, idxs, pcol)
            ov, cov = tag_overlap_at_k(df, tagm, row, idxs, K)
            ovs = f"{ov:.2f} ({cov})" if ov is not None else f"n/a ({cov})"
            print(f"\n  [{label}]  ILD={ild:.3f}  novelty={nov:.2f}  tag@{K}={ovs}")
            for rank, (i, sc) in enumerate(nb, 1):
                print(f"    {rank:2d}. [{sc:.3f}] {_label(df, i, tcol, cfg['year_col'], cfg['lang_col'])}")
        print()


if __name__ == "__main__":
    run()