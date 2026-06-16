"""
eval_neighbors.py  --  embedding A/B harness for the movie recommender (Problem 2)

Runs LOCALLY (needs the real .npy/.index artifacts + corpus parquet). Compares
one or more embedding builds (e.g. v1 MiniLM-rich vs v2 keyword-enriched) on a
fixed seed set, so you can see the title-leakage fix before/after in one run.

It queries the RAW FAISS index only -- no tag re-rank, no title-token penalty,
no director exclusion. That isolates EMBEDDING quality (the serving app layers
those on top; here we want to see the foundation the app is standing on).

Three readouts per seed, per build:
  1. top-K neighbours (title, year, lang, score)        <- the eyeball core
  2. intra-list diversity (ILD)  = 1 - mean pairwise cosine of the K neighbours
     low ILD == near-duplicate cluster (the "every hotel movie" smell, numerically)
  3. novelty = median vote_count percentile of the K neighbours (diagnostic only)
Plus an OPTIONAL objective metric if movie_tags.parquet is present:
  4. tag-overlap@K = |emb top-K  ∩  tag-genome top-K| / K
     uses the MovieLens genome (movie_id,tag,score) as an independent notion of
     "thematically similar", so the A/B isn't purely vibes. Degrades gracefully
     to "n/a" for seeds/neighbours not in the genome (multilingual tail mostly
     won't be -- those stay eyeball-only, stated honestly rather than faked).

Configure BUILDS + paths below, set SEEDS, run from REPO ROOT:
    python notebooks/pipeline/eval_neighbors.py
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import faiss

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "data" / "processed"
CORPUS = PROCESSED / "movies.parquet"
TAGS = PROCESSED / "movie_tags.parquet"        # optional; objective metric if present

# label -> (embeddings .npy, faiss .index). Add a v2 row once you've built it.
BUILDS = {
    "v1_rich":  (PROCESSED / "movie_embeddings_minilm_v1.npy",
                 PROCESSED / "movie_faiss_minilm_v1.index"),
    "v2_kw":  (PROCESSED / "movie_embeddings_allminilml6v_v2_kw.npy",
                PROCESSED / "movie_faiss_allminilml6v_v2_kw.index"),
    "v3_kw":  (PROCESSED / "movie_embeddings_bgem3_v2_kw.npy",
                PROCESSED / "movie_faiss_bgem3_v2_kw.index"),
}

SEEDS = [
    "The Grand Budapest Hotel",
    "Om Shanti Om",
    "Blade Runner 2049",
    "Interstellar",
    "Hera Pheri",   # NB: TV -- will likely miss in a movies corpus; that's fine
    "Dilwale Dulhania Le Jayenge",
    "Parasite",
    "Oldboy",
]
K = 10


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def _title_col(df):
    return "title" if "title" in df.columns else "name"


def find_seed_row(df, title, tcol):
    norm = df[tcol].map(_normalize)
    q = _normalize(title)
    exact = np.where(norm.values == q)[0]
    if len(exact):
        # prefer the most-voted exact match (the canonical film)
        cand = df.iloc[exact]
        if "vote_count" in cand.columns:
            return int(cand["vote_count"].astype("int64").idxmax())
        return int(exact[0])
    contains = df.index[norm.str.contains(re.escape(q), na=False)]
    if len(contains):
        cand = df.loc[contains]
        if "vote_count" in cand.columns:
            return int(cand["vote_count"].astype("int64").idxmax())
        return int(contains[0])
    return None


def neighbours(index, emb, row, k):
    """Raw top-k neighbour rows (excluding the seed itself) + scores."""
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
    """1 - mean pairwise cosine among neighbour vectors (vectors are unit-norm)."""
    if len(idxs) < 2:
        return float("nan")
    V = emb[idxs]
    sims = V @ V.T
    iu = np.triu_indices(len(idxs), k=1)
    return float(1.0 - sims[iu].mean())


def novelty_pct(df, idxs):
    """Median popularity percentile of neighbours (0=obscure,1=blockbuster)."""
    if "vote_count" not in df.columns:
        return float("nan")
    vc = df["vote_count"].astype("float64")
    pct = vc.rank(pct=True).values
    return float(np.median(pct[idxs]))


def load_tag_matrix(path):
    if not path.exists():
        return None
    tdf = pd.read_parquet(path)
    tdf["movie_id"] = tdf["movie_id"].astype(str)
    return tdf.pivot_table(index="movie_id", columns="tag", values="score", fill_value=0.0)


def tag_overlap_at_k(df, tagm, seed_row, emb_neighbor_idxs, k):
    """Overlap@K between embedding neighbours and tag-genome neighbours.

    Returns (overlap_fraction, coverage_note) or (None, reason) if not computable.
    """
    if tagm is None or "id" not in df.columns:
        return None, "no tags"
    ids = df["id"].astype(str)
    seed_id = ids.iloc[seed_row]
    if seed_id not in tagm.index:
        return None, "seed not in genome"
    # genome neighbours: cosine of tag vectors, over the films that HAVE tags
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
    genome_set = set(genome_ids)
    emb_ids = [ids.iloc[i] for i in emb_neighbor_idxs]
    # only credit overlap among emb neighbours that are even IN the genome
    emb_in_genome = [mid for mid in emb_ids if mid in tagm.index]
    if not emb_in_genome:
        return None, "no emb-neighbours in genome"
    hit = len(genome_set & set(emb_ids))
    cov = f"{len(emb_in_genome)}/{k} nbrs tagged"
    return hit / k, cov


def run():
    df = pd.read_parquet(CORPUS).reset_index(drop=True)
    tcol = _title_col(df)
    tagm = load_tag_matrix(TAGS)
    print(f"corpus: {len(df):,} rows | title col: {tcol} | "
          f"tag genome: {'yes (%d movies)' % len(tagm) if tagm is not None else 'no'}\n")

    loaded = {}
    for label, (epath, ipath) in BUILDS.items():
        if not Path(epath).exists() or not Path(ipath).exists():
            print(f"[skip] {label}: artifacts not found")
            continue
        emb = np.load(epath).astype("float32")
        idx = faiss.read_index(str(ipath))
        assert emb.shape[0] == len(df) == idx.ntotal, f"{label}: alignment mismatch"
        loaded[label] = (emb, idx)
    if not loaded:
        raise SystemExit("No builds loaded -- check BUILDS paths.")

    for seed in SEEDS:
        row = find_seed_row(df, seed, tcol)
        print("=" * 74)
        if row is None:
            print(f"SEED  {seed!r}  -> NOT FOUND in corpus")
            continue
        print(f"SEED  {seed!r}  -> row {row}: {df.iloc[row][tcol]} "
              f"({df.iloc[row].get('release_year','?')}, "
              f"{df.iloc[row].get('original_language','?')})")
        for label, (emb, idx) in loaded.items():
            nb = neighbours(idx, emb, row, K)
            idxs = [i for i, _ in nb]
            ild = intra_list_diversity(emb, idxs)
            nov = novelty_pct(df, idxs)
            ov, cov = tag_overlap_at_k(df, tagm, row, idxs, K)
            ovs = f"{ov:.2f} ({cov})" if ov is not None else f"n/a ({cov})"
            print(f"\n  [{label}]  ILD={ild:.3f}  novelty={nov:.2f}  tag-overlap@{K}={ovs}")
            for rank, (i, sc) in enumerate(nb, 1):
                r = df.iloc[i]
                print(f"    {rank:2d}. [{sc:.3f}] {r[tcol]} "
                      f"({r.get('release_year','?')}, {r.get('original_language','?')})")
        print()


if __name__ == "__main__":
    run()