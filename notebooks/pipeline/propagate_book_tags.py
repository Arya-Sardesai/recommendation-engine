"""
propagate_book_tags.py  --  extend book tag coverage by kNN label propagation.

THE IDEA
--------
32,893 books carry LLM-assigned tags. 432,715 books have BGE-M3 embeddings. For
an untagged book, find its nearest TAGGED neighbours in embedding space and
inherit their tags, weighted by similarity and by how many neighbours agree.

Zero API cost. The whole point is to find out how far the existing paid tags can
be stretched for free BEFORE deciding how much more tagging to buy.

SCORING
-------
For untagged book u and candidate tag t, over its k nearest tagged neighbours N:

    score(u, t) = sum_{n in N, n has t} (sim(u,n) * score(n,t))
                  -------------------------------------------
                          sum_{n in N} sim(u,n)

The denominator runs over ALL k neighbours, not just those carrying t. That is
deliberate: it makes the score encode AGREEMENT as well as similarity. A tag on
18 of 20 neighbours scores high; a tag on 2 of 20 scores low even if those two
are very close. Without it, one stray neighbour could hand over its whole tag
set at full confidence.

KNOWN BIAS, REPORTED NOT HIDDEN: this formulation favours COMMON tags. A tag
held by many books is likelier to appear across neighbours and clear the
threshold, so rare-but-precise tags (`epistolary`, `second-person`) will
propagate worse than broad ones (`dark`, `fast-paced`). The eval prints
per-tag recall so the size of that effect is visible rather than assumed.

MODE = "eval"  -- THE HONEST TEST. Holds out N_HOLDOUT already-tagged books,
propagates to them from the REST of the tagged set, and compares predictions
against their real LLM tags. Critically it also scores a POPULARITY BASELINE
(assign every book the globally most frequent tags). If propagation cannot beat
that baseline it is worthless, because a handful of tags dominate the corpus and
guessing them looks deceptively good. Same discipline as the recall@20 /
popularity comparison in the eval chapter.

MODE = "propagate" -- full run over every untagged book, writes
book_tags_knn.parquet (book_id, tag, score, source) WITHOUT touching
book_tags.parquet. Keep them separate: LLM tags are evidence, propagated tags
are inference, and the ranking layer may want to weight them differently.

MEMORY: the embedding matrix is ~1.8GB (432,715 x 1024 float32). It is opened
with mmap_mode='r' and the FAISS index is built over the ~33K TAGGED rows only
(~135MB), then queried in batches. Do not np.load() the whole thing.

Run from REPO ROOT:
    python notebooks/pipeline/propagate_book_tags.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import faiss

# ---- config -----------------------------------------------------------------
MODE = "propagate"              # "eval" first. Only then "propagate".

K = 10                     # neighbours consulted per book
MIN_SIM = 0.55             # a neighbour below this cosine is not evidence
THRESHOLD = 0.25           # same sparse cutoff as book_tags / movie_tags
MAX_TAGS = 8               # same per-item cap as the LLM tagger
N_HOLDOUT = 3_000          # eval only
QUERY_BATCH = 4_096
SEED = 20240725

ROOT = Path(__file__).parent.parent.parent
PROC = ROOT / "data" / "processed"
CORPUS = PROC / "books_v1.parquet"
TAGS = PROC / "book_tags.parquet"
EMB = PROC / "embeddings_bgem3.npy"
OUT = PROC / "book_tags_knn.parquet"


def load_all():
    for p in (CORPUS, TAGS, EMB):
        if not p.exists():
            sys.exit(f"ERROR: missing {p}")
    df = pd.read_parquet(CORPUS, columns=["book_id"])
    tags = pd.read_parquet(TAGS)
    emb = np.load(EMB, mmap_mode="r")
    print(f"corpus {len(df):,} books | embeddings {emb.shape} | "
          f"tags {len(tags):,} rows over {tags.book_id.nunique():,} books")
    if emb.shape[0] != len(df):
        sys.exit(f"ERROR: embedding rows {emb.shape[0]:,} != corpus rows {len(df):,}. "
                 f"Row alignment is the whole contract here -- re-embed or re-check.")
    return df, tags, emb


def tag_lookup(tags):
    """book_id -> {tag: score}"""
    d = defaultdict(dict)
    for b, t, s in zip(tags.book_id, tags.tag, tags.score):
        d[b][t] = float(s)
    return d


def build_index(emb, rows):
    """IndexFlatIP over the given row positions. Vectors are already L2-normalised
    by the embed step, so inner product == cosine."""
    sub = np.ascontiguousarray(emb[rows], dtype="float32")
    idx = faiss.IndexFlatIP(sub.shape[1])
    idx.add(sub)
    return idx


def propagate(emb, query_rows, index, index_row_ids, id_to_tags, label=""):
    """-> dict query_row -> [(tag, score), ...]"""
    out, n_no_neighbour = {}, 0
    for start in range(0, len(query_rows), QUERY_BATCH):
        chunk = query_rows[start:start + QUERY_BATCH]
        q = np.ascontiguousarray(emb[chunk], dtype="float32")
        D, I = index.search(q, K)
        for qi, (sims, nbrs) in enumerate(zip(D, I)):
            # FAISS returns -1 when fewer than K neighbours exist. Guard it --
            # index_row_ids[-1] would silently resolve to the LAST id.
            keep = [(s, n) for s, n in zip(sims, nbrs) if n != -1 and s >= MIN_SIM]
            if not keep:
                n_no_neighbour += 1
                continue
            denom = sum(s for s, _ in keep)
            acc = defaultdict(float)
            for s, n in keep:
                for t, ts in id_to_tags[index_row_ids[n]].items():
                    acc[t] += s * ts
            scored = [(t, v / denom) for t, v in acc.items() if v / denom >= THRESHOLD]
            scored.sort(key=lambda x: -x[1])
            if scored:
                out[chunk[qi]] = scored[:MAX_TAGS]
        print(f"  [{label}] {min(start+QUERY_BATCH, len(query_rows)):,}/{len(query_rows):,}",
              end="\r")
    print()
    print(f"  [{label}] no neighbour above MIN_SIM={MIN_SIM}: {n_no_neighbour:,}")
    return out


def run_eval(df, tags, emb):
    rng = np.random.default_rng(SEED)
    id_to_tags = tag_lookup(tags)
    row_of = {b: i for i, b in enumerate(df.book_id)}
    tagged_rows = np.array(sorted(row_of[b] for b in id_to_tags if b in row_of))
    print(f"tagged books present in corpus: {len(tagged_rows):,}")

    hold = rng.choice(tagged_rows, size=min(N_HOLDOUT, len(tagged_rows)), replace=False)
    hold_set = set(hold.tolist())
    train = np.array([r for r in tagged_rows if r not in hold_set])
    print(f"holdout {len(hold):,} | train {len(train):,}")

    index = build_index(emb, train)
    train_ids = [df.book_id.iloc[r] for r in train]
    preds = propagate(emb, list(hold), index, train_ids, id_to_tags, label="eval")

    # popularity baseline: the globally most frequent tags, same cap
    pop = (tags[tags.book_id.isin(set(train_ids))]
           .tag.value_counts().head(MAX_TAGS).index.tolist())
    print(f"\npopularity baseline tags: {pop}")

    def score_pred(get_pred):
        p_tot = r_tot = n = 0
        top3_hits = top3_tot = 0
        per_tag_hit, per_tag_tot = defaultdict(int), defaultdict(int)
        for row in hold:
            truth = id_to_tags[df.book_id.iloc[row]]
            if not truth:
                continue
            pred = set(get_pred(row))
            if not pred:
                n += 1
                continue
            tset = set(truth)
            inter = pred & tset
            p_tot += len(inter) / len(pred)
            r_tot += len(inter) / len(tset)
            top3 = [t for t, _ in sorted(truth.items(), key=lambda x: -x[1])[:3]]
            top3_hits += len(set(top3) & pred); top3_tot += len(top3)
            for t in tset:
                per_tag_tot[t] += 1
                if t in pred:
                    per_tag_hit[t] += 1
            n += 1
        return (p_tot / max(n, 1), r_tot / max(n, 1),
                top3_hits / max(top3_tot, 1), per_tag_hit, per_tag_tot)

    print("\n" + "=" * 66)
    kp, kr, k3, hit, tot = score_pred(lambda r: [t for t, _ in preds.get(r, [])])
    bp, br, b3, _, _ = score_pred(lambda r: pop)
    print(f"{'':<22}{'precision':>11}{'recall':>9}{'top3-recovery':>15}")
    print(f"{'kNN propagation':<22}{kp:>11.3f}{kr:>9.3f}{k3:>15.3f}")
    print(f"{'popularity baseline':<22}{bp:>11.3f}{br:>9.3f}{b3:>15.3f}")
    print(f"{'lift':<22}{kp-bp:>+11.3f}{kr-br:>+9.3f}{k3-b3:>+15.3f}")
    print(f"\nholdout books receiving >=1 propagated tag: "
          f"{len(preds):,}/{len(hold):,} ({len(preds)/len(hold):.1%})")

    print("\nper-tag recall (propagation) -- best and worst 12:")
    rows = sorted(((t, hit[t] / tot[t], tot[t]) for t in tot if tot[t] >= 20),
                  key=lambda x: -x[1])
    for t, r, n_ in rows[:12]:
        print(f"   {t:<24}{r:>6.2f}  (n={n_})")
    print("   ...")
    for t, r, n_ in rows[-12:]:
        print(f"   {t:<24}{r:>6.2f}  (n={n_})")

    print("\nREAD THIS BEFORE TRUSTING THE NUMBER:")
    print("  Lift over the popularity baseline is what matters, not raw recall.")
    print("  If lift is small, propagation is mostly restating which tags are")
    print("  common and is not worth wiring in. If top3-recovery is high, the")
    print("  tags it does assign are the DEFINING ones, which is what a")
    print("  recommender actually needs.")


def run_propagate(df, tags, emb):
    id_to_tags = tag_lookup(tags)
    row_of = {b: i for i, b in enumerate(df.book_id)}
    tagged_rows = np.array(sorted(row_of[b] for b in id_to_tags if b in row_of))
    tagged_set = set(tagged_rows.tolist())
    untagged = [i for i in range(len(df)) if i not in tagged_set]
    print(f"tagged {len(tagged_rows):,} | untagged {len(untagged):,}")

    index = build_index(emb, tagged_rows)
    tagged_ids = [df.book_id.iloc[r] for r in tagged_rows]
    preds = propagate(emb, untagged, index, tagged_ids, id_to_tags, label="propagate")

    rows = [(df.book_id.iloc[r], t, round(s, 4))
            for r, ts in preds.items() for t, s in ts]
    out = pd.DataFrame(rows, columns=["book_id", "tag", "score"])
    out["source"] = "knn"
    out.to_parquet(OUT, index=False)
    total = len(tagged_rows) + out.book_id.nunique()
    print(f"\nWrote {len(out):,} rows for {out.book_id.nunique():,} books -> {OUT}")
    print(f"Book tag coverage: {len(tagged_rows):,} LLM + {out.book_id.nunique():,} kNN "
          f"= {total:,}/{len(df):,} ({total/len(df):.1%})")
    print("\nNOTE: book_tags.parquet is UNTOUCHED. These are inferred, not observed --")
    print("keep them separate so ranking can weight them differently.")

def run_sweep(df, tags, emb):
    """Grid over K / THRESHOLD / MIN_SIM. Reports overall metrics AND rare-tag
    recall separately, because the headline numbers are dominated by common
    tags and the rare-tag cliff is the actual failure mode."""
    global K, THRESHOLD, MIN_SIM
    rng = np.random.default_rng(SEED)
    id_to_tags = tag_lookup(tags)
    row_of = {b: i for i, b in enumerate(df.book_id)}
    tagged_rows = np.array(sorted(row_of[b] for b in id_to_tags if b in row_of))
    hold = rng.choice(tagged_rows, size=min(N_HOLDOUT, len(tagged_rows)), replace=False)
    hold_set = set(hold.tolist())
    train = np.array([r for r in tagged_rows if r not in hold_set])
    index = build_index(emb, train)
    train_ids = [df.book_id.iloc[r] for r in train]

    freq = tags.tag.value_counts()
    rare = set(freq[freq < freq.quantile(0.40)].index)
    print(f"rare tags (bottom 40% by corpus frequency): {len(rare)}\n")
    print(f"{'K':>3}{'thr':>6}{'sim':>6}{'prec':>8}{'rec':>7}{'top3':>7}"
          f"{'rareRec':>9}{'cover':>8}{'tags/bk':>9}")
    print("-" * 63)
    for k_ in (10, 20, 30):
        for th in (0.20, 0.25, 0.30):
            K, THRESHOLD = k_, th
            preds = propagate(emb, list(hold), index, train_ids, id_to_tags, label="")
            p = r = t3h = t3t = 0.0; n = 0; ntags = 0
            rh = rt = 0
            for row in hold:
                truth = id_to_tags[df.book_id.iloc[row]]
                if not truth: continue
                pred = set(t for t, _ in preds.get(row, []))
                n += 1; ntags += len(pred)
                if pred:
                    ts = set(truth); inter = pred & ts
                    p += len(inter)/len(pred); r += len(inter)/len(ts)
                    top3 = [x for x, _ in sorted(truth.items(), key=lambda y: -y[1])[:3]]
                    t3h += len(set(top3) & pred); t3t += len(top3)
                    for tg in ts & rare:
                        rt += 1
                        if tg in pred: rh += 1
            print(f"{k_:>3}{th:>6.2f}{MIN_SIM:>6.2f}{p/n:>8.3f}{r/n:>7.3f}"
                  f"{t3h/max(t3t,1):>7.3f}{rh/max(rt,1):>9.3f}"
                  f"{len(preds)/len(hold):>8.1%}{ntags/n:>9.1f}")

            
def main():
    df, tags, emb = load_all()
    print(f"MODE={MODE} K={K} MIN_SIM={MIN_SIM} THRESHOLD={THRESHOLD} MAX_TAGS={MAX_TAGS}\n")
    if MODE == "eval":
        run_eval(df, tags, emb)
    elif MODE == "propagate":
        run_propagate(df, tags, emb)
    elif MODE == "sweep":
        run_sweep(df, tags, emb)
    else:
        sys.exit(f"Unknown MODE {MODE!r}")


if __name__ == "__main__":
    main()
