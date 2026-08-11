"""Scale-mismatch probe: are emb_score and tag_sim on comparable scales?

Reproduces recommend_tv's scoring for a small profile, then prints the
distribution of emb_score, tag_sim, and the resulting blended score, split by
whether the candidate is tagged. If the hypothesis holds, tagged candidates
lose score relative to untagged ones purely from the blend.

Run:  python probe_tag_scale.py
"""
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

R = "Arya2305/recommendation-engine-data"
T_CORPUS = "tv_corpus.parquet"
T_EMB = "tv_embeddings_bgem3_v2_kw_notitle.npy"   # matches app.py:56   # <-- match T_EMB in app.py
T_TAGS = "tv_tags.parquet"

TAG_WEIGHT = 0.25
NEUTRAL = 3.0
MIN_VOTES = 300
MIN_VA = 6.0
SHRINK = 500
RETRIEVE_K = 150

# A small profile: (tmdb_id, rating). Edit freely.
PROFILE = [
    (57243, 5),    # Doctor Who (2005)
    (70523, 5),    # Dark
    (95396, 5),    # Severance
    (18347, 5),    # Community
    (8592, 5),     # Parks and Recreation
]

print("loading...")
df = pd.read_parquet(hf_hub_download(R, T_CORPUS, repo_type="dataset")).reset_index(drop=True)
emb = np.load(hf_hub_download(R, T_EMB, repo_type="dataset"))
tdf = pd.read_parquet(hf_hub_download(R, T_TAGS, repo_type="dataset"))
tdf["tv_id"] = tdf["tv_id"].astype(str)
tags_df = tdf.pivot_table(index="tv_id", columns="tag", values="score", fill_value=0.0)
print(f"corpus={len(df):,} emb={emb.shape} tagged={len(tags_df):,}\n")

id_to_pos = {int(t): i for i, t in enumerate(df["tmdb_id"])}

# --- build the signed tag profile exactly as recommend_tv does -------------
t_vecs, t_w = [], []
for tid, rating in PROFILE:
    w = rating - NEUTRAL
    if w == 0:
        continue
    if str(tid) in tags_df.index:
        t_vecs.append(tags_df.loc[str(tid)].values)
        t_w.append(w)
_tw = np.array(t_w, dtype="float32")
tag_profile = (np.array(t_vecs, dtype="float32") * _tw[:, None]).sum(axis=0) / np.abs(_tw).sum()
print(f"tag profile built from {len(t_vecs)}/{len(PROFILE)} rated shows")
print(f"top tags: {', '.join(pd.Series(tag_profile, index=tags_df.columns).nlargest(5).index)}\n")

_C = float(df.loc[df["vote_count"] >= MIN_VOTES, "vote_average"].mean())

rows = []
for tid, rating in PROFILE:
    if tid not in id_to_pos:
        print(f"  !! {tid} not in corpus")
        continue
    anchor = emb[id_to_pos[tid]].astype("float32")
    sims = emb @ anchor                      # full scan; fine for one probe
    top = np.argsort(-sims)[:RETRIEVE_K]
    for idx in top:
        show = df.iloc[idx]
        if int(show["tmdb_id"]) in {p[0] for p in PROFILE}:
            continue
        if int(show["vote_count"]) < MIN_VOTES:
            continue
        va = float(show["vote_average"]) if pd.notna(show["vote_average"]) else None
        if va is None:
            continue
        vc = int(show["vote_count"])
        va_adj = (vc * va + SHRINK * _C) / (vc + SHRINK)
        if va_adj < MIN_VA:
            continue
        quality = (va_adj - MIN_VA) / (10.0 - MIN_VA)
        emb_score = float(sims[idx]) + 0.12 * quality   # no cross/neg: single-anchor probe

        sid = str(int(show["tmdb_id"]))
        tagged = sid in tags_df.index
        tag_sim = np.nan
        blended = emb_score
        if tagged:
            sv = tags_df.loc[sid].values
            nu, ns = np.linalg.norm(tag_profile), np.linalg.norm(sv)
            if nu > 0 and ns > 0:
                tag_sim = float(np.dot(tag_profile, sv) / (nu * ns))
                blended = (1 - TAG_WEIGHT) * emb_score + TAG_WEIGHT * tag_sim
        rows.append({"anchor": tid, "name": show["name"], "tagged": tagged,
                     "emb_score": emb_score, "tag_sim": tag_sim,
                     "blended": blended, "delta": blended - emb_score})

res = pd.DataFrame(rows).drop_duplicates(subset="name")
print(f"candidates: {len(res)}  tagged: {res['tagged'].sum()}  untagged: {(~res['tagged']).sum()}\n")

print("=== SCALE COMPARISON ===")
print(f"emb_score : mean {res['emb_score'].mean():.3f}  "
      f"min {res['emb_score'].min():.3f}  max {res['emb_score'].max():.3f}")
t = res[res["tagged"]]
print(f"tag_sim   : mean {t['tag_sim'].mean():.3f}  "
      f"min {t['tag_sim'].min():.3f}  max {t['tag_sim'].max():.3f}")
print()
print(f"blend effect on tagged candidates: mean delta {t['delta'].mean():+.4f}")
print(f"  helped (delta>0): {(t['delta'] > 0).sum()}   hurt (delta<0): {(t['delta'] < 0).sum()}")
print()

print("=== TOP 15 BY FINAL BLENDED SCORE ===")
top = res.sort_values("blended", ascending=False).head(15)
for _, r in top.iterrows():
    ts = f"{r['tag_sim']:.3f}" if r["tagged"] else "  -  "
    flag = "TAG" if r["tagged"] else "   "
    print(f"  {flag} {r['blended']:.3f}  (emb {r['emb_score']:.3f}, tag {ts})  {r['name'][:45]}")
print()
print(f"tagged share of top 15: {top['tagged'].sum()}/15")
print(f"tagged share of pool  : {res['tagged'].sum()}/{len(res)} "
      f"({100 * res['tagged'].mean():.0f}%)")
