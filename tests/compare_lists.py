"""
compare_lists.py -- your top 20 under the live ranker vs a candidate setting,
side by side, on your real profile. The "titles read better" half of the ship
rule: the evals say whether a change helps on average, this shows what it does
to YOUR list.

    python tests\\compare_lists.py
    python tests\\compare_lists.py --kinds movie --set neg_floor_k=200 tag_idf=1.0

Default candidate: neg_floor_k=200 tag_idf=1.0 (sim_scale stays 1.0), the
setting that won both evals on Oct 19.

Marks:  +  new in the candidate's top 20     -  dropped out of it
        ^3 / v2  moved up / down that many places
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import src.space_ranker as sr  # noqa: E402


def _parse_set(items):
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        v = v.strip()
        try:
            out[k.strip()] = int(v)
        except ValueError:
            out[k.strip()] = float(v)
    return out


def _call(b, ratings, skipped, overrides, n):
    """recommend_* exactly as the Space tab calls it, plus overrides."""
    fn = b["ns"][b["spec"]["rank"]]
    kwargs = dict(b["call_literal"])
    dynamic = {"tags_df": b["tags"], "skip_ids": list(skipped or []),
               "forms": b["forms"], "awards": b.get("awards") or {}}
    for k in b["call_keywords"]:
        if k in dynamic:
            kwargs[k] = dynamic[k]
    kwargs["n"] = n
    kwargs.update(overrides)
    recs, _ = fn(ratings, b["df"], b["emb"], b["index"], **kwargs)
    return recs


def dislike_report(b, ratings, titles, k, neutral=3.0):
    """For each title: the disliked rating it sits closest to, and the dislike
    penalty it pays with the raw cosine vs with the floor at k. Same arithmetic
    as recommend_movies/recommend_tv (before neg_scale)."""
    import numpy as np
    dis = [r for r in ratings if r.get("corpus_index") is not None and r["rating"] < neutral]
    if not dis:
        print("\n  no dislikes in this profile, so the floor cannot change anything")
        return
    tcol = b["spec"]["title_col"]
    D = np.array([b["emb"][int(r["corpus_index"])] for r in dis], dtype="float32")
    w = np.array([(neutral - r["rating"]) / 2.0 for r in dis], dtype="float32")
    tau = b["index"].search(D, int(k))[0][:, -1].astype("float32")
    dname = [str(b["df"].iloc[int(r["corpus_index"])][tcol]) for r in dis]
    pos_by_title = {}
    for i, t in enumerate(b["df"][tcol].astype(str)):
        pos_by_title.setdefault(t, i)
    print(f"\n  dislike penalty, raw cosine vs floor at k={k}  "
          f"({len(dis)} dislikes; the floor only charges cosine above each one's tau)")
    print(f"    {'title':<40}{'raw':>6}{'floor':>7}   closest dislike (cos / its tau)")
    for mark, t in titles:
        p = pos_by_title.get(t)
        if p is None:
            continue
        cos = D @ np.asarray(b["emb"][p], dtype="float32")
        raw = float(np.clip(cos * w, 0, None).max())
        flo = float(np.clip((cos - tau) * w, 0, None).max())
        j = int(np.argmax(cos * w))
        print(f"  {mark} {t[:38]:<40}{raw:6.3f}{flo:7.3f}   {dname[j][:30]} "
              f"({cos[j]:.2f} / {tau[j]:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(sr.DEFAULT_PROFILE))
    ap.add_argument("--kinds", nargs="+", default=["movie", "tv"])
    ap.add_argument("--set", nargs="+", default=["neg_floor_k=200", "tag_idf=1.0"],
                    metavar="KEY=VALUE", help="candidate setting")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--why", action="store_true",
                    help="show each changed title's dislike penalty, raw vs floored")
    args = ap.parse_args()
    cand = _parse_set(args.set)
    raw = Path(args.profile).read_text(encoding="utf-8")

    for kind in args.kinds:
        b = sr.load(kind)
        ratings, skipped = sr.read_profile(raw, kind)
        ratings, _ = sr.normalize_ratings(b, ratings)
        title = b["spec"]["title_col"]
        live = _call(b, ratings, skipped, {}, args.n)
        new = _call(b, ratings, skipped, cand, args.n)
        lt = [str(x) for x in live[title]] if len(live) else []
        nt = [str(x) for x in new[title]] if len(new) else []
        because = dict(zip(nt, new["because_of"])) if len(new) else {}
        because_live = dict(zip(lt, live["because_of"])) if len(live) else {}

        print(f"\n=== {kind.upper()} " + "=" * 78)
        print(f"  live ranker{'':<33}candidate {cand}")
        print("  " + "-" * 92)
        for i in range(max(len(lt), len(nt))):
            a = lt[i] if i < len(lt) else ""
            c = nt[i] if i < len(nt) else ""
            ma = "-" if a and a not in nt else " "
            if c and c not in lt:
                mc = "+"
            elif c:
                d = lt.index(c) - i
                mc = f"^{d}" if d > 0 else (f"v{-d}" if d < 0 else " ")
            else:
                mc = " "
            why = f"   <- {because.get(c, '')}" if mc == "+" else ""
            print(f"  {i + 1:>2}. {ma} {a[:38]:<40}{mc:>3} {c[:38]}{why}")
        gone = [t for t in lt if t not in nt]
        came = [t for t in nt if t not in lt]
        if gone:
            print("\n  dropped, and the rating each one had come from:")
            for t_ in gone:
                print(f"    - {t_[:45]:<47}<- {because_live.get(t_, '')}")
        if args.why and (gone or came):
            dislike_report(b, ratings, [("-", t_) for t_ in gone] + [("+", t_) for t_ in came],
                           cand.get("neg_floor_k") or 200)
        print(f"\n  {len(came)} new, {len(gone)} dropped. Read the + rows (and what each")
        print("  came 'because of') and the - rows. Ship if the new list reads better.")


if __name__ == "__main__":
    main()
