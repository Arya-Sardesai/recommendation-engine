"""
smoke_agents.py -- check the local agents against the Space. No API calls, no
credit spent.

    python tests/smoke_agents.py                    # all three, Downloads profile
    python tests/smoke_agents.py --kinds book
    python tests/smoke_agents.py --profile C:\\path\\profile.json --no-embed

Per kind:
  1. files    the B_/M_/T_ files app.py serves, present and row-aligned
  2. call     the arguments the Space's tab passes, as the agent will pass them
  3. parity   the agent's get_recommendations (no anchors) == recommend_*
              called exactly as the Space tab calls it, on your real profile
  4. tools    every tool runs once without raising
  5. embed    (skip with --no-embed) BGE-M3 encodes a query on this machine
Plus a canned-reply test of the grounding check.

Exit code 1 if anything failed.
"""
import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import space_ranker as sr  # noqa: E402
from agent_common import check_faithfulness, titles_from_result  # noqa: E402

MODULES = {"book": "agent", "movie": "agent_movies", "tv": "agent_tv"}

# (tool, args, needs_embedding). "@liked" is replaced by a title you rated 5.
CALLS = {
    "book": [
        ("search_user_ratings", {}, False),
        ("get_recommendations", {"n": 5}, False),
        ("get_recommendations", {"anchor_titles": "@liked", "n": 5}, False),
        ("get_recommendations", {"n": 5, "min_year": 2015}, False),
        ("get_recommendations", {"n": 10, "avoid_tags": ["romance"]}, False),
        ("search_catalog", {"query": "hobbit"}, False),
        ("search_by_tags", {"positive_tags": ["mystery", "dark"], "limit": 5}, False),
        ("search_by_tags", {"positive_tags": ["no-such-tag"]}, False),
        ("search_user_ratings", {"query": "dark crime"}, True),
        ("search_catalog_semantic", {"query": "a heist with dark humour", "limit": 5}, True),
    ],
    "movie": [
        ("search_user_ratings", {}, False),
        ("get_recommendations", {"n": 5}, False),
        ("get_recommendations", {"anchor_titles": "@liked", "n": 5}, False),
        ("get_recommendations", {"n": 5, "language": "Korean"}, False),
        ("get_recommendations", {"n": 10, "avoid_tags": ["horror"]}, False),
        ("search_catalog", {"query": "kiki"}, False),
        ("search_by_tags", {"positive_tags": ["heist", "dark"], "limit": 5}, False),
        ("search_by_people", {"person": "Christopher Nolan", "role": "director"}, False),
        ("search_by_studio", {"studio": "A24", "limit": 5}, False),
        ("search_catalog_semantic", {"query": "slow-burn revenge", "language": "Korean", "limit": 5}, True),
    ],
    "tv": [
        ("search_user_ratings", {}, False),
        ("get_recommendations", {"n": 5}, False),
        ("get_recommendations", {"anchor_names": "@liked", "n": 5}, False),
        ("get_recommendations", {"n": 10, "avoid_tags": ["animation"]}, False),
        ("search_catalog", {"query": "breaking"}, False),
        ("search_by_tags", {"positive_tags": ["miniseries", "heist"], "limit": 5}, False),
        ("search_catalog_semantic", {"query": "cozy comfort sitcom", "limit": 5}, True),
    ],
}


def space_call(kind, b, ratings, skipped):
    """recommend_* exactly as the Space's tab writes it -- typed out here on
    purpose, so the test does not share the agent's call-site reader."""
    ns = b["ns"]
    if kind == "book":
        return ns["recommend_books"](ratings, b["df"], b["emb"], b["index"], tags_df=b["tags"],
                                     n=20, skip_ids=skipped, forms=b["forms"])
    if kind == "movie":
        return ns["recommend_movies"](ratings, b["df"], b["emb"], b["index"], tags_df=b["tags"],
                                      n=20, skip_ids=skipped)
    return ns["recommend_tv"](ratings, b["df"], b["emb"], b["index"], tags_df=b["tags"],
                              n=20, skip_ids=skipped)


def count(res):
    for k in ("recommendations", "results", "ratings"):
        if isinstance(res.get(k), list):
            return len(res[k])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(sr.DEFAULT_PROFILE))
    ap.add_argument("--kinds", nargs="+", default=["book", "movie", "tv"])
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()

    print(f"app.py:   {sr.APP_PY}")
    print(f"data:     {sr.PROCESSED_DIR}")
    prof = Path(args.profile)
    raw = prof.read_text(encoding="utf-8") if prof.exists() else None
    print(f"profile:  {prof}" + ("" if raw else "   <- NOT FOUND, parity and tools skipped"))
    fails = 0
    shared_state = {}          # one BGE-M3 load for all three kinds

    for kind in args.kinds:
        print(f"\n=== {kind} " + "=" * 50)
        t = time.time()
        try:
            b = sr.load(kind)
        except Exception as e:
            print(f"  FAIL load: {type(e).__name__}: {e}")
            fails += 1
            continue
        print(f"  files    {b['files']}")
        print(f"  rows     {b['rows']}  aligned  ({time.time() - t:.1f}s)")
        print(f"  servable {int(b['servable'].sum()):,} of {len(b['df']):,}")
        tagged = len(b["tags"]) if b["tags"] is not None else 0
        print(f"  tags     {tagged:,} tagged")
        print(f"  call     literal={b['call_literal']}  keywords={sorted(b['call_keywords'])}")
        if b["call_problems"]:
            print(f"  FAIL call: {b['call_problems']}")
            fails += 1
        if kind == "book":
            print(f"  forms    {len(b['forms']):,} form labels loaded locally")
            if b["forms"]:
                print("           NOTE: book_forms_v3.parquet is not in the HF data repo, so the "
                      "live Books tab runs with the shelf gate OFF. Lists can differ for "
                      "non-fiction anchors until it is uploaded.")

        if raw is None:
            continue
        ratings, skipped = sr.read_profile(raw, kind)
        n_in = len(ratings)
        ratings, note = sr.normalize_ratings(b, ratings)
        print(f"  profile  {n_in} in -> {len(ratings)} usable, {len(skipped)} skipped"
              + (f"  ({note})" if note else ""))
        liked = [r for r in ratings if r["rating"] >= 5] or [r for r in ratings if r["rating"] >= 4]
        if not liked:
            print("  no ratings of 4+ in this profile; skipping parity and tools")
            continue

        mod = importlib.import_module(MODULES[kind])
        state = dict(shared_state, bundle=b, ratings=ratings, skipped=skipped)

        # --- parity ---------------------------------------------------------
        key = b["spec"]["title_col"]
        space, _ = space_call(kind, b, ratings, skipped)
        want = [str(x) for x in space[key]] if len(space) else []
        got = [r.get("title") or r.get("name")
               for r in mod.TOOL_IMPLS["get_recommendations"](state, n=20)["recommendations"]]
        if got == want:
            print(f"  PASS parity: agent list == Space list ({len(want)} titles)")
        else:
            fails += 1
            print("  FAIL parity")
            for i in range(max(len(got), len(want))):
                a = got[i] if i < len(got) else "-"
                s = want[i] if i < len(want) else "-"
                print(f"     {i + 1:2d}. {'  ' if a == s else '!='} agent: {a:45.45s} space: {s}")
        for i, t_ in enumerate(want[:10], 1):
            print(f"     {i:2d}. {t_}")

        # --- every tool -----------------------------------------------------
        for name, targs, needs_embed in CALLS[kind]:
            if needs_embed and args.no_embed:
                print(f"  skip     {name} (--no-embed)")
                continue
            targs = {k: ([liked[0]["matched_title"]] if v == "@liked" else v) for k, v in targs.items()}
            t = time.time()
            try:
                res = sr.clean(mod.TOOL_IMPLS[name](state, **targs))
            except Exception:
                fails += 1
                print(f"  FAIL     {name}({targs})")
                traceback.print_exc()
                continue
            if res.get("error"):
                fails += 1
                print(f"  FAIL     {name}({targs}) -> {res['error']}")
                continue
            extra = ""
            if targs.get("avoid_tags"):
                bad = [r.get("title") or r.get("name") for r in res.get("recommendations", [])
                       if any(sr._tagkey(g) in {sr._tagkey(a) for a in targs["avoid_tags"]}
                              for g in r.get("genres", []))]
                if bad:
                    fails += 1
                    print(f"  FAIL     avoid_tags let through: {bad}")
                extra = f"  removed={res.get('removed_by_avoid_tags', 0)}"
            if res.get("invalid_tags"):
                extra = f"  invalid={res['invalid_tags']} suggestions={res.get('suggestions')}"
            sample = list(titles_from_result({k: v for k, v in res.items() if k != "ratings"}))[:3]
            print(f"  ok       {name}({targs}) -> {count(res)}  {sample}{extra}  ({time.time() - t:.1f}s)")
        for k in ("query_embedder",):
            if k in state:
                shared_state[k] = state[k]

    # --- grounding check, canned replies -----------------------------------
    print("\n=== grounding check " + "=" * 41)
    corpus = frozenset({"piranesi", "thehobbit", "dune"})
    cases = [
        ("1. **Piranesi** (Susanna Clarke, 2020) - because you liked X", {"Piranesi"}, 0),
        ("1. **Dune** (Frank Herbert, 1965)", {"Piranesi"}, 1),
        ("- **Made Up Book** by Nobody", set(), 1),
        ("- **Pacing:** fast, and **Mood:** dark", set(), 0),
        ("You asked about Dune (1965), so here goes.", set(), 0),
        ("Try **The Silent Patient** (2019).", {"The English Patient"}, 1),
    ]
    for reply, seen, want_n in cases:
        user = "something like dune" if "asked about Dune" in reply else ""
        got = check_faithfulness(reply, seen, corpus, user)
        ok = len(got) == want_n
        fails += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {reply[:55]:55s} -> {[g['title'] for g in got]}")

    print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILURE(S)'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
