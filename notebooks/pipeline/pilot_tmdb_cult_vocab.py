"""
pilot_tmdb_cult_vocab.py -- v2 pilot 2: what cult vocabulary does TMDB have?

Pilot 1 falsified the design assumption that a 'cult film' keyword anchors the
mask (0/8 canon films carried one, incl. Rocky Horror and Eraserhead). But the
neighbour signal was visible ('psychotronic' on Rocky Horror). This probe
answers the follow-up empirically:

  1. Which cult-adjacent keywords EXIST in TMDB's vocabulary (search/keyword)?
  2. How many films carry each (discover/movie with_keywords total_results)?

That coverage number is the decision data: a neighbour-set mask only works if
the union covers enough films to seat 2 personalized picks per profile.

Also re-checks The Room (2003) properly -- pilot 1's title search mismatched
to The Green Room (1978). Here we exact-match title AND year, then pull its
keywords, since a badfilm-canon title's keywords are high-signal for the mask.

Needs TMDB_API_KEY in env. Run from REPO ROOT:
    python notebooks/pipeline/pilot_tmdb_cult_vocab.py
"""

import os
import sys
import time

import requests

API_KEY = os.environ.get("TMDB_API_KEY")
if not API_KEY:
    sys.exit("Set TMDB_API_KEY in your environment first.")

BASE = "https://api.themoviedb.org/3"
DELAY = 0.25

TERMS = [
    "cult",
    "cult film",
    "cult classic",
    "midnight movie",
    "b movie",
    "camp",
    "psychotronic",
    "exploitation",
    "grindhouse",
    "so bad it's good",
    "troma",
    "video nasty",
]


def get(path, **params):
    params["api_key"] = API_KEY
    r = requests.get(f"{BASE}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    print("TMDB cult-vocabulary probe\n" + "=" * 70)
    found = []

    for term in TERMS:
        data = get("search/keyword", query=term)
        time.sleep(DELAY)
        results = data.get("results", [])[:4]
        if not results:
            print(f"\n'{term}': no keyword exists")
            continue
        print(f"\n'{term}':")
        for kw in results:
            disc = get("discover/movie", with_keywords=kw["id"])
            time.sleep(DELAY)
            n = disc.get("total_results", 0)
            print(f"  [{kw['id']:>6}] {kw['name']:<30} {n:,} films")
            found.append((kw["name"], kw["id"], n))

    # ---- The Room, done right this time: exact title + year match ----
    print("\n" + "=" * 70 + "\nThe Room (2003) sanity check:")
    data = get("search/movie", query="The Room", year=2003)
    time.sleep(DELAY)
    match = None
    for cand in data.get("results", []):
        if cand.get("title") == "The Room" and str(cand.get("release_date", "")).startswith("2003"):
            match = cand
            break
    if match is None:
        print("  exact match not found in search results")
    else:
        kws = get(f"movie/{match['id']}/keywords").get("keywords", [])
        time.sleep(DELAY)
        print(f"  id={match['id']}  votes={match.get('vote_count')}  "
              f"rating={match.get('vote_average')}")
        print(f"  keywords ({len(kws)}): {', '.join(k['name'] for k in kws) or '(none)'}")

    print("\n" + "=" * 70)
    top = sorted(found, key=lambda x: -x[2])[:10]
    print("top keywords by coverage:")
    for name, kid, n in top:
        print(f"  {name:<30} id={kid:<8} {n:,} films")


if __name__ == "__main__":
    main()