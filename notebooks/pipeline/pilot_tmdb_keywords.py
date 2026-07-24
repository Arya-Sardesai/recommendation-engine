"""
pilot_tmdb_keywords.py -- v2 pilot: what do TMDB keywords actually contain?

Pilot-before-full-run. Two questions this answers, both load-bearing for v2:

  1. TONE/SATIRE: do keywords capture what overviews can't? The known residual
     is This Is Spinal Tap -> K-pop documentaries (satire invisible in plot
     summary). If keywords carry 'mockumentary'/'satire', the v2 keyword pass
     fixes it. If not, that residual survives v2.

  2. CULT MEMBERSHIP: does the 'cult film' keyword exist, and does it cover the
     canon? The cult-slot design assumes a reception-derived keyword mask. If
     canonical cult titles come back without it, the mask needs a wider
     neighbour set (midnight movie, b-movie, camp, exploitation) -- or the
     feature needs rethinking before any code gets written.

Also prints keyword COUNT per film, which tells us how dense the tag layer
will be relative to the MovieLens genome (median ~5-7 tags/film).

Needs TMDB_API_KEY in env. Run from REPO ROOT:
    python notebooks/pipeline/pilot_tmdb_keywords.py
"""

import os
import sys
import time

import requests

API_KEY = os.environ.get("TMDB_API_KEY")
if not API_KEY:
    sys.exit("Set TMDB_API_KEY in your environment first.")

BASE = "https://api.themoviedb.org/3"
DELAY = 0.3  # polite; TMDB allows ~50 req/s but there is no need to push

# (title, year, why-it-is-here)
TARGETS = [
    ("This Is Spinal Tap", 1984, "SATIRE residual - overview reads as real doc"),
    ("Om Shanti Om",       2007, "MOOD residual - Bollywood tone invisible"),
    ("The Room",           2003, "CULT canon - paracinema/badfilm, low rating"),
    ("The Rocky Horror Picture Show", 1975, "CULT canon - midnight movie"),
    ("Eraserhead",         1977, "CULT canon - transgressive/arthouse"),
    ("Donnie Darko",       2001, "CULT canon - fractured reception"),
    ("Fight Club",         1999, "CULT-adjacent but mainstream-successful"),
    ("The Godfather",      1972, "CONTROL - canonical, should NOT be cult"),
    ("Parasite",           2019, "CONTROL - recent prestige, non-English"),
]


def search_movie(title, year):
    r = requests.get(f"{BASE}/search/movie",
                     params={"api_key": API_KEY, "query": title, "year": year},
                     timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])
    return results[0] if results else None


def get_keywords(movie_id):
    r = requests.get(f"{BASE}/movie/{movie_id}/keywords",
                     params={"api_key": API_KEY}, timeout=30)
    r.raise_for_status()
    return [k["name"] for k in r.json().get("keywords", [])]


def main():
    print(f"TMDB keyword pilot -- {len(TARGETS)} films\n" + "=" * 70)
    cult_hits, counts = [], []

    for title, year, why in TARGETS:
        hit = search_movie(title, year)
        time.sleep(DELAY)
        if not hit:
            print(f"\n{title} ({year}) -- NOT FOUND")
            continue

        kws = get_keywords(hit["id"])
        time.sleep(DELAY)
        counts.append(len(kws))

        has_cult = any("cult" in k.lower() for k in kws)
        if has_cult:
            cult_hits.append(title)

        print(f"\n{hit['title']} ({hit.get('release_date','?')[:4]})  id={hit['id']}")
        print(f"  why: {why}")
        print(f"  vote_average={hit.get('vote_average')}  votes={hit.get('vote_count')}")
        print(f"  keywords ({len(kws)}): {', '.join(kws) if kws else '(none)'}")
        print(f"  cult-keyword present: {'YES' if has_cult else 'no'}")

    print("\n" + "=" * 70)
    if counts:
        counts_sorted = sorted(counts)
        median = counts_sorted[len(counts_sorted) // 2]
        print(f"keyword count: min={min(counts)} median={median} max={max(counts)}")
    print(f"cult-keyword films: {cult_hits if cult_hits else 'NONE -- mask needs rethink'}")


if __name__ == "__main__":
    main()