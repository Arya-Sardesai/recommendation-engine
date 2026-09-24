"""
why_missing.py — for a title someone reported as missing, say WHY.

Four different causes, four different fixes, and they are indistinguishable
from inside the app:

  A. IN THE CORPUS ALREADY   -> a search/matching problem, not coverage
  B. HARVESTED, NO DESCRIPTION -> the 27K gap; description recovery fixes it
  C. HARVESTED, BELOW THE FLOOR -> in the corpus but ratings_count < 10 so it
     never surfaces; a constant, not a harvest
  D. NEVER HARVESTED         -> below the users_count >= 13 harvest floor, or
     not in Hardcover at all; needs a targeted fetch or another source

Guessing between these is how a week goes into the wrong one.

USAGE
    python notebooks\\pipeline\\why_missing.py "Dongri to Dubai" "The Maltese Falcon"
    python notebooks\\pipeline\\why_missing.py --file reported.txt
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/books/hardcover_v2/raw.jsonl")
CORPUS = Path("data/processed/books_v2_hc2.parquet")
MIN_DESC = 100     # the harvest's description filter
SERVE_FLOOR = 10   # app.py's recommend_books ratings_count floor


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def _core(t):
    t = re.sub(r"\([^)]*\)", " ", str(t))
    t = re.split(r":| - ", t)[0]
    return re.sub(r"^(the|a|an) ", "", _norm(t)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("titles", nargs="*")
    ap.add_argument("--file", help="one title per line")
    args = ap.parse_args()

    titles = list(args.titles)
    if args.file:
        titles += [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines()
                   if l.strip()]
    if not titles:
        sys.exit('give at least one title, e.g. why_missing.py "Dongri to Dubai"')

    if not CORPUS.exists():
        sys.exit(f"{CORPUS} not found")
    df = pd.read_parquet(CORPUS, columns=["title", "primary_author",
                                          "ratings_count", "description"])
    corpus_keys = {}
    for i, (t, a) in enumerate(zip(df["title"], df["primary_author"])):
        corpus_keys.setdefault(_core(t), i)

    wanted = {_core(t): t for t in titles}
    raw_hits = {}
    if RAW.exists():
        with RAW.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    b = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = _core(b.get("title") or "")
                if k in wanted:
                    prev = raw_hits.get(k)
                    # keep the best-described record for this work
                    if prev is None or len((b.get("description") or "")) > len(
                            prev.get("description") or ""):
                        raw_hits[k] = b
    else:
        print(f"NOTE: {RAW} not found -- cannot distinguish B/C/D\n")

    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    print(f"{'title':<44}{'verdict':<34}")
    print("-" * 84)
    for k, orig in wanted.items():
        ci = corpus_keys.get(k)
        raw = raw_hits.get(k)
        if ci is not None:
            rc = df.iloc[ci]["ratings_count"]
            if pd.notna(rc) and rc < SERVE_FLOOR:
                v, code = f"C: in corpus, {int(rc)} ratings < {SERVE_FLOOR} floor", "C"
            else:
                v, code = f"A: IN THE CORPUS ({int(rc)} ratings)", "A"
        elif raw is not None:
            d = len((raw.get("description") or "").strip())
            uc = raw.get("users_count") or 0
            if d < MIN_DESC:
                v, code = f"B: harvested, no description (users {uc})", "B"
            else:
                v, code = f"?: harvested and described but absent (users {uc})", "B"
        else:
            v, code = "D: never harvested -- below floor or not in Hardcover", "D"
        counts[code] += 1
        print(f"{orig[:42]:<44}{v}")

    n = max(sum(counts.values()), 1)
    print("\n" + "=" * 70)
    print("WHAT TO ACTUALLY BUILD")
    print("=" * 70)
    for code, label in (("A", "already there (search/matching problem)"),
                        ("B", "description recovery would fix"),
                        ("C", "below the serving floor -- change a constant"),
                        ("D", "needs a harvest or another source")):
        print(f"  {code}  {counts[code]:>3}  {counts[code]/n:>5.0%}  {label}")
    print()
    top = max(counts, key=counts.get)
    if top == "B":
        print("  Description recovery is the right work. Dump for pre-2018,")
        print("  Google Books for 2018+.")
    elif top == "C":
        print("  These books are ALREADY in the corpus and hidden by the")
        print("  ratings_count floor. Lowering it is one constant and costs")
        print("  nothing to try -- do that before any harvest.")
    elif top == "D":
        print("  Recovery will NOT fix these. They were never harvested, so the")
        print("  fix is a deeper Hardcover sweep or a different catalogue.")
    else:
        print("  These are in the corpus and servable, so the tab is failing to")
        print("  SURFACE them. That is a search/matching bug, not coverage, and")
        print("  it is much cheaper to fix than either harvest.")


if __name__ == "__main__":
    main()
