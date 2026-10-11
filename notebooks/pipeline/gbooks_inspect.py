"""
gbooks_inspect.py -- what did Google actually return for the misses?

The trickle reported 2 of 337 with an English blurb, against 3 of 3 in the
Sept 12 test. That is a lookup problem, not a data finding. This prints, for
the first --n lookups: the title we searched, how many volumes came back, and
for each volume its title / language / blurb length. Read-only, no API calls.

    python notebooks\\pipeline\\gbooks_inspect.py
    python notebooks\\pipeline\\gbooks_inspect.py --n 40
"""
import argparse
import json
from collections import Counter
from pathlib import Path

OUT = Path("data/raw/books/hardcover_v2/gbooks_raw.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()
    rows = [json.loads(l) for l in OUT.open(encoding="utf-8") if l.strip()]

    n_items = Counter(len(r.get("items") or []) for r in rows)
    langs = Counter(i.get("language") for r in rows for i in (r.get("items") or []))
    has_desc = sum(1 for r in rows for i in (r.get("items") or []) if i.get("description"))
    print(f"{len(rows)} lookups")
    print(f"  volumes returned per lookup: {dict(sorted(n_items.items()))}")
    print(f"  languages across all volumes: {dict(langs.most_common(8))}")
    print(f"  volumes with ANY description: {has_desc}")
    paren = sum(1 for r in rows if "(" in r["title"])
    colon = sum(1 for r in rows if ":" in r["title"])
    print(f"  searched titles with '(': {paren}   with ':': {colon}\n")

    for r in rows[:args.n]:
        items = r.get("items") or []
        print(f"[{r['users_count']:>5}] {r['title'][:60]}  /  {r['author'][:25]}"
              f"  -> {len(items)} volumes")
        for i in items[:3]:
            d = i.get("description") or ""
            print(f"         {str(i.get('title'))[:45]:47s} {str((i.get('authors') or ['?'])[0])[:20]:22s}"
                  f" {i.get('language')}  {len(d)} chars")


if __name__ == "__main__":
    main()
