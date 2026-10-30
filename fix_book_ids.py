"""
fix_book_ids.py -- a saved book rating can point at TWO different books.

FOUND (Bourdain probe, Oct 27)
Your backup has
    book_id 265529  "Tony Bourdain Boxset - Kitchen Confidential & Medium Raw"
    corpus_index 756
but row 756 of the current corpus is Kitchen Confidential, a different row.
A rating stores both, and the ranker uses BOTH:
    corpus_index -> the anchor's embedding  (Kitchen Confidential's blurb)
    book_id      -> its tags, "already rated", exclusions  (the boxset)
_migrate_book_ratings only re-points a rating when the index is out of range
or the id is gone. Here both are individually valid, so the mismatch survives
every load. It happens to any rating saved before a corpus rebuild that kept
the row count but moved rows.

It's also why the untagged-anchor setting did nothing under Bourdain: the
ranker checked the BOXSET's id (tagged) while the probe showed Kitchen
Confidential (untagged).

THE FIX (one edit to _migrate_book_ratings in hf-space/app.py)
book_id is the truth (it's what you rated: matched_title agrees with it), so
corpus_index is recomputed from book_id whenever they disagree. Counted as a
migration, so the app's existing "ratings updated" note covers it.

USAGE (repo root):
    python fix_book_ids.py --check C:\\Users\\aryas\\Downloads\\my-recommendation-profile.json
                                 # lists your mismatched ratings, changes nothing
    python fix_book_ids.py       # dry run of the app.py edit
    python fix_book_ids.py --write   # edits ..\\hf-space\\app.py, keeps app.py.bak4
"""
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SPACE = REPO.parent / "hf-space"
APP = Path(sys.argv[sys.argv.index("--app") + 1]) if "--app" in sys.argv else SPACE / "app.py"
CORPUS = REPO / "data" / "processed" / "books_v2_hc3.parquet"

EDITS = [
    ("position table",
     '    valid_ids = set(b_df["book_id"].astype(str))\n',
     '    valid_ids = set(b_df["book_id"].astype(str))\n'
     '    # book_id -> its row now. corpus_index is recomputed from book_id when\n'
     '    # the two disagree: a rating saved before a rebuild that moved rows kept\n'
     '    # an in-range index pointing at a DIFFERENT book (the Bourdain boxset\n'
     '    # rating read Kitchen Confidential\'s embedding with the boxset\'s tags).\n'
     '    _pos = {}\n'
     '    for _i, _b in enumerate(b_df["book_id"].astype(str)):\n'
     '        _pos.setdefault(_b, _i)\n'),
    ("trust book_id",
     "        if isinstance(idx, int) and 0 <= idx < n_rows and bid in valid_ids:\n"
     "            _place(r)\n"
     "            continue\n",
     "        if bid in _pos:\n"
     "            if idx != _pos[bid]:\n"
     "                r = dict(r)\n"
     "                r[\"corpus_index\"] = _pos[bid]\n"
     "                migrated += 1\n"
     "            _place(r)\n"
     "            continue\n"),
]


def check(profile):
    import pandas as pd
    corpus = Path(sys.argv[sys.argv.index("--corpus") + 1]) if "--corpus" in sys.argv else CORPUS
    df = pd.read_parquet(corpus, columns=["book_id", "title"])
    ids = df["book_id"].astype(str).tolist()
    pos = {}
    for i, b in enumerate(ids):
        pos.setdefault(b, i)
    raw = json.loads(Path(profile).read_text(encoding="utf-8"))
    ratings = raw["data"]["ratings"] if isinstance(raw.get("data"), dict) else raw
    bad = 0
    for r in ratings:
        bid, idx = str(r.get("book_id")), r.get("corpus_index")
        if bid in pos and idx != pos[bid]:
            bad += 1
            at = df.iloc[idx]["title"] if isinstance(idx, int) and 0 <= idx < len(df) else "(out of range)"
            print(f"  rated  {str(r.get('matched_title'))[:44]:46s} id {bid}")
            print(f"    index {idx} is  {str(at)[:44]}   -> fixed to {pos[bid]}")
    print(f"\n{bad} of {len(ratings)} ratings point at two different books")


def main():
    if "--check" in sys.argv:
        return check(sys.argv[sys.argv.index("--check") + 1])
    write = "--write" in sys.argv
    src = APP.read_text(encoding="utf-8")
    if "_pos.setdefault(_b, _i)" in src:
        sys.exit("app.py already has this fix -- nothing to do")
    i = src.index("\ndef _migrate_book_ratings(")
    j = src.find("\ndef ", i + 5)
    fn = src[i:j]
    bad = []
    for name, old, _ in EDITS:
        n = fn.count(old)
        print(f"  {'ok ' if n == 1 else 'BAD'}  {name:<16} matches {n} (in _migrate_book_ratings)")
        if n != 1:
            bad.append(name)
    if bad:
        sys.exit("\nnot written. Send me the current app.py.")
    for _, old, rep in EDITS:
        fn = fn.replace(old, rep, 1)
    new = src[:i] + fn + src[j:]
    compile(new, str(APP), "exec")
    if not write:
        print("\ndry run OK. Re-run with --write to apply.")
        return
    shutil.copy2(APP, APP.with_name("app.py.bak4"))
    APP.write_text(new, encoding="utf-8")
    print(f"\nwrote {APP} (backup app.py.bak4)")


if __name__ == "__main__":
    main()
