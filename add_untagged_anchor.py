"""
add_untagged_anchor.py -- no generic mood tags under an anchor that has no tags.

THE BUG (probe_why_books.py --anchor "Bourdain", Oct 26)
Kitchen Confidential has no Hardcover tags. Its raw neighbours are right
(Dirt, Humble Pie, Cork Dork, Service Included), but the tag term judges
every candidate against the WHOLE profile's signature -- funny, emotional,
adventurous -- and that gave Secret Service (a romance) +0.10 and the
untagged Humble Pie -0.05, so the romance won. With tags off the picks were
Humble Pie, Dirt, Cork Dork.

THE FIX (recommend_books only)
New setting untagged_anchor_tags (constant BOOK_UNTAGGED_ANCHOR_TAGS):
the tag weight used for candidates pulled in by an anchor with NO tags,
as a multiple of tag_weight. 1.0 = old behaviour, 0.0 = tag layer off for
those anchors. Anchors with tags are unchanged.

Ships at 1.0 (no change) until tests\\eval_untagged_anchor.py says otherwise.

Edits are made inside recommend_books' source only, each must match exactly
once there, and the film/TV rankers are checked to be byte-identical after.

USAGE (repo root):
    python add_untagged_anchor.py            # dry run
    python add_untagged_anchor.py --write    # edits ..\\hf-space\\app.py, keeps app.py.bak3
"""
import shutil
import sys
from pathlib import Path

SPACE = Path(__file__).resolve().parent.parent / "hf-space"
APP = Path(sys.argv[sys.argv.index("--app") + 1]) if "--app" in sys.argv else SPACE / "app.py"

CONST_OLD = "BOOK_AWARD_NOM = "
CONST_NEW = (
    "# Tag weight under an anchor with no tags, as a multiple of tag_weight. Such\n"
    "# an anchor can only be judged against the profile-wide moods (funny,\n"
    "# emotional ...), which lifted a romance over chef memoirs under Kitchen\n"
    "# Confidential. 1.0 = old; 0.0 = no tag term. See tests/eval_untagged_anchor.py.\n"
    "BOOK_UNTAGGED_ANCHOR_TAGS = 1.0\n"
    "BOOK_AWARD_NOM = ")

# (name, old, new) -- applied inside recommend_books only
FN_EDITS = [
    ("signature",
     "awards=None, award_win=BOOK_AWARD_WIN, award_nom=BOOK_AWARD_NOM):",
     "awards=None, award_win=BOOK_AWARD_WIN, award_nom=BOOK_AWARD_NOM,\n"
     "                    untagged_anchor_tags=BOOK_UNTAGGED_ANCHOR_TAGS):"),

    ("store per-anchor weight",
     '        per_anchor_queues.append({"like": liked[like_i], "queue": queue})',
     '        _a_tagged = (tags_df is not None\n'
     '                     and str(liked[like_i].get("book_id")) in tags_df.index)\n'
     '        per_anchor_queues.append({"like": liked[like_i], "queue": queue,\n'
     '                                  "tw": tag_weight * (1.0 if _a_tagged\n'
     '                                                      else untagged_anchor_tags)})'),

    ("pass 2 reads it",
     "    for pbq in per_anchor_queues:\n        scored = []\n",
     "    for pbq in per_anchor_queues:\n        scored = []\n"
     "        _atw = pbq.get(\"tw\", tag_weight)\n"),

    ("tagged candidate",
     "                adj = tag_weight * (tag_sim - _mu)",
     "                adj = _atw * (tag_sim - _mu)"),

    ("untagged candidate",
     "                adj = tag_weight * (0.0 - _mu)   # untagged reads as below-average",
     "                adj = _atw * (0.0 - _mu)   # untagged reads as below-average"),
]


def _span(src, name):
    i = src.index(f"\ndef {name}(")
    j = src.find("\ndef ", i + 5)
    return i, j


def main():
    write = "--write" in sys.argv
    src = APP.read_text(encoding="utf-8")
    if "BOOK_UNTAGGED_ANCHOR_TAGS" in src:
        sys.exit("app.py already has BOOK_UNTAGGED_ANCHOR_TAGS -- nothing to do")
    if "BOOK_AWARD_WIN" not in src:
        sys.exit("run add_book_awards.py --write first (this patch builds on it)")

    bad = []
    n = src.count(CONST_OLD)
    print(f"  {'ok ' if n == 1 else 'BAD'}  {'constant':<26} matches {n}")
    if n != 1:
        bad.append("constant")
    i, j = _span(src, "recommend_books")
    fn = src[i:j]
    for name, old, _ in FN_EDITS:
        n = fn.count(old)
        print(f"  {'ok ' if n == 1 else 'BAD'}  {name:<26} matches {n} (in recommend_books)")
        if n != 1:
            bad.append(name)
    if bad:
        sys.exit(f"\nnot written: {', '.join(bad)} did not match exactly once. "
                 "Send me the current app.py and I'll redo the anchors.")

    for _, old, rep in FN_EDITS:
        fn = fn.replace(old, rep, 1)
    new = src[:i] + fn + src[j:]
    new = new.replace(CONST_OLD, CONST_NEW, 1)

    for other in ("recommend_movies", "recommend_tv"):
        a, b = _span(src, other)
        c, d = _span(new, other)
        if src[a:b] != new[c:d]:
            sys.exit(f"not written: the edit leaked into {other}")
    compile(new, str(APP), "exec")

    if not write:
        print("\ndry run OK. Re-run with --write to apply.")
        return
    shutil.copy2(APP, APP.with_name("app.py.bak3"))
    APP.write_text(new, encoding="utf-8")
    print(f"\nwrote {APP} (backup app.py.bak3). Ships at 1.0 = no change until the eval.")


if __name__ == "__main__":
    main()
