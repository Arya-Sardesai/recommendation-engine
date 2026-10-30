"""
fix_probe_msg.py -- probe_why_books.py printed "anchor has NO tags" for EVERY
anchor whenever the per-anchor mix is 0 (the default), because the check was
`if mix and anchor in tags`. Kitchen Confidential has tags; the message lied.
After this the probe prints the anchor's own top tags, or says it truly has none.

USAGE (repo root):  python fix_probe_msg.py
"""
from pathlib import Path

P = Path(__file__).resolve().parent / "notebooks" / "pipeline" / "probe_why_books.py"
OLD = '''    if mix and abid in tags_df.index:
        at = tags_df.loc[abid].values.astype("float32")
        prof = (1 - mix) * glob / np.linalg.norm(glob) + mix * at / np.linalg.norm(at)
        print(f"\\n  anchor's own top tags: "
              f"{', '.join(pd.Series(at, index=cols).nlargest(6).index)}")
    else:
        print("\\n  anchor has NO tags -> per-anchor mix falls back to the global profile")'''
NEW = '''    if abid in tags_df.index:
        at = tags_df.loc[abid].values.astype("float32")
        if mix:
            prof = (1 - mix) * glob / np.linalg.norm(glob) + mix * at / np.linalg.norm(at)
        print(f"\\n  anchor's own top tags: "
              f"{', '.join(pd.Series(at, index=cols).nlargest(6).index)}"
              f"{'' if mix else '   (per-anchor mix is 0: candidates are judged on the GLOBAL profile)'}")
    else:
        print("\\n  anchor has NO tags -> judged on the global profile "
              "(BOOK_UNTAGGED_ANCHOR_TAGS scales that term)")'''

src = P.read_text(encoding="utf-8")
n = src.count(OLD)
if n != 1:
    raise SystemExit(f"matched {n} times, not written -- send me probe_why_books.py")
P.write_text(src.replace(OLD, NEW, 1), encoding="utf-8")
print(f"fixed {P}")
