"""
build_union_tags.py -- Hardcover reader tags + LLM register tags, side by side.

WHY
eval_goodbooks --compare-tags measured (100 users, paired):
    live Hardcover tags   recall@20 0.017
    no tag layer                    0.011   t = -2.9   (Hardcover tags are REAL)
    LLM tags alone                  0.013   t = -2.1   (worse than Hardcover)

Hardcover tags are applied by readers, so `young adult` or `classics` encodes
who reads a book together -- close to a collaborative signal, which is exactly
what goodbooks rewards. The LLM tags were designed to describe how a book
READS. If the two carry different information, the union may beat either.

HOW
LLM tags are PREFIXED ("llm:dark") so reader-applied `dark` and model-inferred
`dark` stay separate dimensions. Merging them under one name would average two
different kinds of evidence into one number and lose exactly the distinction
being tested.

FORM tags are left OUT. They are a shelf label for the gate, not a taste
signal, and `novel` sits on two thirds of the corpus -- inside the tag matrix
it would add a near-constant to every book, the same reason `fiction` was
pruned from the Hardcover build.

USAGE
    python notebooks\\pipeline\\build_union_tags.py
    python tests\\eval_goodbooks.py --gb-dir data\\raw\\books\\goodbooks-10k ^
        --compare-tags data\\processed\\book_tags_v3_union.parquet
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tag_taxonomy_v3 import FORM_TAGS  # noqa: E402

PROCESSED = Path("data/processed")
HC = PROCESSED / "book_tags_v2_hc3.parquet"
LLM = PROCESSED / "book_tags_v3_llm.parquet"
OUT = PROCESSED / "book_tags_v3_union.parquet"


def main():
    for p in (HC, LLM):
        if not p.exists():
            raise SystemExit(f"{p} not found")
    hc = pd.read_parquet(HC)
    llm = pd.read_parquet(LLM)
    hc["book_id"] = hc["book_id"].astype(str)
    llm["book_id"] = llm["book_id"].astype(str)

    llm = llm[~llm["tag"].isin(FORM_TAGS)].copy()
    llm["tag"] = "llm:" + llm["tag"]

    out = pd.concat([hc[["book_id", "tag", "score"]],
                     llm[["book_id", "tag", "score"]]], ignore_index=True)
    out = out.drop_duplicates(subset=["book_id", "tag"])

    print(f"hardcover  {len(hc):>9,} rows  {hc.book_id.nunique():>7,} books  "
          f"{hc.tag.nunique():>4} tags")
    print(f"llm        {len(llm):>9,} rows  {llm.book_id.nunique():>7,} books  "
          f"{llm.tag.nunique():>4} tags  (form tags excluded)")
    print(f"union      {len(out):>9,} rows  {out.book_id.nunique():>7,} books  "
          f"{out.tag.nunique():>4} tags")
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
