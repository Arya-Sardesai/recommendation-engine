"""
build_book_forms.py -- one FORM label per book, for the shelf gate.

Pulls the single form tag (novel / memoir / reportage / ...) out of the LLM
tagging run into a small (book_id, form) file. The register tags from that run
are NOT used: on goodbooks they lost to the Hardcover reader tags (t=-2.1) and
tied them when combined, so B_TAGS stays Hardcover. The form label is the part
that is better than anything else in the corpus -- Hardcover's genres call The
Great Gatsby and Brave New World non-fiction.

USAGE
    python notebooks\\pipeline\\build_book_forms.py
    -> data\\processed\\book_forms_v3.parquet   (upload, then it is live: app.py
       already names it as B_FORMS and treats a missing file as gate-off)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tag_taxonomy_v3 import FORM_TAGS, NONFICTION_FORMS  # noqa: E402

PROCESSED = Path("data/processed")
LLM = PROCESSED / "book_tags_v3_llm.parquet"
OUT = PROCESSED / "book_forms_v3.parquet"


def main():
    t = pd.read_parquet(LLM)
    t["book_id"] = t["book_id"].astype(str)
    f = (t[t["tag"].isin(FORM_TAGS)]
         .sort_values("score", ascending=False)
         .drop_duplicates("book_id")[["book_id", "tag"]]
         .rename(columns={"tag": "form"}))
    nf = f["form"].isin(NONFICTION_FORMS).sum()
    print(f"{len(f):,} books with a form label | non-fiction {nf:,} ({nf/len(f):.1%})")
    print(f["form"].value_counts().head(20).to_string())
    f.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
