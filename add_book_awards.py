"""
add_book_awards.py -- trophy badges and an award bonus for BOOKS in hf-space/app.py.

Films and TV already have both (MOVIE_AWARD_WIN / TV_AWARD_WIN). This adds the
same for books, reading data/processed/book_awards.parquet from
build_book_awards.py:

  1. BOOK_AWARD_WIN / BOOK_AWARD_NOM constants, both 0.0 -- the bonus is OFF
     until tests\\eval_book_awards.py picks the values
  2. B_AWARDS = "book_awards.parquet"   (missing file = no badges, no bonus)
  3. book_award_summaries() + load_book_awards(): one tooltip per book like
     "Won: Booker Prize · Shortlisted: Women's Prize for Fiction"
  4. recommend_books gets awards= / award_win= / award_nom=, and adds the
     bonus inside emb_score, next to the quality prior, like the film ranker
  5. the Space loads the awards and passes them to recommend_books
  6. the trophy shows after the title on every book card

Every edit must match exactly once or nothing is written, so a copy of app.py
changed by another session fails loudly instead of being patched in the wrong
place. Only recommend_books is touched; the film and TV rankers are checked
afterwards to be unchanged.

USAGE (repo root):
    python add_book_awards.py            # dry run
    python add_book_awards.py --write    # edits ..\\hf-space\\app.py, keeps app.py.bak2
"""
import shutil
import sys
from pathlib import Path

SPACE = Path(__file__).resolve().parent.parent / "hf-space"
APP = Path(sys.argv[sys.argv.index("--app") + 1]) if "--app" in sys.argv else SPACE / "app.py"

EDITS = [
    ("constants",
     "TV_AWARD_NOM = 0.02\n",
     "TV_AWARD_NOM = 0.02\n"
     "# Books: same mechanism, OFF until tests/eval_book_awards.py picks values.\n"
     "# The books ranker stretches similarity x3 (sim_scale), so its spread inside\n"
     "# an anchor pool is ~3x the films' and the same 0.04 moves books less.\n"
     "BOOK_AWARD_WIN = 0.0\n"
     "BOOK_AWARD_NOM = 0.0\n"),

    ("B_AWARDS",
     'B_FORMS  = "book_forms_v3.parquet"\n',
     'B_FORMS  = "book_forms_v3.parquet"\n'
     "# Prize winners and shortlists from Wikidata + Wikipedia, matched on title +\n"
     "# author (notebooks/pipeline/build_book_awards.py). Missing file = no badges.\n"
     'B_AWARDS = "book_awards.parquet"\n'),

    ("loader",
     '''def load_tv_awards():
    return _load_awards(T_AWARDS, "TV")
''',
     '''def load_tv_awards():
    return _load_awards(T_AWARDS, "TV")


def book_award_summaries(adf):
    """book_id (int) -> {"major", "wins", "noms", "tip"}, the same shape as
    award_summaries so award_bonus and award_badge work unchanged. A book
    prize is given once, so the tip names prizes instead of counting them:
        Won: Booker Prize · Shortlisted: Women's Prize for Fiction"""
    out = {}
    if adf is None or not len(adf):
        return out
    adf = adf.copy()
    adf["_id"] = pd.to_numeric(adf["book_id"], errors="coerce")
    adf = adf.dropna(subset=["_id"])
    for bid, g in adf.groupby("_id"):
        maj = g[g["tier"] == "major"]
        won = g.loc[g["result"] == "won", "award"].tolist()
        nom = g.loc[g["result"] == "nominated", "award"].tolist()
        lau = (g["result"] == "laureate").any()
        if (maj["result"] == "won").any():
            major = "won"
        elif len(maj):
            major = "nominated"      # shortlisted, or a Nobel laureate's book
        else:
            major = None
        parts = []
        if won:
            parts.append("Won: " + ", ".join(won[:3]) + (f" +{len(won) - 3}" if len(won) > 3 else ""))
        if nom:
            parts.append("Shortlisted: " + ", ".join(nom[:3]) + (f" +{len(nom) - 3}" if len(nom) > 3 else ""))
        if lau:
            parts.append("Author: Nobel laureate")
        out[int(bid)] = {"major": major, "wins": len(won), "noms": len(won) + len(nom),
                         "tip": " · ".join(parts)}
    return out


@st.cache_resource
def load_book_awards():
    try:
        return book_award_summaries(pd.read_parquet(_dl(B_AWARDS)))
    except Exception as e:
        print(f"Book awards unavailable (no badges, no bonus): {e}")
        return {}
'''),

    ("recommend_books signature",
     "tag_idf=0.0, genre_weight=0.0, genre_k=30):",
     "tag_idf=0.0, genre_weight=0.0, genre_k=30,\n"
     "                    awards=None, award_win=BOOK_AWARD_WIN, award_nom=BOOK_AWARD_NOM):"),

    ("recommend_books bonus",
     "emb_score = (_gb + _pmu + sim_scale * (float(sim) - _pmu)",
     "# award bonus: best MAJOR award only (see BOOK_AWARD_WIN)\n"
     "            _aw = award_bonus(awards, book[\"book_id\"], award_win, award_nom)\n"
     "            emb_score = (_gb + _aw + _pmu + sim_scale * (float(sim) - _pmu)"),

    ("load at startup",
     "    b_forms = load_books_forms()\n",
     "    b_forms = load_books_forms()\n"
     "    b_awards = load_book_awards()\n"),

    ("pass to ranker",
     "tags_df=b_tags, n=20, skip_ids=st.session_state.skipped,\n"
     "                                             forms=b_forms)",
     "tags_df=b_tags, n=20, skip_ids=st.session_state.skipped,\n"
     "                                             forms=b_forms, awards=b_awards)"),

    ("badge on the card",
     "f\"<span class='item-title'>{rec['title']}</span>{yr}<br>\"\n"
     "                            f\"<span class='item-sub'>{rec['author']}</span>\"",
     "f\"<span class='item-title'>{rec['title']}</span>{yr}\"\n"
     "                            f\"{award_badge(b_awards, rec['book_id'])}<br>\"\n"
     "                            f\"<span class='item-sub'>{rec['author']}</span>\""),
]


def _fn_src(src, name):
    i = src.index(f"\ndef {name}(")
    j = src.find("\ndef ", i + 5)
    return src[i:j]


def main():
    write = "--write" in sys.argv
    src = APP.read_text(encoding="utf-8")
    if "BOOK_AWARD_WIN" in src:
        sys.exit("app.py already has BOOK_AWARD_WIN -- nothing to do")

    bad = []
    for name, old, _ in EDITS:
        n = src.count(old)
        print(f"  {'ok ' if n == 1 else 'BAD'}  {name:<28} matches {n}")
        if n != 1:
            bad.append(name)
    if bad:
        sys.exit(f"\nnot written: {', '.join(bad)} did not match exactly once. "
                 "Send me the current app.py and I'll redo the anchors.")

    new = src
    for _, old, rep in EDITS:
        new = new.replace(old, rep, 1)

    # leak check: the film and TV rankers must come out byte-identical
    for fn in ("recommend_movies", "recommend_tv"):
        if fn in src and _fn_src(src, fn) != _fn_src(new, fn):
            sys.exit(f"not written: the edit leaked into {fn}")
    compile(new, str(APP), "exec")

    if not write:
        print("\ndry run OK. Re-run with --write to apply.")
        return
    shutil.copy2(APP, APP.with_name("app.py.bak2"))
    APP.write_text(new, encoding="utf-8")
    print(f"\nwrote {APP} (backup app.py.bak2). Bonus is OFF (0.0) until the sweep.")


if __name__ == "__main__":
    main()
