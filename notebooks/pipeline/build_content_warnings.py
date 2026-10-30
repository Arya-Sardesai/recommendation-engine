"""
build_content_warnings.py -- "may contain: ..." labels for book, film and TV
cards, from data we already have.

    python notebooks\\pipeline\\build_content_warnings.py --stage review
    python notebooks\\pipeline\\build_content_warnings.py --stage build

SOURCES (check_content_warnings.py, Oct 22)
  films  movies_v2_meta.parquet  keywords_tmdb   73% of films have keywords
  tv     tv_v2_meta.parquet      keywords_tmdb   78% of shows, fewer warning-type ones
  books  hardcover_v2/raw.jsonl  cached_tags["Content Warning"]  only 8% of books

MATCHING
Whole words, never substrings: the first check counted "grape" as rape,
"Spider-Man" as spiders and "Al Gore" as gore. Each warning has word patterns
plus an EXCLUDE list for keywords that contain the word but are not the thing.
--stage review prints, per warning, every distinct keyword/tag that matched,
with counts and example titles: read it, add wrong ones to EXCLUDE, rerun.

A BLANK IS NOT "SAFE"
Most titles carry no warning data at all (92% of books). The card says "may
contain", and shows nothing rather than "none" when there is no data.

OUTPUT  data/processed/content_warnings.parquet  kind, id, warnings (list)
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROC = Path("data/processed")
HC_RAW = Path("data/raw/books/hardcover_v2/raw.jsonl")
# Hardcover records of books added later (merge_new_books.py) live here.
HC_EXTRA = [Path("data/raw/books/hardcover_v2/merge_v4_raw.jsonl")]
BOOK_CORPORA = ["books_v2_hc4.parquet", "books_v2_hc3.parquet"]   # newest first
OUT = PROC / "content_warnings.parquet"

# label -> whole-word regexes. Order = display order on the card (most
# commonly avoided first), so a card that shows 3 shows the 3 that matter most.
WARNINGS = {
    "sexual assault": [r"rape", r"raped", r"sexual assault", r"sexual abuse", r"sexual violence",
                       r"molest\w*"],
    "child abuse": [r"child abuse", r"abuse of children", r"child molest\w*", r"child neglect"],
    "suicide": [r"suicide", r"suicidal \w+", r"suicide attempt"],
    "self-harm": [r"self[- ]harm", r"self[- ]injury", r"self[- ]mutilation"],
    "animal death": [r"animal death", r"animal cruelty", r"animal abuse", r"dog dies",
                     r"death of a dog", r"death of a pet", r"pet death", r"dead dog", r"animal killing"],
    "child death": [r"child death", r"death of a child", r"death of child", r"dead child"],
    "gore": [r"gore", r"gory", r"splatter", r"body horror", r"dismemberment", r"decapitation",
             r"graphic violence", r"blood splatter"],
    "torture": [r"torture", r"tortured"],
    "domestic abuse": [r"domestic abuse", r"domestic violence", r"spousal abuse", r"abusive relationship",
                       r"abusive husband", r"intimate partner violence"],
    "eating disorder": [r"eating disorder", r"anorexia", r"anorexia nervosa", r"bulimia"],
    "drug use": [r"drug use", r"drug abuse", r"drug addiction", r"heroin", r"overdose", r"cocaine",
                 r"meth", r"drug addict"],
    "alcoholism": [r"alcoholism", r"alcoholic", r"alcohol abuse"],
    "pregnancy loss": [r"miscarriage", r"stillbirth", r"pregnancy loss"],
    "incest": [r"incest"],
    "cannibalism": [r"cannibal", r"cannibalism", r"cannibals"],
    "racism": [r"racism", r"racist", r"racial slur\w*"],
    "homophobia": [r"homophobia", r"homophobic", r"transphobia", r"transphobic"],
    "spiders": [r"spider", r"spiders", r"giant spider", r"tarantula", r"arachnid\w*", r"arachnophobia"],
    # not "cobra" alone: in TMDB it is mostly G.I. Joe's villains and Cobra the Animation
    "snakes": [r"snake", r"snakes", r"giant snake", r"python", r"king cobra", r"anaconda",
               r"rattlesnake", r"rattle snakes"],
    "needles": [r"needle", r"needles", r"injection", r"syringe"],
    "clowns": [r"clown", r"clowns", r"killer clown", r"evil clown"],
}
# keywords/tags that contain a pattern's word but are not the warning.
# Grow this from --stage review.
EXCLUDE = {
    # from the Oct 22 review
    "spiders": {"spider-man", "spiderman", "spider-verse", "spider-woman", "spider-gwen",
                "spider sense", "spider-man: miles morales", "spider-ham", "peter parker"},
    "gore": {"al gore", "gore vidal"},
    "snakes": {"python (programming language)", "monty python", "snake eyes", "snake plissken",
               "solid snake", "snake oil", "colt python"},
    "needles": {"space needle", "needlework", "needle in a haystack", "knitting needle",
                "sharp as needle"},
    "pregnancy loss": {"miscarriage of justice"},
    "suicide": {"suicide mission", "suicide squad"},
    "torture": {"tortured genius", "tortured soul", "tortured artist"},
    "clowns": {"class clown", "clown shoes", "rodeo clown"},
    "alcoholism": {"alcoholic drink", "alcoholic drinks", "alcoholic beverage"},
}
# Whole titles a warning must not attach to. Every Spider-Man film carries
# "spider" / "spider bite" in TMDB; a spider warning on all of them is noise.
TITLE_EXCLUDE = {
    "spiders": re.compile(r"spider[- ]?(man|verse|woman|gwen|noir|punk)", re.I),
}
# Hardcover warnings are reader-applied; one reader's tag gave Hitchhiker's
# Guide "self-harm" and Atomic Habits "eating disorder". Require this many.
BOOK_MIN_READERS = 1     # 2 cut book warnings 4,087 -> 1,079 without removing the odd ones (Oct 22)
_RX = {lab: re.compile(r"^(?:.*\b)?(?:" + "|".join(pats) + r")(?:\b.*)?$", re.I)
       for lab, pats in WARNINGS.items()}


@lru_cache(maxsize=None)
def _term_labels(tl):
    """One keyword -> the labels it means. Cached: 150K titles share far fewer
    distinct keywords."""
    return tuple(lab for lab, rx in _RX.items()
                 if tl not in EXCLUDE.get(lab, ()) and rx.match(tl))


def labels_for(terms):
    """terms (keywords or tags) -> {label: [matching terms]}"""
    out = defaultdict(list)
    for t in terms:
        tl = str(t).strip().lower()
        if tl:
            for lab in _term_labels(tl):
                out[lab].append(tl)
    return out


def as_list(v):
    if v is None:
        return []
    if hasattr(v, "tolist") and not isinstance(v, str):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if x is not None]
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return []
    if s.startswith("["):
        try:
            return [str(x) for x in json.loads(s)]
        except json.JSONDecodeError:
            pass
    return [x.strip() for x in s.split(",") if x.strip()]


def screen_items(kind):
    """(id, title, [terms]) for films or TV from their catalogue files."""
    # newest catalogue first: after the November expansion the v3 files hold
    # the new titles, and their ids must be in the warnings file
    if kind == "film":
        names, idc, tc, kcols = (["movies_v3_meta.parquet", "movies_v2_meta.parquet"],
                                 "id", "title", ["keywords_tmdb", "keywords"])
    else:
        names, idc, tc, kcols = (["tv_v3_meta.parquet", "tv_v2_meta.parquet"],
                                 "tmdb_id", "name", ["keywords_tmdb"])
    f = next(n for n in names if (PROC / n).exists())
    print(f"  {kind}: {f}")
    df = pd.read_parquet(PROC / f)
    kcols = [c for c in kcols if c in df.columns]
    cols = [df[c].tolist() for c in kcols]
    for i, (iid, title) in enumerate(zip(df[idc].tolist(), df[tc].tolist())):
        terms = []
        for col in cols:
            terms += as_list(col[i])
        yield str(iid), str(title), list(dict.fromkeys(t.strip() for t in terms))


def _raw_lines():
    for p in [HC_RAW] + HC_EXTRA:
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield line


def book_items():
    corpus = next(PROC / f for f in BOOK_CORPORA if (PROC / f).exists())
    ids = set(pd.read_parquet(corpus, columns=["book_id"])["book_id"].astype(str))
    done = set()
    for line in _raw_lines():
            if not line.strip():
                continue
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                continue
            bid = str(b.get("id"))
            if bid not in ids or bid in done:
                continue          # a merged book is in raw.jsonl AND merge_v4_raw.jsonl
            done.add(bid)
            ct = b.get("cached_tags") or {}
            if isinstance(ct, str):
                try:
                    ct = json.loads(ct)
                except json.JSONDecodeError:
                    ct = {}
            cw = ct.get("Content Warning") if isinstance(ct, dict) else None
            terms = []
            for x in cw or []:
                if isinstance(x, dict):
                    n = x.get("count") or x.get("tagged_count") or 1
                    if int(n) >= BOOK_MIN_READERS and x.get("tag"):
                        terms.append(x["tag"])
                elif x and BOOK_MIN_READERS <= 1:
                    terms.append(str(x))
            yield bid, str(b.get("title") or ""), terms


def collect():
    rows = []
    for kind, gen in (("film", screen_items("film")), ("tv", screen_items("tv")), ("book", book_items())):
        for iid, title, terms in gen:
            labs = labels_for(terms)
            for lab, rx in TITLE_EXCLUDE.items():
                if lab in labs and rx.search(title):
                    del labs[lab]
            rows.append((kind, iid, title, labs))
    return rows


def stage_review(rows, examples):
    by = defaultdict(lambda: defaultdict(Counter))    # kind -> label -> matched term counts
    ex = defaultdict(lambda: defaultdict(list))
    for kind, iid, title, labs in rows:
        for lab, terms in labs.items():
            by[kind][lab].update(set(terms))
            if len(ex[kind][lab]) < examples:
                ex[kind][lab].append(title)
    for kind in ("film", "tv", "book"):
        n = sum(1 for r in rows if r[0] == kind)
        n_any = sum(1 for r in rows if r[0] == kind and r[3])
        print(f"\n{'=' * 90}\n{kind.upper()}: {n_any:,} of {n:,} titles get at least one warning\n{'=' * 90}")
        for lab in WARNINGS:
            c = by[kind].get(lab)
            if not c:
                continue
            titles = sum(1 for r in rows if r[0] == kind and lab in r[3])
            print(f"\n  {lab}  ({titles:,} titles)")
            print("    matched: " + ", ".join(f"{t} {k}" for t, k in c.most_common(25)))
            print("    e.g.     " + " | ".join(ex[kind][lab][:examples]))
    print("\nRead every 'matched' line. Anything that is not the warning (a superhero,")
    print("a politician, a programming language) goes into EXCLUDE for that label.")
    print("Then run --stage review again, and --stage build when it reads clean.")


def stage_build(rows):
    out = [{"kind": k, "id": i, "warnings": [lab for lab in WARNINGS if lab in labs]}
           for k, i, _, labs in rows if labs]
    df = pd.DataFrame(out)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(df):,} titles with at least one warning")
    for k in ("film", "tv", "book"):
        sub = df[df["kind"] == k]
        cnt = Counter(w for ws in sub["warnings"] for w in ws)
        print(f"  {k:<5}{len(sub):>8,} titles   " + ", ".join(f"{w} {n}" for w, n in cnt.most_common(8)))
    print("\nUpload it with scripts\\upload_data.py once the app reads it (add it to FILES).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["review", "build"], required=True)
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()
    rows = collect()
    if args.stage == "review":
        stage_review(rows, args.examples)
    else:
        stage_build(rows)


if __name__ == "__main__":
    main()
