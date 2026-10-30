"""
check_content_warnings.py -- what content-warning data we already have, before
building "may contain: ..." lines on the cards. Read-only, no API calls.

    python notebooks\\pipeline\\check_content_warnings.py

Checks, for films, TV and books:
  1. which columns in the catalogue files could hold warnings (keywords, tags)
  2. Enemy (2013) as a known case: does "spider" appear anywhere for it?
  3. how many titles carry any keyword / warning at all (coverage)
  4. for ~25 common warnings: how many titles match, in which source
  5. raw files on disk that might hold more (TMDB keyword dumps, Hardcover raw)
Paste the whole output back.
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

PROC = Path("data/processed")
RAW = Path("data/raw")
FILES = {
    "films": ["movies_v2_meta.parquet", "movie_tags_v2.parquet"],
    "tv": ["tv_v2_meta.parquet", "tv_tags.parquet"],
    "books": ["books_v2_hc3.parquet", "book_tags_v2_hc3.parquet"],
}
# warning -> words that signal it in a keyword or tag (lowercase, substring)
WARNINGS = {
    "spiders": ["spider", "arachn", "tarantula"],
    "snakes": ["snake", "serpent"],
    "gore": ["gore", "gory", "splatter", "dismember", "body horror", "decapitat"],
    "animal death": ["animal death", "animal cruelty", "dog dies", "death of a dog", "pet death",
                     "animal abuse"],
    "sexual assault": ["rape", "sexual assault", "sexual abuse", "sexual violence"],
    "child abuse": ["child abuse", "abuse of children", "child molest"],
    "suicide": ["suicide"],
    "self-harm": ["self-harm", "self harm", "self-injury", "cutting"],
    "eating disorder": ["eating disorder", "anorexia", "bulimia"],
    "drug use": ["drug use", "drug abuse", "heroin", "overdose", "drug addiction", "cocaine"],
    "alcoholism": ["alcoholism", "alcoholic"],
    "torture": ["torture"],
    "domestic abuse": ["domestic abuse", "domestic violence", "spousal abuse"],
    "cannibalism": ["cannibal"],
    "incest": ["incest"],
    "pregnancy loss": ["miscarriage", "stillbirth", "pregnancy loss"],
    "death of a parent": ["death of a parent", "death of father", "death of mother", "parent death"],
    "terminal illness": ["cancer", "terminal illness"],
    "racism": ["racism", "racist"],
    "homophobia": ["homophobia", "homophobic"],
    "war violence": ["war crime", "genocide", "holocaust"],
    "needles": ["needle", "injection"],
    "clowns": ["clown"],
    "claustrophobia": ["claustrophob", "buried alive", "trapped underground"],
    "jump scares": ["jump scare"],
}


def text_of(v):
    """Any cell -> lowercase text (lists, dicts, JSON strings)."""
    if v is None:
        return ""
    if isinstance(v, float) and v != v:
        return ""
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False).lower()
    return str(v).lower()


def candidate_cols(df):
    return [c for c in df.columns
            if re.search(r"keyword|kw|tag|warning|subject|theme|cached", c, re.I)]


def show_file(name, kind):
    p = PROC / name
    if not p.exists():
        print(f"  {name}: NOT FOUND")
        return None
    df = pd.read_parquet(p)
    print(f"  {name}: {len(df):,} rows")
    print(f"    columns: {', '.join(map(str, df.columns))[:600]}")
    return df


def tag_table_terms(df, label):
    """Tag tables (one row per item x tag): which warning words are tags at all."""
    tcol = next((c for c in df.columns if c.lower() == "tag"), None)
    if tcol is None:
        return
    idcol = next((c for c in df.columns if c.lower().endswith("_id") or c.lower() == "id"), None)
    vocab = df[tcol].astype(str).str.lower()
    print(f"    {vocab.nunique():,} distinct tags; warning words found among them:")
    for w, words in WARNINGS.items():
        hit = vocab[vocab.apply(lambda t: any(x in t for x in words))]
        if len(hit):
            n = df.loc[hit.index, idcol].nunique() if idcol else len(hit)
            print(f"      {w:<20}{n:>7,} titles   tags: {', '.join(sorted(hit.unique())[:5])}")


def corpus_terms(df, cols, label):
    if not cols:
        print("    no keyword/tag-like columns in this file")
        return
    blob = df[cols].apply(lambda r: " ".join(text_of(v) for v in r), axis=1)
    has = (blob.str.len() > 4).sum()
    print(f"    keyword-like columns: {cols}")
    print(f"    titles with anything in them: {has:,} of {len(df):,} ({has / max(len(df), 1):.0%})")
    print("    warning words (titles matching):")
    for w, words in WARNINGS.items():
        n = blob.apply(lambda t: any(x in t for x in words)).sum()
        if n:
            print(f"      {w:<20}{n:>7,}")
    return blob


def find_title(df, titles, year=None):
    tcol = next((c for c in ("title", "name") if c in df.columns), None)
    if tcol is None:
        return df.iloc[0:0]
    m = df[df[tcol].astype(str).str.lower().isin([t.lower() for t in titles])]
    ycol = next((c for c in ("release_year", "start_year", "publication_year") if c in df.columns), None)
    if year and ycol and len(m) > 1:
        m = m[m[ycol] == year] if (m[ycol] == year).any() else m
    return m


def hardcover_warnings():
    """Content Warning tags in the Hardcover raw harvest, for books in the corpus."""
    raw = RAW / "books" / "hardcover_v2" / "raw.jsonl"
    corpus = PROC / "books_v2_hc3.parquet"
    if not raw.exists():
        print(f"  {raw}: NOT FOUND")
        return
    ids = set()
    if corpus.exists():
        ids = set(pd.read_parquet(corpus, columns=["book_id"])["book_id"].astype(str))
    cats, cw_books, cw_tags, seen = Counter(), 0, Counter(), 0
    with raw.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ids and str(b.get("id")) not in ids:
                continue
            seen += 1
            ct = b.get("cached_tags") or {}
            if isinstance(ct, str):
                try:
                    ct = json.loads(ct)
                except json.JSONDecodeError:
                    ct = {}
            if not isinstance(ct, dict):
                continue
            cats.update(ct.keys())
            cw = ct.get("Content Warning") or []
            names = [(x.get("tag") if isinstance(x, dict) else str(x)) for x in cw]
            names = [n for n in names if n]
            if names:
                cw_books += 1
                cw_tags.update(n.lower() for n in names)
    print(f"  raw.jsonl: {seen:,} corpus books read")
    print(f"    cached_tags categories: {dict(cats.most_common())}")
    print(f"    books with any Content Warning tag: {cw_books:,} ({cw_books / max(seen, 1):.0%})")
    print(f"    {len(cw_tags):,} distinct warning tags; most common:")
    for t, n in cw_tags.most_common(40):
        print(f"      {n:>6,}  {t}")
    print("    our warning list, matched against these tags:")
    for w, words in WARNINGS.items():
        n = sum(c for t, c in cw_tags.items() if any(x in t for x in words))
        if n:
            print(f"      {w:<20}{n:>7,} tag uses")


def raw_candidates():
    if not RAW.exists():
        return
    hits = [p for p in RAW.rglob("*")
            if p.is_file() and re.search(r"keyword|kw|warning|tmdb_details|details", p.name, re.I)]
    print(f"  files under data\\raw with keyword/warning/details in the name: {len(hits)}")
    for p in sorted(hits)[:25]:
        print(f"    {p}  ({p.stat().st_size / 1e6:.1f} MB)")


def main():
    for kind in ("films", "tv"):
        print(f"\n=== {kind.upper()} " + "=" * 60)
        meta = show_file(FILES[kind][0], kind)
        tags = show_file(FILES[kind][1], kind)
        if meta is not None:
            cols = candidate_cols(meta)
            corpus_terms(meta, cols, kind)
            probe = (["Enemy"], 2013) if kind == "films" else (["Hannibal"], 2013)
            m = find_title(meta, *probe)
            print(f"    known case {probe[0][0]} ({probe[1]}): {len(m)} row(s)")
            for _, r in m.head(2).iterrows():
                for c in cols:
                    t = text_of(r[c])
                    print(f"      {c}: {t[:300]}")
                    if kind == "films":
                        print(f"        -> 'spider' present: {'spider' in t}")
        if tags is not None:
            tag_table_terms(tags, kind)
    print("\n=== BOOKS " + "=" * 60)
    meta = show_file(FILES["books"][0], "books")
    tags = show_file(FILES["books"][1], "books")
    if meta is not None:
        corpus_terms(meta, candidate_cols(meta), "books")
    if tags is not None:
        tag_table_terms(tags, "books")
    hardcover_warnings()
    print("\n=== RAW FILES " + "=" * 56)
    raw_candidates()


if __name__ == "__main__":
    main()
