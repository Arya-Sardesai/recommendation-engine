"""
build_award_wikipedia.py -- winners and shortlists for the prizes Wikidata
covers thinly, read from their Wikipedia pages.

WHY
Wikidata records winners for most prizes but few shortlists, and some prizes
barely at all (Oct 12 fetch: JCB 0, DSC 0, Agatha 2, Orwell 2, Walter Scott 4,
Women's Prize 25). Their Wikipedia pages carry year-by-year tables with the
full shortlists. This reads those tables and writes one CSV that
build_award_list.py takes with --extra.

HOW IT READS A PAGE
Every table on the page whose header has an author column and a title column
(Author / Writer, and Title / Book / Work / Novel) is read; year is taken when
there is a year column. Footnote marks ([1], *, dagger) are stripped. Pages are
laid out differently, so the report prints, per page, how many tables matched,
how many rows came out, and three sample rows: READ IT. A page with 0 rows has
a layout this does not understand -- leave it out, or add its books by hand.

NEEDS: pip install lxml   (pandas uses it to read HTML tables)

    python notebooks\\pipeline\\build_award_wikipedia.py
    python notebooks\\pipeline\\build_award_wikipedia.py "JCB Prize" "Costa Book Awards"   (rerun some pages)
    python notebooks\\pipeline\\build_award_list.py --stage match --extra data\\raw\\books\\awards\\wikipedia_shortlists.csv

Add or remove pages in PAGES below. The title is the Wikipedia page name;
redirects are followed, and the page actually read is printed.
"""
import csv
import re
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/raw/books/awards/wikipedia_shortlists.csv")
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "rec-engine-awards/1.0 "
                         "(https://github.com/Arya-Sardesai/recommendation-engine)"}

# (Wikipedia page, group)
PAGES = [
    ("Edgar Award for Best Novel", "crime"),
    ("Gold Dagger", "crime"),
    ("Anthony Award for Best Novel", "crime"),
    ("Agatha Award", "crime"),
    ("Macavity Award", "crime"),
    ("Women's Prize for Fiction", "literary"),
    ("Costa Book Awards", "literary"),
    ("Rathbones Folio Prize", "literary"),
    ("Kirkus Prize", "literary"),
    ("National Book Critics Circle Award", "literary"),
    ("Walter Scott Prize", "historical"),
    ("JCB Prize", "south_asia"),
    ("DSC Prize for South Asian Literature", "south_asia"),
    ("Crossword Book Award", "south_asia"),
    ("Warwick Prize for Women in Translation", "translated"),
    ("International Prize for Arabic Fiction", "translated"),
    ("Shirley Jackson Award", "horror"),
    ("Baillie Gifford Prize", "nonfiction"),
    ("Orwell Prize", "nonfiction"),
    ("Royal Society Prizes for Science Books", "nonfiction"),
    ("Wolfson History Prize", "nonfiction"),
    ("Financial Times Business Book of the Year Award", "nonfiction"),
    ("Pulitzer Prize for General Nonfiction", "nonfiction"),
]

# Oct 12 run: 8 pages read 0 rows. Likely headers like "Author(s)" or "Book
# title", so the matcher strips brackets and accepts these variants.
AUTHOR_COLS = ("author", "writer", "authors", "author s", "writer s",
               "author name", "nominee", "nominees")
TITLE_COLS = ("title", "book", "work", "novel", "works", "book title",
              "title of work", "novel title", "books")


def fetch_html(page):
    r = requests.get(API, params={"action": "parse", "page": page, "prop": "text",
                                  "format": "json", "redirects": 1},
                     headers=HEADERS, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        return None, data["error"].get("info", "error")
    return data["parse"]["text"]["*"], data["parse"].get("title", page)


def clean(v):
    s = "" if pd.isna(v) else str(v)
    s = re.sub(r"\[[^\]]*\]", "", s)           # footnotes [1], [a]
    s = re.sub(r"[*†‡§]", "", s)  # * dagger double-dagger section
    s = re.sub(r"\((winner|shortlisted|joint winner)\)", "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip(" \"'“”")


def flat(col):
    if isinstance(col, tuple):
        col = " ".join(str(c) for c in col if not str(c).startswith("Unnamed"))
    c = re.sub(r"\[[^\]]*\]", "", str(col).lower())       # footnotes in headers
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ", c)).strip()   # "author(s)" -> "author s"


def pick(cols, names):
    for i, c in enumerate(cols):
        if any(c == n or c.startswith(n + " ") or c.endswith(" " + n) for n in names):
            return i
    return None


def rows_from_page(html):
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return [], 0
    out, used = [], 0
    for t in tables:
        cols = [flat(c) for c in t.columns]
        ai, ti = pick(cols, AUTHOR_COLS), pick(cols, TITLE_COLS)
        if ai is None or ti is None or ai == ti:
            continue
        yi = pick(cols, ("year",))
        used += 1
        last_year = ""
        for _, r in t.iterrows():
            vals = list(r.values)
            author, title = clean(vals[ai]), clean(vals[ti])
            year = clean(vals[yi])[:4] if yi is not None else ""
            if re.fullmatch(r"\d{4}", year):
                last_year = year
            else:
                year = last_year          # merged year cells come back empty
            if not author or not title or len(title) > 150 or len(author) > 80:
                continue
            if title.lower() in ("title", "book", "work") or author.lower() == "author":
                continue
            out.append({"title": title, "author": author, "year": year})
    return out, used


def main():
    only = {a.lower() for a in sys.argv[1:]}
    pages = [p for p in PAGES if not only or p[0].lower() in only]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total, rows_all = 0, []
    print(f"{'page':<50}{'tables':>7}{'rows':>7}  sample")
    for page, group in pages:
        try:
            html, real = fetch_html(page)
        except requests.RequestException as e:
            print(f"{page[:49]:<50}  FAILED {type(e).__name__}")
            continue
        if html is None:
            print(f"{page[:49]:<50}  NOT FOUND ({real})")
            continue
        rows, used = rows_from_page(html)
        seen, uniq = set(), []
        for r in rows:
            k = (r["title"].lower(), r["author"].lower())
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        for r in uniq:
            rows_all.append({**r, "source": real, "group": group})
        sample = "; ".join(f"{r['title'][:22]} / {r['author'][:16]}" for r in uniq[:3])
        note = f" (read as '{real}')" if real != page else ""
        print(f"{page[:49]:<50}{used:>7}{len(uniq):>7}  {sample}{note}")
        total += len(uniq)
        time.sleep(1.0)
    if only and OUT.exists():
        # rerunning some pages: keep every other page's rows already in the CSV
        done = {p[0].lower() for p in pages}
        with OUT.open(encoding="utf-8", newline="") as fh:
            old = [r for r in csv.DictReader(fh)]
        reread = {r["source"] for r in rows_all}
        rows_all = [r for r in old if r["source"] not in reread
                    and r["source"].lower() not in done] + rows_all
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["title", "author", "year", "source", "group"])
        w.writeheader()
        w.writerows(rows_all)
    print(f"\nwrote {OUT} ({total:,} rows this run, {len(rows_all):,} in the file)")
    print("Check the samples: a page whose sample shows years, prize names or")
    print("other junk in the title or author slot was misread. Remove it from")
    print("PAGES (or delete its rows from the CSV) before running --stage match.")


if __name__ == "__main__":
    sys.exit(main())
