"""
audit_gbooks.py -- how much of the Google Books haul is actually usable.

    python notebooks\\pipeline\\audit_gbooks.py
    python notebooks\\pipeline\\audit_gbooks.py --samples 40

The trickle's "with an English blurb" count is generous on purpose: an English
result of 100+ characters whose author list contains the target's SURNAME. It
does not check that the result is the same BOOK, or that the text is a blurb.
This script reads what has been fetched so far (gbooks_raw.jsonl, never
re-calling Google) and sorts every looked-up book into one outcome:

  usable          same book (author + title match), English, 150+ chars,
                  reads like a description (review quotes stripped if the
                  blurb underneath is long enough)
  wrong book      author matches but the title doesn't (another book by the
                  same author, a box set, a study guide)
  junk text       right book, but the text is a scan notice, review quotes,
                  a study-guide pitch, or literary commentary
  too short       right book, English, but under 150 characters
  not English     results, none in English
  author mismatch English results, none by this author
  no results      Google returned nothing

Then, for the usable ones: how many the Books tab could ever recommend (its
own floor: 10+ ratings, shrunk average 3.6+), the genre mix, and the
popularity bands. Writes, next to gbooks_raw.jsonl:

  gbooks_audit_samples.txt    examples from every outcome, to read by eye
  gbooks_audit_usable.jsonl   id, title, author, chosen blurb -- a first cut
                              of the November merge; nothing reads it yet

Read-only: touches no corpus file and spends no quota.
"""
import argparse
import difflib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

RAW_DIR = Path("data/raw/books/hardcover_v2")
HC_RAW = RAW_DIR / "raw.jsonl"
TARGETS = RAW_DIR / "gbooks_targets.jsonl"
FETCHED = RAW_DIR / "gbooks_raw.jsonl"
SAMPLES = RAW_DIR / "gbooks_audit_samples.txt"
USABLE = RAW_DIR / "gbooks_audit_usable.jsonl"

MIN_CHARS = 150            # the trickle counts 100+; a 100-char blurb embeds poorly
MIN_RATINGS, MIN_AVG, SHRINK = 10, 3.6, 50     # recommend_books defaults
BANDS = [(25, 50), (50, 100), (100, 500), (500, 2000), (2000, 10 ** 9)]
OUTCOMES = ["usable", "wrong book", "junk text", "too short",
            "not English", "author mismatch", "no results"]

# Text that is not a description of the book.
JUNK = [
    ("scan notice", re.compile(
        r"reproduction of (a|an|the) (book|original|work)|originally published (before|prior)"
        r"|public domain|scanned|facsimile|\bOCR\b|missing or blurred pages|forgotten books"
        r"|excerpt from \w", re.I)),
    ("study guide", re.compile(
        r"study guide|summary (and|&) analysis|sparknotes|cliffsnotes|chapter[- ]by[- ]chapter"
        r"|\bbook summary\b|quiz(zes)? and|discussion questions|for students and teachers", re.I)),
    ("commentary", re.compile(
        r"\bis (a|an|one of the) (most )?(seminal|landmark|major|important|key|greatest|great)"
        r" (works?|texts?|novels?)"
        r"|this (critical|scholarly|annotated|norton) edition|with an introduction by"
        r"|introduction and notes by|critical essays|edited (with|and) (an )?introduction", re.I)),
]
REVIEW_SOURCE = re.compile(
    r"[—–-]\s*(the\s+)?(new york times|washington post|guardian|kirkus|publishers weekly"
    r"|booklist|library journal|times|independent|telegraph|npr|usa today|people)", re.I)
WRONG_PRODUCT = re.compile(
    r"\b(box(ed)? set|boxset|omnibus|books? \d+\s*[-–&]\s*\d+|collection|trilogy set"
    r"|study guide|summary of|workbook|sparknotes|cliffsnotes|coloring book|twinpack)\b", re.I)
VOLUME = re.compile(r"(?:\bvol(?:ume)?\.?|#|\bbook|\bpart|\bno\.)\s*(\d+)", re.I)
EDITION = re.compile(r"\b(edition|revised|expanded|updated|anniversary|deluxe|illustrated|"
                     r"unabridged|annotated|special|classic)\b", re.I)
SUFFIX = {"jr", "sr", "ii", "iii", "iv", "phd", "md"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()


def _part(t):
    t = norm(t)
    return re.sub(r"^(the|a|an) ", "", t).strip()


def title_parts(t):
    """(full, [parts], [alts]) -- parentheticals dropped from full/parts, split
    on ':' or ' - '. 'Mistborn: The Final Empire' -> ('mistborn the final
    empire', ['mistborn', 'final empire'], []). A parenthetical with no number
    is kept as an alternative title: 'War of the Fae: Book 1 (The Changelings)'
    has alt 'changelings'."""
    raw = str(t or "")
    alts = [_part(x) for x in re.findall(r"\(([^)]*)\)", raw) if not re.search(r"\d|#", x)]
    t = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", raw)
    t = VOLUME.sub(" ", t)          # volume numbers are compared separately
    parts = [_part(x) for x in re.split(r":| - | — ", t)]
    return _part(t), [x for x in parts if x], [a for a in alts if a]


def main_title(t):
    full, parts, _ = title_parts(t)
    return parts[0] if parts else full


def surname(author):
    toks = [w for w in re.split(r"[\s.,]+", str(author or "")) if len(w) > 1]
    toks = [w for w in toks if norm(w) not in SUFFIX] or toks
    return norm(toks[-1]) if toks else norm(author)


def _close(a, b, r=0.85):
    return bool(a and b) and difflib.SequenceMatcher(None, a, b).ratio() >= r


def _volume(t):
    # a number in parentheses is Hardcover's series position, '(Mistborn, #1)',
    # not part of the title
    m = VOLUME.search(re.sub(r"\([^)]*\)", " ", str(t or "")))
    return int(m.group(1)) if m else None


def title_match(target, item):
    """Same book?
      - if the target has a volume number, the result must have the same one
        ('Otomen, Vol. 1' is not 'Otomen, Vol. 6'; a bare result title only
        stands in for volume 1). A number only on Google's side is ignored:
        it is usually a series position, '(The Ascendance Series, Book 2)'.
      - full titles close ('Dune' vs 'Dune (Deluxe Edition)'), or
      - every word of a 2+-word target inside the result's title + subtitle, or
      - a parenthetical alternative title matches the other side
        ('The Changelings' vs 'War of the Fae: Book 1 (The Changelings)'), or
      - one side has no colon and matches either part of the other
        ('The Final Empire' vs 'Mistborn: The Final Empire'), or
      - both have a colon and the parts AFTER it match -- so
        'Mistborn: The Well of Ascension' is NOT 'Mistborn: The Final Empire'.
    A box set or a study guide never counts, even if it contains the title."""
    it = " ".join(str(x) for x in (item.get("title"), item.get("subtitle")) if x)
    if WRONG_PRODUCT.search(it) and not WRONG_PRODUCT.search(str(target)):
        return False
    vt, vi = _volume(target), _volume(it)
    if vt is not None and vt != (vi if vi is not None else 1):
        return False
    tf, tp, ta_ = title_parts(target)
    if_, ip, ia_ = title_parts(item.get("title"))
    if not tf or not if_:
        return False
    if _close(tf, if_):
        return True
    ta = set(tf.split())
    if len(ta) >= 2 and ta <= set(norm(it).split()):
        return True
    tb = set(if_.split())
    if len(tb) >= 3 and tb <= ta:
        return True     # Google's title is the start of a longer Hardcover one
    if any(_close(a, b) for a in [tf] + ta_ for b in [if_] + ia_ if (a, b) != (tf, if_)):
        return True
    if len(tp) >= 2 and len(ip) >= 2:
        if _close(tp[0], ip[0]) and EDITION.search(ip[-1]):
            return True          # 'The China Study: Revised and Expanded Edition'
        return _close(tp[-1], ip[-1])
    if len(tp) >= 2:
        return any(_close(x, if_) for x in tp)
    if len(ip) >= 2:
        return any(_close(tf, x) for x in ip)
    return False


def exact_long_title(target, item):
    """Exact title of 3+ words. Used when the author does not match: Hardcover
    often lists the translator or the artist first (Sophocles' plays under Edith
    Hall, Velvet under its letterer), and a long exact title is safe on its own."""
    tf, _, _ = title_parts(target)
    if_, _, _ = title_parts(item.get("title"))
    return len(tf.split()) >= 3 and _close(tf, if_, 0.95) and title_match(target, item)


_OUTLET = (r"(?=[A-Z])(?i:(?:the\s+)?(?:new york times(?: book review)?|washington post|guardian|kirkus(?: reviews)?"
           r"|publishers weekly|booklist|library journal|times|independent|telegraph|npr"
           r"|usa today|people|entertainment weekly|o,? the oprah magazine|school library journal"
           r"|bookpage|buzzfeed|bustle|popsugar|daily mail|observer|sunday times|financial times"
           r"|the atlantic|time|vogue|elle|marie claire|cosmopolitan|glamour|harper's bazaar)"
           r"(?:,?\s*\(?starred review\)?)?)")
# Case-SENSITIVE from here: a name is two capitalised words, so the pattern
# cannot run on into the blurb ("Kirkus Reviews When Maya returns...").
_STARTERS = (r"(?!(?:For|In|When|The|A|An|It|After|As|But|On|At|With|She|He|They|I|This|Set|Now|"
             r"From|Born|Years|One|Two|Three|Meet|Welcome|What|Who|If|Once|There|Her|His|Our|My)\b)")
_CAP = r"[A-Z][\w'’-]*"
_PERSON = (r"[A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-zA-Z'’-]+"
           r"(?:(?i:,\s*(?:the\s+)?(?:#1\s+)?(?:new york times\s+|usa today\s+|international\s+|sunday times\s+)?"
           r"(?:and\s+\w+\s+)?(?:bestselling\s+|award-winning\s+)?author)"
           r"(?:\s+of\s+" + _CAP + r"(?:\s+" + _STARTERS + r"(?:of|the|and|a|in|" + _CAP + r")){0,5})?)?")
_ON_TITLE = r"(?:\s+on\s+" + _CAP + r"(?:\s+" + _STARTERS + r"(?:of|the|and|" + _CAP + r")){0,4})?"
QUOTE_ATTR = re.compile(r"[\"“][^\"“”]{10,400}?[\"”]\s*[—–-]+\s*(?:" + _OUTLET + "|" + _PERSON + ")"
                        + _ON_TITLE)
PRAISE = re.compile(r"^\s*(praise for|advance praise|acclaim for)\b.*$", re.I | re.M)
ATTR_ONLY = re.compile(r"[—–]\s*" + _OUTLET + r"(?=\s*(?:[\"“]|$))")
# Publisher copy that Google stores wrapped in quotes with a source tag:
#   "He did not exist...or did he? ..."--Back cover.
#   "..."--Provided by publisher.   "..." - Prové de l'editor.   "..."--
SOURCE_TAG = re.compile(
    r"(?<=[\"”'])\s*[-–—]{1,2}\s*(?:provided by (?:the )?publisher|publisher(?:'s)? (?:description|summary)"
    r"|from the publisher|back cover|jacket|book jacket|cover|from amazon\.com|amazon\.com|amazon"
    r"|prov[ée] de l'editor|publisher)?\.?\s*$", re.I)


def unwrap(desc):
    """Publisher copy wrapped in one quote from start to end -> the copy itself.
    '"The Night Cyclist" by award-winning...' starts with a quote but is not
    wrapped, so it is left alone."""
    t = desc.strip()
    if not t or t[0] not in "\"“'":
        return t
    t2 = SOURCE_TAG.sub("", t).strip()
    if len(t2) < 2 or t2[-1] not in "\"”'":
        return t
    inner = t2[1:-1].strip()
    if re.search(r"[\"“”]", inner):
        return t
    return inner


def strip_reviews(desc):
    """Remove quoted praise with its attribution, 'Praise for ...' lines and
    bare '-- The New York Times' tails; what is left is usually the blurb."""
    t = unwrap(desc)
    t = QUOTE_ATTR.sub(" ", t)
    t = PRAISE.sub(" ", t)
    t = ATTR_ONLY.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def junk_reason(desc):
    for name, rx in JUNK:
        if rx.search(desc):
            return name
    # review quotes are SHORT; a long quoted block is wrapped publisher copy
    quoted = sum(len(m) for m in re.findall(r"[\"“][^\"”]{20,300}[\"”]", desc))
    if len(REVIEW_SOURCE.findall(desc)) >= 2 or quoted > 0.5 * len(desc):
        return "review quotes"
    return None


def _pick(row, cands):
    """From one book's matching editions: (outcome, item, detail)."""
    described = [i for i in cands if len((i.get("description") or "").strip()) >= MIN_CHARS]
    if not described:
        return "too short", cands[0], ""
    judged = []
    for i in described:
        t = strip_reviews(i["description"])
        judged.append((i, t, junk_reason(t) if len(t) >= MIN_CHARS else "too short"))
    good = [(i, t) for i, t, why in judged if why is None]
    if not good:
        i, _, why = judged[0]
        return ("too short", i, "") if why == "too short" else ("junk text", i, why)
    # several clean editions: prefer 300-2500 chars, then the longest
    good.sort(key=lambda x: (300 <= len(x[1]) <= 2500, len(x[1])), reverse=True)
    i, t = good[0]
    changed = t != re.sub(r"\s+", " ", i["description"]).strip()
    return "usable", dict(i, description=t), ("reviews stripped" if changed else "")


def classify(row):
    """(outcome, chosen item or None, detail)"""
    items = [i.get("volumeInfo", i) for i in (row.get("items") or []) if isinstance(i, dict)]
    if not items:
        return "no results", None, ""
    en = [i for i in items if str(i.get("language") or "").lower().startswith("en")]
    if not en:
        return "not English", None, ",".join(sorted({str(i.get("language")) for i in items}))
    sn = surname(row.get("author"))
    by = [i for i in en if sn and any(sn in norm(a) for a in (i.get("authors") or []))]
    if not by:
        exact = [i for i in en if exact_long_title(row.get("title"), i)]
        if exact:
            o, it, d = _pick(row, exact)
            return o, it, ("title-only match" + (f", {d}" if d else "")) if o == "usable" else d
        return "author mismatch", en[0], ""
    same = [i for i in by if title_match(row.get("title"), i)]
    if not same:
        return "wrong book", by[0], ""
    return _pick(row, same)


def load_jsonl(path):
    out = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def hc_genres(b):
    """Genre names from a Hardcover raw record, whatever shape it is in."""
    ct = b.get("cached_tags") or b.get("tags") or {}
    if isinstance(ct, str):
        try:
            ct = json.loads(ct)
        except json.JSONDecodeError:
            ct = {}
    g = ct.get("Genre") if isinstance(ct, dict) else None
    out = []
    for x in g or []:
        name = x.get("tag") if isinstance(x, dict) else x
        if name:
            out.append(str(name))
    return out


def hc_info(ids):
    """ratings_count, rating, genres for the looked-up ids, one pass over raw.jsonl."""
    info = {}
    if not HC_RAW.exists():
        return info
    with HC_RAW.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                continue
            bid = str(b.get("id"))
            if bid in ids:
                info[bid] = {"ratings_count": b.get("ratings_count"),
                             "rating": b.get("rating"),
                             "genres": hc_genres(b)}
    return info


def servable(rc, ar, mean_ar):
    rc = float(rc or 0)
    if rc < MIN_RATINGS:
        return False
    if ar is None:
        return True
    adj = (rc * float(ar) + SHRINK * mean_ar) / (rc + SHRINK)
    return adj >= MIN_AVG


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=12, help="examples per outcome")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    targets = {str(t["id"]): t for t in load_jsonl(TARGETS)}
    # one answer per book: the LATEST v2 row (a --retry-empty row replaces the
    # empty one; old v1 rows are ignored the way the trickle ignores them)
    latest = {}
    for r in load_jsonl(FETCHED):
        if str(r.get("id")) in targets and r.get("v", 2) == 2:
            latest[str(r["id"])] = r
    fetched = list(latest.values())
    if not fetched:
        raise SystemExit(f"nothing fetched yet in {FETCHED} (run from the repo root)")
    fields = Counter(k for r in fetched for i in (r.get("items") or []) for k in i)

    res = []
    for r in fetched:
        t = targets[str(r["id"])]
        row = dict(r, title=r.get("title") or t["title"], author=r.get("author") or t["author"],
                   users_count=int(t.get("users_count") or r.get("users_count") or 0))
        outcome, item, detail = classify(row)
        res.append((row, outcome, item, detail))

    n = len(res)
    by = Counter(o for _, o, _, _ in res)
    trickle_hits = sum(1 for row, _, _, _ in res if any(
        str(i.get("language") or "").startswith("en")
        and len(i.get("description") or "") >= 100
        and surname(row["author"]) in " ".join(norm(a) for a in (i.get("authors") or []))
        for i in (row.get("items") or [])))

    print(f"looked up {n:,} of {len(targets):,} queued books "
          f"({len(targets) - n:,} still to go)")
    print(f"result fields kept: {', '.join(k for k, _ in fields.most_common())}\n")
    print(f"  trickle's count (English, 100+ chars, surname)  {trickle_hits:6,}  {pct(trickle_hits, n)}")
    print("  this audit:")
    for o in OUTCOMES:
        print(f"    {o:<18}{by[o]:7,}  {pct(by[o], n):>4}")
    rs = sum(1 for _, o, _, d in res if o == "usable" and "reviews stripped" in d)
    to = sum(1 for _, o, _, d in res if o == "usable" and "title-only" in d)
    if rs:
        print(f"      ({rs:,} of the usable had review quotes or a source tag stripped)")
    if to:
        print(f"      ({to:,} usable on an exact 3+ word title, author credited differently)")
    jr = Counter(d for _, o, _, d in res if o == "junk text")
    if jr:
        print(f"      junk kinds: {', '.join(f'{k} {v}' for k, v in jr.most_common())}")

    usable = [(row, item) for row, o, item, _ in res if o == "usable"]
    ids = {str(row["id"]) for row, _ in usable}
    info = hc_info(ids)
    if info:
        rated = [(float(v["rating"]), float(v["ratings_count"] or 0)) for v in info.values()
                 if v.get("rating") is not None]
        mean_ar = (sum(a for a, c in rated if c >= MIN_RATINGS)
                   / max(1, sum(1 for a, c in rated if c >= MIN_RATINGS))) if rated else 3.9
        serv = [row for row, _ in usable
                if servable(info.get(str(row["id"]), {}).get("ratings_count"),
                            info.get(str(row["id"]), {}).get("rating"), mean_ar)]
        print(f"\n  of {len(usable):,} usable, recommendable on the Books tab "
              f"(10+ ratings, shrunk avg 3.6+): {len(serv):,}  {pct(len(serv), len(usable))}")
        print("  (the rest would be in the corpus and searchable, but never recommended)")
    else:
        serv = []
        print(f"\n  ({HC_RAW} not found, so no recommendable / genre breakdown)")

    print("\n  by readers on Hardcover     looked up   usable")
    for lo, hi in BANDS:
        band = [(row, o) for row, o, _, _ in res if lo <= row["users_count"] < hi]
        u = sum(1 for _, o in band if o == "usable")
        label = f"{lo}+" if hi > 10 ** 8 else f"{lo}-{hi - 1}"
        print(f"    {label:<26}{len(band):9,}{u:9,}  {pct(u, len(band))}")

    if info:
        g = Counter()
        for row, _ in usable:
            for name in info.get(str(row["id"]), {}).get("genres", [])[:2]:
                g[name] += 1
        if g:
            print("\n  genre mix of the usable books (Hardcover, top 2 genres each):")
            for name, c in g.most_common(15):
                print(f"    {name:<28}{c:6,}  {pct(c, len(usable))}")

    lens = sorted(len(item["description"]) for _, item in usable)
    if lens:
        q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]  # noqa: E731
        print(f"\n  blurb length (usable): median {q(0.5):,} chars, "
              f"10th pct {q(0.1):,}, 90th pct {q(0.9):,}")

    # --- files ---------------------------------------------------------------
    rng = random.Random(args.seed)
    groups = defaultdict(list)
    for row, o, item, d in res:
        groups[o].append((row, item, d))
    with SAMPLES.open("w", encoding="utf-8") as fh:
        for o in OUTCOMES:
            g = groups.get(o) or []
            if not g:
                continue
            fh.write(f"\n{'=' * 78}\n{o.upper()}  ({len(g):,} books, {args.samples} shown)\n{'=' * 78}\n")
            for row, item, d in rng.sample(g, min(args.samples, len(g))):
                fh.write(f"\n> {row['title']}  |  {row['author']}  |  {row['users_count']} readers\n")
                if item:
                    fh.write(f"  google: {item.get('title')}  |  {', '.join(item.get('authors') or [])}"
                             f"  |  {item.get('language')}" + (f"  |  {d}" if d else "") + "\n")
                    desc = re.sub(r"\s+", " ", item.get("description") or "")
                    fh.write(f"  {desc[:600]}{'...' if len(desc) > 600 else ''}\n")
                elif d:
                    fh.write(f"  languages: {d}\n")
    with USABLE.open("w", encoding="utf-8") as fh:
        for row, item in usable:
            fh.write(json.dumps({"id": str(row["id"]), "title": row["title"], "author": row["author"],
                                 "users_count": row["users_count"],
                                 "gbooks_title": item.get("title"),
                                 "description": item["description"]}, ensure_ascii=False) + "\n")
    print(f"\nwrote {SAMPLES}  (read these: is each 'usable' really a blurb for that book?)")
    print(f"wrote {USABLE}  ({len(usable):,} rows)")


if __name__ == "__main__":
    main()
