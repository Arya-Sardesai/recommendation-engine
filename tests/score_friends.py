"""
score_friends.py -- read the profile files testers send back and print, per
person and per tab, how the recommendations actually did.

    python tests\\score_friends.py friends\\
    python tests\\score_friends.py friends\\ --tab books --detail
    python tests\\score_friends.py friends\\ --since 2026-11-10

Put each file in one folder (rename them first: rohan.json, priya.json ...).
Needs the export format from Oct 21 ("version": 2); older files only hold
ratings and are listed as skipped.

WHAT IT PRINTS
  per person x tab   ratings | picks shown | saved | not interested | already
                     seen | hit rate (saved / shown)
  where the saves sit     rank of saved picks in "For you" (top-5 vs 16-20)
  from likes         how many of their liked (4+) ratings the 20 "For you"
                     picks come from (because_of)
  saved vs skipped   popularity, year and similarity of each group, and the
                     tags that showed up most in skipped picks
  which ratings bring in bad picks   because_of ranked by skips
  searches with no results   catalogue gaps, every tab
--detail lists every reacted-to pick with its reason fields.
"""
import argparse
import json
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path

TABS = {"books": "books", "movies": "films", "tv": "TV"}


def load(folder):
    out = []
    for p in sorted(Path(folder).glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"  skip {p.name}: unreadable ({e})")
            continue
        if not isinstance(d, dict) or d.get("version", 1) < 2 or "books" not in d:
            print(f"  skip {p.name}: old format (ratings only) -- ask for a fresh download")
            continue
        out.append((p.stem, d))
    return out


def picks(section, since=None):
    """Every pick this person was shown, once, with their reaction. --since
    drops earlier picks first shown before that date (an older build)."""
    seen = {}
    for r in section.get("recommended", []):
        seen[r["id"]] = r
    for rows in (section.get("other_tabs") or {}).values():
        for r in rows:
            seen.setdefault(r["id"], r)
    for r in section.get("earlier_picks_you_reacted_to", []):
        if since and str(r.get("first_shown", "")) < since:
            continue
        seen.setdefault(r["id"], r)
    return list(seen.values())


def kind_of(reaction):
    r = reaction or "none"
    return ("saved" if r == "saved" else "skipped" if r == "not interested"
            else "seen" if r.startswith("rated") else "none")


def pop(r):
    return r.get("votes") if r.get("votes") is not None else r.get("ratings")


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return f"{stats.median(xs):,.0f}" if xs else "-"


def med_year(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return f"{stats.median(xs):.0f}" if xs else "-"


def med2(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return f"{stats.median(xs):.2f}" if xs else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--tab", choices=list(TABS), help="only this tab")
    ap.add_argument("--detail", action="store_true", help="list every reacted-to pick")
    ap.add_argument("--since", help="YYYY-MM-DD: ignore picks first shown before this "
                                    "(e.g. 2026-11-10, the frozen build)")
    args = ap.parse_args()
    people = load(args.folder)
    if not people:
        sys.exit("no version-2 profile files in that folder")
    versions = Counter(d.get("app_version") for _, d in people)
    print(f"{len(people)} people | app versions: "
          + ", ".join(f"{v} x{n}" for v, n in versions.items()))
    if len(versions) > 1:
        print("  WARNING: files come from different builds -- compare only within one")
    tabs = [args.tab] if args.tab else list(TABS)

    for tab in tabs:
        label = TABS[tab]
        print(f"\n{'=' * 100}\n{label.upper()}\n{'=' * 100}")
        print(f"  {'person':<14}{'ratings':>8}{'shown':>7}{'saved':>7}{'skip':>6}{'seen':>6}"
              f"{'hit rate':>10}   {'For-you saves at rank 1-5/6-10/11-20':<38}{'from likes':>11}")
        all_picks = []
        anchor_skips, anchor_saves = Counter(), Counter()
        misses = []
        for name, d in people:
            sec = d.get(tab) or {}
            ps = picks(sec, args.since)
            if not ps and not sec.get("your_ratings"):
                continue
            k = Counter(kind_of(p.get("your_reaction")) for p in ps)
            shown = len(ps)
            hit = k["saved"] / shown if shown else 0
            fy = [r for r in sec.get("recommended", []) if kind_of(r.get("your_reaction")) == "saved"]
            ranks = [r["rank"] for r in fy]
            bands = (sum(1 for x in ranks if x <= 5), sum(1 for x in ranks if 6 <= x <= 10),
                     sum(1 for x in ranks if x > 10))
            # how many of their liked ratings the 20 picks come from: the ranker
            # takes up to 3 picks per rating, so 20 picks can rest on ~7 of 90
            src = {r.get("because_of") for r in sec.get("recommended", [])}
            liked = sum(1 for r in sec.get("your_ratings", []) if float(r.get("rating") or 0) >= 4)
            top = len(src)
            print(f"  {name[:13]:<14}{len(sec.get('your_ratings', [])):>8}{shown:>7}{k['saved']:>7}"
                  f"{k['skipped']:>6}{k['seen']:>6}{hit:>9.0%}   "
                  f"{'%d / %d / %d' % bands:<38}{f'{top} of {liked}':>11}")
            for p in ps:
                p = dict(p, _who=name)
                all_picks.append(p)
                kk = kind_of(p.get("your_reaction"))
                if kk == "skipped":
                    anchor_skips[(name, p.get("because_of"))] += 1
                elif kk == "saved":
                    anchor_saves[(name, p.get("because_of"))] += 1
            misses += [(name, q) for q in sec.get("searches_with_no_results", [])]

        groups = defaultdict(list)
        for p in all_picks:
            groups[kind_of(p.get("your_reaction"))].append(p)
        print(f"\n  {'':<10}{'picks':>7}{'popularity (median)':>22}{'year':>7}{'similarity':>12}")
        for g in ("saved", "seen", "skipped", "none"):
            ps = groups.get(g, [])
            if ps:
                print(f"  {g:<10}{len(ps):>7}{med([pop(p) for p in ps]):>22}"
                      f"{med_year([p.get('year') for p in ps]):>7}{med2([p.get('similarity') for p in ps]):>12}")
        print("  (seen = they rated it from the card: already watched/read. Many 'seen'\n"
              "   = the picks are too obvious; skipped far more popular or obscure than\n"
              "   saved = the popularity weighting is off.)")

        tag_skip = Counter(t for p in groups.get("skipped", []) for t in p.get("shared_tags") or [])
        tag_save = Counter(t for p in groups.get("saved", []) for t in p.get("shared_tags") or [])
        if tag_skip:
            print("\n  tags behind skipped picks (vs saved):  "
                  + ", ".join(f"{t} {n}/{tag_save.get(t, 0)}" for t, n in tag_skip.most_common(10)))
        gen_skip = Counter(g for p in groups.get("skipped", []) for g in (p.get("genres") or [])[:1])
        gen_save = Counter(g for p in groups.get("saved", []) for g in (p.get("genres") or [])[:1])
        if gen_skip or gen_save:
            print("  first genre, skipped / saved:  "
                  + ", ".join(f"{g} {gen_skip.get(g, 0)}/{gen_save.get(g, 0)}"
                              for g, _ in (gen_skip + gen_save).most_common(10)))
        bad = [(who, a, n, anchor_saves.get((who, a), 0))
               for (who, a), n in anchor_skips.most_common() if n >= 2]
        if bad:
            print("\n  ratings that bring in skipped picks (person: rating -> skipped / saved):")
            for who, a, n, s in bad[:12]:
                print(f"    {who}: {a} -> {n} / {s}")
        if misses:
            print(f"\n  searches with no results ({len(misses)}):")
            for who, q in misses[:40]:
                print(f"    {who}: {q}")

        if args.detail:
            for g in ("saved", "skipped", "seen"):
                ps = groups.get(g, [])
                if not ps:
                    continue
                print(f"\n  --- {g} ---")
                for p in ps:
                    print(f"    {p['_who'][:10]:<11}#{p.get('rank', '?'):<3}{p.get('tab', '')[:10]:<11}"
                          f"{str(p.get('title'))[:38]:<40}<- {str(p.get('because_of'))[:28]:<30}"
                          f"sim {p.get('similarity')}  tags {','.join(p.get('shared_tags') or [])}")


if __name__ == "__main__":
    main()
