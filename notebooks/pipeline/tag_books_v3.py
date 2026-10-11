"""
tag_books.py  --  final book tagging run (pilot confirm + full Batch job).

ONE script, two modes (set MODE below):
  "pilot" : 300 books, REALTIME, instant -- the quick confirm. ~$0.17.
  "full"  : every untagged book >= MIN_RATINGS  + (optionally) re-tag the 8,800,
            via the Batch API (50% off), async, RESUMABLE. ~$9 at floor 10k.

WORKFLOW:
  1. MODE="pilot" -> run -> eyeball dropped-tags (should be ~0) and tag quality.
  2. MODE="full"  -> run -> it submits one batch, saves state, polls, collects,
     and writes book_tags_v2.parquet (old tags + new, re-tagged books replaced).
     Close the terminal anytime after "submitted"; re-run to resume collection.
  3. Upload book_tags_v2.parquet (rename to book_tags.parquet or repoint B_TAGS),
     then raise tag_weight back toward 0.3 now that coverage is dense.

Vocabulary = ALL_BOOK_TAGS from tag_taxonomy.py (the reconciled merged set).
The UNIVERSAL/BOOK_ONLY split does NOT affect tagging -- only the full set does --
so leaving the taxonomy's `# ??` routing calls unsettled is fine for this run.

Run from REPO ROOT (needs ANTHROPIC_API_KEY):
    python notebooks/pipeline/tag_books.py
"""

from pathlib import Path
import os
import sys
import json
import time
import pandas as pd

# import the merged taxonomy that sits next to this script
sys.path.insert(0, str(Path(__file__).parent))
from tag_taxonomy_v3 import ALL_BOOK_TAGS_V3 as ALL_BOOK_TAGS  # noqa: E402
from tag_taxonomy_v3 import FORM_TAGS, NONFICTION_FORMS  # noqa: E402

# ---- config ------------------------------------------------------------------
MODE = "full"              # "pilot" FIRST (~$0.20), then "full" (~$21)
MIN_RATINGS = 0        # v3 tags the WHOLE corpus; see docstring         # full: tag untagged books with >= this many ratings
RETAG_EXISTING = True  # the v1 ids are Goodreads; only 263 survive the swap        # full: also re-tag the existing 8,800 onto merged taxonomy
N_PILOT = 300
BATCH_SIZE = 15
MIN_TAGS, MAX_TAGS = 5, 8
DESC_WORD_CAP = 300
MAX_TOKENS = 1500
MODEL = "claude-haiku-4-5"

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "data" / "processed"
CORPUS = PROCESSED / "books_v2_hc3.parquet"
TAGS = PROCESSED / "book_tags_v2_hc2.parquet"   # only read to report overlap
OUT_MERGED = PROCESSED / "book_tags_v3_llm.parquet"
PILOT_OUT = PROCESSED / "pilot_book_tags_v3.parquet"
STATE_FILE = PROCESSED / "tag_batch_state_v3.json"

# rates ($/M) for the cost report
PRICE_IN, PRICE_CACHE_READ, PRICE_CACHE_WRITE, PRICE_OUT = 1.00, 0.10, 1.25, 5.00

# Near-miss aliases applied BEFORE the whitelist. Whitespace->hyphen and
# lowercasing are handled generically in canon(); this dict is for true synonyms.
ALIASES = {
    "emotional-heavy": "emotionally-heavy",
    "mystery-adjacent": "mystery",
    "philosophical": "thought-provoking",   # soft synonym; delete this line to drop instead
    "comedic": "humorous",
    "suspenseful": "tense",
    "satirical": "satire",
    "epic": "novel",                 # removed from FORM after the pilot
    "romantic": "romance", "tragedy": "heart-wrenching", "tragic": "heart-wrenching",
    "gothic": "atmospheric",
    "short-story": "short-stories", "short-story-collection": "short-stories",
    "essay": "essays", "essay-collection": "essays",
    "graphic-novels": "graphic-novel", "comic": "graphic-novel", "manga": "graphic-novel",
    "journalism": "reportage", "narrative-nonfiction": "reportage",
    "autobiography": "memoir", "memoirs": "memoir",
    "travelogue": "travel-writing", "travel": "travel-writing",
    "popular-science": "science-writing", "poems": "poetry",
}

VOCAB = set(ALL_BOOK_TAGS)


def canon(tag):
    """Normalize a model-emitted tag to a valid vocab tag, or None to drop."""
    t = tag.strip().lower().replace(" ", "-")
    t = ALIASES.get(t, t)
    return t if t in VOCAB else None


def build_system_prompt():
    tag_list = ", ".join(sorted(VOCAB))
    return (
        "You tag books by MOOD, STRUCTURE, PACING, and THEME for a recommender. "
        "You are given a batch of books; tag each one.\n\n"
        "RULES:\n"
        f"- Use ONLY these {len(VOCAB)} tags, exactly as written. Invent nothing, "
        "and do NOT use plain genre words (e.g. 'paranormal') that aren't in the list:\n"
        f"  {tag_list}\n"
        f"- EVERY book gets EXACTLY ONE form tag, chosen from: {', '.join(sorted(FORM_TAGS))}. "
        "Form is what KIND of book it is. A book ABOUT a crime is not a crime novel: "
        "a journalist's account of the Mumbai underworld is `reportage`, a novel set in "
        "it is `novel`. A memoir is `memoir` even when it reads like fiction.\n"
        f"- Then choose {MIN_TAGS - 1}-{MAX_TAGS - 1} more tags that best capture its feel, "
        "prose, pace, and theme.\n"
        "- Score each 0.2-1.0. Reserve 1.0 for a defining, quintessential quality; "
        "most scores are 0.6-0.9. Order each book's tags by score, highest first.\n\n"
        "OUTPUT -- one line per book, nothing else, no preamble, no JSON:\n"
        "<book_id> | tag:score tag:score tag:score\n"
        "Use the exact book_id given. Output every book, in the order given."
    )


def truncate_desc(s):
    return " ".join(str(s).split()[:DESC_WORD_CAP])


def build_user_message(batch):
    out = ["Tag these books:\n"]
    for _, b in batch.iterrows():
        out.append(f"[{b['book_id']}] {b['title']}\n{truncate_desc(b['description'])}\n")
    return "\n".join(out)


def parse_response(text, sent_ids):
    sent_set = set(sent_ids)
    got, dropped = {}, []
    for raw in text.splitlines():
        line = raw.strip()
        if "|" not in line:
            continue
        left, right = line.split("|", 1)
        bid = left.strip().strip("[]")
        if bid not in sent_set:
            continue
        best = {}
        for tok in right.split():
            if ":" not in tok:
                continue
            tag, _, sc = tok.rpartition(":")
            ct = canon(tag)
            if ct is None:
                dropped.append(tag.strip())
                continue
            try:
                score = max(0.2, min(1.0, float(sc)))
            except ValueError:
                continue
            best[ct] = max(best.get(ct, 0.0), score)
        # EXACTLY ONE form tag, enforced here rather than hoped for in the
        # prompt. The pilot got 80% compliance from instructions alone; keeping
        # the highest-scoring form makes it 100% by construction.
        forms = [t for t in best if t in FORM_TAGS]
        if len(forms) > 1:
            keep = max(forms, key=lambda t: best[t])
            for t in forms:
                if t != keep:
                    del best[t]
        pairs = sorted(best.items(), key=lambda x: -x[1])[:MAX_TAGS]
        if pairs:
            got[bid] = pairs
    return got, {"missing": sorted(sent_set - set(got)), "dropped": dropped}


def build_candidates():
    """Every book in the corpus, most-read first.

    v1 skipped already-tagged books. That logic is inert now: the v1 tag rows
    are keyed to Goodreads book_ids and only 263 of 32,893 survive the Hardcover
    corpus swap. The Hardcover tag file is a different vocabulary entirely
    (shelf categories, not register), so it is not a skip list either -- it is
    the thing this run REPLACES.

    Most-read first matters for a resumable batch: if the run is interrupted or
    the balance runs out, the books people actually reach are already done.
    """
    df = pd.read_parquet(CORPUS)
    if MODE == "pilot":
        return df.sort_values("ratings_count", ascending=False).head(N_PILOT).reset_index(drop=True)
    return df.sort_values("ratings_count", ascending=False).reset_index(drop=True)


def batches_of(cand):
    for i in range(0, len(cand), BATCH_SIZE):
        yield cand.iloc[i:i + BATCH_SIZE]


# ---------------- PILOT (realtime) -------------------------------------------
def run_pilot(client, system, cand):
    rows, dropped, tot = [], [], dict(inp=0, out=0, cr=0, cw=0)
    misaligned = 0
    blist = list(batches_of(cand))
    for bi, batch in enumerate(blist):
        sent = batch["book_id"].tolist()
        r = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=system,
                                   messages=[{"role": "user", "content": build_user_message(batch)}])
        u = r.usage
        tot["inp"] += u.input_tokens; tot["out"] += u.output_tokens
        tot["cr"] += getattr(u, "cache_read_input_tokens", 0) or 0
        tot["cw"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        text = "".join(b.text for b in r.content if b.type == "text")
        got, prob = parse_response(text, sent)
        dropped += prob["dropped"]
        misaligned += 1 if prob["missing"] else 0
        for bid, pairs in got.items():
            for t, s in pairs:
                rows.append({"book_id": bid, "tag": t, "score": round(s, 2)})
        print(f"  batch {bi+1}/{len(blist)}: {len(got)}/{len(sent)} | out={u.output_tokens}")
        time.sleep(0.3)
    out = pd.DataFrame(rows); out.to_parquet(PILOT_OUT, index=False)
    cost = (tot["inp"] * PRICE_IN + tot["cr"] * PRICE_CACHE_READ +
            tot["cw"] * PRICE_CACHE_WRITE + tot["out"] * PRICE_OUT) / 1e6
    report(out, dropped, misaligned, len(blist))
    titles = dict(zip(cand["book_id"].astype(str), cand["title"]))
    print("\nSAMPLE -- read these. Do they describe how the book READS?")
    ids = list(dict.fromkeys(out["book_id"].astype(str)))
    for bid in ids[:: max(1, len(ids) // 14)][:14]:
        tg = out[out["book_id"].astype(str) == bid].sort_values("score", ascending=False)
        form = next((t for t in tg["tag"] if t in FORM_TAGS), "?")
        rest = ", ".join(t for t in tg["tag"] if t not in FORM_TAGS)
        print(f"  [{form:>13}] {str(titles.get(bid, bid))[:38]:40s} {rest}")
    print(f"\nPILOT cost: ${cost:.4f}  (realtime). Wrote {PILOT_OUT}")
    print("If dropped~0 and tags look right -> set MODE='full' and run again.")


# ---------------- FULL (Batch API, resumable) --------------------------------
def run_full(client, system, cand):
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        batch_id, batch_map = state["batch_id"], state["batch_map"]
        print(f"Resuming batch {batch_id} ({len(batch_map)} requests)")
    else:
        print(f"Submitting {len(cand):,} books in {(len(cand)+BATCH_SIZE-1)//BATCH_SIZE:,} requests...")
        reqs, batch_map = [], {}
        for bi, batch in enumerate(batches_of(cand)):
            cid = f"b{bi}"
            batch_map[cid] = batch["book_id"].tolist()
            reqs.append({"custom_id": cid, "params": {
                "model": MODEL, "max_tokens": MAX_TOKENS, "system": system,
                "messages": [{"role": "user", "content": build_user_message(batch)}]}})
        job = client.messages.batches.create(requests=reqs)
        batch_id = job.id
        STATE_FILE.write_text(json.dumps({"batch_id": batch_id, "batch_map": batch_map}))
        print(f"  submitted: {batch_id}  (state saved -> safe to close terminal)")

    while True:
        job = client.messages.batches.retrieve(batch_id)
        c = job.request_counts
        print(f"  status={job.processing_status}  done={c.succeeded} err={c.errored} proc={c.processing}")
        if job.processing_status == "ended":
            break
        time.sleep(30)

    rows, dropped, failed, tot_out = [], [], 0, 0
    for res in client.messages.batches.results(batch_id):
        sent = batch_map.get(res.custom_id, [])
        if res.result.type != "succeeded":
            failed += 1
            continue
        msg = res.result.message
        tot_out += msg.usage.output_tokens
        text = "".join(b.text for b in msg.content if b.type == "text")
        got, prob = parse_response(text, sent)
        dropped += prob["dropped"]
        for bid, pairs in got.items():
            for t, s in pairs:
                rows.append({"book_id": bid, "tag": t, "score": round(s, 2)})

    new_df = pd.DataFrame(rows)
    # v3 REPLACES rather than merges. The Hardcover file is a different
    # vocabulary (shelf categories); mixing the two would put `young adult` and
    # `unreliable-narrator` in one matrix and the pool-centred adjustment would
    # average across two incompatible axes.
    merged = new_df
    merged.to_parquet(OUT_MERGED, index=False)

    print(f"\nfailed requests: {failed}")
    report(new_df, dropped, 0, len(batch_map), label="NEW")
    print(f"\nMERGED tag file: {merged['book_id'].nunique():,} books, {len(merged):,} rows -> {OUT_MERGED}")
    # batch is already 50% off; report measured output as the dominant term
    print(f"(batch output tokens: {tot_out:,})")
    print(f"\nNEXT: run eval_goodbooks with B_TAGS -> {OUT_MERGED.name} against 0.016,")
    print("then upload it and flip B_TAGS in app.py if it holds up.")
    print(f"Delete {STATE_FILE.name} once you've confirmed the merged file is good.")


def report(df, dropped, misaligned, n_batches, label="TAGGED"):
    if df.empty:
        print("  (no rows produced)"); return
    # THE CHECK THAT MATTERS: the shelf gate needs one form tag per book.
    n_form = df[df["tag"].isin(FORM_TAGS)].groupby("book_id").size()
    books = df["book_id"].nunique()
    exactly_one = int((n_form == 1).sum())
    none = books - len(n_form)
    print(f"\nFORM TAGS: {exactly_one}/{books} books have exactly one "
          f"({exactly_one / max(books, 1):.0%}); {none} have none; "
          f"{int((n_form > 1).sum())} have several")
    fc = df[df["tag"].isin(FORM_TAGS)]["tag"].value_counts()
    nf = int(df[df["tag"].isin(NONFICTION_FORMS)]["book_id"].nunique())
    print(f"  non-fiction share: {nf}/{books} ({nf / max(books, 1):.0%})")
    print("  " + ", ".join(f"{t} {c}" for t, c in fc.head(12).items()))
    if exactly_one < 0.9 * books:
        print("  >> under 90% -- the shelf gate cannot rely on this yet. Fix the")
        print("     prompt before paying for the full run.")
    per = df.groupby("book_id").size()
    print("\n" + "=" * 56)
    print(f"{label} {df['book_id'].nunique():,} books | {len(df):,} rows | "
          f"avg {per.mean():.1f}/book (min {per.min()}, max {per.max()})")
    print(f"misaligned batches: {misaligned}/{n_batches}")
    if dropped:
        from collections import Counter
        print(f"dropped after alias+whitelist: {len(dropped)} -> {dict(Counter(dropped).most_common(6))}")
    else:
        print("dropped after alias+whitelist: 0")


def main():
    if not CORPUS.exists():
        raise SystemExit(f"Missing {CORPUS.name} in data/processed.")
    try:
        from anthropic import Anthropic
    except ImportError:
        raise SystemExit("pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY first.")

    print(f"MODE={MODE} | vocab={len(VOCAB)} tags")
    cand = build_candidates()
    print(f"candidates: {len(cand):,} books "
          f"(ratings {cand['ratings_count'].min():,}-{cand['ratings_count'].max():,})")
    if MODE == "full":
        est = len(cand) * 0.000279
        print(f"est. full-run cost (Batch API): ~${est:.2f}")

    client = Anthropic()
    system = [{"type": "text", "text": build_system_prompt(),
               "cache_control": {"type": "ephemeral"}}]
    (run_pilot if MODE == "pilot" else run_full)(client, system, cand)


if __name__ == "__main__":
    main()