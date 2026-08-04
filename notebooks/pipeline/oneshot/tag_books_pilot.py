"""
tag_books_pilot.py  --  pilot: tag the top 300 UNTAGGED books with Haiku 4.5.

Purpose of THIS pilot (not the full run):
  1. Prove tag QUALITY on books you actually know (top-300 = 1.4M+ ratings each).
  2. Prove batched output stays ALIGNED (book 7's tags don't land on book 8).
  3. MEASURE real output tokens/book -- the number that decides whether the full
     20K fits your $6.68 balance. Output costs 5x input, so tag output dominates.

WHY REALTIME, NOT THE BATCH API (for the pilot):
  The pilot's whole value is fast iteration -- run, read tags, fix prompt, rerun.
  The Batch API is asynchronous (submit a job, poll, collect minutes-to-hours
  later) and is where the 50% discount lives. That latency fights iteration. So:
    pilot      = realtime calls (full rate, but 300 short books = pennies)
    full 20K   = Batch API (flip the client call; 50% off where it matters)
  Caching is on in BOTH (taxonomy block is identical every call).

COST LEVERS baked in here:
  - COMPACT output: "book_id | tag:score tag:score ..." not JSON. ~halves output
    tokens vs JSON. Output is 5x input price, so this is the biggest lever.
  - Prompt caching: the 66-tag taxonomy + instructions are a cached system block.
    (If the block is under Haiku's min cacheable size, caching just no-ops -- the
    usage report prints cache_read/creation so you can SEE if it engaged.)
  - 15 books/call: amortizes the taxonomy across the batch.

SAFETY:
  - Vocabulary is the 66 tags FROM book_tags.parquet (authoritative, not hardcoded).
  - Every returned tag not in those 66 is DROPPED (a hallucinated tag can never
    enter the corpus).
  - Output is book_id-keyed; we assert every book_id sent comes back, and flag any
    batch that misaligns rather than writing bad rows.

Run from REPO ROOT (needs ANTHROPIC_API_KEY in env):
    python notebooks/pipeline/tag_books_pilot.py
"""

from pathlib import Path
import os
import time
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "data" / "processed"
CORPUS = PROCESSED / "books_v1.parquet"
TAGS = PROCESSED / "book_tags.parquet"
OUT = PROCESSED / "pilot_book_tags.parquet"

MODEL = "claude-haiku-4-5"
N_PILOT = 300
BATCH_SIZE = 15
MIN_TAGS, MAX_TAGS = 5, 8
DESC_WORD_CAP = 300          # truncate long descriptions (median ~173 words)
MAX_TOKENS = 1500

# Haiku 4.5 rates ($/million tokens) -- for the cost report only
PRICE_IN = 1.00
PRICE_CACHE_READ = 0.10
PRICE_CACHE_WRITE = 1.25     # 5-min cache creation = 1.25x input
PRICE_OUT = 5.00


def load_vocab_and_tagged():
    tags = pd.read_parquet(TAGS)
    return sorted(tags["tag"].unique()), set(tags["book_id"])


def pick_candidates(tagged_ids):
    df = pd.read_parquet(CORPUS)
    df = df[~df["book_id"].isin(tagged_ids)]
    return df.sort_values("ratings_count", ascending=False).head(N_PILOT).reset_index(drop=True)


def truncate_desc(s):
    return " ".join(str(s).split()[:DESC_WORD_CAP])


def build_system_prompt(vocab):
    tag_list = ", ".join(vocab)
    return (
        "You tag books by MOOD, STRUCTURE, and PACING for a recommender. "
        "You are given a batch of books; tag each one.\n\n"
        "RULES:\n"
        f"- Use ONLY these {len(vocab)} tags, exactly as written. Invent nothing:\n"
        f"  {tag_list}\n"
        f"- Choose {MIN_TAGS}-{MAX_TAGS} tags per book that best capture its feel, "
        "structure, and pace (not its genre/topic).\n"
        "- Give each tag a confidence score from 0.2 to 1.0. Reserve 1.0 for a "
        "defining, quintessential quality; most scores are 0.6-0.9.\n"
        "- Order each book's tags by score, highest first.\n\n"
        "OUTPUT FORMAT -- one line per book, nothing else, no preamble, no JSON:\n"
        "<book_id> | tag:score tag:score tag:score\n"
        "Use the exact book_id given. Output every book, in the order given."
    )


def build_user_message(batch):
    out = ["Tag these books:\n"]
    for _, b in batch.iterrows():
        out.append(f"[{b['book_id']}] {b['title']}\n{truncate_desc(b['description'])}\n")
    return "\n".join(out)


def parse_response(text, sent_ids, vocab):
    vocab_set, sent_set = set(vocab), set(sent_ids)
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
            tag = tag.strip()
            if tag not in vocab_set:
                dropped.append(tag)
                continue
            try:
                score = max(0.2, min(1.0, float(sc)))
            except ValueError:
                continue
            best[tag] = max(best.get(tag, 0.0), score)
        pairs = sorted(best.items(), key=lambda x: -x[1])[:MAX_TAGS]
        if pairs:
            got[bid] = pairs
    return got, {"missing": sorted(sent_set - set(got)), "dropped_tags": dropped}


def main():
    if not CORPUS.exists() or not TAGS.exists():
        raise SystemExit("Missing books_v1.parquet or book_tags.parquet in data/processed.")
    try:
        from anthropic import Anthropic
    except ImportError:
        raise SystemExit("pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY in your environment first.")

    vocab, tagged_ids = load_vocab_and_tagged()
    print(f"Vocabulary: {len(vocab)} tags | already-tagged skipped: {len(tagged_ids):,}")
    cand = pick_candidates(tagged_ids)
    print(f"Pilot: {len(cand)} untagged books "
          f"(ratings {cand['ratings_count'].min():,}-{cand['ratings_count'].max():,})")

    client = Anthropic()
    system = [{"type": "text", "text": build_system_prompt(vocab),
               "cache_control": {"type": "ephemeral"}}]

    all_rows, all_dropped = [], []
    tot = dict(inp=0, out=0, cr=0, cw=0)
    misaligned = 0
    n_batches = (len(cand) + BATCH_SIZE - 1) // BATCH_SIZE

    for bi in range(n_batches):
        batch = cand.iloc[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
        sent_ids = batch["book_id"].tolist()
        resp = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=system,
            messages=[{"role": "user", "content": build_user_message(batch)}],
        )
        u = resp.usage
        tot["inp"] += u.input_tokens
        tot["out"] += u.output_tokens
        tot["cr"] += getattr(u, "cache_read_input_tokens", 0) or 0
        tot["cw"] += getattr(u, "cache_creation_input_tokens", 0) or 0

        text = "".join(b.text for b in resp.content if b.type == "text")
        got, prob = parse_response(text, sent_ids, vocab)
        all_dropped += prob["dropped_tags"]
        if prob["missing"]:
            misaligned += 1
            print(f"  batch {bi+1}/{n_batches}: MISSING {len(prob['missing'])} books (skipped)")
        for bid, pairs in got.items():
            for tag, score in pairs:
                all_rows.append({"book_id": bid, "tag": tag, "score": round(score, 2)})
        print(f"  batch {bi+1}/{n_batches}: {len(got)}/{len(sent_ids)} tagged "
              f"| out={u.output_tokens} cached={tot['cr']>0}")
        time.sleep(0.3)

    out_df = pd.DataFrame(all_rows)
    out_df.to_parquet(OUT, index=False)

    print("\n" + "=" * 60)
    n_done = out_df["book_id"].nunique()
    per_book = out_df.groupby("book_id").size()
    print(f"TAGGED {n_done}/{len(cand)} | rows={len(out_df)} | "
          f"avg {per_book.mean():.1f}/book (min {per_book.min()}, max {per_book.max()})")
    print(f"misaligned batches: {misaligned}/{n_batches}")
    if all_dropped:
        from collections import Counter
        print(f"dropped invalid tags: {len(all_dropped)} -> {dict(Counter(all_dropped).most_common(5))}")
    else:
        print("dropped invalid tags: 0 (model stayed in vocabulary)")

    print("\n--- TOKEN USAGE (pilot, actual) ---")
    print(f"  input={tot['inp']:,}  cache_read={tot['cr']:,}  cache_write={tot['cw']:,}  output={tot['out']:,}")
    cost = (tot["inp"] * PRICE_IN + tot["cr"] * PRICE_CACHE_READ
            + tot["cw"] * PRICE_CACHE_WRITE + tot["out"] * PRICE_OUT) / 1e6
    print(f"  pilot cost: ${cost:.4f}  ({tot['out']/max(n_done,1):.0f} output tokens/book)")

    scale = 20000 / max(n_done, 1)
    full_batch = cost * scale * 0.5
    affordable = int(6.68 / (full_batch / 20000)) if full_batch > 0 else 0
    print("\n--- EXTRAPOLATION to 20,000 (Batch API, 50% off) ---")
    print(f"  est. full-run cost: ${full_batch:.2f}   | $6.68 covers ~{affordable:,} books")

    print(f"\nWrote {OUT}\nSample tags to eyeball:")
    titles = dict(zip(cand["book_id"], cand["title"]))
    for bid in out_df["book_id"].drop_duplicates().head(8):
        tg = ", ".join(f"{r.tag}({r.score})" for r in out_df[out_df.book_id == bid].itertuples())
        print(f"  {titles.get(bid, bid)}: {tg}")


if __name__ == "__main__":
    main()