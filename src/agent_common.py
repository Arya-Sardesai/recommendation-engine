"""
agent_common.py -- the loop and the grounding check, shared by all three agents.

The three agents used to carry three copies of run_agent and the faithfulness
checker, and the copies had already drifted (TV printed its warnings, books
and films ran with verbose=False so theirs went nowhere). One copy now.

GROUNDING CHECK ("self-checks for hallucinations")
After the model's final answer, every title-shaped phrase in the reply is
compared against everything tools returned this conversation, plus the user's
own ratings and whatever the user typed. Three shapes count as titles:

    Title (Author, 2006)          a parenthetical with a year
    1. **Title** ...              a bold title opening a list item
    **Title** by Someone          a bold title followed by "by"

A bold label ending in a colon ("- **Pacing:** ...") is ignored, which is what
kept the first, greedier version of this checker from flagging prose.

Anything left over is checked against the whole catalogue and labelled
"in the catalogue, but no tool returned it" (the model freelancing from
memory) or "not in the catalogue" (invented, or the wrong edition). The model
then gets ONE chance to fix it: it is told which titles were ungrounded and
asked to verify them with a tool or drop them. Whatever is still ungrounded
after that is reported to the app, which shows it under the reply.
"""
import json
import re

from anthropic import Anthropic

import space_ranker as sr

MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Query encoder (BGE-M3, the model every corpus was embedded with)
# ---------------------------------------------------------------------------
def get_query_embedder(state):
    if "query_embedder" not in state:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state["query_embedder"] = SentenceTransformer("BAAI/bge-m3", device=device)
    return state["query_embedder"]


def embed_query(state, text):
    model = get_query_embedder(state)
    return model.encode([text], convert_to_numpy=True,
                        normalize_embeddings=True).astype("float32")


# ---------------------------------------------------------------------------
# Grounding check
# ---------------------------------------------------------------------------
_YEAR_PAREN = re.compile(
    r"(?:\*\*)?([A-Z0-9][^\n*()]{1,80}?)(?:\*\*)?"
    r"\s*\((?:[^)]*?\b(?:1[5-9]|20)\d{2}\b[^)]*)\)")
_LIST_BOLD = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*\*\*([^*\n]{2,90}?)\*\*", re.M)
_BY_AUTHOR = re.compile(r"\*\*([^*\n]{2,90}?)\*\*\s+by\s+[A-Z]")


def _candidates(reply):
    out = []
    for pat in (_YEAR_PAREN, _LIST_BOLD, _BY_AUTHOR):
        for m in pat.finditer(reply):
            raw = m.group(1)
            if raw.rstrip().endswith(":") or reply[m.end():m.end() + 1] == ":":
                continue                       # "**Pacing:**" is a label, not a title
            c = raw.strip().strip("*").rstrip(":–-—,").strip()
            c = re.sub(r"^\d+[.)]\s*", "", c).strip()
            if " by " in c:
                c = c.split(" by ")[0].strip()
            if c:
                out.append(c)
    return out


def check_faithfulness(reply, seen_titles, corpus_norms, user_text=""):
    """[{'title', 'in_catalogue'}] for titles in the reply no tool returned."""
    seen_norm = {sr.norm_title(t) for t in seen_titles if t}
    user_norm = sr.norm_title(user_text)

    def grounded(cn, loose):
        if cn in seen_norm:
            return True
        if loose and any(cn in s or s in cn for s in seen_norm if len(s) >= 4):
            return True                                  # subtitle / edition differences
        return len(cn) >= 4 and cn in user_norm          # the user named it themselves

    def in_catalogue(cn):
        return cn in corpus_norms or (
            len(cn) >= 6 and any(t.startswith(cn) for t in corpus_norms))

    issues, done = [], set()
    for c in _candidates(reply):
        cn = sr.norm_title(c)
        if len(cn) < 3 or cn in done:
            continue
        done.add(cn)
        # An unbolded "Title (2006)" match can drag in the prose before it
        # ("You asked about Dune (1965)"), so every word-suffix is a possible
        # title: "about Dune", "Dune". The full phrase gets the loose
        # (substring) match; shorter suffixes must match exactly, or "Patient"
        # inside an ungrounded "The Silent Patient" would hide behind a
        # grounded "The English Patient".
        words = c.split()
        suffixes = [" ".join(words[i:]) for i in range(len(words))]
        if any(grounded(sr.norm_title(s), loose=(i == 0))
               for i, s in enumerate(suffixes) if len(sr.norm_title(s)) >= 3):
            continue
        hit = next((s for s in suffixes if in_catalogue(sr.norm_title(s))), None)
        issues.append({"title": hit or c, "in_catalogue": hit is not None})
    return issues


def titles_from_result(result, keys=("title", "name")):
    """Every title/name string anywhere in a tool result."""
    found = set()

    def walk(x, depth=0):
        if depth > 4:
            return
        if isinstance(x, dict):
            for k in keys:
                v = x.get(k)
                if isinstance(v, str) and v:
                    found.add(v)
            for v in x.values():
                if isinstance(v, (dict, list)):
                    walk(v, depth + 1)
        elif isinstance(x, list):
            for v in x:
                walk(v, depth + 1)

    walk(result)
    return found


_CORRECTION = (
    "[grounding check] Your answer named {names}, which no tool returned in this "
    "conversation. For each one, either verify it with a tool call or remove it. "
    "Then give your full final answer again. Do not mention this check.")


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def run_agent_loop(user_message, state, *, system, tools, impls, noun,
                   conversation_history=None, max_turns=8, verbose=True,
                   self_correct=True):
    """Run one user turn. Returns (reply, updated_history).

    Leaves state["_last_check"] = {"issues", "corrected", "seen"} for the app.
    """
    client = Anthropic()
    history = list(conversation_history or []) + [{"role": "user", "content": user_message}]

    # Conversation-scoped, not turn-scoped: "tell me about the third one"
    # makes no tool calls and must not be flagged. The user's own ratings are
    # legitimately nameable, so they seed the ledger.
    seen = state.setdefault("_seen_titles", set())
    seen |= {str(r.get("matched_title", "")) for r in state.get("ratings", []) if r.get("matched_title")}
    user_texts = state.setdefault("_user_text", [])
    user_texts.append(user_message)
    corpus = state["bundle"]["title_norms"]
    corrected = False

    for turn in range(max_turns):
        resp = client.messages.create(
            model=MODEL, max_tokens=1500, system=system,
            tools=tools, messages=history)
        calls = [b for b in resp.content if b.type == "tool_use"]
        texts = [b.text for b in resp.content if b.type == "text"]
        history.append({"role": "assistant", "content": resp.content})

        if not calls:
            reply = "\n".join(texts)
            issues = check_faithfulness(reply, seen, corpus, " ".join(user_texts))
            if issues and self_correct and not corrected and turn < max_turns - 1:
                corrected = True
                names = ", ".join(f'"{i["title"]}"' for i in issues)
                if verbose:
                    print(f"  [!] ungrounded: {names} -- asking the model to verify or drop")
                history.append({"role": "user", "content": _CORRECTION.format(names=names)})
                continue
            state["_last_check"] = {"issues": issues, "corrected": corrected, "seen": len(seen)}
            if verbose:
                if issues:
                    print(f"  [!] WARNING: {len(issues)} title(s) not returned by any tool:")
                    for i in issues:
                        where = "in catalogue" if i["in_catalogue"] else "NOT in catalogue"
                        print(f"        - {i['title']}  ({where})")
                else:
                    fixed = " after one correction" if corrected else ""
                    print(f"  [ok] every named title was grounded{fixed} "
                          f"({len(seen)} {noun} known from tools and ratings)")
            return reply, history

        results = []
        for tc in calls:
            impl = impls.get(tc.name)
            try:
                result = impl(state, **tc.input) if impl else {"error": f"unknown tool {tc.name}"}
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            result = sr.clean(result)
            returned = titles_from_result(result)
            seen |= returned
            if verbose:
                print(f"  [agent calls {tc.name}({json.dumps(tc.input, default=str)})]")
                if result.get("error") if isinstance(result, dict) else False:
                    print(f"        -> ERROR {result['error']}")
                elif returned:
                    preview = list(returned)[:8]
                    more = f" (+{len(returned) - 8} more)" if len(returned) > 8 else ""
                    print(f"        -> {len(returned)} {noun}: {', '.join(preview)}{more}")
                else:
                    print(f"        -> (no {noun} in result)")
            results.append({"type": "tool_result", "tool_use_id": tc.id,
                            "content": json.dumps(result, default=str)})
        history.append({"role": "user", "content": results})

    state["_last_check"] = {"issues": [], "corrected": corrected, "seen": len(seen)}
    return "(the agent ran out of steps without finishing; try a narrower question)", history
