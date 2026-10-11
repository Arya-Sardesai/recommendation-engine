"""
agent_ui.py -- the Streamlit page shared by the three local agent apps.

Left: the taste profile. Load the profile you exported from the website (one
click if it is in Downloads), or search and rate by hand. Search is the
Space's own search_books / search_movies / search_tv.
Right: the chat. Under every answer the app shows the grounding check: a
quiet tick when every named title came from a tool, a warning naming any title
that didn't.

Ratings are stored in the Space's exact shape, so a profile moves between the
website and the agent without conversion.
"""
import html
import os
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*__path__.*")

import streamlit as st  # noqa: E402

import space_ranker as sr  # noqa: E402

LABELS = {
    "book": dict(title="Books", icon="📚", noun="books",
                 search="Search a book to rate", ph="e.g. The Hobbit or Brandon Sanderson",
                 chat="What are you in the mood to read?",
                 tips='Try: "what should I read next?", "something like my noir books but lighter", '
                      'or "non-fiction about crime in Mumbai".'),
    "movie": dict(title="Films", icon="🎬", noun="films",
                  search="Search a film to rate", ph="e.g. Inception, Nolan or Bong Joon-ho",
                  chat="What are you in the mood to watch?",
                  tips='Try: "what should I watch tonight?", "Korean revenge thrillers like my crime '
                       'films", or "more A24".'),
    "tv": dict(title="TV", icon="📺", noun="series",
               search="Search a series to rate", ph="e.g. Breaking Bad or Vince Gilligan",
               chat="What are you in the mood to watch?",
               tips='Try: "a limited series about a con", "something like Breaking Bad but anime", '
                    'or "a long sitcom to live in".'),
}

CSS = """
<style>
.stApp { background-color: #14110f; }
h1, h2, h3 { color: #f4f1ea; font-family: Georgia, 'Times New Roman', serif; }
.card { background: #1f1b18; border: 1px solid #34302b; border-radius: 8px;
        padding: 10px 12px; margin-bottom: 6px; }
.card-title { font-weight: 600; color: #f4f1ea; }
.card-sub { color: #c9a86a; font-size: 0.85rem; }
</style>
"""


@st.cache_resource(show_spinner="Loading the catalogue and the Space's ranker (first run takes a minute)...")
def _load(kind):
    return sr.load(kind)


def _ss():
    return st.session_state


def _label(b, pos):
    spec, row = b["spec"], b["df"].iloc[int(pos)]
    y = sr.year_of(row.get(spec["year_col"]))
    head = str(row[spec["title_col"]]) + (f" ({y})" if y else "")
    return head, sr.text(row.get(spec["people_col"]))


# --- callbacks --------------------------------------------------------------
def _add_rating(b, pos, key):
    val = _ss().get(key)
    if val is None:
        return
    r = sr.catalogue_rating(b, pos, float(val))
    idk = b["spec"]["rating_id"]
    if any(str(x.get(idk)) == str(r[idk]) for x in _ss().ratings):
        return
    _ss().ratings.append(r)


def _remove(b, rid):
    idk = b["spec"]["rating_id"]
    _ss().ratings = [x for x in _ss().ratings if str(x.get(idk)) != str(rid)]


def _reset_chat():
    _ss().chat = []
    _ss().agent_history = []
    _ss().agent_state = {}       # drops the seen-titles ledger with the conversation


def _clear():
    _ss().ratings = []
    _ss().skipped = []
    _ss().load_note = ""
    _reset_chat()


def _load_profile(b, raw, label):
    try:
        ratings, skipped = sr.read_profile(raw, b["kind"])
    except Exception as e:
        _ss().load_note = f"Couldn't read {label}: {e}"
        return
    clean, note = sr.normalize_ratings(b, ratings)
    _ss().ratings = clean
    _ss().skipped = list(skipped)
    _reset_chat()
    msg = f"Loaded {len(clean)} ratings"
    if skipped:
        msg += f" and {len(skipped)} 'not interested'"
    msg += f" from {label}."
    if note:
        msg += f" ({note})"
    _ss().load_note = msg


def _load_default(b):
    _load_profile(b, sr.DEFAULT_PROFILE.read_text(encoding="utf-8"), sr.DEFAULT_PROFILE.name)


def _render_check(check):
    if not check:
        return
    issues = check.get("issues") or []
    if issues:
        parts = [f"**{i['title']}** ("
                 + ("in the catalogue, but no tool returned it" if i["in_catalogue"]
                    else "not in the catalogue") + ")"
                 for i in issues]
        st.warning("Grounding check: " + "; ".join(parts))
    elif check.get("corrected"):
        st.caption("✓ Grounding check: caught an unverified title and fixed it before answering.")
    else:
        st.caption("✓ Grounding check: every title named came from a tool or your ratings.")


# --- page -------------------------------------------------------------------
def run(kind, run_agent):
    L = LABELS[kind]
    st.set_page_config(page_title=f"{L['title']} — Agent", page_icon=L["icon"], layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        b = _load(kind)
    except Exception as e:
        st.error(f"Could not load the {L['noun']} catalogue: {e}")
        st.stop()

    ss = _ss()
    for k, v in (("ratings", []), ("skipped", []), ("chat", []),
                 ("agent_history", []), ("agent_state", {}), ("load_note", "")):
        if k not in ss:
            ss[k] = v
    spec, idk = b["spec"], b["spec"]["rating_id"]

    st.title(f"{L['title']} — ask the agent")
    lit = ", ".join(f"{k}={v}" for k, v in sorted(b["call_literal"].items()))
    st.caption(f"{len(b['df']):,} {L['noun']} · ranked by the Space's own "
               f"`{spec['rank']}({lit})` · corpus `{b['files'].get('CORPUS')}`")
    if b["call_problems"]:
        st.warning("The Space passes arguments the agent can't copy: "
                   + "; ".join(b["call_problems"]))

    left, right = st.columns([1, 1.4], gap="large")

    with left:
        st.subheader("Your taste profile")
        c1, c2 = st.columns(2)
        if sr.DEFAULT_PROFILE.exists():
            c1.button("Load my profile", use_container_width=True,
                      help=f"Reads {sr.DEFAULT_PROFILE}", on_click=_load_default, args=(b,))
        c2.button("Clear all", use_container_width=True, on_click=_clear)
        up = st.file_uploader("Or import a profile exported from the website", type=["json"])
        if up is not None:
            tag = f"_imported_{up.name}_{up.size}"
            if not ss.get(tag):
                ss[tag] = True
                _load_profile(b, up.getvalue().decode("utf-8", "replace"), up.name)
        if ss.load_note:
            st.info(ss.load_note)

        q = st.text_input(L["search"], key="search_box", placeholder=L["ph"])
        if q:
            hits = b["ns"][spec["search"]](q, b["df"], b["search_idx"], limit=12)
            rated = {str(r.get(idk)): r["rating"] for r in ss.ratings}
            if not len(hits):
                st.caption("No matches.")
            for pos in hits.index:
                head, who = _label(b, pos)
                rid = b["df"].iloc[int(pos)][spec["id_col"]]
                a, c = st.columns([3, 1.2])
                a.markdown(f"**{head}**" + (f"  \n*{who}*" if who else ""))
                if str(rid) in rated:
                    c.caption(f"rated {int(rated[str(rid)])}/5")
                else:
                    key = f"rate_{rid}"
                    c.selectbox("Rate", [None, 1, 2, 3, 4, 5], key=key,
                                label_visibility="collapsed",
                                on_change=_add_rating, args=(b, int(pos), key))

        st.markdown("---")
        if ss.ratings:
            likes = sum(1 for r in ss.ratings if r["rating"] >= 4)
            extra = f" · {len(ss.skipped)} not interested" if ss.skipped else ""
            st.markdown(f"**{len(ss.ratings)} rated** · {likes} at 4★ or more{extra}")
            with st.expander("Show ratings", expanded=len(ss.ratings) <= 15):
                for r in sorted(ss.ratings, key=lambda x: -x["rating"]):
                    rid = r.get(idk)
                    who = html.escape(str(r.get(spec["people_key"]) or ""))
                    x1, x2 = st.columns([5, 1])
                    x1.markdown(
                        f"<div class='card'><span class='card-title'>"
                        f"{html.escape(str(r['matched_title']))}</span><br>"
                        f"<span class='card-sub'>{who}</span> · "
                        f"<span style='color:#c9a86a'>{int(r['rating'])}/5</span></div>",
                        unsafe_allow_html=True)
                    x2.button("X", key=f"rm_{rid}", on_click=_remove, args=(b, rid))
        else:
            st.info(f"Rate some {L['noun']}, or load the profile you exported from the website.")

    with right:
        st.subheader("Ask the agent")
        st.caption(L["tips"])
        for msg in ss.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                _render_check(msg.get("check"))

        prompt = st.chat_input(L["chat"])
        if prompt:
            if not ss.ratings:
                st.warning(f"Rate a few {L['noun']} first so the agent knows your taste.")
                return
            ss.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                check = None
                with st.spinner("Thinking (tool trace in the terminal)..."):
                    state = ss.agent_state
                    state.update(bundle=b, ratings=ss.ratings, skipped=ss.skipped)
                    state.pop("_last_check", None)
                    try:
                        reply, ss.agent_history = run_agent(
                            prompt, state, conversation_history=ss.agent_history, verbose=True)
                        check = state.get("_last_check")
                    except Exception as e:
                        reply = f"Something went wrong: {type(e).__name__}: {e}"
                st.markdown(reply)
                _render_check(check)
            ss.chat.append({"role": "assistant", "content": reply, "check": check})
