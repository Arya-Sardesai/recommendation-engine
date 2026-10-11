"""
Film recommendation agent — local chat app.

Run from the repo root:
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    python -m streamlit run src/app_agent_movies.py

Reads the files hf-space/app.py serves (its M_* constants)
from data/processed, and ranks with the Space's own recommend_* function.
The page itself lives in agent_ui.py, shared by all three apps.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent_ui  # noqa: E402
from agent_movies import run_agent  # noqa: E402

agent_ui.run("movie", run_agent)
