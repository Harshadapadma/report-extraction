"""
pipeline/
─────────
Pure business logic — no Streamlit, no UI.

Modules
───────
mapping.py   — label mapping (unified LLM+KB, smart_map_sheet helpers)
pool.py      — pool building and deduplication

web_only_app.py imports from here, so all logic is testable independently.
"""
from pipeline.mapping import llm_map_with_kb

__all__ = ["llm_map_with_kb"]
