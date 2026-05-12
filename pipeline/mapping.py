"""
pipeline/mapping.py
───────────────────
Core financial label mapping logic — fully decoupled from Streamlit.

Key functions
─────────────
llm_map_with_kb()       — unified LLM + knowledge-base mapper (replaces
                          separate FinancialMappingAgent + llm_align_labels)
smart_map_sheet()       — full multi-stage sheet mapper
targeted_pdf_fill()     — Phase 4: LLM searches PDF text for still-empty rows
llm_map_fields()        — full structured-data LLM mapping (fallback)

All functions accept a plain openai.OpenAI (or compatible) client.
No Streamlit imports anywhere in this file.
"""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from web.financial_mapping_agent import _kb_entry as _kb_entry_fn
    _KB_AVAILABLE = True
except ImportError:
    _KB_AVAILABLE = False
    def _kb_entry_fn(label: str):   # type: ignore[misc]
        return None

try:
    from web.canonical_mapper import map_to_canonical as _canon_map
    _CANONICAL_AVAILABLE = True
except ImportError:
    _CANONICAL_AVAILABLE = False
    def _canon_map(label, source="_all", min_confidence=0.50):  # type: ignore[misc]
        return None, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Unified LLM + Knowledge-Base mapper
# ─────────────────────────────────────────────────────────────────────────────

def llm_map_with_kb(
    sheet_name: str,
    unresolved_labels: list[str],
    pool_labels: list[str],
    pool: dict[str, dict[str, Any]],
    years: list[str],
    sector: str,
    client: Any,
    model: str,
    row_meta: dict[str, dict] | None = None,
    batch_size: int = 55,
) -> dict[str, str | None]:
    """
    Single-call KB-aware LLM mapping.

    Replaces two separate stages:
      • FinancialMappingAgent  (per-label, N API calls)
      • llm_align_labels       (batch, but no KB context)

    Now: ONE batch call where each template label has its KB facts embedded
    inline. The LLM sees all labels simultaneously so it can avoid duplicate
    pool-label assignments and use financial reasoning for every row.

    Returns {template_label: best_pool_label_or_None}
    """
    if not unresolved_labels or not pool_labels or client is None:
        return {}

    row_meta = row_meta or {}
    sample_yrs = years[:2]

    # ── Build pool label block (with sample values) ───────────────────────────
    # Pre-rank pool by string similarity to the batch as a whole (faster for LLM)
    from difflib import SequenceMatcher as _SM
    all_tmpl = " ".join(unresolved_labels).lower()
    ranked_pool = sorted(
        pool_labels,
        key=lambda pl: _SM(None, all_tmpl, pl.lower()).ratio(),
        reverse=True,
    )[:120]   # cap at 120 to stay within context budget

    pool_lines = []
    for i, pl in enumerate(ranked_pool):
        samples = ", ".join(
            f"{yr}={pool[pl][yr]:,.0f}"
            for yr in sample_yrs
            if yr in pool.get(pl, {}) and pool[pl][yr] is not None
        )
        pool_lines.append(f"P{i+1}: {pl}" + (f"  [{samples}]" if samples else ""))
    pool_block = "\n".join(pool_lines)
    pl_keys = {f"P{i+1}": pl for i, pl in enumerate(ranked_pool)}

    results: dict[str, str | None] = {}

    # ── Process in batches ────────────────────────────────────────────────────
    for batch_start in range(0, len(unresolved_labels), batch_size):
        batch = unresolved_labels[batch_start: batch_start + batch_size]

        # Build template label block with KB facts inline
        tmpl_lines = []
        for i, lbl in enumerate(batch, 1):
            meta    = row_meta.get(lbl, {})
            section = meta.get("section", "")
            kb      = _kb_entry_fn(lbl) if _KB_AVAILABLE else None

            kb_parts = []
            if kb:
                if kb.get("definition"):
                    kb_parts.append(f"Def: {kb['definition']}")
                if kb.get("formula"):
                    kb_parts.append(f"Formula: {kb['formula']}")
                if kb.get("ind_as_note"):
                    kb_parts.append(f"Ind AS: {kb['ind_as_note']}")
                if kb.get("never_confuse_with"):
                    kb_parts.append(f"NEVER confuse with: {', '.join(kb['never_confuse_with'])}")
                if kb.get("statement") and kb.get("section"):
                    kb_parts.append(f"Belongs to: {kb['statement']} / {kb['section']}")

            line = f"{i}. \"{lbl}\""
            if section:
                line += f"  [section: {section}]"
            if kb_parts:
                line += "\n   KB: " + " | ".join(kb_parts)
            tmpl_lines.append(line)

        tmpl_block = "\n".join(tmpl_lines)

        prompt = f"""You are a financial data expert for Indian listed companies (Ind AS / IGAAP).

Map {len(batch)} template labels to the BEST matching pool label for "{sheet_name}" ({sector} company).

TEMPLATE LABELS (with financial knowledge-base context):
{tmpl_block}

POOL LABELS (P-key: label [sample values in INR Crores]):
{pool_block}

RULES
1. Each pool label (P-key) may be used AT MOST ONCE across all mappings.
2. Use KB facts to resolve ambiguity — e.g. if KB says "NEVER confuse with X", don't pick X.
3. Section context is binding — a NON-CURRENT ASSETS label must match a non-current pool item.
4. If no pool label is a genuine match, return null (do NOT force a wrong match).
5. Prefer the pool label whose magnitude matches what this item should be.

Return a JSON array, one entry per template label in order:
[{{"t": "<template label>", "m": "<exact pool label string or null>"}}, ...]"""

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content

            # Response might be wrapped in {"mappings": [...]} or just [...]
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                arr = parsed.get("mappings") or parsed.get("items") or list(parsed.values())[0]
            else:
                arr = parsed

            # Build a lookup from P-key → actual label
            for entry in arr:
                t = entry.get("t", "")
                m = entry.get("m")
                if not t:
                    continue
                # m could be a P-key ("P12") or the actual label string
                if isinstance(m, str):
                    if m in pl_keys:
                        m = pl_keys[m]
                    # Verify it's actually in the pool
                    if m not in pool:
                        m = None
                results[t] = m if m else None

        except Exception as exc:
            logger.warning(f"llm_map_with_kb batch [{batch_start}] failed: {exc}")
            for lbl in batch:
                results.setdefault(lbl, None)

    return results
