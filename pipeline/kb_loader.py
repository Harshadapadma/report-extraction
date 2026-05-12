"""
pipeline/kb_loader.py
─────────────────────
Sector-aware knowledge-base loader.

Combines the base `web/financial_kb.json` (Ind AS general industry) with the
sector-specific overlay (`financial_kb_banks.json`, `financial_kb_nbfc.json`,
`financial_kb_insurance.json`) into a unified KB dict + alias index that the
mapping agents use.

Design notes
────────────
• Base + overlay are MERGED; sector-overlay entries override base entries on
  conflict (e.g. an NBFC's "interest income" definition wins over the generic).
• The merged KB is cached per-sector so repeated calls don't re-parse JSON.
• `kb_entry_for(label, sector)` is a drop-in replacement for the legacy
  `web.financial_mapping_agent._kb_entry` that's sector-aware.
• `kb_context_for(label, sector)` returns the short prompt-ready fact string.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path resolution
_ROOT     = Path(__file__).parent.parent
_BASE_KB  = _ROOT / "web" / "financial_kb.json"
_OVERLAYS = {
    "bank":      _ROOT / "web" / "financial_kb_banks.json",
    "nbfc":      _ROOT / "web" / "financial_kb_nbfc.json",
    "insurance": _ROOT / "web" / "financial_kb_insurance.json",
}


# ── Loading ──────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"kb_loader: failed to read {path}: {exc}")
        return {}


@lru_cache(maxsize=None)
def load_merged_kb(sector: str = "general") -> dict:
    """
    Return the merged KB for the given sector (base + overlay).
    Strips any '_meta' top-level key so callers get only concept entries.
    """
    base = _load_json(_BASE_KB) or {}

    overlay_path = _OVERLAYS.get(sector)
    overlay = _load_json(overlay_path) if overlay_path else {}

    merged = {}
    for k, v in base.items():
        if k.startswith("_"):
            continue
        merged[k] = v

    # Overlay wins on conflict
    for k, v in overlay.items():
        if k.startswith("_"):
            continue
        merged[k] = v

    return merged


@lru_cache(maxsize=None)
def alias_index(sector: str = "general") -> dict[str, str]:
    """
    Build a {alias_lowercased → canonical_key} lookup over the merged KB.
    """
    kb = load_merged_kb(sector)
    idx: dict[str, str] = {}
    for ck, info in kb.items():
        idx[ck.lower()] = ck
        # Also index a humanised version of the canonical key
        humanised = ck.replace("_", " ")
        idx.setdefault(humanised.lower(), ck)
        for a in info.get("aliases", []):
            idx.setdefault(str(a).lower().strip(), ck)
    return idx


# Operator-prefix / readability noise patterns — stripped before KB lookup
# so labels like "Less: Tax" or "Add: Other Income (Net)" resolve cleanly.
_OPERATOR_PREFIX = re.compile(
    r"^\s*(?:less\s*:|add\s*:|plus\s*:|minus\s*:)\s*",
    re.IGNORECASE,
)
_NOISE_SUFFIX_PATTERNS = [
    re.compile(r"\(net\)\s*$", re.IGNORECASE),
    re.compile(r"\(gross\)\s*$", re.IGNORECASE),
    re.compile(r"\(consolidated\)\s*$", re.IGNORECASE),
    re.compile(r"\(standalone\)\s*$", re.IGNORECASE),
    re.compile(r"&\s*impairment\b", re.IGNORECASE),
]


def _strip_label_noise(label: str) -> str:
    s = _OPERATOR_PREFIX.sub("", label or "")
    for pat in _NOISE_SUFFIX_PATTERNS:
        s = pat.sub("", s)
    return s.strip()


# ── Public lookups ───────────────────────────────────────────────────────────

def _morphological_variants(s: str) -> list[str]:
    """Return label variants — singular/plural, with/without "expense"/"cost"
    suffix etc. Lets us match 'Tax' to 'tax expense' and 'Finance cost' to
    'finance costs'."""
    s = (s or "").strip().lower()
    if not s:
        return []
    variants = {s}
    # Plural ↔ singular on last word
    words = s.split()
    if words:
        last = words[-1]
        if last.endswith("s") and not last.endswith("ss") and len(last) > 3:
            variants.add(" ".join(words[:-1] + [last[:-1]]))
        else:
            variants.add(" ".join(words[:-1] + [last + "s"]))
    # Common pairings — Tax ↔ Tax Expense; Finance ↔ Finance Cost
    if not s.endswith(("expense", "expenses")):
        variants.add(f"{s} expense")
        variants.add(f"{s} expenses")
    if not s.endswith(("cost", "costs")):
        variants.add(f"{s} cost")
        variants.add(f"{s} costs")
    return list(variants)


def kb_entry_for(label: str, sector: str = "general") -> Optional[dict]:
    """Return the KB entry for *label*, considering sector overlay first."""
    if not label:
        return None
    # Strip operator prefixes / noise suffixes first ("Less: Tax" → "Tax")
    label_stripped = _strip_label_noise(label)
    key_lc = label_stripped.lower().strip()
    if not key_lc:
        return None
    idx = alias_index(sector)
    kb  = load_merged_kb(sector)

    # 1. Exact alias match (also try morphological variants)
    if key_lc in idx:
        return kb.get(idx[key_lc])
    for variant in _morphological_variants(key_lc):
        if variant in idx:
            return kb.get(idx[variant])

    # 2. Token overlap fallback — raised to 0.75 to stop false positives.
    #
    # The old 0.60 threshold caused dangerous mis-mappings, e.g.:
    #   "Total Liabilities"   → total_borrowings  (shared word "total" + "liabilities")
    #   "Working Capital Days" → short_term_borrowings (shared "working capital")
    #   "Gross Margin %"       → gross_profit (shared "gross")
    #
    # A higher threshold (0.75) keeps clearly related pairs (e.g. "Interest
    # Expense" → finance_costs) while blocking accidental overlaps.
    # NOTE: The exact-alias path above handles all legitimate partial-name
    # matches already; the overlap path is only a last-resort safety net.
    words = set(re.split(r"[\s/\-_,():]+", key_lc))
    words = {w for w in words if w and len(w) > 1}  # skip single-char tokens
    if not words:
        return None

    best_ck: Optional[str] = None
    best_score = 0.0
    for alias_lc, ck in idx.items():
        alias_words = set(re.split(r"[\s/\-_,():]+", alias_lc))
        alias_words = {w for w in alias_words if w and len(w) > 1}
        if not alias_words:
            continue
        overlap = len(words & alias_words) / max(len(alias_words), len(words))
        if overlap > best_score:
            best_score, best_ck = overlap, ck

    if best_score >= 0.75 and best_ck:
        return kb.get(best_ck)
    return None


def kb_context_for(label: str, sector: str = "general") -> str:
    """Compose a short fact-string for *label* suitable for inline LLM prompts."""
    entry = kb_entry_for(label, sector)
    if not entry:
        return ""
    parts: list[str] = []
    if entry.get("definition"):
        parts.append(f"Def: {entry['definition']}")
    if entry.get("formula"):
        f = entry["formula"]
        f_str = f if isinstance(f, str) else " + ".join(f) if isinstance(f, list) else str(f)
        parts.append(f"Formula: {f_str}")
    if entry.get("ind_as_note"):
        parts.append(f"Ind AS: {entry['ind_as_note']}")
    if entry.get("never_confuse_with"):
        parts.append(f"NEVER confuse with: {', '.join(entry['never_confuse_with'])}")
    if entry.get("statement") and entry.get("section"):
        parts.append(f"Belongs to: {entry['statement']} / {entry['section']}")
    if entry.get("is_percentage"):
        rng = entry.get("expected_range")
        if rng:
            parts.append(f"% expected in {rng[0]}–{rng[1]}")
    return " | ".join(parts)


def all_canonical_keys(sector: str = "general") -> list[str]:
    """List every canonical key in the merged KB for *sector*."""
    return list(load_merged_kb(sector).keys())


def all_aliases(sector: str = "general") -> list[str]:
    """List every alias known to the merged KB for *sector*."""
    return list(alias_index(sector).keys())


def reset_cache() -> None:
    """Drop cached KBs (useful in tests)."""
    load_merged_kb.cache_clear()
    alias_index.cache_clear()


__all__ = [
    "load_merged_kb",
    "alias_index",
    "kb_entry_for",
    "kb_context_for",
    "all_canonical_keys",
    "all_aliases",
    "reset_cache",
]
