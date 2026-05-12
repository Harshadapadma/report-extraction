"""
pipeline/pool.py
────────────────
Pool building helpers — pure logic, no Streamlit.
Imported by web_only_app._pool_web_labels for the OCR garbage filter.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

_OCR_SPLIT_PAT = re.compile(r'[a-z] [a-z]{2,}')

_NOTES_JUNK_PHRASES = frozenset([
    "attending every meeting", "options were exercised", "options were granted",
    "not applicable", "refer note", "per the", "as per the",
    "in years", "year 1", "year 2", "year 3", "year 4", "year 5",
    "subtotal", "opening balance", "closing balance",
    "weighted average", "exercise price", "grant date",
    "defined benefit", "post-retirement", "pension plan",
    "sensitivity analysis", "discount rate", "mortality rate",
    "not later than", "later than one", "financial year ended",
])


def is_ocr_garbage(label: str) -> bool:
    """Return True if label is an OCR word-split artifact or notes junk."""
    if not label or len(label) < 3:
        return True
    ll = label.lower().strip()
    if _OCR_SPLIT_PAT.search(ll) and len(label) > 35:
        return True
    if len(label) > 90:
        return True
    if any(phrase in ll for phrase in _NOTES_JUNK_PHRASES):
        return True
    if re.match(r'^[\(\[ivxIVX\d\s\)\]]+$', label.strip()):
        return True
    return False


def clean_label_dict(
    data: dict[str, dict],
    filter_ocr: bool = False,
) -> dict:
    """Filter garbage labels from a {label: {year: val}} dict."""
    if not filter_ocr:
        return data
    return {k: v for k, v in data.items() if not is_ocr_garbage(k)}
