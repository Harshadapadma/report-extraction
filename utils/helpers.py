"""
Utility helpers shared across the extraction pipeline.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from config.settings import YEAR_ALIASES, QUARTER_ALIASES

logger = logging.getLogger(__name__)


# ── Logging setup ──────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured logger."""
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    return log


# ── Year / period normalisation ────────────────────────────────────────────────

def normalise_year(raw: str) -> Optional[str]:
    """
    Convert a raw year string extracted from a document into the template
    column header format (e.g. 'FY2025', '2024-25' → 'F2025').

    Returns None if the string cannot be matched.
    """
    if not raw:
        return None
    key = raw.strip().lower()

    # Direct alias lookup
    if key in YEAR_ALIASES:
        return YEAR_ALIASES[key]

    # Quarterly alias lookup
    if key in QUARTER_ALIASES:
        return QUARTER_ALIASES[key]

    # Pattern: F2025 / F2024 etc. already in template format
    if re.match(r"^f\d{4}$", key, re.IGNORECASE):
        return key.upper()

    # Pattern: FY25 / FY2025
    m = re.match(r"^fy(\d{2,4})$", key, re.IGNORECASE)
    if m:
        yr = m.group(1)
        if len(yr) == 2:
            yr = "20" + yr
        return f"F{yr}"

    # Pattern: FY2024-25 or FY2024-2025 (common in Indian Excel templates)
    m = re.match(r"^fy(\d{4})[-/](\d{2,4})$", key, re.IGNORECASE)
    if m:
        end = m.group(2)
        if len(end) == 2:
            end = m.group(1)[:2] + end
        return f"F{end}"

    # Pattern: 2024-25 or 2024-2025
    m = re.match(r"^(\d{4})[-/](\d{2,4})$", key)
    if m:
        end = m.group(2)
        if len(end) == 2:
            end = m.group(1)[:2] + end
        return f"F{end}"

    # Pattern: year ended march 31, 2025
    m = re.search(r"march\s+31,?\s+(\d{4})", key, re.IGNORECASE)
    if m:
        return f"F{m.group(1)}"

    # Pattern: "Mar 2025" / "March 2025" (4-digit year) → annual F-year.
    # "Mar-25" / "Mar 25" (2-digit year) falls through to normalise_quarter → 4QF2025.
    # Heuristic: 4-digit year in an annual context means the fiscal year; 2-digit means quarter header.
    m = re.match(r"^mar(?:ch)?[-\s](\d{4})$", key, re.IGNORECASE)
    if m:
        return f"F{m.group(1)}"

    # Pattern: "F.Y. 2025" / "F.Y.2025" / "F.Y.25" — dotted abbreviation in templates
    m = re.match(r"^f\.y\.?\s*(\d{2,4})$", key, re.IGNORECASE)
    if m:
        yr = m.group(1)
        if len(yr) == 2:
            yr = "20" + yr
        return f"F{yr}"

    # Pattern: "F.Y. 2024-25" / "F.Y.2024-25"
    m = re.match(r"^f\.y\.?\s*(\d{4})[-/](\d{2,4})$", key, re.IGNORECASE)
    if m:
        end = m.group(2)
        if len(end) == 2:
            end = m.group(1)[:2] + end
        return f"F{end}"

    # Pattern: plain 4-digit year 2025 — only accept plausible fiscal years
    m = re.match(r"^(\d{4})$", key)
    if m:
        yr = int(m.group(1))
        if 2000 <= yr <= 2040:
            return f"F{yr}"
        return None   # reject e.g. "1000", "9999", "1234"

    # Last resort: try quarterly period (e.g. "Mar-25" → "4QF2025")
    # This allows read_template_structure to recognise quarter column headers in templates.
    return normalise_quarter(raw)


# ── Quarter normalisation ──────────────────────────────────────────────────────

# Month → (quarter_num, add_year):
#   Q1 = Apr/May/Jun  (FY starts Apr, add_year=1 to convert calendar year → FY)
#   Q2 = Jul/Aug/Sep  (add_year=1)
#   Q3 = Oct/Nov/Dec  (add_year=1)
#   Q4 = Jan/Feb/Mar  (add_year=0 — same calendar year as FY end)
_MONTH_TO_QTR: dict[str, tuple[int, int]] = {
    "jan": (4, 0), "feb": (4, 0), "mar": (4, 0),
    "apr": (1, 1), "may": (1, 1), "jun": (1, 1),
    "jul": (2, 1), "aug": (2, 1), "sep": (2, 1),
    "oct": (3, 1), "nov": (3, 1), "dec": (3, 1),
}


def normalise_quarter(raw: str) -> Optional[str]:
    """
    Convert a raw quarterly period string to the canonical ``NQF YYYY`` format.

    Examples
    --------
    "Mar-25"   -> "4QF2025"
    "Jun-24"   -> "1QF2025"   (Jun 2024 is Q1 of FY2025)
    "Sep 2024" -> "2QF2025"
    "Dec 2024" -> "3QF2025"
    "Q4FY25"   -> "4QF2025"
    "Q1FY24"   -> "1QF2024"
    "1QF2025"  -> "1QF2025"   (already canonical)
    Returns None if the string is not recognisable as a quarter.
    """
    if not raw:
        return None
    t = raw.strip().lower()

    # Already canonical: "1QF2025" / "4QF2025"
    m = re.match(r"^(\d)qf(\d{4})$", t)
    if m:
        return f"{m.group(1)}QF{m.group(2)}"

    # Q4FY25 / Q1FY2024 / "Q4 FY25" style
    m = re.match(r"^q(\d)\s*fy\s*(\d{2,4})$", t)
    if m:
        q  = m.group(1)
        yr = m.group(2)
        yr = ("20" + yr) if len(yr) == 2 else yr
        return f"{q}QF{yr}"

    # 4QFY25 / 1QFY2024 style (digit+Q then FY)
    m = re.match(r"^(\d)q\s*fy\s*(\d{2,4})$", t)
    if m:
        q  = m.group(1)
        yr = m.group(2)
        yr = ("20" + yr) if len(yr) == 2 else yr
        return f"{q}QF{yr}"

    # Month[-/space]YY or Month[-/space]YYYY: "Mar-25", "Jun 24", "Sep 2024"
    m = re.match(
        r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        r"[-\s](\d{2,4})$",
        t,
    )
    if m:
        mon    = m.group(1)
        yr_raw = m.group(2)
        yr_int = int(yr_raw) if len(yr_raw) == 4 else (
            2000 + int(yr_raw) if int(yr_raw) < 50 else 1900 + int(yr_raw)
        )
        q_num, add_yr = _MONTH_TO_QTR[mon]
        return f"{q_num}QF{yr_int + add_yr}"

    return None


def extract_all_years(text: str) -> list[str]:
    """Return all unique template-format year labels found in a block of text."""
    found: set[str] = set()

    # Look for FY25, FY2025 patterns
    for m in re.finditer(r"\bFY\s*(\d{2,4})\b", text, re.IGNORECASE):
        y = normalise_year(f"FY{m.group(1)}")
        if y:
            found.add(y)

    # Look for 2024-25 patterns
    for m in re.finditer(r"\b(\d{4})[-/](\d{2,4})\b", text):
        y = normalise_year(m.group(0))
        if y:
            found.add(y)

    # Look for "March 31, 2025"
    for m in re.finditer(r"March\s+31,?\s+(\d{4})", text, re.IGNORECASE):
        y = normalise_year(f"march 31, {m.group(1)}")
        if y:
            found.add(y)

    return sorted(found)


# ── Unit detection ────────────────────────────────────────────────────────────

# Each tuple: (compiled_regex, divisor_to_convert_to_crores)
# Ordered from most-specific to least-specific so the first match wins.
_UNIT_PATTERNS: list[tuple] = [
    # Already in crores — no conversion needed
    (re.compile(r'(?:rs\.?|₹|inr|amount)\s*in\s*(?:indian\s+)?(?:rs\.?\s*)?crore', re.I), 1.0),
    (re.compile(r'\bcrore', re.I), 1.0),

    # Millions → crores: 1 crore = 10 million (approx for Indian Rupee context)
    (re.compile(r'(?:rs\.?|₹|inr)\s*in\s*million', re.I), 10.0),
    (re.compile(r'\brs\.\s*million\b', re.I), 10.0),

    # Lakhs → crores: 1 crore = 100 lakhs
    (re.compile(r'(?:rs\.?|₹|inr|amount)\s*in\s*(?:rs\.?\s*)?lakh', re.I), 100.0),
    (re.compile(r'\(?rs\.?\s*lakh\)?', re.I), 100.0),
    (re.compile(r'\blakh', re.I), 100.0),

    # Thousands → crores: 1 crore = 10,000 thousands
    (re.compile(r'(?:rs\.?|₹|inr|amount)\s*in\s*(?:rs\.?\s*)?thousand', re.I), 10_000.0),
    (re.compile(r'\(?(?:rs\.?|₹)\s*000\)?', re.I), 10_000.0),          # (₹ 000) or (Rs. 000)
    (re.compile(r'₹\s*in\s*000', re.I), 10_000.0),
    (re.compile(r'\bthousand', re.I), 10_000.0),
    # "000's of ₹" / "000's of `" — curly/straight apostrophe variants
    (re.compile(r"000[‘’ʼ`'`]\s*s\s+of", re.I), 10_000.0),
    (re.compile(r"in\s+000[‘’ʼ`'`]\s*s\b", re.I), 10_000.0),
    (re.compile(r"'000", re.I), 10_000.0),                               # '000
    (re.compile(r'\b000s\b', re.I), 10_000.0),                           # 000s
    (re.compile(r'amounts?\s+in\s+000', re.I), 10_000.0),               # "amounts in 000"
]


def detect_unit_divisor(text: str) -> float:
    """
    Scan ``text`` for a currency-unit declaration and return the divisor
    that converts reported values into INR Crores.

    Searches progressively larger windows so that unit declarations buried
    after titles / disclaimers are still found.

    Examples
    --------
    "₹ in Thousands"  → 10_000.0   (divide all values by 10,000)
    "Rs. in Lakhs"    → 100.0
    "₹ in Crores"     → 1.0        (already crores, no change)
    Not found         → 1.0        (assume crores)
    """
    # Try progressively larger windows. Crore/lakh declarations almost always
    # appear within the first few thousand characters of a section, but some PDFs
    # bury them after long preambles.
    for window in (2_000, 5_000, 15_000, len(text)):
        sample = text[:window]
        for pattern, divisor in _UNIT_PATTERNS:
            if pattern.search(sample):
                return divisor
    return 1.0   # default: assume values are already in crores


# ── Numeric helpers ────────────────────────────────────────────────────────────

def parse_number(raw: str) -> Optional[float]:
    """
    Parse a financial number string to float.
    Handles commas, brackets (negatives), crore suffixes, etc.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in ("-", "—", "N/A", "NA", "Nil", "nil", "–"):
        return None

    negative = False

    # Bracketed negatives: (1,234.56)
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
        negative = True

    # Remove currency symbols and spaces
    s = re.sub(r"[₹$€£,\s]", "", s)

    # Handle Cr / Crores / Mn / Bn suffixes
    multiplier = 1.0
    m = re.match(r"^([\d.]+)\s*(Cr|Crs|Crore|Crores|Mn|Mn\.|Bn)?$", s, re.IGNORECASE)
    if m:
        s = m.group(1)
        suffix = (m.group(2) or "").lower()
        if suffix in ("mn", "mn."):
            multiplier = 0.1       # convert Mn to Crores (1 Cr = 10 Mn roughly ... but context dependent)
        elif suffix in ("bn",):
            multiplier = 100.0     # Bn → Crores
        # 'cr', 'crs', etc. → already in crores, multiplier stays 1

    try:
        val = float(s) * multiplier
        return -val if negative else val
    except ValueError:
        return None


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_label(label: str) -> str:
    """Normalise a row/field label for comparison."""
    if not label:
        return ""
    label = label.strip()
    label = re.sub(r"\s+", " ", label)
    # Remove leading dashes / bullets
    label = re.sub(r"^[-–—•·]+\s*", "", label)
    return label


def clean_for_llm(text: str) -> str:
    """
    Strip boilerplate that wastes tokens before sending to an LLM:
    - Blank lines (collapse to single blank)
    - Lines that are purely dashes / dots / underscores (table rules)
    - Page headers that repeat on every page (CIN numbers, company names, etc.)
    - Lines shorter than 3 chars (noise)
    Result: typically 30-40% fewer characters → faster + cheaper LLM calls.
    """
    _JUNK_LINE = re.compile(
        r"^[\s\-–—_=\.·•|/\\]{3,}$"         # rule lines
        r"|^\s*\d+\s*$"                        # bare page numbers
        r"|^.{0,2}$"                           # very short lines
    )
    lines = text.splitlines()
    cleaned: list[str] = []
    prev_blank = False
    for line in lines:
        stripped = line.strip()
        if _JUNK_LINE.match(stripped):
            continue
        is_blank = not stripped
        if is_blank and prev_blank:
            continue          # collapse consecutive blanks
        cleaned.append(stripped if stripped else "")
        prev_blank = is_blank
    return "\n".join(cleaned)


def chunk_text(text: str, max_chars: int = 12_000) -> list[str]:
    """
    Split a long text into overlapping chunks for LLM processing.
    Splits on newlines to avoid breaking numbers mid-line.
    """
    lines = text.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        if current_len + len(line) > max_chars and current:
            chunks.append("\n".join(current))
            # Keep last 20 lines as context overlap
            current = current[-20:]
            current_len = sum(len(l) for l in current)
        current.append(line)
        current_len += len(line)

    if current:
        chunks.append("\n".join(current))

    return chunks
