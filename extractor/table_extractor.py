"""
Table Extractor — direct PDF table parsing using pdfplumber.

Generic fixes applied (works for any company's annual report):

  1. Auto-crop sidebars  — detects narrow right/left margin banners by checking
     for a small cluster of words with (a) no numeric characters, (b) narrow
     x-range, (c) tall y-span — the signature of rotated section-label text.
     Never crops value columns, which always contain numbers.

  2. Three extraction strategies per page (in order):
       a. pdfplumber text-strategy   — whitespace-aligned PDFs (most common)
       b. pdfplumber line-strategy   — grid/boxed PDFs
       c. Word-position spatial grouping — last resort for unusual layouts

  3. Post-process rows to fix split cells:
       • Joins adjacent cells whose concatenation forms a valid number
         (e.g. "6,31,17,25" + "0" → "6,31,17,250")
       • Joins adjacent label fragments into a single label string
       • Splits cells where pdfplumber MERGED two adjacent columns into one
         - Concatenated year headers: "2024-252023-24" → ["2024-25","2023-24"]
         - Concatenated values: "3,251.613,079.76" → ["3,251.61","3,079.76"]

  4. Multi-row year header detection:
       If the standard single-row scan finds no year, the first 15 rows are
       combined column-by-column and re-searched (handles "Year ended" on row N
       and "March 31, 2025" on row N+1).

  5. Year range filter: only F2005–F2035 accepted; F1961, F5000, etc. rejected.

  6. ESG/CSR page filter: sustainability and non-financial report pages skipped.

  7. Unit detection covers all common Indian formats including
     "000's of ₹" / "000's of `" (curly apostrophe variant).

Values come directly from the PDF's own cells — no LLM for value extraction.
"""

from __future__ import annotations

import io as _io
import re
import subprocess
import sys
from collections import defaultdict
from typing import Optional

from utils.helpers import get_logger, parse_number, normalise_year, detect_unit_divisor
from extractor.llm_extractor import ExtractionResult

logger = get_logger(__name__)


# ── Auto-install pdfplumber ────────────────────────────────────────────────────

def _require_pdfplumber():
    try:
        import pdfplumber as _p
        return _p
    except ImportError:
        logger.info("Installing pdfplumber…")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "pdfplumber>=0.10.0"],
        )
        import importlib, site
        importlib.invalidate_caches()
        for sp in site.getsitepackages():
            if sp not in sys.path:
                sys.path.insert(0, sp)
        import pdfplumber as _p
        return _p


# ── Valid fiscal year range ────────────────────────────────────────────────────
# Rejects spurious year labels like F1961, F5000 that come from table cell
# content (employee IDs, reference numbers, etc.).

_VALID_FY_YEARS: set[int] = set(range(2005, 2036))


# ── Year cell detection ────────────────────────────────────────────────────────

def _cell_to_year(text: str) -> Optional[str]:
    if not text:
        return None
    yr = normalise_year(text.strip())
    if yr:
        try:
            y = int(yr[1:])          # strip leading 'F'
            if y not in _VALID_FY_YEARS:
                return None
        except (ValueError, IndexError):
            return None
    return yr

def _is_year_cell(text: str) -> bool:
    return _cell_to_year(text) is not None


# ── Note reference filter ──────────────────────────────────────────────────────

def _is_note_ref(text: str) -> bool:
    """Small whole integers 1–300 with no decimal = note reference, not a value."""
    t = text.strip().replace(",", "")
    try:
        v = float(t)
        return v == int(v) and 1 <= int(v) <= 300 and "." not in text
    except (ValueError, AttributeError):
        return False


# ── Section classification ─────────────────────────────────────────────────────

_SECTION_KW: dict[str, set[str]] = {
    "Annual P&L": {
        "profit and loss", "profit & loss", "income statement",
        "statement of profit", "statement of operations",
        "revenue from operations", "interest income", "interest earned",
        "total income", "net revenue",
    },
    "Annual Balance Sheet": {
        "balance sheet", "financial position", "assets and liabilities",
        "total assets", "total liabilities", "equity and liabilities",
        "sources of funds", "application of funds",
    },
    "Annual Cash Flow": {
        "cash flow", "cash flows", "statement of cash",
        "operating activities", "investing activities", "financing activities",
    },
    # ── Additional sections captured from Indian bank/NBFC annual reports ──────
    "Operating Metrics": {
        "key performance indicator", "financial highlight", "key financial indicator",
        "key indicator", "performance highlight", "business highlight",
        "asset quality", "capital adequacy", "crar",
        "gnpa", "nnpa", "gross npa", "net npa",
        "net interest margin", "nim", "return on asset", "return on equity",
        "casa ratio", "loan book", "book value", "credit cost",
        "provision coverage", "slippage", "cost to income",
        "disbursement", "aum", "collection efficiency",
    },
    "Quarterly": {
        "quarterly result", "quarter ended", "results for the quarter",
        "unaudited result", "limited review",
        # Note: do NOT add q1/q2/q3/q4 here — too common in table cells
    },
}

# Order matters: check Operating Metrics BEFORE Balance Sheet
# (some pages mention "total assets" in a KPI context, not as a BS heading)
_SECTION_PRIORITY = [
    "Annual P&L",
    "Annual Cash Flow",
    "Operating Metrics",
    "Quarterly",
    "Annual Balance Sheet",   # broadest catch-all last
]

def _classify_text(text: str) -> Optional[str]:
    t = text.lower()
    for section in _SECTION_PRIORITY:
        kws = _SECTION_KW.get(section, set())
        if any(kw in t for kw in kws):
            return section
    return None


# ── Consolidation detection ────────────────────────────────────────────────────

_CONS_RE = re.compile(r"\bconsolidated\b", re.I)
_STAN_RE = re.compile(r"\bstandalone\b|\bseparate\b", re.I)

def _detect_stmt_type(text: str, caller_default: str = "consolidated") -> str:
    has_c = bool(_CONS_RE.search(text[:1500]))
    has_s = bool(_STAN_RE.search(text[:1500]))
    if has_c and not has_s:
        return "consolidated"
    if has_s and not has_c:
        return "standalone"
    return caller_default


# ── Generic sidebar detection & crop ─────────────────────────────────────────
#
# Many Indian annual reports print vertical section banners ("CORPORATE OVERVIEW",
# "STATUTORY REPORTS", "FINANCIAL STATEMENTS") along one or both page margins.
# These are rotated text fragments — they appear at a narrow x-band but span a
# tall y range.  Crucially, they contain NO numeric characters, which distinguishes
# them from value columns.
#
# Detection logic (all generic):
#   1. Find words whose x0 is significantly beyond the 85th-percentile x-position.
#   2. Check that the outlier cluster:
#        a. Is small (<12% of all words) — sidebars are sparse
#        b. Has x-range < 25 pt — the characters are stacked vertically
#        c. Has y-span > 60 pt  — rotated text spans a large height
#        d. Contains NO digit characters — section labels are pure text
#   3. If all conditions met, crop to just before the cluster.
#   4. Never crop more than 10% of page width (safety valve).

def _auto_crop_bbox(page) -> tuple[float, float, float, float]:
    """
    Return a (x0, top, x1, bottom) bounding box that excludes narrow
    marginal sidebars.  Falls back to the full page if no sidebar found.
    """
    try:
        words = page.extract_words(x_tolerance=3, y_tolerance=3)
    except Exception:
        return (0, 0, page.width, page.height)

    if len(words) < 5:
        return (0, 0, page.width, page.height)

    xs = sorted(w["x0"] for w in words)
    n  = len(xs)
    p85 = xs[min(n - 1, int(n * 0.85))]

    # ── Right-margin sidebar detection ─────────────────────────────────────────
    right_words = [w for w in words if w["x0"] > p85 + 30]
    right_crop  = page.width

    if right_words and len(right_words) < n * 0.12:
        rw_xs    = [w["x0"] for w in right_words]
        rw_ys    = [w["top"] for w in right_words]
        x_range  = max(rw_xs) - min(rw_xs) if rw_xs else 0
        y_span   = (max(rw_ys) - min(rw_ys)) if len(rw_ys) > 1 else 0
        has_nums = any(re.search(r"\d", w["text"]) for w in right_words)

        if x_range < 25 and y_span > 60 and not has_nums:
            # Confirmed sidebar — crop to just before it
            right_crop = min(rw_xs) - 5
            # Safety: never remove more than 10% of page width
            right_crop = max(right_crop, page.width * 0.90)

    # ── Left-margin sidebar detection (less common) ────────────────────────────
    p15      = xs[max(0, int(n * 0.15))]
    left_words = [w for w in words if w["x0"] < p15 - 30]
    left_crop  = 0

    if left_words and len(left_words) < n * 0.12:
        lw_xs    = [w["x0"] for w in left_words]
        lw_ys    = [w["top"] for w in left_words]
        x_range  = max(lw_xs) - min(lw_xs) if lw_xs else 0
        y_span   = (max(lw_ys) - min(lw_ys)) if len(lw_ys) > 1 else 0
        has_nums = any(re.search(r"\d", w["text"]) for w in left_words)

        if x_range < 25 and y_span > 60 and not has_nums:
            left_crop = max(lw_xs) + 5
            left_crop = min(left_crop, page.width * 0.10)

    return (left_crop, 0, right_crop, page.height)


# ── Concatenated cell splitting ───────────────────────────────────────────────
#
# pdfplumber sometimes MERGES two adjacent columns into one cell when they are
# very close together (e.g. two year columns in a busy layout).
#
#   Year-header merge: "2024-252023-24"     → ["2024-25",  "2023-24"]
#   Value merge:       "3,251.613,079.76"   → ["3,251.61", "3,079.76"]
#
# Both splits are detected generically (no company-specific knowledge).

def _split_concat_year(text: str) -> list[str]:
    """
    Detect and split concatenated year strings.
    E.g. "2024-252023-24" → ["2024-25", "2023-24"]
         "F2025F2024"      → ["F2025",   "F2024"]
    Returns the original single-item list if no split is found.
    """
    t = text.strip()

    # Pattern A: two Indian FY ranges "YYYY-YYYYYY-YY"
    m = re.match(r"^(\d{4}-\d{2,4})(\d{4}-\d{2,4})$", t)
    if m:
        y1 = normalise_year(m.group(1))
        y2 = normalise_year(m.group(2))
        if y1 and y2 and int(y1[1:]) in _VALID_FY_YEARS and int(y2[1:]) in _VALID_FY_YEARS:
            return [m.group(1), m.group(2)]

    # Pattern B: two F/FY labels "F2025F2024" or "FY2025FY2024"
    m = re.match(r"^(F[Y]?\d{4})(F[Y]?\d{4})$", t, re.I)
    if m:
        y1 = normalise_year(m.group(1))
        y2 = normalise_year(m.group(2))
        if y1 and y2 and int(y1[1:]) in _VALID_FY_YEARS and int(y2[1:]) in _VALID_FY_YEARS:
            return [m.group(1), m.group(2)]

    return [t]


def _split_concat_number(text: str) -> list[str]:
    """
    Detect and split a cell that contains two concatenated financial numbers.
    E.g. "3,251.613,079.76" → ["3,251.61", "3,079.76"]

    Strategy: try every split point; prefer splits where the first part ends
    with exactly 2 decimal digits (standard Indian financial formatting).
    Falls back to the first valid split found.

    Only attempts splitting when the cell contains more than one decimal point
    (after comma-removal), indicating two numbers have been merged.
    """
    t = text.strip()

    # Only consider purely numeric-looking cells
    if not re.match(r"^[\d,.()\-]+$", t) or len(t) < 8:
        return [t]

    # Count decimal points after removing commas; >1 → merged numbers
    clean = t.replace(",", "")
    if clean.count(".") <= 1:
        return [t]          # single number, nothing to split

    # Valid number-format checker (no leading comma, proper structure)
    _num_fmt = re.compile(r"^\(?[\d,]+(\.\d+)?\)?$")

    best_fallback: Optional[list[str]] = None

    for i in range(3, len(t) - 3):
        p1, p2 = t[:i], t[i:]

        # p2 must start with a digit (not comma or dot)
        if not p2[0].isdigit():
            continue

        # Both parts must look like well-formed numbers
        if not (_num_fmt.match(p1) and _num_fmt.match(p2)):
            continue

        # Both parts must actually parse
        if parse_number(p1) is None or parse_number(p2) is None:
            continue

        # Prefer split where p1 ends with exactly 2 decimal digits
        if re.search(r"\.\d{2}$", p1):
            return [p1, p2]

        if best_fallback is None:
            best_fallback = [p1, p2]

    return best_fallback or [t]


# ── Cell post-processing: fix split cells ─────────────────────────────────────
#
# pdfplumber sometimes splits a single number or word across adjacent cells
# because the underlying PDF text stream places characters in separate runs.

_NUMERIC_CHARS = re.compile(r"^[\d,.()\-]+$")

def _looks_partial_number(s: str) -> bool:
    """True if s looks like a number fragment (could be continued)."""
    s = s.strip()
    if not s:
        return False
    return bool(_NUMERIC_CHARS.match(s)) and (s[-1].isdigit() or s[-1] == ",")

def _looks_number_start(s: str) -> bool:
    s = s.strip()
    return bool(s) and bool(_NUMERIC_CHARS.match(s))

def _merge_row_cells(row: list[str | None]) -> list[str | None]:
    """
    Scan a row and merge adjacent cells that together form a single token.
    Handles:
      - Split numbers: "6,31,17,25" + "0"  → "6,31,17,250"
      - Split words:   "Net Interest Incom" + "e"  → "Net Interest Income"
    Returns a new row (may be shorter than the input).
    """
    result: list[str | None] = []
    i = 0
    cells = [str(c).strip() if c is not None else "" for c in row]

    while i < len(cells):
        cur = cells[i]

        if cur and i + 1 < len(cells):
            nxt = cells[i + 1]
            # Case 1: numeric split — "6,31,17,25" + "0"
            if _looks_partial_number(cur) and nxt and nxt[0].isdigit():
                result.append(cur + nxt)
                i += 2
                continue
            # Case 2: word split — short alphabetic continuation
            if (cur and nxt and len(nxt) <= 3 and nxt.isalpha()
                    and cur[-1].isalpha() and not _looks_partial_number(cur)):
                result.append(cur + nxt)
                i += 2
                continue

        result.append(cur if cur else None)
        i += 1

    return result


def _expand_row_cells(row: list[str | None]) -> list[str | None]:
    """
    Expand cells that contain two concatenated tokens (year or number).
    Applied BEFORE _merge_row_cells so the two passes don't fight each other.
    """
    expanded: list[str | None] = []
    for cell in row:
        if not cell:
            expanded.append(cell)
            continue
        s = str(cell).strip()
        # Try year concatenation split first
        parts = _split_concat_year(s)
        if len(parts) > 1:
            expanded.extend(parts)
            continue
        # Try number concatenation split
        parts = _split_concat_number(s)
        expanded.extend(parts)
    return expanded


def _clean_table(table: list[list[str | None]]) -> list[list[str | None]]:
    """Expand split/merged cells, then merge fragmented adjacent cells, on every row."""
    result = []
    for row in table:
        expanded = _expand_row_cells(row)
        merged   = _merge_row_cells(expanded)
        result.append(merged)
    return result


# ── pdfplumber table settings ─────────────────────────────────────────────────

_TABLE_SETTINGS_TEXT = {
    "vertical_strategy":        "text",
    "horizontal_strategy":      "text",
    "intersection_x_tolerance": 15,
    "intersection_y_tolerance": 15,
    "snap_tolerance":           5,
    "join_tolerance":           3,
    "edge_min_length":          50,
    "min_words_vertical":       3,
    "min_words_horizontal":     1,
    "text_x_tolerance":         15,   # wider → fewer split columns
    "text_y_tolerance":         5,
}

_TABLE_SETTINGS_LINES = {
    "vertical_strategy":        "lines",
    "horizontal_strategy":      "lines",
    "intersection_x_tolerance": 5,
    "intersection_y_tolerance": 5,
}


def _extract_page_tables(page) -> list[list[list[str | None]]]:
    """
    Try multiple strategies to extract tables from a cropped pdfplumber page.
    Returns the result with the most data cells.
    """
    best: list[list[list]] = []

    def _cell_count(tables):
        return sum(
            1 for t in tables for row in t
            for c in row if c and str(c).strip()
        )

    # Strategy 1: text-based (whitespace-aligned, most common for Indian PDFs)
    try:
        tables = page.extract_tables(_TABLE_SETTINGS_TEXT)
        if tables and _cell_count(tables) > _cell_count(best):
            best = tables
    except Exception:
        pass

    # Strategy 2: explicit lines (grid PDFs)
    if not best:
        try:
            tables = page.extract_tables(_TABLE_SETTINGS_LINES)
            if tables and _cell_count(tables) > _cell_count(best):
                best = tables
        except Exception:
            pass

    # Strategy 3: pdfplumber auto-detect
    if not best:
        try:
            tables = page.extract_tables()
            if tables:
                best = tables
        except Exception:
            pass

    # Normalise cells to str | None, expand merged columns, merge split cells
    normalised = []
    for table in best:
        norm_table = []
        for row in table:
            norm_row = [str(c).strip() if c is not None else None for c in row]
            norm_table.append(norm_row)
        normalised.append(norm_table)

    return [_clean_table(t) for t in normalised]


# ── Multi-row year header detection ──────────────────────────────────────────
#
# Some PDFs print the year header across two rows:
#   Row N:   "Particulars"  |  "Year ended"   |  "Year ended"
#   Row N+1: (blank)        |  "March 31,2025" |  "March 31,2024"
#
# Single-row detection misses "Year ended" (not a year) and then "March 31,2025"
# (which IS a year) on the next row.  Fix: if single-row scan finds nothing,
# concatenate each column's text from the first max_header_rows rows and retry.

def _detect_year_columns(
    table: list[list[str | None]],
    max_header_rows: int = 15,
) -> tuple[Optional[int], dict[int, str]]:
    """
    Return (header_row_idx, {col_idx: year_label}) for the year header row.

    Pass 1 (fast): scan each row individually for year cells.
    Pass 2 (slow): combine text from the first max_header_rows rows per column.
    """
    # ── Pass 1: single-row detection ───────────────────────────────────────────
    for row_idx, row in enumerate(table[:max_header_rows]):
        if not row:
            continue
        hits: dict[int, str] = {}
        for col_idx, cell in enumerate(row):
            if not cell or col_idx == 0:
                continue
            yr = _cell_to_year(str(cell))
            if not yr:
                # Multi-token cell: "Year ended\nMarch 31, 2025"
                for token in re.split(r"[\n|,]", str(cell)):
                    yr = _cell_to_year(token.strip())
                    if yr:
                        break
            if yr:
                hits[col_idx] = yr
        if hits:
            return row_idx, hits

    # ── Pass 2: column-wise text accumulation ──────────────────────────────────
    max_scan = min(max_header_rows, len(table))
    col_combined: dict[int, str] = defaultdict(str)

    for row in table[:max_scan]:
        for col_idx, cell in enumerate(row):
            if cell:
                col_combined[col_idx] += " " + str(cell).strip()

    hits = {}
    for col_idx, combined in col_combined.items():
        if col_idx == 0:
            continue
        combined = combined.strip()

        # Try direct normalise
        yr = normalise_year(combined)
        if yr and int(yr[1:]) in _VALID_FY_YEARS:
            hits[col_idx] = yr
            continue

        # Search for "March 31, YYYY" pattern
        m = re.search(r"march\s+31,?\s*(\d{4})", combined, re.I)
        if m:
            candidate = f"F{m.group(1)}"
            y = int(m.group(1))
            if y in _VALID_FY_YEARS:
                hits[col_idx] = candidate
                continue

        # Search for YYYY-YY or FY\d+ patterns
        for pat in [r"\b(\d{4}-\d{2,4})\b", r"\bFY\s*(\d{2,4})\b"]:
            for match in re.finditer(pat, combined, re.I):
                candidate = normalise_year(match.group(0))
                if candidate and int(candidate[1:]) in _VALID_FY_YEARS:
                    hits[col_idx] = candidate
                    break
            if col_idx in hits:
                break

    if hits:
        return max_scan - 1, hits

    return None, {}


# ── Table → ExtractionResult ──────────────────────────────────────────────────

def _parse_table(
    table: list[list[str | None]],
    default_section: str,
    statement_type: str,
    unit_divisor: float,
) -> list[ExtractionResult]:
    """Parse one pdfplumber table into ExtractionResult objects."""
    results: list[ExtractionResult] = []
    if not table or len(table) < 2:
        return results

    # Detect year columns (single-row first, then multi-row fallback)
    header_idx, year_col_map = _detect_year_columns(table)

    if header_idx is None or not year_col_map:
        return results

    # Classify the section from the table's own text
    table_text = " ".join(
        str(cell) for row in table[:30] for cell in row if cell
    )
    section = _classify_text(table_text) or default_section

    # Parse data rows
    year_data: dict[str, dict[str, float]] = {yr: {} for yr in year_col_map.values()}

    for row in table[header_idx + 1:]:
        if not row:
            continue

        # Build label from ALL leading non-numeric cells
        label_parts: list[str] = []
        for cell in row:
            s = str(cell).strip() if cell else ""
            if not s:
                if label_parts:
                    break
                continue
            if parse_number(s) is not None and not _is_note_ref(s):
                break
            if not _is_note_ref(s):
                label_parts.append(s)

        label = " ".join(label_parts).strip()
        if not label or len(label) < 2:
            continue

        # Extract values for each year column
        for col_idx, year_label in year_col_map.items():
            if col_idx >= len(row):
                continue
            cell = row[col_idx]
            if not cell:
                continue
            s = str(cell).strip()
            if not s or _is_note_ref(s):
                continue
            val = parse_number(s)
            if val is None:
                continue
            converted = round(val / unit_divisor, 4) if unit_divisor != 1.0 else val
            if label not in year_data[year_label]:
                year_data[year_label][label] = converted

    unit_note = f"unit_divisor={unit_divisor}"
    for year_label, data in year_data.items():
        if data:
            results.append(ExtractionResult(
                section=section,
                statement_type=statement_type,
                year=year_label,
                currency="INR Crores",
                data=data,
                notes=f"mode=table_direct | {unit_note}",
                confidence=0.95,
            ))
    return results


# ── Word-position fallback ────────────────────────────────────────────────────

def _words_to_rows(words: list[dict], y_tolerance: float = 4.0) -> list[list[dict]]:
    """Group pdfplumber word objects into logical rows by y-coordinate proximity."""
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    rows: list[list[dict]] = []
    current_row: list[dict] = [sorted_words[0]]
    current_y: float = sorted_words[0]["top"]

    for word in sorted_words[1:]:
        if abs(word["top"] - current_y) <= y_tolerance:
            current_row.append(word)
        else:
            rows.append(sorted(current_row, key=lambda w: w["x0"]))
            current_row = [word]
            current_y   = word["top"]

    if current_row:
        rows.append(sorted(current_row, key=lambda w: w["x0"]))

    return rows


def _looks_numeric(text: str) -> bool:
    t = text.strip().replace(",", "").replace("(", "").replace(")", "")
    try:
        float(t)
        return True
    except ValueError:
        return False


def _find_year_xs_from_words(
    rows_of_words: list[list[dict]],
    max_rows: int = 15,          # strict header area only — beyond this, years are dates
) -> dict[float, str]:
    """
    Identify year column x-positions from word-level analysis.
    Handles multi-word year labels ("March 31, 2025" split across words/rows).

    IMPORTANT: Only searches the first max_rows rows.  Years appearing later in
    body text are date references (e.g. "January 2024"), not column headers.

    Returns {x_center: year_label} for all found year columns.
    Requirement: at least 2 distinct year columns must be found; otherwise
    the "year" is likely a date in running text, not a table header, and {}
    is returned to prevent garbage extraction.
    """
    year_x_map: dict[float, str] = {}

    # Pass 1: individual word matches (FY2025, 2024-25, plain 2025)
    for row_words in rows_of_words[:max_rows]:
        for w in row_words:
            yr = _cell_to_year(w["text"])
            if yr:
                x = (w["x0"] + w["x1"]) / 2
                year_x_map[x] = yr

    if len(year_x_map) >= 2:
        return year_x_map

    # Pass 2: search combined row text for "March 31, YYYY" and "YYYY-YY"
    year_x_map = {}
    for row_words in rows_of_words[:max_rows]:
        row_text = " ".join(w["text"] for w in row_words)

        # "March 31, 2025"
        for m in re.finditer(r"March\s+31,?\s*(\d{4})", row_text, re.I):
            yr = f"F{m.group(1)}"
            if int(m.group(1)) not in _VALID_FY_YEARS:
                continue
            for w in row_words:
                if m.group(1) in w["text"]:
                    x = (w["x0"] + w["x1"]) / 2
                    year_x_map[x] = yr
                    break

        # "2024-25" or "2024-2025"
        for m in re.finditer(r"\b(\d{4}-\d{2,4})\b", row_text):
            candidate = normalise_year(m.group(1))
            if not candidate or int(candidate[1:]) not in _VALID_FY_YEARS:
                continue
            for w in row_words:
                if m.group(1) in w["text"] or w["text"] in m.group(1):
                    x = (w["x0"] + w["x1"]) / 2
                    year_x_map[x] = candidate
                    break

    # Require ≥2 distinct year columns — lone years are likely date references
    if len(year_x_map) < 2:
        return {}

    return year_x_map


def _parse_page_via_words(
    page,
    default_section: str,
    statement_type: str,
    unit_divisor: float,
) -> list[ExtractionResult]:
    """
    Fallback: reconstruct table structure from individual word positions.
    Groups words by y-coordinate → rows, identifies year columns, extracts values.
    Enhanced to handle multi-word / multi-row year labels.
    """
    results: list[ExtractionResult] = []

    words = page.extract_words(
        x_tolerance=5, y_tolerance=5,
        keep_blank_chars=False, use_text_flow=False,
    )
    if not words:
        return results

    rows_of_words = _words_to_rows(words, y_tolerance=5.0)
    if not rows_of_words:
        return results

    # Detect year column x-positions (handles multi-row / multi-word headers)
    year_x_map = _find_year_xs_from_words(rows_of_words)

    if not year_x_map:
        return results

    sorted_year_xs = sorted(year_x_map.keys())
    sorted_years   = [year_x_map[x] for x in sorted_year_xs]

    # X-tolerance for assigning a word to a year column
    if len(sorted_year_xs) > 1:
        col_width = min(
            abs(sorted_year_xs[i + 1] - sorted_year_xs[i])
            for i in range(len(sorted_year_xs) - 1)
        )
        x_tol = col_width * 0.5
    else:
        x_tol = 50.0

    # Find the row after which data rows begin (skip header rows)
    header_end = 0
    for ri, row_words in enumerate(rows_of_words[:25]):
        if any(year_x_map.get((w["x0"] + w["x1"]) / 2) for w in row_words):
            header_end = ri + 1

    year_data: dict[str, dict[str, float]] = {yr: {} for yr in sorted_years}

    for row_words in rows_of_words[header_end:]:
        if not row_words:
            continue

        text_words = [w for w in row_words if not _looks_numeric(w["text"])]
        num_words  = [
            w for w in row_words
            if _looks_numeric(w["text"]) and not _is_note_ref(w["text"])
        ]

        if not text_words or not num_words:
            continue

        label = " ".join(w["text"] for w in text_words).strip()
        if not label or len(label) < 2:
            continue

        # Assign each numeric word to nearest year column by x-coordinate
        for nw in num_words:
            x_center = (nw["x0"] + nw["x1"]) / 2
            best_yr   = None
            best_dist = float("inf")
            for yx, yr in year_x_map.items():
                d = abs(x_center - yx)
                if d < best_dist:
                    best_dist = d
                    best_yr   = yr
            if best_yr and best_dist <= x_tol:
                val = parse_number(nw["text"])
                if val is not None:
                    converted = round(val / unit_divisor, 4) if unit_divisor != 1.0 else val
                    if label not in year_data[best_yr]:
                        year_data[best_yr][label] = converted

    section = _classify_text(
        " ".join(w["text"] for row in rows_of_words[:30] for w in row)
    ) or default_section

    for yr, data in year_data.items():
        if data:
            results.append(ExtractionResult(
                section=section,
                statement_type=statement_type,
                year=yr,
                currency="INR Crores",
                data=data,
                notes=f"mode=table_words | unit_divisor={unit_divisor}",
                confidence=0.90,
            ))
    return results


# ── Notes / non-financial page filter ────────────────────────────────────────
#
# Skip pages that are clearly NOT core financial statements:
#   • Notes to accounts / accounting policies
#   • Auditor's report
#   • ESG / sustainability / CSR report pages
#   • Basel / capital adequacy disclosures
#   • Board / governance sections
#
# We check the first 1 500 characters (covers multi-paragraph headings) and
# also a secondary scan of the first 3 000 characters for stronger signals.

# ── Pattern A: financial statement TITLE keywords ─────────────────────────────
# If a page's FIRST 250 chars contain these, it's definitely a financial page.
# This overrides all exclusion filters below.

_FINANCIAL_TITLE_RE = re.compile(
    r"balance\s+sheet"
    r"|profit\s+(?:and|&)\s+loss"
    r"|profit\s+and\s+loss\s+account"
    r"|cash\s+flow\s+statement"
    r"|income\s+statement"
    r"|statement\s+of\s+(?:profit|financial\s+position)"
    r"|revenue\s+from\s+operations"
    r"|interest\s+earned"
    r"|interest\s+income",
    re.I,
)

# ── Pattern B: page-TITLE-level exclusions (checked in first 350 chars only) ──
# These keywords appear as the title/heading of non-financial pages.
# They should NOT match footer references like "See note 17" on financial pages.

_NOTES_TITLE_RE = re.compile(
    r"^\s*(?:notes?\s+(?:to|on|forming\s+part)|"
    r"schedule\s+\d+\b|"
    r"significant\s+accounting\s+polic|"
    r"basis\s+of\s+(?:preparation|consolidation)|"
    r"auditor.s\s+report)",
    re.I | re.MULTILINE,
)

# ── Pattern C: strong non-financial signals (checked up to first 1 500 chars) ──
# These only appear on pages that are clearly not financial statements,
# regardless of where they appear on the page.

_NON_FINANCIAL_RE = re.compile(
    # Regulatory disclosures
    r"capital\s+adequacy"
    r"|risk\s+weighted\s+assets"
    r"|\bcrar\b"
    r"|\bbasel\s+(?:ii|iii)\b"

    # ESG / sustainability / CSR
    r"|energy\s+(?:consumption|intensity|generated)"
    r"|ghg\s+emissions?"
    r"|greenhouse\s+gas"
    r"|carbon\s+footprint"
    r"|\besg\s+(?:report|framework|polic)"
    r"|sustainability\s+report"
    r"|environmental\s+(?:performance|impact|data)"
    r"|waste\s+(?:generated|disposed|recycled)"
    r"|water\s+(?:consumption|withdrawal|discharge)"
    r"|csr\s+(?:activit|expenditure|spend|committee)"
    r"|business\s+responsibility\s+(?:and\s+sustainability\s+)?report"
    r"|\bbrsr\b"

    # Board / governance
    r"|directors.\s+report"
    r"|corporate\s+governance\s+report"
    r"|management\s+discussion\s+(?:and|&)\s+analysis"

    # Employee / HR
    r"|training\s+(?:hours?|programme|beneficiar)"
    r"|employee\s+(?:engagement|wellness|strength)",

    re.I,
)

# ── Pattern D: operational / non-monetary tables (checked up to first 3 000 chars) ──
# These patterns indicate tables with operational counts, headcounts, or regulatory
# references — NOT financial statements.  All are generic: any Indian annual report
# that contains these in its early content is not a financial-statement page.
_CSR_TABLE_RE = re.compile(
    # Headcount / beneficiary count tables (always operational, never financial)
    r"no\.\s+of\s+beneficiar"
    r"|number\s+of\s+beneficiar"
    r"|beneficiar(?:y|ies)\s+\(in\s+numbers?\)"
    # Operational count rows: "No. of <noun>" where noun is non-financial
    r"|no\.\s+of\s+(?:camps?|programmes?|schools?|clinics?|villages?|households?)"
    # SEBI ESOS / sweat equity references (appear only in Director's Report)
    r"|\bsebi\s+\(share\s+based\s+employee\b"
    r"|\bsweat\s+equity\s+shares?\b",
    re.I,
)


def _is_non_financial_page(page_text: str) -> bool:
    """
    Return True if this page should be skipped (not a core financial statement).

    Logic (in order):
      1. If the first 250 chars contain a financial statement title → KEEP (False).
      2. If the first 350 chars contain a notes/disclosures TITLE → SKIP (True).
      3. If the first 1 500 chars contain a strong non-financial signal → SKIP (True).
      4. If the first 3 000 chars contain CSR/operational table data → SKIP (True).
      5. Otherwise → KEEP (False).
    """
    # Step 1: financial title override
    if _FINANCIAL_TITLE_RE.search(page_text[:250]):
        return False

    # Step 2: notes/disclosures page title
    if _NOTES_TITLE_RE.search(page_text[:350]):
        return True

    # Step 3: strong non-financial page signals
    if _NON_FINANCIAL_RE.search(page_text[:1500]):
        return True

    # Step 4: CSR/operational table markers
    if _CSR_TABLE_RE.search(page_text[:3000]):
        return True

    return False


# ── Result quality validation ─────────────────────────────────────────────────
#
# After extraction, filter out result blocks that are clearly garbage:
#   1. Fewer than 3 values — financial statements always have many rows.
#      A block with 1–2 values is almost always from stray numbers in running text.
#   2. More than 60% of labels are long (>70 chars) — these come from prose,
#      not table rows.  Financial label rows are short (e.g. "Interest earned").
#   3. Year not plausible relative to the last 3 fiscal years — e.g. if this is
#      a 2024-25 annual report, years older than F2020 in minor blocks are suspect.
#      (We use a soft rule: keep any block with ≥5 values regardless of year.)

_MIN_VALUES_PER_BLOCK = 5        # raised: financial tables always have many rows
_MAX_LONG_LABEL_RATIO = 0.60
_MAX_LABEL_LEN = 70


def _is_quality_result(r: "ExtractionResult") -> bool:
    """Return True if an ExtractionResult block looks like real financial data."""
    if len(r.data) < _MIN_VALUES_PER_BLOCK:
        return False
    labels = list(r.data.keys())
    long_count = sum(1 for lbl in labels if len(lbl) > _MAX_LABEL_LEN)
    if long_count / len(labels) > _MAX_LONG_LABEL_RATIO:
        return False
    return True


# ── Main entry point ───────────────────────────────────────────────────────────

def extract_tables_from_pdf(
    pdf_source,
    parsed_doc,
    statement_type: str = "consolidated",
) -> list[ExtractionResult]:
    """
    Extract financial data directly from PDF table structures.

    Generic pipeline:
      1. Auto-detect & crop marginal sidebars (no-number check + narrow x-band)
      2. Try pdfplumber text-strategy table extraction
      3. Try pdfplumber line-strategy table extraction
      4. Fall back to word-position spatial grouping
      5. Apply unit conversion (once per page, from page text)
      6. Filter spurious years outside F2005–F2035
      7. Filter ESG/CSR/notes pages by heading keywords
      8. Merge results across pages

    Parameters
    ----------
    pdf_source    : file path (str/Path), bytes, or file-like object
    parsed_doc    : ParsedDocument from pdf_parser
    statement_type: "consolidated" | "standalone"
    """
    pdfplumber = _require_pdfplumber()

    if isinstance(pdf_source, (bytes, bytearray)):
        pdf_source = _io.BytesIO(pdf_source)

    all_results: list[ExtractionResult] = []

    # Collect financial page numbers from pdf_parser output
    all_pages: Optional[set[int]] = set()
    for sec in parsed_doc.sections:
        all_pages.update(sec.page_numbers)
    for pg_num, pg_text in parsed_doc.raw_pages.items():
        if _classify_text(pg_text) is not None:
            all_pages.add(pg_num)

    if not all_pages:
        logger.warning(
            f"table_extractor: no financial pages found by pdf_parser in "
            f"'{parsed_doc.filename}' — scanning ALL pages"
        )
        all_pages = None

    logger.info(
        f"table_extractor: '{parsed_doc.filename}' — "
        + (f"scanning {len(all_pages)} financial page(s)" if all_pages
           else "scanning all pages")
    )

    try:
        with pdfplumber.open(pdf_source) as pdf:
            total = len(pdf.pages)
            pages_to_scan = (
                sorted(p for p in all_pages if 1 <= p <= total)
                if all_pages
                else list(range(1, total + 1))
            )

            for pg_num in pages_to_scan:
                page      = pdf.pages[pg_num - 1]
                page_text = page.extract_text() or ""

                if not page_text.strip():
                    logger.debug(f"  Page {pg_num}: blank/image page — skipping")
                    continue

                # Skip notes / disclosures / ESG / governance pages
                if _is_non_financial_page(page_text):
                    logger.debug(f"  Page {pg_num}: notes/disclosures/ESG — skipping")
                    continue

                # ── Generic sidebar crop ────────────────────────────────────
                bbox    = _auto_crop_bbox(page)
                cropped = page.crop(bbox) if bbox != (0, 0, page.width, page.height) else page

                # Unit divisor and statement type from FULL page text
                # (unit declaration may be in the sidebar we cropped out)
                unit_divisor = detect_unit_divisor(page_text)
                stmt_type    = _detect_stmt_type(page_text, statement_type)
                section_hint = _classify_text(page_text) or "Annual Balance Sheet"

                # ── Strategies 1 & 2: pdfplumber table extraction ───────────
                tables       = _extract_page_tables(cropped)
                page_results: list[ExtractionResult] = []

                for tbl in tables:
                    r = _parse_table(tbl, section_hint, stmt_type, unit_divisor)
                    page_results.extend(r)

                # ── Strategy 3: word-position fallback ──────────────────────
                if not page_results:
                    page_results = _parse_page_via_words(
                        cropped, section_hint, stmt_type, unit_divisor
                    )

                # Quality filter: drop blocks that look like prose / non-financial data
                page_results = [r for r in page_results if _is_quality_result(r)]

                if page_results:
                    n = sum(len(r.data) for r in page_results)
                    logger.info(
                        f"  Page {pg_num}: {n} values"
                        + (f" [÷{unit_divisor:g}]" if unit_divisor != 1.0 else "")
                        + f" ({page_results[0].notes.split('|')[0].strip()})"
                    )
                    all_results.extend(page_results)
                else:
                    logger.debug(f"  Page {pg_num}: 0 values extracted")

    except Exception as e:
        logger.error(
            f"table_extractor: failed on '{parsed_doc.filename}': {e}",
            exc_info=True,
        )
        return []

    merged     = _merge(all_results)
    total_vals = sum(len(r.data) for r in merged)
    logger.info(
        f"table_extractor: '{parsed_doc.filename}' → "
        f"{total_vals} total values, {len(merged)} result block(s)"
    )
    return merged


def _merge(results: list[ExtractionResult]) -> list[ExtractionResult]:
    """Merge ExtractionResults with the same (section, type, year). Later values fill gaps."""
    merged: dict[tuple, ExtractionResult] = {}
    for r in results:
        key = (r.section, r.statement_type, r.year)
        if key not in merged:
            merged[key] = r
        else:
            for k, v in r.data.items():
                if k not in merged[key].data:
                    merged[key].data[k] = v
    return list(merged.values())
