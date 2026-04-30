"""
Rule-Based Financial Data Extractor

Extracts financial data directly from structured PDF text using regex and
label matching — no LLM required for well-formatted statements.

Strategy
--------
1. Detect year columns from header lines (e.g. "March 31, 2025  March 31, 2024").
2. Scan each line for known field labels.
3. Extract numeric values from the same line (or following lines).
4. Return ExtractionResult objects + a coverage score (0.0–1.0).

The caller uses coverage to decide whether to supplement with LLM:
  - coverage >= 0.6  → rule-based is good enough, skip LLM
  - coverage <  0.6  → hand off to LLM for the gaps
"""

from __future__ import annotations

import re
from typing import Optional

from utils.helpers import get_logger, normalise_year, parse_number

logger = get_logger(__name__)


# ── Number regex ──────────────────────────────────────────────────────────────
# Matches: 9,151.18  |  (1,234.56)  |  123  |  0.00
_NUM_RE = re.compile(r"\(?[\d,]+\.?\d*\)?")

# Strips "face value" parenthetical from EPS label lines, e.g.:
#   "Earnings Per Equity Share (Diluted) (₹) (Face value of ₹ 10/- each)"
# Without this, "10" from the face value is mistakenly extracted as the EPS figure.
_FACE_VALUE_RE = re.compile(r"\(?face\s+value[^)]*\)", re.IGNORECASE)

# ── Year header patterns ──────────────────────────────────────────────────────
_YEAR_HEADER_PATTERNS = [
    # "Year ended March 31, 2025"  /  "As at March 31, 2025"
    re.compile(r"(?:year\s+ended|as\s+at|for\s+the\s+year)\s+march\s+31,?\s*(\d{4})", re.IGNORECASE),
    # "2024-25"  /  "FY2025"  /  "FY 25"
    re.compile(r"\bFY\s*(\d{2,4})\b", re.IGNORECASE),
    re.compile(r"\b(\d{4})[-/](\d{2,4})\b"),
    # Plain 4-digit year used as a column header
    re.compile(r"\b(20\d{2})\b"),
]


_BARE_MARCH_RE = re.compile(r"march\s+31,?\s*(\d{4})", re.IGNORECASE)

def _find_years_in_text(text: str) -> list[str]:
    """
    Scan the first 50 lines for year patterns. Return them in left-to-right
    (most-recent-first) order as template column labels (e.g. ["F2025","F2024"]).
    Handles both "Year ended March 31, 2025" and bare "March 31, 2025" formats.
    """
    found: list[str] = []
    seen: set[str] = set()

    for line in text.splitlines()[:50]:
        line_low = line.lower()

        # Try "Year ended / As at March 31, YYYY"
        for m in _YEAR_HEADER_PATTERNS[0].finditer(line_low):
            y = normalise_year(f"march 31, {m.group(1)}")
            if y and y not in seen:
                found.append(y); seen.add(y)

        # Try bare "March 31, YYYY" (balance sheet header line)
        for m in _BARE_MARCH_RE.finditer(line):
            y = normalise_year(f"march 31, {m.group(1)}")
            if y and y not in seen:
                found.append(y); seen.add(y)

        # Try FY25 / FY2025
        for m in _YEAR_HEADER_PATTERNS[1].finditer(line):
            y = normalise_year(m.group(0))
            if y and y not in seen:
                found.append(y); seen.add(y)

        # Try 2024-25
        for m in _YEAR_HEADER_PATTERNS[2].finditer(line):
            y = normalise_year(m.group(0))
            if y and y not in seen:
                found.append(y); seen.add(y)

    # If nothing found with fancy patterns, fall back to bare 4-digit years
    if not found:
        for line in text.splitlines()[:50]:
            for m in _YEAR_HEADER_PATTERNS[3].finditer(line):
                y = normalise_year(m.group(1))
                if y and y not in seen:
                    found.append(y); seen.add(y)

    return found[:3]   # at most 3 years (current + 2 prior)


def _extract_numbers(line: str) -> list[float]:
    """
    Pull all numeric tokens from a line and return them as floats.

    Indian annual report balance sheets include a "Note No." column between the
    row label and the financial values, e.g.:
        "Property, Plant and Equipment   3   4,175.98   3,872.64"
    The small integer (3) is a schedule/note reference, NOT a financial value.
    We strip leading note-reference integers so the year-column assignment is
    correct: [4175.98, 3872.64] rather than [3, 4175.98].

    A number is treated as a note reference when ALL of:
      - It is a positive whole number (no decimal part)
      - Its value is < 200  (note refs rarely exceed ~100)
      - At least one of the remaining numbers has a decimal or is ≥ 200
        (i.e. the rest look like real financial values)
    """
    results = []
    for m in _NUM_RE.finditer(line):
        raw = m.group(0)
        val = parse_number(raw)
        if val is not None:
            results.append(val)

    # Note-reference filter: only fires when note_ref + at least 2 year values present.
    # Using len(rest) >= 2 prevents accidentally stripping a real first value when
    # only two numbers appear on the line (both are year values, no note ref).
    if len(results) >= 3:
        first = results[0]
        rest  = results[1:]
        if (first == int(first)   # whole number (no decimal part)
                and 0 < first < 100):  # small enough to be a note/schedule ref (1–99)
            results = rest

    return results


# ── Field label dictionaries ──────────────────────────────────────────────────
#
# Each entry: (canonical_field_name, [regex_patterns_to_match])
# Patterns are case-insensitive and matched against cleaned line text.
#
# Keep patterns specific enough to avoid false positives.

# ── COGS component patterns ───────────────────────────────────────────────────
# "Less: Cost of goods sold" is a derived value = sum of three separate P&L lines.
# These patterns detect the individual components; their values are summed after
# the full section text has been scanned.  The key "_cogs_*" is internal and never
# written to the output — only the summed "Less: Cost of goods sold" is.

_COGS_COMPONENT_KEYS: list[tuple[str, str]] = [
    ("_cogs_materials",        r"cost of materials consumed"),
    ("_cogs_purchases",        r"purchases? of stock.in.trade|purchase of traded goods"),
    ("_cogs_inventory_change", r"changes? in inventories?"),
]
_COGS_DIRECT_PATTERNS: list[str] = [
    r"^cost of goods sold\b",
    r"^cost of revenues?\b",
    r"^cost of sales?\b",
    r"^cost of services?\b",
]

# ──────────────────────────────────────────────────────────────────────────────
# P&L FIELDS
# Aliases cover Ind AS, older GAAP, IT, FMCG, pharma, manufacturing companies.
# Patterns are kept specific enough to avoid matching ratio tables / note text.
# ──────────────────────────────────────────────────────────────────────────────
_PNL_FIELDS: list[tuple[str, list[str]]] = [
    ("Revenue from operations", [
        r"revenue from operations",
        r"net revenue from operations",
        r"total revenue from operations",
        r"revenue from contracts? with customers?",
        r"revenue from software\b",           # IT companies
        r"revenue from services?\b",          # service companies
        r"\bnet\s+sales?\b",                  # FMCG / older companies
        r"\btotal\s+sales?\b",
        r"\bnet\s+turnover\b",                # "net turnover" is specific enough
        r"\bgross\s+revenue\s+from\s+operations?\b",
        r"\btotal\s+income\s+from\s+operations?\b",
        r"\btotal\s+operating\s+(?:revenue|income)\b",
        r"\bsales\s+(?:and\s+)?other\s+operating\s+(?:revenue|income)\b",
    ]),

    ("Less: Cost of goods sold",
     _COGS_DIRECT_PATTERNS + [r"cost of goods sold"]),

    # ── COGS components (internal — summed in _flush_segment) ─────────────────
    ("_cogs_materials",        [r"cost of materials consumed",
                                 r"raw materials? consumed",
                                 r"consumption of raw materials?"]),
    ("_cogs_purchases",        [r"purchases? of stock.in.trade",
                                 r"purchase of traded goods",
                                 r"purchases? of finished goods"]),
    ("_cogs_inventory_change", [r"changes? in inventories?",
                                 r"(?:increase|decrease) in inventories?"]),

    ("Employee benefits expense", [
        r"employee benefits? expense",
        r"\bstaff costs?\b",
        r"personnel (?:expense|costs?)",
        r"manpower (?:expense|costs?)",
        r"salaries.*wages.*bonus",
        r"wages.*salaries",
    ]),

    ("Other expenses (Net)", [
        r"other expenses?\b(?!\s*income)",
        r"other operating expenses?",
        r"selling.*general.*(?:and\s+)?administrative",
        r"general.*(?:and\s+)?administrative.*expenses?",
        r"administration\s+expenses?",
        r"selling.*distribution.*expenses?",
    ]),

    ("Less: Depreciation, amortization & impairment", [
        r"depreciation.*amorti[sz]ation",
        r"depreciation\s+and\s+amorti[sz]ation",
        r"depreciation.*impairment",
        r"\bdepreciation\b.*\bexpense\b",
        r"\bamorti[sz]ation\b.*\bexpense\b",
    ]),

    ("Less: Finance cost", [
        r"finance costs?",
        r"interest expense",
        r"finance charges?",
        r"borrowing costs?",
        r"interest on borrowings?",
        r"financial (?:expense|charges?|costs?)",
        r"interest\s+(?:and\s+)?finance\s+charges?",
    ]),

    ("Add: Other Income (Net)", [
        r"other income\b",
        r"other income \(net\)",
        r"non.operating income",
        r"miscellaneous\s+income\b",
    ]),

    ("Add: Share of net profit of associates", [
        r"^share of (?:net )?profit (?:of|from) associates?",
        r"equity in (?:net\s+)?earnings",
        r"^share of profit.*associates?",
        r"^share of profit.*joint ventures?",
        r"share of (?:net\s+)?profit.*equity method",
    ]),

    ("Less: Tax", [
        r"\btotal\s+(?:income\s+)?tax\s*(?:expense)?\b",
        r"^provision\s+for\s+(?:income\s+)?tax\b",
        r"less:\s*income\s+tax\s+expense",
        r"income\s+tax\s+expense\b",
        r"^tax\s+expense\b",
        r"^income\s+tax\b(?!\s+assets?)",
    ]),

    ("Profit after Tax", [
        r"profit after (?:income\s+)?tax\b",
        r"profit for the (?:year|period)\b",
        r"net profit after tax\b",
        r"profit \(after tax\)",
        r"net (?:income|earnings?) after tax\b",
    ]),

    ("Diluted EPS", [
        r"diluted\s+(?:earnings per share|eps|e\.p\.s\.)",
        r"earnings per equity share.*diluted",
        r"diluted\s+eps\b",
    ]),
]

# ──────────────────────────────────────────────────────────────────────────────
# BALANCE SHEET FIELDS
# Rule: only use patterns that uniquely identify a financial-statement row.
# Patterns like bare "stocks", "reserves", "net worth", "plant and machinery",
# "long-term borrowings" are intentionally omitted — they also appear in risk
# disclosures, ratio tables, and notes pages that may be included in the section
# text window, causing false matches with note-reference numbers.
# ──────────────────────────────────────────────────────────────────────────────
_BS_FIELDS: list[tuple[str, list[str]]] = [
    # ── Non-Current Assets ────────────────────────────────────────────────────
    ("Property, Plant & Equipment", [
        r"property,?\s+plant\s+(?:and|&)\s+equipment\b(?!\s+net)",
    ]),
    ("Capital Work in Progress", [
        r"capital work.in.progress",
        r"\bcwip\b",
    ]),
    ("Goodwill", [
        r"\bgoodwill\b(?!\s+on\s+(?:impairment|testing))",  # allow "on consolidation"
    ]),
    ("Intangible Asset", [
        r"intangible assets?\b(?!\s+under)",
    ]),
    ("Intangible Assets under development", [
        r"intangible assets?\s+under\s+development",
    ]),
    ("Right of Use Assets", [
        r"right.of.use assets?",
        r"\brou\s+assets?\b",
    ]),
    ("Investments - equity method", [
        r"investments?.*equity method",
        r"investments?.*associates.*equity",
        r"investments?.*associates?\b",
        r"investments?.*joint ventures?",
    ]),
    ("Other Investments", [
        r"other investments?\b",
        r"current investments?\b",
        r"investments?.*fair value",
        r"investments?.*mutual\s+funds?\b",
    ]),
    ("Other Financial Assets",  [r"other financial assets?\b"]),
    ("Income Tax Assets (Net)", [
        r"income tax assets?\s*\(?net\)?",
        r"advance\s+income.tax",
        r"current tax assets?\s*\(?net\)?",
    ]),
    ("Deferred tax assets (Net)", [r"deferred tax assets?\s*\(?net\)?"]),
    ("Other non-current assets",  [r"other non.current assets?", r"other\s+noncurrent\s+assets?"]),
    ("Total Non-Current Assets",  [r"total non.current assets?"]),

    # ── Current Assets ────────────────────────────────────────────────────────
    ("Inventories", [
        r"\binventories\b",
        r"\bstock.in.trade\b",               # trading companies
    ]),
    ("Trade Receivables", [
        r"trade receivables?",
        r"sundry\s+debtors?\b",              # older GAAP / pre-Ind AS
        r"trade\s+debtors?\b",
        r"accounts\s+receivable\b",
    ]),
    ("Cash & Cash Equivalents", [
        r"cash and cash equivalents?",
        r"cash\s*&\s*cash equivalents?",
        r"cash\s+and\s+bank\s+balances?\b(?!\s+other)",
    ]),
    ("Bank Balances Other Than above", [
        r"bank balances?\s+other than",
        r"other\s+bank\s+balances?",
    ]),
    ("Other Current Assets",  [r"other current assets?\b"]),
    ("Total Current Assets",  [r"total current assets?"]),
    ("Total Assets",          [r"total assets\b"]),

    # ── Equity ────────────────────────────────────────────────────────────────
    ("Equity Share capital", [
        r"(?:equity\s+)?share capital\b",
        r"paid.up\s+(?:equity\s+)?share\s+capital",
        r"paid.up\s+capital\b",
    ]),
    ("Other Equity", [
        r"other equity\b",
        r"reserves\s+and\s+surplus",         # pre-Ind AS / older companies
        r"shareholders.?\s*funds?\b",         # older GAAP
        r"total\s+other\s+equity\b",
    ]),
    ("Total Equity", [
        r"total equity\b",
        r"total shareholders.?\s*equity",
    ]),

    # ── Non-Current Liabilities ───────────────────────────────────────────────
    ("Borrowings (Non-Current)", [
        r"borrowings?\b.*non.current",
        r"non.current.*borrowings?\b",
    ]),
    ("Lease Liabilities (Non-Current)", [
        r"lease liabilit(?:y|ies).*non.current",
        r"non.current.*lease liabilit",
    ]),
    ("Provisions (Non-Current)", [
        r"provisions?\s+non.current",
        r"non.current.*provisions?\b",
    ]),
    ("Deferred tax liabilities (Net)", [r"deferred tax liabilit(?:y|ies)\s*\(?net\)?"]),
    ("Other Non-Current Liabilities",  [r"other non.current liabilit", r"other noncurrent liabilit"]),
    ("Total Non-Current Liabilities",  [r"total non.current liabilit"]),

    # ── Current Liabilities ───────────────────────────────────────────────────
    ("Borrowings (Current)", [
        r"borrowings?\b.*current(?!\s+assets)",
        r"current.*borrowings?\b(?!.*non)",
    ]),
    ("Dues of Micro & small enterprises", [
        r"dues.*micro.*small",
        r"\bmsme\b",
        r"micro.*small.*enterprises?",
    ]),
    ("Dues of other creditors", [
        r"dues.*other\s+creditors?",
        r"dues.*creditors?\s+other\s+than",
        r"creditors?\s+other\s+than\s+micro",
        r"other\s+than\s+micro.*small.*enterprise",
    ]),
    ("Other Financial Liabilities", [r"other financial liabilit"]),
    ("Other Current Liabilities",   [r"other current liabilit"]),
    ("Provisions (Current)", [
        r"provisions?\s+(?:\(current\)|current)",
    ]),
    ("Current Tax Liabilities (Net)", [r"current tax liabilit"]),
    ("Total Current Liabilities",       [r"total current liabilit"]),
    ("Total Equity & Liabilities",      [r"total equity\s*(?:and|&)\s*liabilit"]),
]

_CF_FIELDS: list[tuple[str, list[str]]] = [
    ("Net Cash from Operating Activities", [
        r"net cash (?:generated |flow )?from operating",
        r"net cash.*operating activities?",
        r"cash (?:generated|used) (?:in|from) operations?",
        r"net cash(?:flow)?\s+from\s+operations?\b",
    ]),
    ("Net Cash from Investing Activities", [
        r"net cash (?:used |flow )?(?:in|from) investing",
        r"net cash.*investing activities?",
        r"cash (?:used|generated) in investing",
    ]),
    ("Net Cash from Financing Activities", [
        r"net cash (?:used |flow )?(?:in|from) financing",
        r"net cash.*financing activities?",
        r"cash (?:used|generated) in financing",
    ]),
    ("Net Change in Cash", [
        r"net (?:increase|decrease|change) in cash",
        r"net change in cash",
        r"(?:increase|decrease) in cash and cash equivalents?",
        r"net (?:increase|decrease) in cash and bank",
    ]),
    ("Opening Cash", [
        r"cash.*(?:beginning|opening)\s+of\s+(?:year|period)",
        r"opening balance.*cash",
        r"cash.*balance.*(?:beginning|opening)",
        r"cash and cash equivalents?.*beginning",
    ]),
    ("Closing Cash", [
        r"cash.*(?:end|close)\s+of\s+(?:year|period)",
        r"closing balance.*cash",
        r"cash.*balance.*(?:end|close)",
        r"cash and cash equivalents?.*end\s+of",
    ]),
]

_SECTION_FIELD_MAP = {
    "Annual P&L":            _PNL_FIELDS,
    "Annual Balance Sheet":  _BS_FIELDS,
    "Annual Cash Flow":      _CF_FIELDS,
}

# Pre-compile patterns for the STATIC sections (P&L, BS, CF).
# Operating metrics is excluded here because it changes per active company.
_COMPILED: dict[str, list[tuple[str, list[re.Pattern]]]] = {}
for _sec, _fields in _SECTION_FIELD_MAP.items():
    _COMPILED[_sec] = [
        (label, [re.compile(p, re.IGNORECASE) for p in patterns])
        for label, patterns in _fields
    ]


def _get_compiled_fields(section_type: str, sector: str = "manufacturing") -> list[tuple[str, list[re.Pattern]]]:
    """
    Return compiled field patterns for *section_type*.
    For non-manufacturing sectors, loads sector-specific patterns from config/sectors.py
    and falls back to the default manufacturing patterns if none defined.
    """
    if sector == "manufacturing":
        return _COMPILED.get(section_type, [])

    # Load sector-specific patterns
    try:
        from config.sectors import get_sector_fields
        fields_map = get_sector_fields(sector)
        key = {"Annual P&L": "pl", "Annual Balance Sheet": "bs", "Annual Cash Flow": "cf"}.get(section_type)
        sector_fields = fields_map.get(key) if key else None

        if sector_fields is None:
            # Sector has no custom patterns for this section — use manufacturing default
            return _COMPILED.get(section_type, [])

        # Compile sector-specific patterns on demand (cached per call via module-level dict)
        cache_key = f"{sector}:{section_type}"
        if cache_key not in _SECTOR_COMPILED:
            _SECTOR_COMPILED[cache_key] = [
                (label, [re.compile(p, re.IGNORECASE) for p in patterns])
                for label, patterns in sector_fields
            ]
        return _SECTOR_COMPILED[cache_key]
    except Exception:
        return _COMPILED.get(section_type, [])


# Cache for sector-specific compiled patterns
_SECTOR_COMPILED: dict[str, list[tuple[str, list[re.Pattern]]]] = {}


# ── Look-ahead behaviour flags ────────────────────────────────────────────────

# Fields where the TOTAL value is the LAST set of numbers in their block rather
# than the first.  For these fields the look-ahead collects the full block and
# then takes the last len(years) financial values.
# Example: "Less: Income Tax Expense" header → current tax → deferred tax → TOTAL
_TAKE_LAST_FIELDS: frozenset[str] = frozenset({
    "Less: Tax",
})

# When the BS subsection transitions from non-current → current assets, reset
# these canonical names from found_labels so they can be matched again.
# NOTE: Only include fields where the CURRENT-assets value is DIFFERENT in the
# template from the non-current value (i.e. separate template rows).
# "Other Financial Assets" deliberately NOT included — the NC value is the primary
# one and must not be overwritten by the current-assets occurrence.
_RESET_ON_CURRENT_ASSETS: frozenset[str] = frozenset()

# ── Balance Sheet subsection tracking ────────────────────────────────────────
#
# Balance sheets are organised into sections whose headers appear on standalone
# lines ("Non-Current Assets", "Current Liabilities", etc.).  Many row labels
# repeat across sections (e.g. "Borrowings", "Provisions") so we must track
# which section we are currently inside to assign the correct canonical name.

_BS_SUBSECTION_HEADERS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\s*non.current\s+assets?\s*$",           re.IGNORECASE), "non_current_assets"),
    (re.compile(r"^\s*current\s+assets?\s*$",               re.IGNORECASE), "current_assets"),
    # "Equity" standalone header — must NOT match "Equity Share capital" etc.
    # Require the word "equity" to be the whole line (optionally followed by colon/spaces).
    (re.compile(r"^\s*equity\s*:?\s*$",                     re.IGNORECASE), "equity"),
    (re.compile(r"^\s*non.current\s+liabilit",              re.IGNORECASE), "non_current_liabilities"),
    # "Current Liabilities" — must not match "Non-Current Liabilities"
    (re.compile(r"^\s*(?:total\s+)?current\s+liabilit",     re.IGNORECASE), "current_liabilities"),
]

# Bare-label patterns whose canonical name depends on subsection context.
# Each entry: (bare_regex, {subsection_name: canonical_field_name})
_BS_CONTEXT_FIELDS: list[tuple[re.Pattern, dict[str, str]]] = [
    (
        re.compile(r"^borrowings?\b", re.IGNORECASE),
        {
            "non_current_liabilities": "Borrowings (Non-Current)",
            "current_liabilities":     "Borrowings (Current)",
        },
    ),
    (
        re.compile(r"^lease\s+liabilit", re.IGNORECASE),
        {
            "non_current_liabilities": "Lease Liabilities (Non-Current)",
            "current_liabilities":     "Lease Liabilities (Current)",
        },
    ),
    (
        re.compile(r"^provisions?\b", re.IGNORECASE),
        {
            "non_current_liabilities": "Provisions (Non-Current)",
            "current_liabilities":     "Provisions (Current)",
        },
    ),
    (
        re.compile(r"^other\s+financial\s+assets?\b", re.IGNORECASE),
        {
            "non_current_assets": "Other Financial Assets",
            "current_assets":     "Other Financial Assets",
        },
    ),
    (
        re.compile(r"^other\s+(?:non.current|noncurrent)\s+assets?\b", re.IGNORECASE),
        {
            "non_current_assets": "Other non-current assets",
        },
    ),
    # Bare "Investments" label (no qualifier) — routed by subsection context.
    # NC assets → equity-method investments (associates/JVs in most Indian companies)
    # Current assets → short-term / other investments (mutual funds etc.)
    (
        re.compile(r"^investments?\b", re.IGNORECASE),
        {
            "non_current_assets": "Investments - equity method",
            "current_assets":     "Other Investments",
        },
    ),
]


def _bs_subsection_for_line(line: str) -> Optional[str]:
    """
    If *line* is a Balance Sheet section header, return the subsection key;
    otherwise return None.
    """
    stripped = line.strip()
    for pat, key in _BS_SUBSECTION_HEADERS:
        if pat.match(stripped):
            return key
    return None


def _bs_context_match(line: str, subsection: Optional[str]) -> Optional[str]:
    """
    Try to resolve a line to a canonical BS field using subsection context.
    Returns the canonical field name, or None if no match.
    """
    if not subsection:
        return None
    stripped = line.strip()
    for pat, sub_map in _BS_CONTEXT_FIELDS:
        if pat.match(stripped):
            return sub_map.get(subsection)
    return None


# ── Consolidation detection ────────────────────────────────────────────────────

def _detect_statement_type(text: str) -> str:
    """Return 'consolidated' or 'standalone' based on text signals."""
    text_low = text.lower()
    has_consolidated = bool(re.search(r"\bconsolidated\b", text_low))
    has_standalone   = bool(re.search(r"\bstandalone\b",   text_low))
    if has_consolidated and not has_standalone:
        return "consolidated"
    if has_standalone and not has_consolidated:
        return "standalone"
    # Default: consolidated (most common for investor-facing docs)
    return "consolidated"


# ── Statement segment boundary patterns ───────────────────────────────────────
#
# Indian annual reports embed BOTH standalone and consolidated statements in
# the same page range.  When we encounter one of these exact header lines we
# flush the current extraction segment and start a new one with the appropriate
# statement_type.  Patterns are matched against the stripped line text.
#
# Each tuple: (pattern, stmt_type, section_category)
# section_category is used to detect when a different section type bleeds into
# the current section's text (e.g. Cash Flow pages inside a Balance Sheet
# section due to WINDOW expansion). In that case we enter a "skip" state so
# that CF values are not written into BS results.

_STMT_BOUNDARY_RE: list[tuple[re.Pattern, str, str]] = [
    # "Balance Sheet" / "Standalone Balance Sheet" (own line, not "Consolidated …")
    (re.compile(r"^(?:standalone\s+)?balance\s+sheet$",          re.IGNORECASE), "standalone",    "bs"),
    # "Consolidated Balance Sheet"
    (re.compile(r"^consolidated\s+balance\s+sheet$",             re.IGNORECASE), "consolidated",  "bs"),
    # "Statement of Profit and Loss" / "Standalone Statement of …"
    (re.compile(r"^(?:standalone\s+)?statement\s+of\s+profit",   re.IGNORECASE), "standalone",    "pl"),
    # "Consolidated Statement of Profit and Loss"
    (re.compile(r"^consolidated\s+statement\s+of\s+profit",      re.IGNORECASE), "consolidated",  "pl"),
    # "Statement of Cash Flows"
    (re.compile(r"^(?:standalone\s+)?statement\s+of\s+cash",     re.IGNORECASE), "standalone",    "cf"),
    # "Consolidated Statement of Cash Flows"
    (re.compile(r"^consolidated\s+statement\s+of\s+cash",        re.IGNORECASE), "consolidated",  "cf"),
]

# Maps section_type → the boundary category that belongs to it.
# When a boundary whose category does NOT match the current section_type is
# detected, we flush and enter "_skip_" mode to avoid writing values from a
# different statement type into this section's results.
_SECTION_BOUNDARY_CATEGORY: dict[str, str] = {
    "Annual Balance Sheet": "bs",
    "Annual P&L":           "pl",
    "Annual Cash Flow":     "cf",
}

_SKIP_STMT = "_skip_"   # sentinel: extraction paused (irrelevant section boundary)


# ── Core extraction ────────────────────────────────────────────────────────────

def extract_section(
    text: str,
    section_type: str,
    year_override: str | None = None,
    stmt_override: str | None = None,
    sector: str = "manufacturing",
) -> tuple[list[dict], float]:
    """
    Extract financial fields from text using rule-based matching.

    Returns:
        (year_data_list, coverage)

        year_data_list: list of dicts, one per discovered year:
            {year: str, statement_type: str, data: {field: float}}

        coverage: fraction of expected fields that were found (0.0–1.0).
    """
    # Generic sector has no rule-based fields — signal 0% so LLM always runs
    if sector == "generic":
        return [], 0.0

    if section_type not in _COMPILED and sector == "manufacturing":
        return [], 0.0

    compiled_fields = _get_compiled_fields(section_type, sector=sector)
    if not compiled_fields:
        return [], 0.0
    # Internal (_-prefixed) fields are implementation helpers, not real output fields.
    expected_count  = sum(1 for lbl, _ in compiled_fields if not lbl.startswith("_"))

    # Determine years — user override takes priority
    if year_override:
        # Derive prior year from the override.
        try:
            yr_int = int(year_override[1:])   # "F2025" → 2025
            prior_year = f"F{yr_int - 1}"     # "F2024"
        except (ValueError, IndexError):
            years = [year_override]
        else:
            expected = {year_override, prior_year}
            # Detect the ACTUAL column order from the text header.
            # PDFs vary: some show current year first, others prior year first.
            # _find_years_in_text returns years in left-to-right order as they
            # appear on the header line(s), so we trust that order.
            text_years = _find_years_in_text(text)
            ordered = [y for y in text_years if y in expected]
            if len(ordered) == 2:
                years = ordered
                logger.info(f"  Rule-based: column order detected from text: {years}")
            elif len(ordered) == 1:
                # Only one year found in header — still add the other
                other = prior_year if ordered[0] == year_override else year_override
                years = ordered + [other]
                logger.info(f"  Rule-based: partial detection, using: {years}")
            else:
                # No header years detected — default [current, prior]
                years = [year_override, prior_year]
                logger.info(f"  Rule-based: no header detected, defaulting to: {years}")
    else:
        years = _find_years_in_text(text)
        if not years:
            years = _find_years_in_text("\n".join(text.splitlines()))
        if not years:
            logger.debug(f"  Rule-based: no year columns detected in {section_type}")
            return [], 0.0

    # Determine statement type — user override takes priority
    stmt_type = stmt_override if stmt_override else _detect_statement_type(text)
    logger.debug(f"  Rule-based: {section_type} | years={years} | type={stmt_type}")

    lines = text.splitlines()
    is_balance_sheet = (section_type == "Annual Balance Sheet")

    # ── Segment-aware extraction ───────────────────────────────────────────────
    # Indian annual reports embed both standalone AND consolidated statements in
    # the same text block.  We detect header lines (e.g. "Balance Sheet",
    # "Consolidated Balance Sheet") to split the text into segments and produce
    # separate results for each statement type.
    #
    # all_segment_data: (stmt_type, year_str) → {field: value}
    all_segment_data: dict[tuple[str, str], dict[str, float]] = {}
    all_found_labels: set[str] = set()

    # Current segment state
    current_stmt: str = stmt_type   # initial detection; overridden by boundaries
    per_year: list[dict[str, float]] = [{} for _ in years]
    found_labels: set[str] = set()
    current_bs_subsection: Optional[str] = None

    def _flush_segment() -> None:
        """Persist the current segment's data into all_segment_data."""
        # ── COGS derived sum (P&L only) ───────────────────────────────────────
        # If the three COGS component fields were matched in this segment, sum them
        # into "Less: Cost of goods sold" UNLESS a direct COGS line was already
        # captured.  Internal (_-prefixed) keys are removed from the output dict.
        if section_type == "Annual P&L":
            cogs_written = False
            for year_data in per_year:
                comp_vals = [
                    year_data.pop("_cogs_materials",        None),
                    year_data.pop("_cogs_purchases",        None),
                    year_data.pop("_cogs_inventory_change", None),
                ]
                valid = [v for v in comp_vals if v is not None]
                if len(valid) >= 2 and "Less: Cost of goods sold" not in year_data:
                    year_data["Less: Cost of goods sold"] = sum(valid)
                    cogs_written = True
            if cogs_written:
                all_found_labels.add("Less: Cost of goods sold")
                logger.debug(
                    f"    COGS derived sum flushed for segment '{current_stmt}'"
                )

        for year_str, year_data in zip(years, per_year):
            if year_data:
                key = (current_stmt, year_str)
                if key not in all_segment_data:
                    all_segment_data[key] = {}
                all_segment_data[key].update(year_data)
        all_found_labels.update(found_labels)

    for line_idx, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Skip page-marker lines inserted by the pdf_parser ("[PAGE N]")
        if line_stripped.startswith("[PAGE ") and line_stripped.endswith("]"):
            continue

        # ── Detect statement segment boundary ─────────────────────────────────
        if not stmt_override:
            for bnd_pat, bnd_stmt, bnd_sec in _STMT_BOUNDARY_RE:
                if bnd_pat.match(line_stripped):
                    _flush_segment()
                    per_year = [{} for _ in years]
                    found_labels = set()
                    current_bs_subsection = None
                    # Check if this boundary belongs to the same section category.
                    # E.g. when processing "Annual Balance Sheet", a "Statement of
                    # Cash Flows" boundary (bnd_sec="cf") is irrelevant — we enter
                    # skip mode so CF values aren't written into BS results.
                    expected_sec = _SECTION_BOUNDARY_CATEGORY.get(section_type)
                    if expected_sec is None or bnd_sec == expected_sec:
                        current_stmt = bnd_stmt
                    else:
                        current_stmt = _SKIP_STMT
                    logger.debug(
                        f"  Segment boundary '{line_stripped}': "
                        f"bnd_sec={bnd_sec} expected={expected_sec} → {current_stmt}"
                    )
                    break

        # Skip all field matching when we're in a different-section-type block
        if current_stmt == _SKIP_STMT:
            continue

        # ── Balance Sheet: detect subsection headers ──────────────────────────
        if is_balance_sheet:
            new_sub = _bs_subsection_for_line(line_stripped)
            if new_sub:
                # When crossing from non-current → current assets, allow re-matching
                # of fields that legitimately appear in both sections (e.g. OFA).
                if (new_sub == "current_assets"
                        and current_bs_subsection in ("non_current_assets", None)):
                    found_labels -= _RESET_ON_CURRENT_ASSETS
                    logger.debug(
                        f"    Subsection transition → current_assets; "
                        f"reset: {_RESET_ON_CURRENT_ASSETS & found_labels | _RESET_ON_CURRENT_ASSETS}"
                    )
                current_bs_subsection = new_sub
                logger.debug(f"    BS subsection → {current_bs_subsection}")
                continue   # header row has no numeric values; skip to next line

        # Try to match a known field label
        matched_label: Optional[str] = None

        # For Balance Sheet, first try context-aware match for ambiguous labels
        if is_balance_sheet:
            ctx_label = _bs_context_match(line_stripped, current_bs_subsection)
            if ctx_label and ctx_label not in found_labels:
                matched_label = ctx_label

        # Fall back to compiled regex patterns
        if not matched_label:
            for label, patterns in compiled_fields:
                if label in found_labels:
                    continue
                for pat in patterns:
                    if pat.search(line_stripped):
                        matched_label = label
                        break
                if matched_label:
                    break

        if not matched_label:
            continue

        # Strip "face value" parentheticals before number extraction so that
        # EPS rows like "…(Face value of ₹ 10/- each)" don't yield 10 as the value.
        clean_line = _FACE_VALUE_RE.sub("", line_stripped).strip()

        # Extract numbers from this line first (handles "Label  note  val1  val2" on one line)
        nums = _extract_numbers(clean_line)

        if not nums:
            # Multi-line PDF layout: values appear on separate lines after the label.
            #
            # Extended look-ahead strategy
            # ─────────────────────────────
            # We collect up to len(years)*2 financial values (generous budget so we
            # can detect interleaved sub-totals and multi-component tax blocks) but
            # still stop immediately at the next field-label line.
            #
            # After collection we apply smart value selection:
            #  • _TAKE_LAST_FIELDS  (e.g. Tax): take the LAST len(years) values —
            #    the total appears after all sub-components.
            #  • Interleaved sub-total pattern: some PDFs render
            #    [val_F25, SUBTOTAL_F25, val_F24, SUBTOTAL_F24] for the last item
            #    in a financial-assets sub-group.  When we see exactly 2*N values
            #    and every pair has sub-total >> value (ratio > 3×), take even-indexed.
            #  • Otherwise: take first len(years) values.
            #
            # Nil values: a bare "-" or "–" line means the value is nil / zero.
            raw_ahead: list[float] = []
            # For _TAKE_LAST_FIELDS we need to collect the whole multi-component block;
            # use a much higher ceiling.
            fin_ceil = len(years) * (6 if matched_label in _TAKE_LAST_FIELDS else 2)
            for look_ahead in range(1, 18):
                if line_idx + look_ahead >= len(lines):
                    break
                la_line = lines[line_idx + look_ahead].strip()
                if not la_line:
                    continue
                # Stop at the next BS subsection boundary marker
                if is_balance_sheet and _bs_subsection_for_line(la_line):
                    break
                # Stop if this line matches the next context field
                if is_balance_sheet and _bs_context_match(la_line, current_bs_subsection):
                    break
                # Stop if any compiled field pattern matches this line (next row starts).
                # Skip matched_label itself — for fields like "Less: Tax", the total-line
                # (e.g. "Total Tax Expense") matches the same pattern as the header; we
                # must NOT stop there or we'd never collect the total value.
                hit_next_label = False
                for _lbl, _pats in compiled_fields:
                    if _lbl == matched_label:
                        continue   # don't self-stop mid look-ahead block
                    if any(p.search(la_line) for p in _pats):
                        hit_next_label = True
                        break
                if hit_next_label:
                    break

                # Nil / dash value → zero for that year column
                if la_line in ("-", "–", "—", "Nil", "nil"):
                    raw_ahead.append(0.0)
                    continue

                # Collect raw numbers (no note-ref filter yet — apply after combining)
                line_nums: list[float] = []
                for m in _NUM_RE.finditer(la_line):
                    v = parse_number(m.group(0))
                    if v is not None:
                        line_nums.append(v)
                raw_ahead.extend(line_nums)

                fin_count = sum(1 for v in raw_ahead if v >= 100 or v != int(v))
                if fin_count >= fin_ceil:
                    break

            # ── Note-ref stripping (iterative, handles "3 & 43" double refs) ───
            while len(raw_ahead) >= 2:
                first_la = raw_ahead[0]
                if first_la == int(first_la) and 0 < first_la < 200:
                    rest_la = raw_ahead[1:]
                    if any(v >= 100 or v != int(v) for v in rest_la[:2]):
                        raw_ahead = rest_la
                    else:
                        break
                else:
                    break

            # ── Smart value selection ────────────────────────────────────────────
            n = len(years)
            if not raw_ahead:
                pass   # nothing found
            elif matched_label in _TAKE_LAST_FIELDS:
                # Take LAST n financial values (total appears after sub-components)
                fin_vals = [v for v in raw_ahead if v >= 100 or v != int(v)]
                raw_ahead = fin_vals[-n:] if len(fin_vals) >= n else fin_vals
            elif len(raw_ahead) == 2 * n and n >= 1:
                # Check for interleaved sub-total pattern:
                # [val_F1, SUBTOTAL_F1, val_F2, SUBTOTAL_F2]
                # where SUBTOTAL_Fi >> val_Fi (ratio > 3×)
                evens = raw_ahead[::2]   # candidate year values
                odds  = raw_ahead[1::2]  # candidate sub-totals
                if all(
                    evens[i] != 0 and odds[i] > 3 * abs(evens[i])
                    for i in range(n)
                ):
                    raw_ahead = evens
                # else fall through to first-n selection below

            nums = raw_ahead[:n] if len(raw_ahead) > n else raw_ahead

        if not nums:
            continue

        # Assign numbers to years in column order
        for col_idx, year_data in enumerate(per_year):
            if col_idx < len(nums):
                year_data[matched_label] = nums[col_idx]

        found_labels.add(matched_label)
        logger.debug(
            f"    Matched '{matched_label}'"
            + (f" [sub={current_bs_subsection}]" if is_balance_sheet else "")
            + f": {nums[:3]}"
        )

    # Flush the final segment
    _flush_segment()

    public_found = sum(1 for l in all_found_labels if not l.startswith("_"))
    coverage = public_found / expected_count if expected_count else 0.0
    logger.info(
        f"  Rule-based {section_type}: {len(all_found_labels)}/{expected_count} fields "
        f"({coverage:.0%}) | years={years}"
    )

    # ── Unit normalisation ─────────────────────────────────────────────────────
    # Apply divisor based on unit declaration in source text (e.g. ₹ in Thousands).
    from utils.helpers import detect_unit_divisor
    divisor = detect_unit_divisor(text)
    if divisor != 1.0:
        logger.info(f"  Rule-based unit: ÷{divisor:,.0f}")

    # Build result list from all discovered segments
    result_list = []
    for (seg_stmt, year_str), data in all_segment_data.items():
        if data:
            if divisor != 1.0:
                data = {k: round(v / divisor, 4) for k, v in data.items()}
            result_list.append({
                "year": year_str,
                "statement_type": seg_stmt,
                "data": data,
            })

    return result_list, coverage


# ── Public API ────────────────────────────────────────────────────────────────

def extract_all_rule_based(
    parsed_doc,
) -> tuple[list, float]:
    """
    Run rule-based extraction on all sections of a ParsedDocument.
    Respects _year_override and _stmt_override attributes set by the UI.
    """
    from extractor.llm_extractor import ExtractionResult  # avoid circular at import time

    year_override = getattr(parsed_doc, "_year_override", None)
    stmt_override = getattr(parsed_doc, "_stmt_override", None)

    all_rb: list[ExtractionResult] = []
    coverages: list[float] = []

    sector = getattr(parsed_doc, "sector", "manufacturing")

    for section in parsed_doc.sections:
        raw_results, cov = extract_section(
            section.text,
            section.section_type,
            year_override=year_override,
            stmt_override=stmt_override,
            sector=sector,
        )
        coverages.append(cov)
        for entry in raw_results:
            all_rb.append(ExtractionResult(
                section=section.section_type,
                statement_type=entry["statement_type"],
                year=entry["year"],
                currency="INR Crores",
                data=entry["data"],
                notes="rule-based extraction",
                confidence=cov,
            ))

    mean_cov = sum(coverages) / len(coverages) if coverages else 0.0
    logger.info(
        f"Rule-based extraction: {len(all_rb)} result(s), "
        f"mean coverage={mean_cov:.0%}"
        + (f" (year override: {year_override})" if year_override else "")
    )
    return all_rb, mean_cov
