"""
Tickertape Connector — fetches structured annual financial data from
api.tickertape.in for NSE/BSE-listed Indian companies.

Tickertape exposes a public JSON API that returns annual P&L, Balance Sheet,
and Cash Flow data, typically in INR Crores.  It covers a different dataset
than Screener.in and serves as an independent validation source.

URL pattern:
  https://api.tickertape.in/stocks/{NSE_SYMBOL}/financials

Confidence: HIGH (annual data from regulatory filings)
"""

from __future__ import annotations

import re
from typing import Optional

from utils.helpers import get_logger
from web.base import CompanyIdentifier, WebDataResult

logger = get_logger(__name__)


def _require_requests():
    try:
        import requests as _r
        return _r
    except ImportError:
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "requests>=2.31.0"]
        )
        import importlib, site
        importlib.invalidate_caches()
        for sp in site.getsitepackages():
            if sp not in sys.path:
                sys.path.insert(0, sp)
        import requests as _r
        return _r


_TIMEOUT = 15
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin":          "https://tickertape.in",
    "Referer":         "https://tickertape.in/",
}

_API_BASE = "https://api.tickertape.in/stocks"

# ── Field maps: Tickertape JSON keys → template hints ────────────────────────

# Tickertape returns keys like "revenue", "netProfit", "totalAssets" etc.
# We map them to the same template hint labels used by other connectors.

_PL_MAP: dict[str, str] = {
    "revenue":              "Revenue from Operations",
    "netRevenue":           "Revenue from Operations",
    "totalRevenue":         "Total Revenue",
    "otherIncome":          "Other Income",
    "totalExpenses":        "Total Expenses",
    "operatingProfit":      "Operating Profit",
    "ebitda":               "EBITDA",
    "interestExpense":      "Finance Costs",
    "depreciation":         "Depreciation and Amortisation",
    "pbt":                  "Profit Before Tax",
    "tax":                  "Tax Expense",
    "pat":                  "Profit After Tax",
    "netProfit":            "Profit After Tax",
    "eps":                  "EPS (Basic)",
    "epsBasic":             "EPS (Basic)",
    "epsDiluted":           "EPS (Diluted)",
    # Banking
    "interestIncome":       "Interest Income",
    "interestExpended":     "Interest Expended",
    "netInterestIncome":    "Net Interest Income",
    "provisions":           "Provisions and Contingencies",
}

_BS_MAP: dict[str, str] = {
    "totalAssets":          "Total Assets",
    "totalEquity":          "Total Equity",
    "totalLiabilities":     "Total Liabilities",
    "debt":                 "Total Borrowings",
    "borrowings":           "Total Borrowings",
    "cash":                 "Cash and Cash Equivalents",
    "cashEquivalents":      "Cash and Cash Equivalents",
    "investments":          "Investments",
    "netFixedAssets":       "Fixed Assets",
    "reserves":             "Reserves and Surplus",
    "shareCapital":         "Share Capital",
    # Banking
    "deposits":             "Deposits",
    "advances":             "Advances",
}

_CF_MAP: dict[str, str] = {
    "operatingCashflow":    "Net Cash from Operating Activities",
    "cfo":                  "Net Cash from Operating Activities",
    "investingCashflow":    "Net Cash from Investing Activities",
    "cfi":                  "Net Cash from Investing Activities",
    "financingCashflow":    "Net Cash from Financing Activities",
    "cff":                  "Net Cash from Financing Activities",
    "capex":                "Capital Expenditure",
    "freeCashflow":         "Free Cash Flow",
}

_ALL_MAPS = {**_PL_MAP, **_BS_MAP, **_CF_MAP}


# ── Year label parser ─────────────────────────────────────────────────────────

def _parse_tt_year(period_str: str) -> Optional[str]:
    """
    Tickertape period labels:
      "FY25", "FY2025", "Mar 2025", "2025", "TTM"
    """
    from utils.helpers import normalise_year
    s = str(period_str).strip()
    if s.upper() in ("TTM", "LTM"):
        return None   # trailing 12 months — skip
    return normalise_year(s)


# ── Value parser ──────────────────────────────────────────────────────────────

def _to_crore(value, multiplier: float = 1.0) -> Optional[float]:
    """Convert a Tickertape value to INR Crores."""
    try:
        v = float(str(value).replace(",", ""))
        if v != v:   # NaN check
            return None
        return round(v * multiplier, 4)
    except (TypeError, ValueError):
        return None


# ── Main connector ────────────────────────────────────────────────────────────

class TickertapeConnector:
    """
    Fetches annual financial data from api.tickertape.in.
    Uses the NSE symbol (e.g. HDFCBANK) as the identifier.
    """

    def __init__(self):
        pass

    def fetch(
        self,
        company: CompanyIdentifier,
        years: list[str],
    ) -> list[WebDataResult]:
        """Fetch annual financials from Tickertape for the given company + years."""
        symbol = company.nse_symbol
        if not symbol:
            logger.info(f"TickertapeConnector: no NSE symbol for '{company.name}' — skipping")
            return []

        year_set = set(years)
        results: list[WebDataResult] = []

        # Fetch all three statement types
        for stmt_type in ("income-statement", "balance-sheet", "cash-flow"):
            data = self._fetch_statement(symbol, stmt_type)
            if data:
                parsed = self._parse_statement(data, stmt_type, year_set)
                results.extend(parsed)

        if results:
            logger.info(
                f"TickertapeConnector: '{company.name}' ({symbol}) → "
                f"{len(results)} data points"
            )
        else:
            logger.info(
                f"TickertapeConnector: no data for '{company.name}' ({symbol})"
            )
        return results

    def _fetch_statement(
        self,
        symbol: str,
        stmt_type: str,
    ) -> Optional[dict]:
        """Fetch one statement type from Tickertape API.

        Tries multiple URL / ticker formats because Tickertape has changed
        their API structure over time.
        """
        requests = _require_requests()
        sym_upper = symbol.upper()

        # URL candidates — Tickertape uses ":NSI" suffix for NSE symbols
        url_candidates = [
            (f"{_API_BASE}/{sym_upper}:NSI/financials",  {"type": stmt_type, "period": "annual"}),
            (f"{_API_BASE}/{sym_upper}/financials",       {"type": stmt_type, "period": "annual"}),
            (f"{_API_BASE}/{sym_upper}:NSI/financials",  {"period": "annual"}),
            # Legacy endpoint used in some Tickertape integrations
            (f"https://api.tickertape.in/stocks/{sym_upper}:NSI/financials/{stmt_type}", {"period": "annual"}),
        ]

        for url, params in url_candidates:
            try:
                resp = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
                if resp.status_code == 404:
                    continue
                if resp.status_code != 200:
                    continue
                data = resp.json()
                # Quick sanity: must be a non-empty dict or list
                if data and (isinstance(data, (dict, list))):
                    logger.debug(f"TickertapeConnector: {stmt_type} OK via {url}")
                    return data
            except Exception as e:
                logger.debug(f"TickertapeConnector: {url} — {e}")
                continue

        return None

    def _parse_statement(
        self,
        data: dict,
        stmt_type: str,
        year_set: set[str],
    ) -> list[WebDataResult]:
        """Parse a Tickertape API response into WebDataResult objects."""
        results: list[WebDataResult] = []

        # Tickertape response structure varies — handle common shapes
        # Shape 1: {"data": {"financials": [{"period": "FY25", "revenue": 1234, ...}]}}
        # Shape 2: {"data": [{"period": "FY25", "revenue": 1234, ...}]}
        # Shape 3: {"financials": [...]}
        records = []
        if isinstance(data, dict):
            d = data.get("data", data)
            if isinstance(d, dict):
                records = d.get("financials", d.get("result", []))
            elif isinstance(d, list):
                records = d

        if not records:
            return []

        # Determine field map and statement section name
        field_map = {
            "income-statement": (_PL_MAP,  "Annual P&L"),
            "balance-sheet":    (_BS_MAP,  "Annual Balance Sheet"),
            "cash-flow":        (_CF_MAP,  "Annual Cash Flow"),
        }.get(stmt_type, (_ALL_MAPS, "Annual P&L"))
        fmap, section = field_map

        for rec in records:
            if not isinstance(rec, dict):
                continue

            # Get period / year
            period = rec.get("period") or rec.get("year") or rec.get("date") or ""
            year = _parse_tt_year(str(period))
            if not year or year not in year_set:
                continue

            # Detect multiplier — Tickertape sometimes reports in Cr, sometimes Lakhs
            # The API usually includes a "unit" or "reportedCurrency" field
            unit_str = str(rec.get("unit", rec.get("currencyUnit", "crore"))).lower()
            multiplier = 1.0
            if "lakh" in unit_str:
                multiplier = 0.01   # lakhs → crores
            elif "million" in unit_str:
                multiplier = 0.1    # millions → crores
            elif "thousand" in unit_str:
                multiplier = 0.0001 # thousands → crores

            for api_key, template_hint in fmap.items():
                raw_val = rec.get(api_key)
                if raw_val is None:
                    continue
                val = _to_crore(raw_val, multiplier)
                if val is None:
                    continue

                results.append(WebDataResult(
                    source="tickertape",
                    raw_field=f"tickertape/{stmt_type}/{api_key}",
                    template_field=template_hint,
                    value=val,
                    year=year,
                    confidence="HIGH",
                    unit_applied="INR Crores (Tickertape annual)",
                ))

        return results
