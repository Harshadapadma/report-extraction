"""
Screener.in Connector — fetches structured annual financial data for listed
Indian companies from screener.in.

URL pattern:
  Consolidated: https://www.screener.in/company/{NSE_SYMBOL}/consolidated/
  Standalone:   https://www.screener.in/company/{NSE_SYMBOL}/

Screener publishes 10-year P&L, Balance Sheet, and Cash Flow tables with
values already in INR Crores — no unit conversion required for most companies.
The data comes directly from XBRL filings submitted to BSE/NSE.

Confidence: HIGH (annual XBRL-sourced, same data as regulatory filings)
"""

from __future__ import annotations

import re
from typing import Optional

from utils.helpers import get_logger
from web.base import CompanyIdentifier, WebDataResult

logger = get_logger(__name__)

# ── Lazy-import helpers ────────────────────────────────────────────────────────

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


def _require_bs4():
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup
    except ImportError:
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "beautifulsoup4>=4.12.0"]
        )
        import importlib, site
        importlib.invalidate_caches()
        for sp in site.getsitepackages():
            if sp not in sys.path:
                sys.path.insert(0, sp)
        from bs4 import BeautifulSoup
        return BeautifulSoup


_TIMEOUT = 20
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.screener.in/",
}

_SCREENER_BASE = "https://www.screener.in/company"


# ── Field mapping: Screener row labels → template hints ───────────────────────

# Screener uses plain-English row labels that are very close to template labels.
# We map known aliases; anything not in this map goes through as-is (usually fine
# because Screener labels are already close to standard template labels).

_LABEL_ALIAS: dict[str, str] = {
    # P&L
    "Sales":                            "Revenue from Operations",
    "Revenue":                          "Revenue from Operations",
    "Net Sales":                        "Revenue from Operations",
    "Revenue from Operations":          "Revenue from Operations",
    "Other Income":                     "Other Income",
    "Total Revenue":                    "Total Revenue",
    "Expenses":                         "Total Expenses",
    "Total Expenses":                   "Total Expenses",
    "Operating Profit":                 "Operating Profit",
    "OPM %":                            None,                        # skip ratio rows
    "Interest":                         "Finance Costs",
    "Finance Costs":                    "Finance Costs",
    "Depreciation":                     "Depreciation and Amortisation",
    "Profit before tax":                "Profit Before Tax",
    "Tax %":                            None,
    "Net Profit":                       "Profit After Tax",
    "Net Profit after minority interest and share of associates": "Profit After Tax",
    "EPS in Rs":                        "EPS (Basic)",
    # Balance Sheet
    "Share Capital":                    "Share Capital",
    "Reserves":                         "Reserves and Surplus",
    "Borrowings":                       "Total Borrowings",
    "Other Liabilities":                "Other Liabilities",
    "Total Liabilities":                "Total Equity and Liabilities",
    "Fixed Assets":                     "Fixed Assets",
    "CWIP":                             "Capital Work in Progress",
    "Investments":                      "Investments",
    "Other Assets":                     "Other Assets",
    "Total Assets":                     "Total Assets",
    "Cash Equivalents":                 "Cash and Cash Equivalents",
    "Cash":                             "Cash and Cash Equivalents",
    # Cash Flow
    "Cash from Operating Activity":     "Net Cash from Operating Activities",
    "Cash from Investing Activity":     "Net Cash from Investing Activities",
    "Cash from Financing Activity":     "Net Cash from Financing Activities",
    "Net Cash Flow":                    "Net Increase / (Decrease) in Cash",
    "Capex":                            "Capital Expenditure",
    # Banking-specific
    "Interest Earned":                  "Interest Income",
    "Interest Expended":                "Interest Expended",
    "Net NPA":                          "Net NPA",
    "Gross NPA":                        "Gross NPA",
    "Deposits":                         "Deposits",
    "Advances":                         "Advances",
    "Net Interest Income":              "Net Interest Income",
}

# Section IDs in the Screener HTML
_SECTION_IDS = {
    "pl":         "profit-loss",
    "bs":         "balance-sheet",
    "cf":         "cash-flow",
    "quarters":   "quarters",       # skip quarterly table
    "ratios":     "ratios",         # skip ratios
    "shareholding": "shareholding", # skip
}

# Only parse these sections
_WANTED_SECTIONS = {"profit-loss", "balance-sheet", "cash-flow"}


# ── Year header parser ─────────────────────────────────────────────────────────

def _parse_year_header(header_text: str) -> Optional[str]:
    """
    Convert Screener year headers to FY label.
    Screener uses "Mar 2025", "Mar 2024", "TTM", etc.
    Indian FY ends in March — "Mar 2025" → F2025.
    """
    text = header_text.strip()
    # Match "Mar 2025", "Mar-25", "2025", "FY25", "FY2025"
    m = re.search(r"(\d{4})", text)
    if m:
        yr = int(m.group(1))
        if yr < 100:
            yr += 2000
        return f"F{yr}"
    m = re.search(r"(\d{2})\b", text)
    if m:
        yr = int(m.group(1)) + 2000
        return f"F{yr}"
    return None


# ── Value cleaner ──────────────────────────────────────────────────────────────

def _parse_value(text: str) -> Optional[float]:
    """Clean and parse a Screener cell value to float (INR Crores)."""
    t = text.strip().replace(",", "").replace(" ", "").replace("%", "")
    if not t or t in ("-", "—", "N/A", ""):
        return None
    # Negative values shown as "(123.45)"
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    try:
        v = float(t)
        return -v if neg else v
    except ValueError:
        return None


# ── Main connector ─────────────────────────────────────────────────────────────

class ScreenerConnector:
    """
    Fetches annual financial data from screener.in for Indian listed companies.

    Requires the NSE symbol (stored in CompanyIdentifier.nse_symbol).
    Fetches consolidated data first; falls back to standalone if not available.

    Returns WebDataResult objects with:
      source      = "screener"
      confidence  = "HIGH"
      unit_applied = "INR Crores (as reported by Screener)"
    """

    def __init__(self):
        pass

    def fetch(
        self,
        company: CompanyIdentifier,
        years: list[str],
    ) -> list[WebDataResult]:
        """
        Fetch annual financials from screener.in for the given company and years.
        Returns empty list on any error (never crashes the pipeline).
        """
        symbol = company.nse_symbol
        if not symbol:
            # Try BSE code as fallback lookup key (screener also accepts BSE code)
            symbol = company.bse_code
        if not symbol:
            logger.info(
                f"ScreenerConnector: no NSE symbol for '{company.name}' — skipping"
            )
            return []

        year_set = set(years)
        results: list[WebDataResult] = []

        # Try consolidated first, then standalone
        for path_suffix in ("/consolidated/", "/"):
            url = f"{_SCREENER_BASE}/{symbol.upper()}{path_suffix}"
            logger.info(f"ScreenerConnector: fetching {url}")
            html = self._fetch_page(url)
            if html:
                parsed = self._parse_page(html, year_set, url)
                if parsed:
                    results = parsed
                    break
            # If consolidated 404 / empty, try standalone
        if results:
            logger.info(
                f"ScreenerConnector: '{company.name}' ({symbol}) → "
                f"{len(results)} data points"
            )
        else:
            logger.info(
                f"ScreenerConnector: no data for '{company.name}' ({symbol}) "
                "— page unavailable or no matching years"
            )
        return results

    def _fetch_page(self, url: str) -> Optional[str]:
        """GET the Screener page HTML, return None on failure."""
        requests = _require_requests()
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            if resp.status_code == 404:
                logger.debug(f"ScreenerConnector: 404 for {url}")
                return None
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning(f"ScreenerConnector: GET {url} failed — {e}")
            return None

    def _parse_page(
        self,
        html: str,
        year_set: set[str],
        source_url: str,
    ) -> list[WebDataResult]:
        """Parse the screener.in company page and extract financial tables."""
        BeautifulSoup = _require_bs4()
        soup = BeautifulSoup(html, "html.parser")
        results: list[WebDataResult] = []

        for section_id in _WANTED_SECTIONS:
            section = soup.find("section", {"id": section_id})
            if section is None:
                # Some pages wrap in <div id="...">
                section = soup.find(id=section_id)
            if section is None:
                continue

            table = section.find("table")
            if table is None:
                continue

            table_results = self._parse_table(
                table, section_id, year_set, source_url
            )
            results.extend(table_results)

        return results

    def _parse_table(
        self,
        table,
        section_id: str,
        year_set: set[str],
        source_url: str,
    ) -> list[WebDataResult]:
        """Parse a single financial table and return WebDataResult objects."""
        results: list[WebDataResult] = []

        # Parse header row → year columns
        thead = table.find("thead")
        if not thead:
            return []
        header_cells = thead.find_all("th")
        # First cell is the row label header; the rest are years
        year_columns: list[Optional[str]] = []
        for th in header_cells[1:]:
            yr = _parse_year_header(th.get_text(strip=True))
            year_columns.append(yr)

        if not year_columns:
            return []

        # Parse data rows
        tbody = table.find("tbody")
        if not tbody:
            return []

        rows = tbody.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue

            # Row label is in the first cell (may be inside an <a> or <span>)
            raw_label = cells[0].get_text(separator=" ", strip=True)
            # Remove trailing +/- or % signs that screener appends
            raw_label = re.sub(r"\s*[+\-]\s*$", "", raw_label).strip()

            # Map label to template hint
            template_field = _LABEL_ALIAS.get(raw_label)
            if template_field is None and raw_label not in _LABEL_ALIAS:
                # Not in alias map — pass through as-is, mapper will match
                template_field = raw_label
            if template_field is None:
                # Explicitly skipped (mapped to None)
                continue

            # Parse each year column
            for col_idx, yr in enumerate(year_columns):
                if yr not in year_set:
                    continue
                cell_idx = col_idx + 1  # offset by label column
                if cell_idx >= len(cells):
                    continue

                cell_text = cells[cell_idx].get_text(strip=True)
                value = _parse_value(cell_text)
                if value is None:
                    continue

                results.append(WebDataResult(
                    source="screener",
                    raw_field=f"screener/{section_id}/{raw_label}",
                    template_field=template_field,
                    value=value,
                    year=yr,
                    confidence="HIGH",
                    unit_applied="INR Crores (as reported by Screener)",
                ))

        return results
