"""
Yahoo Finance connector via the `yfinance` Python library.

Why yfinance:
- Zero scraping — uses Yahoo's official data feed via their API
- Covers P&L, Balance Sheet, Cash Flow for all NSE/BSE listed companies
- No rate limits for reasonable usage, no cookie management
- Returns annual + quarterly data

Unit conversion:
  yfinance returns raw INR values (e.g. 8,990,420,000,000 for ~900,000 crore revenue).
  We divide by 1e7 (10 million) to convert to INR Crores.
  1 crore = 10,000,000 rupees = 1e7 rupees.

Year mapping:
  yfinance annual data is indexed by the fiscal year END date.
  Indian fiscal year ends March 31.
  FY2025 = year ending 2025-03-31 → "F2025"
"""

from __future__ import annotations

import subprocess
import sys
from typing import Optional

from utils.helpers import get_logger
from web.base import CompanyIdentifier, WebDataResult

logger = get_logger(__name__)

# ── Auto-install yfinance ──────────────────────────────────────────────────────

def _require_yfinance():
    try:
        import yfinance as yf
        return yf
    except ImportError:
        logger.info("Installing yfinance…")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "yfinance>=0.2.40"]
        )
        import importlib, site
        importlib.invalidate_caches()
        for sp in site.getsitepackages():
            if sp not in sys.path:
                sys.path.insert(0, sp)
        import yfinance as yf
        return yf


# ── INR → Crores conversion ────────────────────────────────────────────────────

_INR_TO_CRORE = 1e7   # 1 crore = 10,000,000 rupees


def _to_crore(value) -> Optional[float]:
    """Convert a raw INR value (as returned by yfinance) to INR Crores."""
    try:
        v = float(value)
        if v != v:   # NaN check
            return None
        return round(v / _INR_TO_CRORE, 4)
    except (TypeError, ValueError):
        return None


# ── Year extraction from yfinance date index ───────────────────────────────────

def _date_to_fyear(date_val) -> Optional[str]:
    """
    Convert a yfinance date index to our F20XX format.
    Indian FY ends March 31: date 2025-03-31 → "F2025".
    For non-March year-ends, we map to the calendar year of the end date.
    """
    try:
        import pandas as pd
        ts = pd.Timestamp(date_val)
        # Indian FY: if month is March, year is the fiscal year
        # If month is not March (e.g. Dec for some MNCs), still use end year
        return f"F{ts.year}"
    except Exception:
        return None


# ── Field catalogue ────────────────────────────────────────────────────────────
#
# yfinance uses pandas DataFrames with string row indices.
# The exact row labels differ slightly across companies and periods.
# We normalise them to a canonical yfinance label set, then apply
# a GENERIC mapping to common financial template terms.
#
# This mapping is intentionally broad and uses lower-cased keys for
# case-insensitive lookup.  The FieldMapper or cross-validator will
# still do the final template-label resolution.
#
# Mapping: lower(yfinance label) → canonical_name (used as template_field hint)

_PL_MAP: dict[str, str] = {
    "total revenue":                    "Revenue from Operations",
    "operating revenue":                "Revenue from Operations",
    "net revenue":                      "Revenue from Operations",
    "revenue":                          "Revenue from Operations",
    "gross profit":                     "Gross Profit",
    "operating income":                 "EBIT",
    "ebit":                             "EBIT",
    "ebitda":                           "EBITDA",
    "net income":                       "Profit After Tax",
    "net income common stockholders":   "Profit After Tax",
    "net income from continuing operations": "Profit After Tax",
    "tax provision":                    "Tax Expense",
    "income tax expense":               "Tax Expense",
    "pretax income":                    "Profit Before Tax",
    "income before tax":                "Profit Before Tax",
    "total expenses":                   "Total Expenses",
    "operating expense":                "Operating Expenses",
    "research and development":         "R&D Expenses",
    "selling general administrative":   "SG&A",
    "depreciation amortization depletion": "Depreciation & Amortization",
    "reconciled depreciation":          "Depreciation & Amortization",
    "interest expense":                 "Finance Costs",
    "interest expense non operating":   "Finance Costs",
    "total other income expense net":   "Other Income",
    "other income expense":             "Other Income",
    "diluted eps":                      "EPS (Diluted)",
    "basic eps":                        "EPS (Basic)",
    # Banking-specific
    "net interest income":              "Net Interest Income",
    "non interest income":              "Non-Interest Income",
    "provision for loan losses":        "Provisions",
}

_BS_MAP: dict[str, str] = {
    "total assets":                     "Total Assets",
    "total liabilities":                "Total Liabilities",
    "total liabilities net minority interest": "Total Liabilities",
    "stockholders equity":              "Total Equity",
    "total equity gross minority interest": "Total Equity",
    "common stock equity":              "Shareholders Equity",
    "retained earnings":                "Retained Earnings",
    "cash and cash equivalents":        "Cash and Cash Equivalents",
    "cash cash equivalents and short term investments": "Cash and Equivalents",
    "total debt":                       "Total Borrowings",
    "long term debt":                   "Borrowings (Non-Current)",
    "current debt":                     "Borrowings (Current)",
    "net ppe":                          "Property, Plant & Equipment",
    "properties":                       "Property, Plant & Equipment",
    "net tangible assets":              "Net Tangible Assets",
    "goodwill":                         "Goodwill",
    "investments and advances":         "Investments",
    "investmentsand advances":          "Investments",
    "inventory":                        "Inventories",
    "receivables":                      "Trade Receivables",
    "accounts receivable":              "Trade Receivables",
    "payables":                         "Trade Payables",
    "accounts payable":                 "Trade Payables",
    "current assets":                   "Total Current Assets",
    "current liabilities":              "Total Current Liabilities",
    "capital lease obligations":        "Lease Liabilities",
    # Banking
    "loans":                            "Advances",
    "net loans":                        "Advances",
    "deposits":                         "Deposits",
}

_CF_MAP: dict[str, str] = {
    "operating cash flow":                          "Cash Flow from Operating Activities",
    "cash flow from continuing operating activities": "Cash Flow from Operating Activities",
    "investing cash flow":                          "Cash Flow from Investing Activities",
    "cash flow from continuing investing activities": "Cash Flow from Investing Activities",
    "financing cash flow":                          "Cash Flow from Financing Activities",
    "cash flow from continuing financing activities": "Cash Flow from Financing Activities",
    "free cash flow":                               "Free Cash Flow",
    "capital expenditure":                          "Capital Expenditure",
    "capex":                                        "Capital Expenditure",
    "changes in working capital":                   "Changes in Working Capital",
    "net income from continuing operations":        "Profit After Tax",
    "depreciation amortization depletion":          "Depreciation & Amortization",
    "end cash position":                            "Closing Cash Balance",
    "beginning cash position":                      "Opening Cash Balance",
    "net issuance payments of debt":                "Net Debt Raised / (Repaid)",
    "dividends paid":                               "Dividends Paid",
}


def _map_label(raw: str, table: dict[str, str]) -> str:
    """Return template_field hint from the lookup table, or the raw label."""
    return table.get(raw.lower().strip(), raw)


# ── Main connector ─────────────────────────────────────────────────────────────

class YFinanceConnector:
    """
    Fetches annual financial data from Yahoo Finance via yfinance.
    Returns WebDataResult objects with values in INR Crores.
    """

    def __init__(self):
        self._yf = _require_yfinance()

    def fetch(
        self,
        company: CompanyIdentifier,
        years: list[str],
    ) -> list[WebDataResult]:
        """
        Fetch P&L, Balance Sheet, and Cash Flow for the given company and years.
        Returns a flat list of WebDataResult objects (one per field per year).
        """
        ticker_sym = company.yfinance_ticker
        if not ticker_sym:
            logger.warning(
                f"YFinanceConnector: no ticker for '{company.name}' — "
                "provide nse_symbol or bse_code"
            )
            return []

        logger.info(f"YFinanceConnector: fetching {ticker_sym} for years {years}")

        try:
            ticker = self._yf.Ticker(ticker_sym)
            results: list[WebDataResult] = []

            results.extend(self._fetch_table(ticker.financials,   "P&L",     _PL_MAP, years))
            results.extend(self._fetch_table(ticker.balance_sheet, "Balance Sheet", _BS_MAP, years))
            results.extend(self._fetch_table(ticker.cashflow,      "Cash Flow",    _CF_MAP, years))

            logger.info(
                f"YFinanceConnector: {ticker_sym} → "
                f"{len(results)} data points across {len(years)} year(s)"
            )
            return results

        except Exception as e:
            logger.error(f"YFinanceConnector: fetch failed for {ticker_sym}: {e}")
            return []

    def _fetch_table(
        self,
        df,
        section_label: str,
        field_map: dict[str, str],
        years: list[str],
    ) -> list[WebDataResult]:
        """Parse a yfinance DataFrame (rows=fields, cols=dates) into WebDataResults."""
        if df is None or df.empty:
            return []

        results: list[WebDataResult] = []
        year_set = set(years)

        for col in df.columns:
            fyear = _date_to_fyear(col)
            if fyear not in year_set:
                continue

            for raw_field in df.index:
                raw_val = df.loc[raw_field, col]
                crore_val = _to_crore(raw_val)
                if crore_val is None:
                    continue

                template_hint = _map_label(str(raw_field), field_map)

                results.append(WebDataResult(
                    source="yfinance",
                    raw_field=str(raw_field),
                    template_field=template_hint,
                    value=crore_val,
                    year=fyear,
                    confidence="HIGH",
                    unit_applied=f"÷{_INR_TO_CRORE:.0e} (raw INR → INR Crores)",
                ))

        return results

    def fetch_info(self, company: CompanyIdentifier) -> dict:
        """Return basic company info dict (sector, market cap, description)."""
        ticker_sym = company.yfinance_ticker
        if not ticker_sym:
            return {}
        try:
            ticker = self._yf.Ticker(ticker_sym)
            info = ticker.info or {}
            return {
                "name":        info.get("longName", company.name),
                "sector":      info.get("sector", ""),
                "industry":    info.get("industry", ""),
                "market_cap":  _to_crore(info.get("marketCap")),
                "currency":    info.get("currency", "INR"),
                "country":     info.get("country", ""),
            }
        except Exception as e:
            logger.warning(f"YFinanceConnector.fetch_info failed: {e}")
            return {}
