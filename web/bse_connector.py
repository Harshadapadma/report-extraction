"""
BSE Connector — fetches structured financial data from BSE India.

Two data paths:
  1. BSE Company API  — basic company info, scrip details (works, public endpoint)
  2. BSE XBRL quarterly results — structured XML financial data per quarter
     Aggregate 4 quarters → approximate annual figures for cross-validation.

BSE API base: https://api.bseindia.com/BseIndiaAPI/api/

Note: BSE's public API endpoints are undocumented but stable. We use only
GET endpoints that return JSON/XML with no authentication required.
If BSE changes endpoint structure, the connector degrades gracefully and
returns an empty list (never crashes the main pipeline).
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from typing import Optional

from utils.helpers import get_logger
from web.base import CompanyIdentifier, WebDataResult

logger = get_logger(__name__)

# ── Auto-install requests ──────────────────────────────────────────────────────

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


_BSE_API_BASE = "https://api.bseindia.com/BseIndiaAPI/api"
_BSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FinancialDataExtractor/1.0)",
    "Referer":    "https://www.bseindia.com",
    "Accept":     "application/json, text/xml, */*",
}
_TIMEOUT = 15   # seconds


# ── Unit conversion ────────────────────────────────────────────────────────────

def _to_crore(value, unit_label: str = "") -> Optional[float]:
    """Convert BSE-reported value to INR Crores."""
    try:
        v = float(str(value).replace(",", ""))
        if v != v:
            return None
        ul = unit_label.lower()
        if "lakh" in ul:
            return round(v / 100.0, 4)
        if "thousand" in ul or "000" in ul:
            return round(v / 10_000.0, 4)
        if "million" in ul:
            return round(v / 10.0, 4)
        # Default: already in crores or unknown — return as-is
        return round(v, 4)
    except (TypeError, ValueError):
        return None


# ── BSE API helpers ────────────────────────────────────────────────────────────

def _safe_get(url: str, params: dict = None) -> Optional[dict | list]:
    """GET request to BSE API with graceful failure."""
    requests = _require_requests()
    try:
        resp = requests.get(url, params=params, headers=_BSE_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if "json" in ct:
            return resp.json()
        if "xml" in ct or resp.text.strip().startswith("<"):
            return {"_xml": resp.text}
        return None
    except Exception as e:
        logger.warning(f"BSE GET {url} failed: {e}")
        return None


# ── Field mapping (BSE XBRL tags → template hints) ────────────────────────────

_XBRL_FIELD_MAP: dict[str, str] = {
    # P&L tags used in BSE quarterly XBRL submissions
    "TotalRevenue":                         "Revenue from Operations",
    "RevenueFromOperationsNet":             "Revenue from Operations",
    "RevenueFromOperations":                "Revenue from Operations",
    "OtherIncome":                          "Other Income",
    "TotalExpenses":                        "Total Expenses",
    "ProfitBeforeTax":                      "Profit Before Tax",
    "TaxExpense":                           "Tax Expense",
    "ProfitForPeriod":                      "Profit After Tax",
    "ProfitAfterTax":                       "Profit After Tax",
    "EarningsPerShareBasic":                "EPS (Basic)",
    "EarningsPerShareDiluted":              "EPS (Diluted)",
    # Balance Sheet
    "TotalAssets":                          "Total Assets",
    "TotalEquity":                          "Total Equity",
    "TotalLiabilities":                     "Total Liabilities",
    "Borrowings":                           "Total Borrowings",
    "CashandCashEquivalents":              "Cash and Cash Equivalents",
    # Banking-specific XBRL tags (RBI-mandated format)
    "InterestEarned":                       "Interest Income",
    "InterestExpended":                     "Interest Expended",
    "NetInterestIncome":                    "Net Interest Income",
    "OperatingProfit":                      "Operating Profit",
    "Provisions":                           "Provisions and Contingencies",
    "Deposits":                             "Deposits",
    "Advances":                             "Advances",
    "Investments":                          "Investments",
    "CapitalAndReserves":                   "Total Equity",
}


# ── Quarter → fiscal year mapping ─────────────────────────────────────────────

def _quarter_to_fyear(quarter_end: str) -> Optional[str]:
    """
    Map a quarter end date string (YYYYMMDD or YYYY-MM-DD) to an FY label.
    Indian FY: April → March.  Quarter ending March 2025 is Q4FY25 → F2025.
    """
    try:
        d = re.sub(r"[-/]", "", quarter_end.strip())[:8]
        year, month = int(d[:4]), int(d[4:6])
        # If month <= 3 (Jan/Feb/Mar), it's Q4 of the FY ending that year
        # If month >= 4 (Apr onwards), it's Q1-Q3 of the FY ending NEXT year
        fy = year if month <= 3 else year + 1
        return f"F{fy}"
    except Exception:
        return None


# ── Main connector ─────────────────────────────────────────────────────────────

class BSEConnector:
    """
    Fetches financial data from BSE India public endpoints.

    Two modes:
      - Annual results (from BSE annual report metadata)
      - Quarterly XBRL aggregation (4 quarters summed for annual validation)
    """

    def __init__(self):
        pass

    def fetch(
        self,
        company: CompanyIdentifier,
        years: list[str],
    ) -> list[WebDataResult]:
        """
        Fetch BSE-sourced financial data for the given company and years.
        Tries multiple endpoints in order; returns whatever is available.
        """
        if not company.bse_code:
            logger.info(
                f"BSEConnector: no BSE code for '{company.name}' — skipping"
            )
            return []

        results: list[WebDataResult] = []

        # Try quarterly XBRL results — most reliable BSE endpoint
        quarterly = self._fetch_quarterly_results(company, years)
        results.extend(quarterly)

        if results:
            logger.info(
                f"BSEConnector: '{company.name}' ({company.bse_code}) → "
                f"{len(results)} data points"
            )
        else:
            logger.info(
                f"BSEConnector: no data retrieved for '{company.name}' "
                f"({company.bse_code}) — BSE endpoint may be unavailable"
            )
        return results

    def _fetch_quarterly_results(
        self,
        company: CompanyIdentifier,
        years: list[str],
    ) -> list[WebDataResult]:
        """
        Fetch quarterly P&L results from BSE's financial results API.
        Endpoint: /FinancialResult4/w?scripcode={code}&typeflag=C (consolidated)
        Returns quarterly data; we aggregate to approximate annual figures.
        """
        url = f"{_BSE_API_BASE}/FinancialResult4/w"
        params = {
            "scripcode": company.bse_code,
            "typeflag": "C",   # C = Consolidated, S = Standalone
        }
        data = _safe_get(url, params)
        if not data or isinstance(data, dict) and not data:
            # Try standalone if consolidated not available
            params["typeflag"] = "S"
            data = _safe_get(url, params)

        if not data:
            return []

        results: list[WebDataResult] = []
        year_buckets: dict[str, dict[str, list[float]]] = {}

        # BSE API returns a list of quarterly result records
        records = data if isinstance(data, list) else data.get("Table", [])
        if not records:
            return []

        year_set = set(years)

        for rec in records:
            try:
                qend = rec.get("TO_DATE") or rec.get("QuarterEndDate", "")
                fyear = _quarter_to_fyear(str(qend))
                if not fyear or fyear not in year_set:
                    continue

                stmt_type = rec.get("TypeFlag", "C")   # "C" or "S"

                for bse_key, tmpl_hint in _XBRL_FIELD_MAP.items():
                    raw_val = rec.get(bse_key) or rec.get(bse_key.upper())
                    if raw_val is None:
                        continue
                    val = _to_crore(raw_val, "crore")   # BSE reports in INR Crores
                    if val is None:
                        continue
                    bucket_key = f"{fyear}||{tmpl_hint}||{stmt_type}"
                    year_buckets.setdefault(bucket_key, {}).setdefault("vals", []).append(val)
                    year_buckets[bucket_key]["tmpl_hint"] = tmpl_hint
                    year_buckets[bucket_key]["fyear"] = fyear
                    year_buckets[bucket_key]["bse_key"] = bse_key
                    year_buckets[bucket_key]["stmt_type"] = stmt_type

            except Exception as e:
                logger.debug(f"BSEConnector: error parsing record: {e}")
                continue

        # Aggregate quarterly buckets to annual figures
        # Revenue/P&L items: sum 4 quarters.
        # Balance sheet items: use latest quarter (point-in-time).
        _CUMULATIVE_FIELDS = {
            "Revenue from Operations", "Other Income", "Total Expenses",
            "Profit Before Tax", "Tax Expense", "Profit After Tax",
            "Interest Income", "Interest Expended", "Net Interest Income",
            "Operating Profit", "Provisions and Contingencies",
        }
        for bucket_key, bucket in year_buckets.items():
            vals = bucket.get("vals", [])
            if not vals:
                continue
            tmpl_hint = bucket["tmpl_hint"]
            fyear = bucket["fyear"]
            bse_key = bucket["bse_key"]
            stmt_type = bucket.get("stmt_type", "C")

            if tmpl_hint in _CUMULATIVE_FIELDS:
                annual_val = round(sum(vals), 4)   # sum all quarters
            else:
                annual_val = round(vals[-1], 4)    # latest quarter for balance sheet

            results.append(WebDataResult(
                source="bse_xbrl",
                raw_field=f"{bse_key} ({stmt_type}, 4Q sum)" if tmpl_hint in _CUMULATIVE_FIELDS
                          else f"{bse_key} ({stmt_type}, latest Q)",
                template_field=tmpl_hint,
                value=annual_val,
                year=fyear,
                confidence="MEDIUM",   # quarterly aggregation → medium confidence
                unit_applied="INR Crores (as reported by BSE)",
            ))

        return results

    def get_scrip_info(self, bse_code: str) -> dict:
        """Return basic company info from BSE scrip details endpoint."""
        url = f"{_BSE_API_BASE}/ComHeader/w"
        params = {"quotetype": "EQ", "scripcode": bse_code}
        data = _safe_get(url, params)
        if not data:
            return {}
        rec = data if isinstance(data, dict) else (data[0] if data else {})
        return {
            "name":       rec.get("compname", ""),
            "isin":       rec.get("ISIN_Number", ""),
            "sector":     rec.get("industry", ""),
            "market_cap": _to_crore(rec.get("mktcap"), "crore"),
        }
