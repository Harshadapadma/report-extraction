"""
NSE Connector — fetches quarterly financial results from NSE India's public API.

NSE India provides structured quarterly P&L and balance sheet data for all
listed companies via undocumented but stable JSON endpoints.  We aggregate
four quarters to approximate annual figures for P&L, and take the latest
quarter's snapshot for balance sheet items.

Endpoint:
  GET https://www.nseindia.com/api/financial-results?index=equities&symbol={SYMBOL}&period=Quarterly

NSE requires valid session cookies (acquired by first visiting the main site).
This connector handles session bootstrap automatically.

Confidence: MEDIUM (quarterly aggregation, same as BSE XBRL)
"""

from __future__ import annotations

import re
import time
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


_TIMEOUT = 20

# NSE requires a session cookie obtained by visiting the main site first
_NSE_HOME = "https://www.nseindia.com"
_NSE_API  = "https://www.nseindia.com/api"

_HOME_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_API_HEADERS = {
    **_HOME_HEADERS,
    "Accept":  "application/json, */*",
    "Referer": "https://www.nseindia.com/",
}


# ── Field mapping: NSE result keys → template hints ───────────────────────────

# NSE quarterly results use these key names in their JSON response.
# The "xbrl" field names vary by period; we use the "consolidated" nested object.

_PL_FIELD_MAP: dict[str, str] = {
    # Revenue — camelCase (new NSE API) + snake_case variants
    "netSales":                     "Revenue from Operations",
    "net_sales":                    "Revenue from Operations",
    "totalRevenue":                 "Total Revenue",
    "total_revenue":                "Total Revenue",
    "otherIncome":                  "Other Income",
    "other_income":                 "Other Income",
    # Expenses
    "totalExpenditure":             "Total Expenses",
    "total_expenditure":            "Total Expenses",
    "totalExpenses":                "Total Expenses",
    "operatingProfit":              "Operating Profit",
    "operating_profit":             "Operating Profit",
    "pbdt":                         "Profit Before Depreciation & Tax",
    "depreciation":                 "Depreciation and Amortisation",
    "interest":                     "Finance Costs",
    "financeCharges":               "Finance Costs",
    "pbt":                          "Profit Before Tax",
    "profit_before_tax":            "Profit Before Tax",
    "profitBeforeTax":              "Profit Before Tax",
    "tax":                          "Tax Expense",
    "taxExpense":                   "Tax Expense",
    "pat":                          "Profit After Tax",
    "netProfit":                    "Profit After Tax",
    "net_profit":                   "Profit After Tax",
    "eps":                          "EPS (Basic)",
    "epsBasic":                     "EPS (Basic)",
    "epsDiluted":                   "EPS (Diluted)",
    # Banking (both camelCase and underscore)
    "interestEarned":               "Interest Income",
    "interest_earned":              "Interest Income",
    "interestExpended":             "Interest Expended",
    "interest_expended":            "Interest Expended",
    "netInterestIncome":            "Net Interest Income",
    "net_interest_income":          "Net Interest Income",
    "provisions":                   "Provisions and Contingencies",
    "provisionsAndContingencies":   "Provisions and Contingencies",
    "operatingExpenditure":         "Operating Expenses",
    "operating_expenditure":        "Operating Expenses",
}

_BS_FIELD_MAP: dict[str, str] = {
    "paidUpCapital":                "Share Capital",
    "paid_up_capital":              "Share Capital",
    "shareCapital":                 "Share Capital",
    "reserves":                     "Reserves and Surplus",
    "reservesSurplus":              "Reserves and Surplus",
    "borrowings":                   "Total Borrowings",
    "totalBorrowings":              "Total Borrowings",
    "totalLiabilities":             "Total Liabilities",
    "total_liabilities":            "Total Liabilities",
    "fixedAssets":                  "Fixed Assets",
    "fixed_assets":                 "Fixed Assets",
    "investments":                  "Investments",
    "totalAssets":                  "Total Assets",
    "total_assets":                 "Total Assets",
    "cash":                         "Cash and Cash Equivalents",
    "cashAndEquivalents":           "Cash and Cash Equivalents",
    # Banking
    "deposits":                     "Deposits",
    "advances":                     "Advances",
    "netNpa":                       "Net NPA",
    "net_npa":                      "Net NPA",
    "grossNpa":                     "Gross NPA",
    "gross_npa":                    "Gross NPA",
    "netNPA":                       "Net NPA",
    "grossNPA":                     "Gross NPA",
}

_CUMULATIVE_FIELDS = {
    "Revenue from Operations", "Total Revenue", "Other Income",
    "Total Expenses", "Operating Profit", "Profit Before Tax",
    "Tax Expense", "Profit After Tax", "Finance Costs",
    "Depreciation and Amortisation", "Interest Income",
    "Interest Expended", "Net Interest Income",
    "Provisions and Contingencies", "Operating Expenses",
}


# ── Quarter → FY mapping ──────────────────────────────────────────────────────

def _quarter_end_to_fyear(period_str: str) -> Optional[str]:
    """
    Convert NSE period string to FY label.
    NSE uses "Mar 2025", "Dec 2024", "Sep 2024", "Jun 2024" etc.
    Indian FY: Apr → Mar.  Quarter ending Mar 2025 → F2025.
    """
    period_str = str(period_str).strip()
    # Try "Mar 2025" or "March 2025" or "31-03-2025"
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*[,\s]\s*(\d{4})",
                  period_str, re.I)
    if m:
        month_str, year = m.group(1).lower(), int(m.group(2))
        _MONTH_NUM = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                      "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
        month = _MONTH_NUM.get(month_str[:3], 0)
        if month == 0:
            return None
        # Apr–Mar fiscal year: if month <= 3 → Q4, belongs to same calendar year
        # if month >= 4 → Q1-Q3, belongs to year+1
        fy = year if month <= 3 else year + 1
        return f"F{fy}"

    # Try "DD-MM-YYYY"
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})", period_str)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        fy = year if month <= 3 else year + 1
        return f"F{fy}"

    return None


# ── Value converter ───────────────────────────────────────────────────────────

def _to_crore(raw_value, multiplier: float = 1.0) -> Optional[float]:
    """Parse a raw NSE value (string or number) to INR Crores."""
    if raw_value is None:
        return None
    try:
        v = float(str(raw_value).replace(",", "").replace(" ", ""))
        if v != v:
            return None
        return round(v * multiplier, 4)
    except (ValueError, TypeError):
        return None


# ── NSE session manager ───────────────────────────────────────────────────────

class _NSESession:
    """
    Manages an NSE session.  NSE blocks API calls without a valid cookie
    obtained by first loading the home page.
    """
    def __init__(self):
        self._session = None
        self._last_refresh: float = 0.0
        self._REFRESH_INTERVAL = 1800   # 30 minutes

    def get(self) -> "requests.Session":
        requests = _require_requests()
        now = time.time()
        if self._session is None or (now - self._last_refresh) > self._REFRESH_INTERVAL:
            s = requests.Session()
            try:
                # Bootstrap: visit home page to get session cookies
                resp = s.get(_NSE_HOME, headers=_HOME_HEADERS, timeout=_TIMEOUT)
                resp.raise_for_status()
                # Also visit the company pages section to get the right cookies
                s.get(
                    f"{_NSE_HOME}/companies-listing/corporate-filings-financial-results",
                    headers=_HOME_HEADERS,
                    timeout=_TIMEOUT,
                )
                self._session = s
                self._last_refresh = now
                logger.debug("NSEConnector: session bootstrapped")
            except Exception as e:
                logger.warning(f"NSEConnector: session bootstrap failed — {e}")
                self._session = s   # use even on partial failure
        return self._session


_session_mgr = _NSESession()


# ── Main connector ────────────────────────────────────────────────────────────

class NSEConnector:
    """
    Fetches quarterly financial results from NSE India and aggregates to annual.

    Requires NSE symbol (e.g. HDFCBANK).
    Returns WebDataResult objects with:
      source     = "nse"
      confidence = "MEDIUM" (quarterly aggregation)
    """

    def __init__(self):
        pass

    def fetch(
        self,
        company: CompanyIdentifier,
        years: list[str],
    ) -> list[WebDataResult]:
        symbol = company.nse_symbol
        if not symbol:
            logger.info(f"NSEConnector: no NSE symbol for '{company.name}' — skipping")
            return []

        year_set = set(years)

        # Fetch consolidated quarterly results
        raw_records = self._fetch_quarterly(symbol, consolidated=True)
        if not raw_records:
            raw_records = self._fetch_quarterly(symbol, consolidated=False)

        if not raw_records:
            logger.info(f"NSEConnector: no data for '{company.name}' ({symbol})")
            return []

        results = self._aggregate_to_annual(raw_records, year_set)
        logger.info(
            f"NSEConnector: '{company.name}' ({symbol}) → {len(results)} data points"
        )
        return results

    def _fetch_quarterly(
        self,
        symbol: str,
        consolidated: bool = True,
    ) -> list[dict]:
        """Fetch quarterly financial result records from NSE API."""
        session = _session_mgr.get()
        period_type = "Consolidated" if consolidated else "Standalone"
        url = f"{_NSE_API}/financial-results"
        params = {
            "index":  "equities",
            "symbol": symbol.upper(),
            "period": "Quarterly",
        }
        try:
            resp = session.get(
                url, params=params, headers=_API_HEADERS, timeout=_TIMEOUT
            )
            if resp.status_code in (404, 403):
                return []
            resp.raise_for_status()
            data = resp.json()

            # NSE returns {"data": [...records...]} or a flat list
            records = data if isinstance(data, list) else data.get("data", [])
            if not isinstance(records, list):
                # Some NSE responses use different wrapper keys
                for wrap_key in ("results", "financialResults", "quarterlyResults"):
                    if isinstance(data, dict) and wrap_key in data:
                        records = data[wrap_key]
                        break
            if not isinstance(records, list) or not records:
                logger.info(f"NSEConnector: empty/unexpected response shape: {type(data)}, keys={list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                return []

            # Log first record keys so we can debug field mismatches
            if records:
                logger.info(f"NSEConnector: first record keys = {list(records[0].keys())[:25]}")

            # Filter to the consolidation type we want
            filtered = []
            for rec in records:
                rtype = str(
                    rec.get("consolidatedOrStandalone") or
                    rec.get("xbrlAttachment", {}).get("consolidatedOrStandalone", "") or ""
                ).strip().lower()
                if consolidated and rtype in ("consolidated", "con", "c"):
                    filtered.append(rec)
                elif not consolidated and rtype in ("standalone", "std", "s"):
                    filtered.append(rec)
            return filtered or records   # return all if filtering got nothing

        except Exception as e:
            logger.warning(f"NSEConnector: quarterly fetch failed for {symbol}: {e}")
            return []

    def _aggregate_to_annual(
        self,
        records: list[dict],
        year_set: set[str],
    ) -> list[WebDataResult]:
        """Aggregate quarterly records into annual figures per FY."""
        # Bucket: (fyear, template_field) → list[float]
        pl_buckets:  dict[tuple, list[float]] = {}
        bs_snapshot: dict[tuple, float]       = {}   # latest Q for BS fields

        for rec in records:
            # Get period
            period = (
                rec.get("period") or rec.get("toDate") or
                rec.get("periodEnded") or ""
            )
            fyear = _quarter_end_to_fyear(str(period))
            if not fyear or fyear not in year_set:
                continue

            # Determine multiplier (NSE usually reports in Lakhs or Crores)
            unit_str = str(
                rec.get("unit", rec.get("reportingCurrency", "lakh"))
            ).lower()
            multiplier = 1.0
            if "lakh" in unit_str or "100000" in unit_str:
                multiplier = 0.01     # lakhs → crores
            elif "crore" in unit_str:
                multiplier = 1.0
            elif "million" in unit_str:
                multiplier = 0.1

            # Parse P&L fields (cumulative over quarters)
            for nse_key, tmpl_hint in _PL_FIELD_MAP.items():
                raw = rec.get(nse_key) or rec.get(nse_key[0].upper() + nse_key[1:])
                if raw is None:
                    continue
                val = _to_crore(raw, multiplier)
                if val is None:
                    continue
                key = (fyear, tmpl_hint)
                pl_buckets.setdefault(key, []).append(val)

            # Parse BS fields (use latest quarter only — point-in-time)
            for nse_key, tmpl_hint in _BS_FIELD_MAP.items():
                raw = rec.get(nse_key) or rec.get(nse_key[0].upper() + nse_key[1:])
                if raw is None:
                    continue
                val = _to_crore(raw, multiplier)
                if val is None:
                    continue
                key = (fyear, tmpl_hint)
                # Keep the record with the latest quarter (dict order = API order)
                bs_snapshot[key] = val

        results: list[WebDataResult] = []

        # P&L: sum all 4 quarters for cumulative fields, else take latest
        for (fyear, tmpl_hint), vals in pl_buckets.items():
            if not vals:
                continue
            annual = round(sum(vals), 4) if tmpl_hint in _CUMULATIVE_FIELDS else vals[-1]
            results.append(WebDataResult(
                source="nse",
                raw_field=f"nse/quarterly/{tmpl_hint}",
                template_field=tmpl_hint,
                value=annual,
                year=fyear,
                confidence="MEDIUM",
                unit_applied="INR Crores (NSE quarterly aggregated)",
            ))

        # Balance Sheet: latest quarter snapshot
        for (fyear, tmpl_hint), val in bs_snapshot.items():
            results.append(WebDataResult(
                source="nse",
                raw_field=f"nse/bs/{tmpl_hint}",
                template_field=tmpl_hint,
                value=val,
                year=fyear,
                confidence="MEDIUM",
                unit_applied="INR Crores (NSE latest quarter)",
            ))

        return results
