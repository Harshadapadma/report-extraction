"""
Web data collection module.

Sources (in priority order):
  1. yfinance     — Yahoo Finance; annual P&L, BS, CF; HIGH confidence
  2. screener     — Screener.in; 10-year tables; annual; HIGH confidence
  3. tickertape   — Tickertape API; annual P&L, BS, CF; HIGH confidence
  4. nse          — NSE India quarterly results aggregated to annual; MEDIUM confidence
  5. bse_xbrl     — BSE structured XML quarterly results; MEDIUM confidence (4Q aggregate)
  6. mca_xbrl     — MCA XBRL annual filings; most authoritative (stub — future)
  7. rbi_dbie     — RBI DBIE banking datasets; banking sector only (stub — future)

Usage:
    from web.collector import WebCollector, CompanyIdentifier
    collector = WebCollector()
    results = collector.collect(
        company=CompanyIdentifier(name="HDFC Bank", nse_symbol="HDFCBANK"),
        years=["F2025", "F2024"],
    )
    validation = collector.cross_validate(pdf_results, results)
"""

from web.base import CompanyIdentifier, WebDataResult, CrossValidationResult
from web.collector import WebCollector

__all__ = ["CompanyIdentifier", "WebDataResult", "CrossValidationResult", "WebCollector"]
