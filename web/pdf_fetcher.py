"""
Annual Report PDF Fetcher
==========================
Automatically finds and downloads annual report PDFs for Indian listed companies.

Sources tried in order:
  1. NSE India  — /api/annual-reports (most reliable, direct exchange link)
  2. BSE India  — corporate filings API
  3. Screener   — links embedded in company page

Returns the raw PDF bytes so the caller can pipe them into pdfplumber / pdf_parser.
"""

from __future__ import annotations

import re
import time
from datetime import date
from typing import Optional

from utils.helpers import get_logger

logger = get_logger(__name__)

_TIMEOUT = 30


# ── Report validity helpers ────────────────────────────────────────────────────

def _current_max_fy() -> int:
    """
    Return the latest fiscal year that could plausibly have an annual report.
    Indian FY ends March 31.  If today is before July 1 we allow the FY that
    just ended; otherwise we add 1 to account for reports published mid-year.
    (e.g. in April 2026 → max FY = 2026; in August 2026 → still 2026 because
    the FY 2026 report would just have been published.)
    """
    today = date.today()
    # Indian FY ends March 31.  Reports are usually available by July-August.
    fy = today.year if today.month >= 4 else today.year - 1
    return fy   # cap at this value — F(fy+1) cannot exist yet


# Filenames that indicate the document is NOT an annual report.
# These patterns are generic: earnings calls, investor presentations, quarterly
# results, credit ratings, prospectus, etc. all appear across companies.
_NON_AR_PATTERNS = re.compile(
    r"transcript|earnings.?call|investor.?present|roadshow"
    r"|quarterly|q[1-4]fy|fy\d{2}q[1-4]"      # quarterly docs
    r"|q[1-4].results?|half.?year|h[12]fy"     # half-yearly
    r"|credit.?rat|rating.?report"              # rating reports
    r"|prospectus|offer.?document|red.?herring"  # IPO docs
    r"|agm.?notice|postal.?ballot|notice.?agm"  # AGM notices
    r"|press.?release|presentation",            # press releases
    re.I,
)

# Filenames / link text that CONFIRM the document is an annual report.
_AR_CONFIRM = re.compile(
    r"annual.?report|annualreport|\bAR[_\-]\d|\b_A_\d{8}",
    re.I,
)


def _is_annual_report(filename: str, link_text: str = "", year_label: Optional[str] = None) -> bool:
    """
    Return True if the document looks like a genuine annual report.
    Rejects: earnings call transcripts, investor presentations, quarterly results,
             rating reports, AGM notices, press releases, etc.
    Also rejects documents whose inferred year is in the future.
    """
    combined = f"{filename} {link_text}"

    # Reject future years
    if year_label:
        try:
            y = int(year_label[1:])          # "F2027" → 2027
            if y > _current_max_fy():
                return False
        except (ValueError, IndexError):
            pass

    # Explicit non-AR pattern → reject
    if _NON_AR_PATTERNS.search(combined):
        return False

    # UUID-only filename (no meaningful name) with no confirming link text
    # e.g. "cc24f889-a466-463e-951a-0c193ee04689.pdf"
    uuid_re = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.pdf$",
        re.I,
    )
    if uuid_re.match(filename) and not _AR_CONFIRM.search(link_text):
        return False

    return True

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_API_HEADERS = {
    **_HEADERS,
    "Accept":  "application/json, */*",
    "Referer": "https://www.nseindia.com/",
}

_NSE_HOME = "https://www.nseindia.com"
_NSE_API  = "https://www.nseindia.com/api"


# ── Lazy imports ───────────────────────────────────────────────────────────────

def _get_requests():
    try:
        import requests
        return requests
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "requests>=2.31.0"])
        import requests
        return requests


# ── NSE session (reuses the same cookie management as nse_connector) ──────────

class _SimpleSession:
    """Lightweight NSE session with cookie bootstrap."""
    def __init__(self):
        self._session = None
        self._refreshed = 0.0

    def get(self):
        requests = _get_requests()
        now = time.time()
        if self._session is None or (now - self._refreshed) > 1800:
            s = requests.Session()
            try:
                r = s.get(_NSE_HOME, headers=_HEADERS, timeout=_TIMEOUT)
                r.raise_for_status()
                s.get(
                    f"{_NSE_HOME}/companies-listing/corporate-filings-annual-reports",
                    headers=_HEADERS, timeout=_TIMEOUT,
                )
                self._session  = s
                self._refreshed = now
                logger.debug("pdf_fetcher: NSE session bootstrapped")
            except Exception as e:
                logger.warning(f"pdf_fetcher: NSE session bootstrap failed — {e}")
                self._session = s
        return self._session


_nse_session = _SimpleSession()


# ── NSE annual report list ─────────────────────────────────────────────────────

def fetch_nse_report_list(symbol: str) -> list[dict]:
    """
    Fetch list of available annual reports from NSE for a given symbol.

    Returns list of dicts:
        {
          "year":     "F2025",
          "filename": "HDFCBANK_Annual_Report_2024-25.pdf",
          "url":      "https://...",
          "size_mb":  12.3,
        }
    Sorted newest-first.
    """
    session = _nse_session.get()
    url     = f"{_NSE_API}/annual-reports"
    params  = {"index": "equities", "symbol": symbol.upper()}

    try:
        resp = session.get(url, params=params, headers=_API_HEADERS, timeout=_TIMEOUT)
        if resp.status_code in (403, 404):
            logger.info(f"pdf_fetcher: NSE annual-reports endpoint returned {resp.status_code} for {symbol}")
            return []
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"pdf_fetcher: NSE annual-report list failed for {symbol}: {e}")
        return []

    records = data if isinstance(data, list) else data.get("data", [])
    if not isinstance(records, list):
        return []

    results = []
    for rec in records:
        # NSE response fields (field names may vary — handle both)
        link = (
            rec.get("pdfLink") or rec.get("fileName") or
            rec.get("downloadLink") or rec.get("attachmentURL") or ""
        )
        if not link:
            continue

        # Make absolute URL if relative
        if link.startswith("/"):
            link = f"https://www.nseindia.com{link}"
        if not link.startswith("http"):
            link = f"https://www.nseindia.com/{link}"

        # Extract year from the record
        from_date = rec.get("fromDate", "") or rec.get("startDate", "") or ""
        to_date   = rec.get("toDate",   "") or rec.get("endDate",   "") or ""
        year_label = _infer_fy_from_dates(from_date, to_date) or _infer_fy_from_text(link)

        filename = link.split("/")[-1].split("?")[0] or f"{symbol}_annual_report.pdf"

        try:
            size_bytes = int(rec.get("fileSize", 0) or 0)
            size_mb    = round(size_bytes / 1_048_576, 1) if size_bytes else 0
        except (TypeError, ValueError):
            size_mb = 0

        if not _is_annual_report(filename, year_label=year_label):
            logger.debug(f"pdf_fetcher: NSE skip non-AR/future: {filename}")
            continue

        results.append({
            "year":     year_label,
            "filename": filename,
            "url":      link,
            "size_mb":  size_mb,
        })

    results.sort(key=lambda r: r["year"] or "", reverse=True)
    return _dedup_by_year(results)


def _infer_fy_from_dates(from_date: str, to_date: str) -> Optional[str]:
    """Infer FY label from date strings like '2024-04-01' and '2025-03-31'."""
    for text in (to_date, from_date):
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(text))
        if m:
            year, month = int(m.group(1)), int(m.group(2))
            # Indian FY ends in March; if month ≤ 3 the fiscal year = same calendar year
            fy = year if month <= 3 else year + 1
            return f"F{fy}"
    return None


def _infer_fy_from_text(text: str) -> Optional[str]:
    """Infer FY label from a filename or URL string like '2024-25' or '2025'."""
    # Pattern: 2024-25 or 2024-2025
    m = re.search(r"20(\d{2})[-_](\d{2,4})", text)
    if m:
        end_part = m.group(2)
        end_year = int(end_part) if len(end_part) == 4 else int(f"20{end_part}")
        return f"F{end_year}"
    # Plain year: 2025
    m = re.search(r"\b(20\d{2})\b", text)
    if m:
        return f"F{m.group(1)}"
    return None


# ── BSE annual report list (fallback) ─────────────────────────────────────────

def fetch_bse_report_list(bse_code: str) -> list[dict]:
    """
    Fetch annual report list from BSE for a given scrip code.
    BSE API: https://api.bseindia.com/BseIndiaAPI/api/AnnualReport/w
    """
    requests = _get_requests()
    url = "https://api.bseindia.com/BseIndiaAPI/api/AnnualReport/w"
    params = {"scripcode": bse_code, "type": "AR"}
    bse_headers = {
        **_HEADERS,
        "Referer": "https://www.bseindia.com/",
        "Origin":  "https://www.bseindia.com",
    }

    try:
        resp = requests.get(url, params=params, headers=bse_headers, timeout=_TIMEOUT)
        if resp.status_code in (403, 404):
            return []
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"pdf_fetcher: BSE annual-report list failed for {bse_code}: {e}")
        return []

    records = data if isinstance(data, list) else data.get("Table", data.get("data", []))
    if not isinstance(records, list):
        return []

    results = []
    for rec in records:
        link = rec.get("PDFLINKURL") or rec.get("pdfLink") or rec.get("URL") or ""
        if not link:
            continue
        if link.startswith("/"):
            link = f"https://www.bseindia.com{link}"

        # Year from field like "2024-2025"
        year_text = rec.get("YEAR") or rec.get("year") or rec.get("Period") or ""
        year_label = _infer_fy_from_text(str(year_text)) or _infer_fy_from_text(link)
        filename   = link.split("/")[-1].split("?")[0] or f"bse_{bse_code}_ar.pdf"

        if not _is_annual_report(filename, year_label=year_label):
            logger.debug(f"pdf_fetcher: BSE skip non-AR/future: {filename}")
            continue

        results.append({
            "year":     year_label,
            "filename": filename,
            "url":      link,
            "size_mb":  0,
        })

    results.sort(key=lambda r: r["year"] or "", reverse=True)
    return _dedup_by_year(results)


# ── Screener annual report links (secondary fallback) ────────────────────────

def fetch_screener_report_links(symbol: str) -> list[dict]:
    """
    Scrape annual report PDF links from screener.in company page.

    Screener's document list mixes annual reports with earnings call transcripts,
    investor presentations, and quarterly results.  We filter strictly:
      - Only links whose text OR filename clearly indicates an annual report
      - Reject any link matching the non-AR patterns (transcripts, presentations…)
      - Reject future fiscal years
      - Deduplicate per year
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    requests = _get_requests()
    # Try both consolidated and standalone pages
    for suffix in ("/consolidated/", "/"):
        url = f"https://www.screener.in/company/{symbol.upper()}{suffix}"
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            if resp.status_code == 200:
                break
        except Exception:
            continue
    else:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # ── Strategy 1: look specifically inside the "Annual Reports" section ──────
    # Screener renders annual reports in a <section> or <div> with id/class
    # containing "annual-reports" or heading text "Annual Reports".
    results = []
    ar_section = None

    for tag in soup.find_all(["section", "div", "ul"]):
        heading = tag.find(["h2", "h3", "h4", "span"])
        if heading and "annual report" in heading.get_text(strip=True).lower():
            ar_section = tag
            break

    search_scope = ar_section if ar_section else soup

    for a_tag in search_scope.find_all("a", href=True):
        href      = a_tag["href"]
        link_text = a_tag.get_text(strip=True)

        # Must be a PDF link
        if not href.lower().endswith(".pdf"):
            continue

        if not href.startswith("http"):
            href = f"https://www.screener.in{href}"

        filename   = href.split("/")[-1].split("?")[0] or "report.pdf"
        year_label = _infer_fy_from_text(link_text) or _infer_fy_from_text(href)

        if not _is_annual_report(filename, link_text=link_text, year_label=year_label):
            logger.debug(f"pdf_fetcher: Screener skip non-AR/future: {filename}")
            continue

        results.append({
            "year":     year_label,
            "filename": filename,
            "url":      href,
            "size_mb":  0,
        })

    results.sort(key=lambda r: r["year"] or "", reverse=True)
    return _dedup_by_year(results)


# ── Deduplication helper ──────────────────────────────────────────────────────

def _dedup_by_year(reports: list[dict]) -> list[dict]:
    """
    Keep only the first (highest-priority) report per fiscal year.
    Input must be sorted newest-first; order is preserved.
    """
    seen: set[str] = set()
    deduped = []
    for r in reports:
        yr = r.get("year") or "__none__"
        if yr not in seen:
            seen.add(yr)
            deduped.append(r)
    return deduped


# ── Download a single PDF ─────────────────────────────────────────────────────

def download_pdf(url: str, progress_cb=None) -> Optional[bytes]:
    """
    Download a PDF from a URL and return its raw bytes.

    Tries multiple strategies so that BSE, NSE, and Screener-sourced links
    all work regardless of which session/cookie setup they need:
      1. Direct open request (works for BSE/Screener direct links)
      2. NSE session with cookie bootstrap (needed for NSE-hosted files)
      3. Fallback with minimal headers (last resort)

    progress_cb: optional callable(bytes_downloaded, total_bytes) for UI updates.
    Returns None if all strategies fail.
    """
    requests = _get_requests()

    def _stream(resp) -> Optional[bytes]:
        """Read a streaming response and return bytes, or None if not a PDF."""
        total = int(resp.headers.get("Content-Length", 0))
        chunks = []
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                chunks.append(chunk)
                downloaded += len(chunk)
                if progress_cb:
                    progress_cb(downloaded, total)
        data = b"".join(chunks)
        return data if data.startswith(b"%PDF") else None

    # ── Strategy 1: direct request (no session) ────────────────────────────────
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=60, stream=True)
        if resp.status_code == 200:
            data = _stream(resp)
            if data:
                logger.info(f"pdf_fetcher: ✓ direct  {len(data)/1_048_576:.1f} MB  {url[:80]}")
                return data
    except Exception as e:
        logger.debug(f"pdf_fetcher: strategy 1 failed: {e}")

    # ── Strategy 2: NSE session (cookie-bootstrapped) ─────────────────────────
    try:
        session = _nse_session.get()
        resp = session.get(url, headers=_API_HEADERS, timeout=60, stream=True)
        if resp.status_code == 200:
            data = _stream(resp)
            if data:
                logger.info(f"pdf_fetcher: ✓ NSE-session  {len(data)/1_048_576:.1f} MB  {url[:80]}")
                return data
        # Force session refresh and retry once
        _nse_session._session = None
        session = _nse_session.get()
        resp = session.get(url, headers=_API_HEADERS, timeout=60, stream=True)
        if resp.status_code == 200:
            data = _stream(resp)
            if data:
                logger.info(f"pdf_fetcher: ✓ NSE-refresh  {len(data)/1_048_576:.1f} MB  {url[:80]}")
                return data
    except Exception as e:
        logger.debug(f"pdf_fetcher: strategy 2 failed: {e}")

    # ── Strategy 3: BSE-specific headers ──────────────────────────────────────
    try:
        bse_headers = {
            **_HEADERS,
            "Referer": "https://www.bseindia.com/",
            "Origin":  "https://www.bseindia.com",
        }
        resp = requests.get(url, headers=bse_headers, timeout=60, stream=True)
        if resp.status_code == 200:
            data = _stream(resp)
            if data:
                logger.info(f"pdf_fetcher: ✓ BSE-headers  {len(data)/1_048_576:.1f} MB  {url[:80]}")
                return data
    except Exception as e:
        logger.debug(f"pdf_fetcher: strategy 3 failed: {e}")

    logger.warning(f"pdf_fetcher: all strategies failed for {url}")
    return None


# ── Convenience: get best available report for a given FY ────────────────────

def get_report_for_year(
    symbol: str,
    fy_label: str,           # e.g. "F2025"
    bse_code: Optional[str] = None,
) -> Optional[dict]:
    """
    Find the best available annual report download link for a given FY.
    Returns the report dict {year, filename, url, size_mb} or None.
    """
    # Try NSE first
    nse_reports = fetch_nse_report_list(symbol)
    for rep in nse_reports:
        if rep["year"] == fy_label:
            logger.info(f"pdf_fetcher: found {fy_label} report on NSE: {rep['filename']}")
            return rep

    # Try BSE
    if bse_code:
        bse_reports = fetch_bse_report_list(bse_code)
        for rep in bse_reports:
            if rep["year"] == fy_label:
                logger.info(f"pdf_fetcher: found {fy_label} report on BSE: {rep['filename']}")
                return rep

    # Try Screener links
    screener_reports = fetch_screener_report_links(symbol)
    for rep in screener_reports:
        if rep["year"] == fy_label:
            logger.info(f"pdf_fetcher: found {fy_label} report on Screener: {rep['filename']}")
            return rep

    logger.info(f"pdf_fetcher: no report found for {symbol} {fy_label}")
    return None
