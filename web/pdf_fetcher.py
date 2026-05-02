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

import io
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
                r = s.get(_NSE_HOME, headers=_HEADERS, timeout=(8, 15))
                r.raise_for_status()
                # Visit relevant pages to get the right cookies for each API
                for warm_url in [
                    f"{_NSE_HOME}/companies-listing/corporate-filings-annual-reports",
                    f"{_NSE_HOME}/companies-listing/corporate-filings-announcements",
                    f"{_NSE_HOME}/companies-listing/corporate-filings-financial-results",
                ]:
                    try:
                        s.get(warm_url, headers=_HEADERS, timeout=(8, 15))
                    except Exception:
                        pass
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

    def _extract_pdf_from_zip(data: bytes) -> Optional[bytes]:
        """If data is a ZIP archive, extract the largest PDF inside it."""
        try:
            import zipfile
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                pdf_names = [n for n in zf.namelist() if n.lower().endswith(".pdf")]
                if not pdf_names:
                    return None
                # Pick the largest PDF (likely the annual report, not a cover letter)
                pdf_names.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
                pdf_bytes = zf.read(pdf_names[0])
                if pdf_bytes.startswith(b"%PDF"):
                    logger.info(f"pdf_fetcher: extracted {pdf_names[0]} from ZIP ({len(pdf_bytes)//1024} KB)")
                    return pdf_bytes
        except Exception as e:
            logger.debug(f"pdf_fetcher: zip extraction failed: {e}")
        return None

    def _stream(resp) -> Optional[bytes]:
        """Read a streaming response and return bytes, or None if not a PDF/ZIP."""
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
        if data.startswith(b"%PDF"):
            return data
        # NSE sometimes packages annual reports as ZIP files containing a PDF
        if data[:4] == b"PK\x03\x04":
            return _extract_pdf_from_zip(data)
        return None

    # Per-strategy connect+read timeouts (connect_timeout, read_timeout).
    # Using a tuple prevents hanging when server accepts connection but never sends data.
    _DL_TIMEOUT = (10, 45)   # 10s to connect, 45s to read — total max ~55s per strategy

    # ── Strategy 1: direct request (no session) ────────────────────────────────
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_DL_TIMEOUT, stream=True)
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
        resp = session.get(url, headers=_API_HEADERS, timeout=_DL_TIMEOUT, stream=True)
        if resp.status_code == 200:
            data = _stream(resp)
            if data:
                logger.info(f"pdf_fetcher: ✓ NSE-session  {len(data)/1_048_576:.1f} MB  {url[:80]}")
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
        resp = requests.get(url, headers=bse_headers, timeout=_DL_TIMEOUT, stream=True)
        if resp.status_code == 200:
            data = _stream(resp)
            if data:
                logger.info(f"pdf_fetcher: ✓ BSE-headers  {len(data)/1_048_576:.1f} MB  {url[:80]}")
                return data
    except Exception as e:
        logger.debug(f"pdf_fetcher: strategy 3 failed: {e}")

    logger.warning(f"pdf_fetcher: all strategies failed for {url}")
    return None


# ── Investor Presentation helpers ────────────────────────────────────────────

# Patterns that positively identify investor / analyst presentations.
# NOTE: use .{0,4} instead of .? so we handle "Investors' Presentation",
# "Investor's Presentation", "Investors Presentation" etc.
_PRESENTATION_CONFIRM = re.compile(
    r"investor.{0,4}present|analyst.{0,4}(day|present|meet|brief)"
    r"|investor.{0,4}(day|meet|brief|update|deck|call)"
    r"|corporate.{0,4}present|results?.{0,4}present|earnings.{0,4}present"
    r"|quarterly.{0,4}(update|present|deck)|fact.{0,4}sheet"
    r"|investor.{0,4}relation.{0,10}present|ir.{0,4}present"
    r"|q[1-4].{0,10}(present|update|deck)|fy\d{2}.{0,10}present",
    re.I,
)


def _is_investor_presentation(filename: str, link_text: str = "") -> bool:
    """Return True if the document looks like an investor/analyst presentation."""
    combined = f"{filename} {link_text}"
    return bool(_PRESENTATION_CONFIRM.search(combined))


def fetch_nse_presentation_list(symbol: str) -> list[dict]:
    """
    Fetch list of investor presentations from NSE Corporate Announcements API.

    NSE's Corporate Announcements tab is where companies file investor
    presentations.  We query the announcements endpoint and filter by
    subject keywords matching "Investor Presentation".

    Returns list of dicts:
        {
          "year":     "F2025",
          "filename": "EQUITASBNK_InvestorPresentation_Q4FY25.pdf",
          "url":      "https://...",
          "size_mb":  3.2,
          "source":   "NSE",
        }
    Sorted newest-first.
    """
    session = _nse_session.get()

    # NSE Corporate Announcements API — returns all corporate announcements
    # (results, presentations, boardmeeting notices, etc.) for a symbol.
    url    = f"{_NSE_API}/corporate-announcements"
    params = {"index": "equities", "symbol": symbol.upper()}

    try:
        resp = session.get(url, params=params, headers=_API_HEADERS, timeout=_TIMEOUT)
        if resp.status_code in (403, 404):
            logger.info(f"pdf_fetcher: NSE corporate-announcements returned {resp.status_code} for {symbol}")
            return []
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"pdf_fetcher: NSE corporate-announcements failed for {symbol}: {e}")
        return []

    records = data if isinstance(data, list) else data.get("data", [])
    if not isinstance(records, list):
        return []

    results = []
    for rec in records:
        # Subject field identifies the filing type
        subject   = rec.get("subject", "") or rec.get("desc", "") or ""
        link_text = subject

        # Only take records whose subject matches investor presentation patterns
        if not _is_investor_presentation("", subject):
            continue

        # Attachment file — the actual PDF URL
        link = (
            rec.get("attchmntFile") or rec.get("pdfLink") or
            rec.get("attachmentURL") or rec.get("fileName") or ""
        )
        if not link:
            continue

        if link.startswith("/"):
            link = f"https://www.nseindia.com{link}"
        elif not link.startswith("http"):
            link = f"https://nsearchives.nseindia.com/{link.lstrip('/')}"

        # Date of broadcast
        bcast_dt  = rec.get("bcastDt") or rec.get("exchdisstime") or ""
        year_label = _infer_fy_from_text(subject) or _infer_fy_from_text(link) or _infer_fy_from_text(str(bcast_dt))
        filename   = link.split("/")[-1].split("?")[0] or f"{symbol}_investor_presentation.pdf"

        results.append({
            "year":     year_label,
            "filename": filename,
            "url":      link,
            "size_mb":  0,
            "source":   "NSE",
            "desc":     subject,
        })

    results.sort(key=lambda r: r["year"] or "", reverse=True)
    logger.info(f"pdf_fetcher: NSE corporate-announcements found {len(results)} presentations for {symbol}")
    return results


def fetch_bse_presentation_list(bse_code: str) -> list[dict]:
    """
    Fetch investor presentation list from BSE using type=PRESN.

    BSE separates presentations (PRESN) from annual reports (AR) in the same
    corporate filings API — so we just change the type parameter.
    """
    requests = _get_requests()
    url = "https://api.bseindia.com/BseIndiaAPI/api/AnnualReport/w"
    params = {"scripcode": bse_code, "type": "PRESN"}
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
        logger.warning(f"pdf_fetcher: BSE presentation list failed for {bse_code}: {e}")
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

        year_text  = rec.get("YEAR") or rec.get("year") or rec.get("Period") or ""
        year_label = _infer_fy_from_text(str(year_text)) or _infer_fy_from_text(link)
        filename   = link.split("/")[-1].split("?")[0] or f"bse_{bse_code}_presentation.pdf"

        results.append({
            "year":     year_label,
            "filename": filename,
            "url":      link,
            "size_mb":  0,
            "source":   "BSE",
        })

    results.sort(key=lambda r: r["year"] or "", reverse=True)
    return results


def _extract_website_from_html(html: str, skip_domains: tuple = ()) -> Optional[str]:
    """
    Extract the company's official website URL from a page's HTML.
    Searches for external links near 'website' / 'visit' label text.
    Skips known aggregator domains.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None

    _SKIP = {
        "screener.in", "bseindia.com", "nseindia.com", "moneycontrol.com",
        "tickertape.in", "google.com", "yahoo.com", "economictimes.com",
        "bloomberg.com", "reuters.com", "trendlyne.com", "valueresearchonline.com",
        *skip_domains,
    }

    soup = BeautifulSoup(html, "html.parser")

    # Strategy 1: find a link immediately following a "Website" label
    for tag in soup.find_all(string=re.compile(r"\bwebsite\b", re.I)):
        parent = tag.parent
        for container in [parent, parent.parent if parent else None]:
            if not container:
                continue
            a = container.find("a", href=True)
            if a:
                href = a["href"].strip()
                if href.startswith("http") and not any(d in href for d in _SKIP):
                    return href.rstrip("/")

    # Strategy 2: any external link in company-info / about sections
    for section_id in ("company-info", "about", "company-details", "about-company"):
        sec = soup.find(id=re.compile(section_id, re.I)) or \
              soup.find(class_=re.compile(section_id.replace("-", ".?"), re.I))
        if sec:
            for a in sec.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith("http") and not any(d in href for d in _SKIP):
                    return href.rstrip("/")

    return None


# Module-level cache: symbol → website URL (avoids repeated lookups per session)
_website_cache: dict[str, Optional[str]] = {}


def _get_company_website(symbol: str, bse_code: Optional[str] = None) -> Optional[str]:
    """
    Resolve a company's official website URL.  Results are cached in-process
    so repeated calls for the same company cost nothing.

    Sources tried in order (most → least reliable for server/cloud deployment):
      1. yfinance Ticker.info['website']  — designed for programmatic use,
                                            works from any server IP
      2. BSE ComHeader API                — proper JSON API, generally accessible
      3. Screener.in company page         — explicit 'Website' link, may rate-limit
                                            on cloud IPs but usually works
      4. BSE stock page HTML scrape       — last resort HTML scrape
    """
    cache_key = f"{symbol}:{bse_code}"
    if cache_key in _website_cache:
        return _website_cache[cache_key]

    requests = _get_requests()

    def _cache_and_return(url: Optional[str]) -> Optional[str]:
        _website_cache[cache_key] = url
        return url

    # ── 0. NSE quote-equity API — best source, returns JSON with website field ──
    try:
        nse_s  = _nse_session.get()
        resp   = nse_s.get(
            f"{_NSE_API}/quote-equity",
            params={"symbol": symbol.upper()},
            headers=_API_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            data   = resp.json()
            info   = data.get("info", data)
            for key in ("companyWebSite", "website", "companyWebsite",
                        "officialSite", "Website", "companyWeb"):
                site = info.get(key, "")
                if site and str(site).startswith("http"):
                    logger.info(f"pdf_fetcher: company website from NSE quote-equity: {site}")
                    return _cache_and_return(str(site).rstrip("/"))
    except Exception as e:
        logger.debug(f"pdf_fetcher: NSE quote-equity website lookup failed: {e}")

    # ── 1. yfinance — works from cloud/server IPs, no scraping needed ─────────
    try:
        import yfinance as yf
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

        def _yf_website():
            for suffix in (".NS", ".BO"):
                info = yf.Ticker(symbol.upper() + suffix).info
                site = info.get("website") or info.get("companyOfficialSite")
                if site and site.startswith("http"):
                    return site
            return None

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_yf_website)
            try:
                site = fut.result(timeout=15)
                if site:
                    logger.info(f"pdf_fetcher: company website from yfinance: {site}")
                    return _cache_and_return(site.rstrip("/"))
            except FutTimeout:
                logger.debug("pdf_fetcher: yfinance website lookup timed out")
    except Exception as e:
        logger.debug(f"pdf_fetcher: yfinance website lookup failed: {e}")

    # ── 2. BSE ComHeader JSON (proper API — accessible from servers) ──────────
    if bse_code:
        try:
            bse_h = {**_HEADERS, "Accept": "application/json, */*",
                     "Referer": "https://www.bseindia.com/",
                     "Origin": "https://www.bseindia.com"}
            resp = requests.get(
                "https://api.bseindia.com/BseIndiaAPI/api/ComHeader/w",
                params={"quotetype": "EQ", "scripcode": bse_code},
                headers=bse_h, timeout=_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                rec  = data if isinstance(data, dict) else (data[0] if isinstance(data, list) and data else {})
                for v in rec.values():
                    if isinstance(v, str) and v.startswith("http") and "." in v:
                        if not any(d in v for d in ("bseindia", "nseindia")):
                            logger.info(f"pdf_fetcher: company website from BSE ComHeader: {v}")
                            return _cache_and_return(v.rstrip("/"))
        except Exception as e:
            logger.debug(f"pdf_fetcher: BSE ComHeader failed: {e}")

    # ── 3. Screener.in (reliable but may rate-limit on cloud IPs) ────────────
    for suffix in ("/consolidated/", "/"):
        try:
            url  = f"https://www.screener.in/company/{symbol.upper()}{suffix}"
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            if resp.status_code == 200:
                site = _extract_website_from_html(resp.text)
                if site:
                    logger.info(f"pdf_fetcher: company website from Screener: {site}")
                    return _cache_and_return(site)
        except Exception as e:
            logger.debug(f"pdf_fetcher: Screener website lookup failed: {e}")

    # ── 4. BSE stock page HTML scrape (last resort) ───────────────────────────
    if bse_code:
        try:
            bse_page = (
                f"https://www.bseindia.com/stock-share-price/"
                f"placeholder/placeholder/{bse_code}/"
            )
            resp = requests.get(bse_page, headers=_HEADERS, timeout=_TIMEOUT)
            if resp.status_code == 200:
                site = _extract_website_from_html(resp.text)
                if site:
                    logger.info(f"pdf_fetcher: company website from BSE page: {site}")
                    return _cache_and_return(site)
        except Exception as e:
            logger.debug(f"pdf_fetcher: BSE page scrape failed: {e}")

    logger.warning(f"pdf_fetcher: could not resolve company website for {symbol}")
    return _cache_and_return(None)


def fetch_company_ir_presentations(
    symbol: str,
    bse_code: Optional[str] = None,
) -> list[dict]:
    """
    Scrape the company's own Investor Relations page for presentation PDFs.

    Strategy:
      1. Resolve company website from BSE/NSE profile.
      2. Try common IR page paths (/investor-relations, /investors, /ir, …)
      3. Also try subpaths for presentations specifically.
      4. Collect all PDF links matching investor presentation patterns.
      5. Return same dict format as other fetchers.

    This is entirely generic — no hardcoded company names or URLs.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    requests = _get_requests()

    company_site = _get_company_website(symbol, bse_code)
    if not company_site:
        logger.info(f"pdf_fetcher: IR scraper — no company website found for {symbol}")
        return []

    # Common IR page path patterns used by Indian listed companies
    _IR_PATHS = [
        "/investor-relations/presentations",
        "/investor-relations/investor-presentation",
        "/investor-relations/financial-results",
        "/investor-relations",
        "/investors/presentations",
        "/investors/investor-presentation",
        "/investors",
        "/ir/presentations",
        "/ir",
        "/corporate/investor-relations",
        "/about-us/investor-relations",
    ]

    found_pdfs: dict[str, dict] = {}   # url → record

    for path in _IR_PATHS:
        page_url = f"{company_site}{path}"
        try:
            resp = requests.get(page_url, headers=_HEADERS, timeout=_TIMEOUT, allow_redirects=True)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")

            for a in soup.find_all("a", href=True):
                href      = a["href"].strip()
                link_text = a.get_text(strip=True)

                # Make absolute URL
                if href.startswith("//"):
                    href = "https:" + href
                elif href.startswith("/"):
                    href = company_site + href
                elif not href.startswith("http"):
                    continue

                # Must be a PDF
                if not (href.lower().endswith(".pdf") or "pdf" in href.lower()):
                    continue

                # Must look like a presentation
                if not _is_investor_presentation(href, link_text):
                    continue

                if href not in found_pdfs:
                    year_label = _infer_fy_from_text(link_text) or _infer_fy_from_text(href)
                    filename   = href.split("/")[-1].split("?")[0] or f"{symbol}_ir_presentation.pdf"
                    found_pdfs[href] = {
                        "year":     year_label,
                        "filename": filename,
                        "url":      href,
                        "size_mb":  0,
                        "source":   "Company IR",
                    }
                    logger.debug(f"pdf_fetcher: IR scraper found: {filename}")

            if found_pdfs:
                # Found PDFs on this path — try a couple more subpaths but stop deep search
                if len(found_pdfs) >= 5:
                    break
        except Exception as e:
            logger.debug(f"pdf_fetcher: IR scraper {page_url}: {e}")
            continue

    results = list(found_pdfs.values())
    results.sort(key=lambda r: r["year"] or "", reverse=True)
    logger.info(f"pdf_fetcher: IR scraper found {len(results)} presentations for {symbol}")
    return results


def get_available_presentations(symbol: str, bse_code: Optional[str] = None) -> list[dict]:
    """
    Return all available investor presentation links, newest-first.

    Sources tried in order:
      1. NSE Corporate Announcements API  (dedicated presentations tab)
      2. BSE PRESN filings               (BSE type=PRESN category)
      3. Company IR website              (generic scraper — most reliable for recent quarterly decks)
    """
    seen_urls: set[str] = set()
    all_pres: list[dict] = []

    # 1. NSE Corporate Announcements (primary — the correct NSE tab for presentations)
    for rep in fetch_nse_presentation_list(symbol):
        url = rep["url"]
        if url and url not in seen_urls:
            all_pres.append(rep)
            seen_urls.add(url)

    # 2. BSE PRESN category
    if bse_code:
        for rep in fetch_bse_presentation_list(bse_code):
            url = rep["url"]
            if url and url not in seen_urls:
                all_pres.append(rep)
                seen_urls.add(url)

    # 3. Company's own IR page — always try; often has the most recent quarterly deck
    for rep in fetch_company_ir_presentations(symbol, bse_code):
        url = rep["url"]
        if url and url not in seen_urls:
            all_pres.append(rep)
            seen_urls.add(url)

    all_pres.sort(key=lambda r: r.get("year") or "", reverse=True)
    logger.info(f"pdf_fetcher: total presentations found for {symbol}: {len(all_pres)}")
    return all_pres


# ── Quarterly Financial Results PDF helpers ───────────────────────────────────

# Patterns that identify quarterly / half-yearly financial result PDFs.
_QRESULT_CONFIRM = re.compile(
    r"financial.?result|quarterly.?result|unaudited.?result|audited.?result"
    r"|quarter.?ended|quarter.?end|q[1-4].{0,10}result"
    r"|half.?year.?result|h[12].{0,5}result"
    r"|results?.for.the.quarter|results?.for.the.half",
    re.I,
)

# Indian FY months → quarter map  (Apr=Q1, Jul=Q2, Oct=Q3, Jan=Q4)
_MONTH_TO_QUARTER: dict[int, str] = {
    6: "Q1", 7: "Q1",        # Jun/Jul → Q1 ending
    9: "Q2", 10: "Q2",       # Sep/Oct → Q2 ending
    12: "Q3", 1: "Q3",       # Dec/Jan → Q3 ending
    3: "Q4", 4: "Q4",        # Mar/Apr → Q4 ending (annual)
}

_MONTH_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


_QRESULT_QUARTER_TAG = re.compile(
    r"(?<![0-9])q[1-4](?![0-9]).{0,30}result"   # "Q4FY25_Financial_Results"
    r"|result.{0,30}(?<![0-9])q[1-4](?![0-9])", # "results Q4"
    re.I,
)


def _is_quarterly_result(filename: str, link_text: str = "") -> bool:
    """Return True if the document is a quarterly / half-yearly financial result.
    Explicitly excludes investor presentations even if they mention a quarter."""
    combined = f"{filename} {link_text}"
    if _PRESENTATION_CONFIRM.search(combined):
        return False   # presentations are not financial result filings
    # Primary: explicit "financial result", "quarterly result" etc.
    if _QRESULT_CONFIRM.search(combined):
        return True
    # Secondary: Qn tag + "result" word anywhere in the string
    if _QRESULT_QUARTER_TAG.search(combined):
        return True
    return False


def _infer_quarter(text: str) -> Optional[str]:
    """
    Extract quarter label from a string.
    Returns "Q1", "Q2", "Q3", or "Q4", or None.
    Priority: explicit Qn tag > month name > date.
    """
    # Explicit Q1-Q4  (handles Q4FY25, Q4 FY25, Q4-FY25, Q4/FY25, Q4.)
    m = re.search(r"(?<![0-9])q([1-4])(?![0-9])", text, re.I)
    if m:
        return f"Q{m.group(1)}"

    # Month name (quarter end months: Jun/Sep/Dec/Mar)
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*", text, re.I)
    if m:
        mon = _MONTH_NUM.get(m.group(1).lower()[:3])
        if mon:
            return _MONTH_TO_QUARTER.get(mon)

    # DD-MM-YYYY or YYYY-MM-DD date
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if m:
        mon = int(m.group(2))
        return _MONTH_TO_QUARTER.get(mon)
    m = re.search(r"\b(\d{2})-(\d{2})-(\d{4})\b", text)
    if m:
        mon = int(m.group(2))
        return _MONTH_TO_QUARTER.get(mon)

    return None


def fetch_nse_quarterly_result_pdfs(symbol: str) -> list[dict]:
    """
    Fetch quarterly result PDF + XBRL links from NSE's financial-results API.

    NSE's Financial Results tab has one record per quarter with both a PDF
    link (human-readable filing) and an XBRL XML link (machine-readable).
    We capture both so callers can prefer XBRL when available.

    Returns list of dicts:
        {
          "year":      "F2025",
          "quarter":   "Q4",       # Q1/Q2/Q3/Q4
          "is_annual": True,       # True only for Q4 (full-year figures included)
          "filename":  "EQUITASBNK_Q4FY25_Results.pdf",
          "url":       "https://...",   # PDF link
          "xbrl_url":  "https://...",   # XBRL XML link (may be None)
          "source":    "NSE",
        }
    Sorted newest-first.
    """
    session = _nse_session.get()
    url     = f"{_NSE_API}/financial-results"
    params  = {"index": "equities", "symbol": symbol.upper(), "period": "Quarterly"}

    try:
        resp = session.get(url, params=params, headers=_API_HEADERS, timeout=_TIMEOUT)
        if resp.status_code in (403, 404):
            return []
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"pdf_fetcher: NSE quarterly result PDFs failed for {symbol}: {e}")
        return []

    records = data if isinstance(data, list) else data.get("data", [])
    if not isinstance(records, list):
        return []

    def _abs(link: str) -> str:
        if not link:
            return ""
        if link.startswith("/"):
            return f"https://www.nseindia.com{link}"
        if not link.startswith("http"):
            return f"https://nsearchives.nseindia.com/{link.lstrip('/')}"
        return link

    results = []
    for rec in records:
        # ── PDF link (primary, human-readable filing) ──────────────────────────
        pdf_link = _abs(
            rec.get("pdfLink") or rec.get("attachmentURL") or
            rec.get("fileName") or rec.get("pdfname") or ""
        )

        # ── XBRL link (machine-readable structured data) ───────────────────────
        xbrl_link = _abs(
            rec.get("xbrlLink") or rec.get("xbrl") or
            rec.get("xbrlFile") or rec.get("xbrlName") or ""
        )

        # Need at least one of PDF or XBRL
        if not pdf_link and not xbrl_link:
            continue

        # Period / date fields
        period     = rec.get("period") or rec.get("toDate") or rec.get("periodEnded") or ""
        to_date    = rec.get("toDate") or ""
        year_label = _infer_fy_from_dates("", str(to_date)) or _infer_fy_from_text(str(period))
        quarter    = _infer_quarter(str(period)) or _infer_quarter(str(to_date))

        primary_link = pdf_link or xbrl_link
        filename     = primary_link.split("/")[-1].split("?")[0] or f"{symbol}_quarterly_result.pdf"

        results.append({
            "year":      year_label,
            "quarter":   quarter,
            "is_annual": quarter == "Q4",
            "filename":  filename,
            "url":       primary_link,
            "xbrl_url":  xbrl_link if xbrl_link != primary_link else None,
            "size_mb":   0,
            "source":    "NSE",
        })

    results.sort(key=lambda r: (r["year"] or "", r["quarter"] or ""), reverse=True)
    logger.info(f"pdf_fetcher: NSE financial-results found {len(results)} quarterly results for {symbol}")
    return results


def fetch_bse_quarterly_result_pdfs(bse_code: str) -> list[dict]:
    """
    Fetch quarterly result PDF links from BSE's financial results API.
    BSE returns quarterly result metadata including a PDF download link.
    """
    requests = _get_requests()
    bse_headers = {
        **_HEADERS,
        "Accept":  "application/json, */*",
        "Referer": "https://www.bseindia.com/",
        "Origin":  "https://www.bseindia.com",
    }

    # Try consolidated first, then standalone
    results = []
    for typeflag in ("C", "S"):
        url    = "https://api.bseindia.com/BseIndiaAPI/api/FinancialResult4/w"
        params = {"scripcode": bse_code, "typeflag": typeflag}
        try:
            resp = requests.get(url, params=params, headers=bse_headers, timeout=_TIMEOUT)
            if resp.status_code in (403, 404):
                continue
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"pdf_fetcher: BSE quarterly results failed for {bse_code}: {e}")
            continue

        records = data if isinstance(data, list) else data.get("Table", data.get("data", []))
        if not isinstance(records, list):
            continue

        seen_urls = {r["url"] for r in results}
        for rec in records:
            # BSE result record fields
            link = rec.get("PDFLINKURL") or rec.get("pdfLink") or rec.get("XBRL_LINK") or ""
            if not link:
                continue
            if link.startswith("/"):
                link = f"https://www.bseindia.com{link}"

            period     = rec.get("TO_DATE") or rec.get("QuarterEndDate") or ""
            year_label = _infer_fy_from_text(str(period)) or _infer_fy_from_dates("", str(period))
            quarter    = _infer_quarter(str(period))
            filename   = link.split("/")[-1].split("?")[0] or f"bse_{bse_code}_result.pdf"

            if link not in seen_urls:
                # Also capture XBRL link if present in same record
                xbrl_link = rec.get("XBRL_LINK") or rec.get("XBRLLink") or ""
                if xbrl_link and xbrl_link.startswith("/"):
                    xbrl_link = f"https://www.bseindia.com{xbrl_link}"
                results.append({
                    "year":      year_label,
                    "quarter":   quarter,
                    "is_annual": quarter == "Q4",
                    "filename":  filename,
                    "url":       link,
                    "xbrl_url":  xbrl_link or None,
                    "size_mb":   0,
                    "source":    "BSE",
                })
                seen_urls.add(link)

        if results:
            break   # consolidated found — skip standalone

    results.sort(key=lambda r: (r["year"] or "", r["quarter"] or ""), reverse=True)
    return results


def fetch_company_annual_reports(
    symbol: str,
    bse_code: Optional[str] = None,
) -> list[dict]:
    """
    Scrape annual report PDF links from the company's own IR page.
    Used as a fallback when NSE/BSE/Screener don't have the report.

    Returns same dict format as other fetchers:
      {year, filename, url, size_mb, source}
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    requests = _get_requests()

    company_site = _get_company_website(symbol, bse_code)
    if not company_site:
        return []

    _AR_IR_PATHS = [
        "/investor-relations/annual-reports",
        "/investor-relations/annual-report",
        "/investors/annual-reports",
        "/investors/annual-report",
        "/ir/annual-reports",
        "/annual-reports",
        "/annual-report",
        "/investor-relations",
        "/investors",
        "/ir",
    ]

    seen_urls: set[str] = set()
    results: list[dict] = []

    for path in _AR_IR_PATHS:
        page_url = f"{company_site}{path}"
        try:
            resp = requests.get(page_url, headers=_HEADERS, timeout=_TIMEOUT)
            if resp.status_code not in (200, 301, 302):
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                link_text = a.get_text(" ", strip=True)
                if not href.lower().endswith(".pdf"):
                    continue
                if href.startswith("//"):
                    href = "https:" + href
                elif href.startswith("/"):
                    href = company_site + href
                elif not href.startswith("http"):
                    continue
                if href in seen_urls:
                    continue
                if not _is_annual_report(href.split("/")[-1], link_text):
                    continue
                year = _infer_fy_from_text(href.split("/")[-1] + " " + link_text)
                seen_urls.add(href)
                results.append({
                    "year":     year,
                    "filename": href.split("/")[-1],
                    "url":      href,
                    "size_mb":  None,
                    "source":   "company_ir",
                })
                logger.info(f"pdf_fetcher: company IR annual report [{year}]: {href}")
        except Exception as e:
            logger.debug(f"pdf_fetcher: company IR annual report scrape failed at {page_url}: {e}")
            continue

    results.sort(key=lambda r: r["year"] or "", reverse=True)
    return results


def fetch_company_quarterly_result_pdfs(
    symbol: str,
    bse_code: Optional[str] = None,
) -> list[dict]:
    """
    Scrape quarterly financial result PDF links from the company's own IR page.

    Tries common IR subpaths for the financial results section, then
    collects any PDF whose filename / link text matches quarterly result
    patterns.  Entirely generic — no hardcoded company URLs.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    requests = _get_requests()

    company_site = _get_company_website(symbol, bse_code)
    if not company_site:
        logger.info(f"pdf_fetcher: qresult IR scraper — no website for {symbol}")
        return []

    _IR_RESULT_PATHS = [
        "/investor-relations/financial-results",
        "/investor-relations/quarterly-results",
        "/investor-relations/results",
        "/investors/financial-results",
        "/investors/quarterly-results",
        "/investors/results",
        "/financial-results",
        "/quarterly-results",
        "/investor-relations",          # broad fallback — will filter by pattern
        "/investors",
    ]

    found: dict[str, dict] = {}

    for path in _IR_RESULT_PATHS:
        page_url = f"{company_site}{path}"
        try:
            resp = requests.get(page_url, headers=_HEADERS, timeout=_TIMEOUT, allow_redirects=True)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")

            for a in soup.find_all("a", href=True):
                href      = a["href"].strip()
                link_text = a.get_text(strip=True)

                if href.startswith("//"):
                    href = "https:" + href
                elif href.startswith("/"):
                    href = company_site + href
                elif not href.startswith("http"):
                    continue

                if not (href.lower().endswith(".pdf") or "pdf" in href.lower()):
                    continue

                if not _is_quarterly_result(href, link_text):
                    continue

                if href not in found:
                    combined   = f"{href} {link_text}"
                    year_label = _infer_fy_from_text(link_text) or _infer_fy_from_text(href)
                    quarter    = _infer_quarter(combined)
                    filename   = href.split("/")[-1].split("?")[0] or f"{symbol}_result.pdf"
                    found[href] = {
                        "year":      year_label,
                        "quarter":   quarter,
                        "is_annual": quarter == "Q4",
                        "filename":  filename,
                        "url":       href,
                        "size_mb":   0,
                        "source":    "Company IR",
                    }
                    logger.debug(f"pdf_fetcher: qresult scraper found: {filename}")

            if len(found) >= 8:   # enough results, stop scanning
                break
        except Exception as e:
            logger.debug(f"pdf_fetcher: qresult scraper {page_url}: {e}")
            continue

    results = list(found.values())
    results.sort(key=lambda r: (r["year"] or "", r["quarter"] or ""), reverse=True)
    logger.info(f"pdf_fetcher: qresult scraper found {len(results)} results for {symbol}")
    return results


def _parse_xbrl_xml(xml_bytes: bytes, years: Optional[list] = None) -> dict:
    """
    Parse an XBRL XML file (IndAS / Indian GAAP format used by BSE/NSE filings).

    Returns:  {template_label: {fy_label: value_in_crores}, ...}

    The XBRL format has:
      - <context> elements defining time periods (instant or duration)
      - Fact elements like <in-bfsi:InterestEarned contextRef="..." decimals="-5">1234</in-bfsi:InterestEarned>

    We strip the namespace prefix and match the local tag name against
    a known-label map, then convert values to INR Crores.
    """
    import xml.etree.ElementTree as ET

    # Map of XBRL local tag names (case-insensitive, stripped of namespace) → template label
    # Banking-specific tags follow RBI Schedule 13-18 naming conventions.
    _XBRL_TAG_MAP: dict[str, str] = {
        # ── P&L — Interest ──────────────────────────────────────────────────────
        "interestearned":                      "Interest Income",
        "interestincome":                      "Interest Income",
        "incomeonadvances":                    "Interest Income",
        "interestexpended":                    "Interest Expended",
        "interestexpense":                     "Interest Expended",
        "netinterestincome":                   "Net Interest Income",
        # ── P&L — Other income ──────────────────────────────────────────────────
        "otheroperatingrevenue":               "Other Income",
        "otherincome":                         "Other Income",
        "feeincome":                           "Other Income",
        "totalrevenuefromoperations":          "Revenue from Operations",
        "revenuefromoperations":               "Revenue from Operations",
        "totalrevenue":                        "Total Revenue",
        "totalincome":                         "Total Revenue",
        # ── P&L — Operating expenses (two distinct sub-lines for banks) ────────
        # Schedule 16 (i): Payments to employees  → "Employee expenses"
        "paymentstoandfprovisionforemployees": "Employee Expenses",
        "paymentstoandfemployees":             "Employee Expenses",
        "employeebenefitexpense":              "Employee Expenses",
        "staffexpenses":                       "Employee Expenses",
        "employeecost":                        "Employee Expenses",
        "salarywagesandallowances":            "Employee Expenses",
        # Schedule 16 (ii): Other operating expenses → "Operating expenses"
        "otheroperatingexpenses":              "Operating Expenses",
        "operatingexpenses":                   "Operating Expenses",
        "operatingexpenditure":                "Operating Expenses",
        "rentleasestaxes":                     "Operating Expenses",          # sub-item proxy
        # Total opex (both sub-lines combined — only use if neither above matched)
        "totaloperatingexpenses":              "Total Operating Expenses",    # NOT mapped to template col
        # ── P&L — Provisions / credit cost ─────────────────────────────────────
        "provisions":                          "Provisions and Contingencies",
        "provisionandcontingencies":           "Provisions and Contingencies",
        "provisionsandcontingencies":          "Provisions and Contingencies",
        "provisionfornonperformingassets":     "Provisions and Contingencies",
        "creditcost":                          "Provisions and Contingencies",
        # ── P&L — Profit lines ──────────────────────────────────────────────────
        "operatingprofit":                     "Operating Profit",
        "preprovisioningoperatingprofit":      "Pre-provisioning Operating Profit",
        "ppop":                                "Pre-provisioning Operating Profit",
        "profitbeforetax":                     "Profit Before Tax",
        "profitbeforeincometax":               "Profit Before Tax",
        "taxexpense":                          "Tax Expense",
        "incometaxexpense":                    "Tax Expense",
        "profitaftertax":                      "Profit After Tax",
        "profitfortheyear":                    "Profit After Tax",
        "profitfortheperiod":                  "Profit After Tax",
        "netprofit":                           "Profit After Tax",
        "earningspersharebasic":               "EPS (Basic)",
        "basicearningspershare":               "EPS (Basic)",
        "earningspersharediluted":             "EPS (Diluted)",
        "dilutedearningspershare":             "EPS (Diluted)",
        # ── Balance Sheet ────────────────────────────────────────────────────────
        "deposits":                            "Deposits",
        "advances":                            "Advances",
        "investments":                         "Investments",
        "borrowings":                          "Total Borrowings",
        "capitalandreserves":                  "Total Equity",
        "reservesandsurplus":                  "Reserves and Surplus",
        "sharecapital":                        "Share Capital",
        "totalassets":                         "Total Assets",
        "balancesheetsize":                    "Total Assets",
        "cashandcashequivalents":              "Cash and Cash Equivalents",
        "cashandbalancewithrbi":               "Cash and Balance with RBI",
        # ── Asset Quality ─────────────────────────────────────────────────────────
        "grossnpa":                            "Gross NPA",
        "grossnonperformingassets":            "Gross NPA",
        "grossnpassets":                       "Gross NPA",
        "netnonperformingassets":              "Net NPA",
        "netnpa":                              "Net NPA",
        "netnonperformingadvances":            "Net NPA",
        "gnparatio":                           "Gross NPA %",
        "grossnparatio":                       "Gross NPA %",
        "nnparatio":                           "Net NPA %",
        "netnparatio":                         "Net NPA %",
        "provisioncoverageratio":              "PCR %",
        # ── Key Ratios ───────────────────────────────────────────────────────────
        "returnonassets":                      "ROA",
        "returnonequity":                      "ROE",
        "netinterestmargin":                   "NIM",
        "capitaladequacyratio":                "CRAR",
        "tier1capitalratio":                   "Tier 1 Ratio",
        "tier2capitalratio":                   "Tier 2 Ratio",
    }

    year_set = set(years) if years else None

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        logger.warning(f"xbrl_parser: XML parse error: {e}")
        return {}

    ns_re = re.compile(r"\{[^}]+\}")   # strip {namespace} prefix

    # ── Parse contexts: contextId → (fy_label, duration_months) ──────────────
    # XBRL dates are always YYYY-MM-DD.
    # For Q4 filings, there are TWO period types that end on the same March 31:
    #   • Full year:   startDate=2024-04-01, endDate=2025-03-31  → 12-month duration
    #   • Q4 quarter:  startDate=2025-01-01, endDate=2025-03-31  → 3-month duration
    # We track duration so that when a label has both, we prefer the 12-month (annual) value.
    from datetime import datetime

    def _duration_months(start: str, end: str) -> int:
        """Approximate number of months between two YYYY-MM-DD strings."""
        try:
            s = datetime.strptime(start.strip(), "%Y-%m-%d")
            e = datetime.strptime(end.strip(),   "%Y-%m-%d")
            return (e.year - s.year) * 12 + (e.month - s.month)
        except Exception:
            return 0

    # contexts maps ctx_id → (fy_label, duration_months)
    contexts: dict[str, tuple[str, int]] = {}

    for ctx in root.iter():
        if ns_re.sub("", ctx.tag).lower() != "context":
            continue
        ctx_id     = ctx.get("id", "")
        start_date = ""
        end_date   = ""
        for child in ctx:
            if ns_re.sub("", child.tag).lower() == "period":
                for gc in child:
                    gtag = ns_re.sub("", gc.tag).lower()
                    if gtag == "startdate":
                        start_date = (gc.text or "").strip()
                    elif gtag in ("enddate", "instant"):
                        end_date = (gc.text or "").strip()
                break
        if end_date:
            fy = _infer_fy_from_dates("", end_date)
            if fy:
                dur = _duration_months(start_date, end_date) if start_date else 0
                contexts[ctx_id] = (fy, dur)

    if not contexts:
        logger.warning("xbrl_parser: no contexts parsed — XBRL structure may differ")
        return {}

    # ── Detect reporting unit ─────────────────────────────────────────────────
    # Indian NSE/BSE XBRL filings use two conventions:
    #   A) Values in actual INR Rupees with unit="iso4217:INR"   → 1e-7 multiplier
    #   B) Values in INR Lakhs with unit="INR" and custom scale  → 0.01 multiplier
    #   C) Values already in Crores                              → 1.0 multiplier
    #
    # Strategy:
    #  1. Look for explicit currency label declarations (measure, ScaleOfPresentation, etc.)
    #  2. Fall back: inspect `decimals` attribute of the FIRST numeric fact element.
    #     NSE XBRL in rupees → decimals="-5" or "-7" (rounded to lakh/crore precision)
    #     NSE XBRL in lakhs  → decimals="-2" or  "0" (rounded to hundred lakhs)
    #  3. Auto-detect from actual value magnitudes as final fallback.

    multiplier = 0.01   # conservative default: lakhs → crores

    _scale_found = False
    for elem in root.iter():
        local = ns_re.sub("", elem.tag).lower()
        text  = (elem.text or "").strip().lower()
        if not text:
            continue
        if local in ("measure", "unit", "reportingcurrencyunit",
                     "scaleofpresentation", "unitnumerator", "scalefactor"):
            if "crore" in text:
                multiplier = 1.0
                _scale_found = True
            elif "lakh" in text or "hundredthousand" in text or "100000" in text:
                multiplier = 0.01
                _scale_found = True
            elif "million" in text:
                multiplier = 0.1
                _scale_found = True
            elif "thousand" in text and "hundred" not in text:
                multiplier = 1e-4
                _scale_found = True
            elif "iso4217:inr" in text or text == "inr" or "rupee" in text:
                # Raw rupees — divide by 10 million (1 crore = 10^7 rupees)
                multiplier = 1e-7
                _scale_found = True
            if _scale_found:
                break

    # Fallback: inspect `decimals` attribute on first meaningful numeric fact
    if not _scale_found:
        for elem in root.iter():
            decimals_attr = elem.get("decimals", "")
            raw_text = (elem.text or "").strip().replace(",", "")
            if not decimals_attr or not raw_text:
                continue
            try:
                val     = float(raw_text)
                dec_int = int(decimals_attr)
            except (ValueError, TypeError):
                continue
            if val == 0:
                continue
            # `decimals=-5` in a pure-rupee filing → value precision is ±100,000 (1 lakh)
            # Real magnitude is just the raw number (e.g., 12,345,678,900 rupees)
            # `decimals=-2` in a lakh filing → value precision is ±100 (lakhs)
            # Use the combination of actual value size + decimals to determine:
            if dec_int <= -6 and abs(val) > 1e7:
                multiplier = 1e-7   # raw rupees
            elif dec_int <= -4 and abs(val) > 1e4:
                multiplier = 1e-7   # likely rupees
            elif dec_int >= -2 and abs(val) < 1e7:
                multiplier = 0.01   # likely lakhs
            else:
                multiplier = 0.01   # safe default
            _scale_found = True
            break

    # ── Parse fact elements ────────────────────────────────────────────────────
    # For each (label, fy) pair, prefer the value from the LONGEST duration context.
    # This ensures Q4 annual figures (12-month) win over Q4 quarterly figures (3-month).
    # result structure: {label: {fy: (value, duration_months)}}
    _raw: dict[str, dict[str, tuple[float, int]]] = {}

    for elem in root.iter():
        local          = ns_re.sub("", elem.tag).lower()
        template_label = _XBRL_TAG_MAP.get(local)
        if not template_label:
            continue

        ctx_ref  = elem.get("contextRef", "")
        ctx_info = contexts.get(ctx_ref)
        if not ctx_info:
            continue
        fy, dur = ctx_info
        if year_set and fy not in year_set:
            continue

        raw_text = (elem.text or "").strip().replace(",", "")
        if not raw_text:
            continue

        try:
            raw_val   = float(raw_text)
            val_crore = round(raw_val * multiplier, 4)
        except (ValueError, OverflowError):
            continue

        label_data = _raw.setdefault(template_label, {})
        existing   = label_data.get(fy)
        # Keep the value from the longest-duration context (prefer annual over quarterly)
        if existing is None or dur > existing[1]:
            label_data[fy] = (val_crore, dur)

    # Flatten to {label: {fy: value}}
    result: dict[str, dict[str, float]] = {
        lbl: {fy: val for fy, (val, _dur) in yr_map.items()}
        for lbl, yr_map in _raw.items()
    }

    logger.info(f"xbrl_parser: extracted {len(result)} fields from XBRL")
    return result


def fetch_xbrl_from_url(url: str, years: Optional[list] = None) -> dict:
    """
    Download an XBRL file (XML or ZIP containing XML) from a URL and parse it.
    Returns {template_label: {fy_label: value_crores}}.
    """
    requests = _get_requests()

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=60)
        if resp.status_code != 200:
            return {}
        content = resp.content
    except Exception as e:
        logger.warning(f"xbrl_fetch: download failed for {url}: {e}")
        return {}

    # If ZIP, extract the main XBRL XML inside
    if url.lower().endswith(".zip") or content[:4] == b"PK\x03\x04":
        try:
            import zipfile
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                # Find the first .xml file (usually the main XBRL instance)
                xml_files = [n for n in zf.namelist()
                             if n.lower().endswith(".xml") and not n.lower().endswith("_def.xml")
                             and "_lab" not in n.lower() and "_pre" not in n.lower()]
                if not xml_files:
                    return {}
                with zf.open(xml_files[0]) as f:
                    content = f.read()
        except Exception as e:
            logger.warning(f"xbrl_fetch: ZIP extraction failed: {e}")
            return {}

    return _parse_xbrl_xml(content, years)


def _fetch_qresults_from_nse_announcements(symbol: str) -> list[dict]:
    """
    Fallback: fetch quarterly result PDFs from NSE corporate-announcements endpoint
    (same endpoint used for investor presentations — reliable cookie-wise).
    Filters announcements whose subject/desc contains "Financial Results".
    """
    session = _nse_session.get()
    all_results: list[dict] = []
    seen_urls: set[str] = set()

    _FIN_RESULT_RE = re.compile(
        r"financial\s+results?|quarterly\s+results?|unaudited\s+results?|audited\s+results?",
        re.I,
    )

    try:
        resp = session.get(
            f"{_NSE_API}/corporate-announcements",
            params={"index": "equities", "symbol": symbol.upper()},
            headers=_API_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        records = resp.json() if isinstance(resp.json(), list) else resp.json().get("data", [])
        if not isinstance(records, list):
            return []
    except Exception:
        return []

    for rec in records:
        subject = str(rec.get("subject") or rec.get("desc") or "")
        if not _FIN_RESULT_RE.search(subject):
            continue
        # Skip investor presentations that match financial keywords accidentally
        if re.search(r"investor\s+present|analyst", subject, re.I):
            continue

        an_body = rec.get("attchmntFile") or rec.get("attachment") or rec.get("attachmentURL") or ""
        if not an_body:
            continue
        link = an_body if an_body.startswith("http") else f"https://nsearchives.nseindia.com/{an_body.lstrip('/')}"
        if link in seen_urls:
            continue
        seen_urls.add(link)

        period = str(rec.get("bm_desc") or rec.get("period") or rec.get("dispDt") or "")
        year_label = _infer_fy_from_text(link + " " + period + " " + subject)
        quarter    = _infer_quarter(period) or _infer_quarter(subject) or _infer_quarter(link)
        filename   = link.split("/")[-1].split("?")[0] or f"{symbol}_qresult.pdf"

        all_results.append({
            "year":      year_label,
            "quarter":   quarter,
            "is_annual": quarter == "Q4",
            "filename":  filename,
            "url":       link,
            "xbrl_url":  None,
            "size_mb":   0,
            "source":    "NSE-announcements",
        })

    all_results.sort(key=lambda r: (r["year"] or "", r["quarter"] or ""), reverse=True)
    logger.info(f"pdf_fetcher: NSE-announcements fallback found {len(all_results)} quarterly results for {symbol}")
    return all_results


def get_available_quarterly_results(
    symbol: str,
    bse_code: Optional[str] = None,
) -> list[dict]:
    """
    Return all available quarterly financial result PDF links, newest-first.

    Sources tried (in order):
      1. NSE financial-results API  (pdfLink per quarter + xbrlLink)
      2. NSE corporate-announcements filtered for "Financial Results" (reliable fallback)
      3. BSE FinancialResult4 API   (PDFLINKURL per quarter)
      4. Company IR page            (generic scraper)
    """
    seen_urls: set[str] = set()
    all_results: list[dict] = []

    for rep in fetch_nse_quarterly_result_pdfs(symbol):
        if rep["url"] not in seen_urls:
            all_results.append(rep)
            seen_urls.add(rep["url"])

    # If primary NSE financial-results API returned nothing, try announcements fallback
    if not all_results:
        for rep in _fetch_qresults_from_nse_announcements(symbol):
            if rep["url"] not in seen_urls:
                all_results.append(rep)
                seen_urls.add(rep["url"])

    if bse_code:
        for rep in fetch_bse_quarterly_result_pdfs(bse_code):
            if rep["url"] not in seen_urls:
                all_results.append(rep)
                seen_urls.add(rep["url"])

    # Always also try company IR — most likely to have the latest quarter
    for rep in fetch_company_quarterly_result_pdfs(symbol, bse_code):
        if rep["url"] not in seen_urls:
            all_results.append(rep)
            seen_urls.add(rep["url"])

    all_results.sort(
        key=lambda r: (r["year"] or "", r["quarter"] or ""),
        reverse=True,
    )
    return all_results


# ── Convenience: get best available report for a given FY ────────────────────

def get_report_for_year(
    symbol: str,
    fy_label: str,           # e.g. "F2025"
    bse_code: Optional[str] = None,
) -> Optional[dict]:
    """
    Find the best available annual report download link for a given FY.
    Returns the report dict {year, filename, url, size_mb} or None.

    Order of preference:
      1. Company's own IR page  (direct PDFs — no zips, best quality)
      2. NSE
      3. BSE
      4. Screener
    """
    # ── 1. Company's own IR page first (direct PDFs, not zip-packaged) ──────
    try:
        company_reports = fetch_company_annual_reports(symbol, bse_code)
        for rep in company_reports:
            if rep["year"] == fy_label:
                logger.info(f"pdf_fetcher: found {fy_label} report on company IR: {rep['filename']}")
                return rep
    except Exception:
        pass

    # ── 2. NSE ───────────────────────────────────────────────────────────────
    nse_reports = fetch_nse_report_list(symbol)
    for rep in nse_reports:
        if rep["year"] == fy_label:
            logger.info(f"pdf_fetcher: found {fy_label} report on NSE: {rep['filename']}")
            return rep

    # ── 3. BSE ───────────────────────────────────────────────────────────────
    if bse_code:
        bse_reports = fetch_bse_report_list(bse_code)
        for rep in bse_reports:
            if rep["year"] == fy_label:
                logger.info(f"pdf_fetcher: found {fy_label} report on BSE: {rep['filename']}")
                return rep

    # ── 4. Screener ──────────────────────────────────────────────────────────
    screener_reports = fetch_screener_report_links(symbol)
    for rep in screener_reports:
        if rep["year"] == fy_label:
            logger.info(f"pdf_fetcher: found {fy_label} report on Screener: {rep['filename']}")
            return rep

    logger.info(f"pdf_fetcher: no report found for {symbol} {fy_label}")
    return None
