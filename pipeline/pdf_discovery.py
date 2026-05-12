"""
pipeline/pdf_discovery.py
─────────────────────────
Multi-tier annual-report discovery.

Layers (tried in order):
  1. NSE annual-reports API   (existing — web.pdf_fetcher.fetch_nse_report_list)
  2. BSE annual-reports API   (existing — web.pdf_fetcher.fetch_bse_report_list)
  3. Screener filings page    (existing — web.pdf_fetcher.fetch_screener_report_links)
  4. Company website IR page  (NEW — scrape /investors / /annual-reports etc.)
  5. DuckDuckGo PDF search    (NEW — "<name> annual report <year> filetype:pdf")

Each layer reports its own status (URLs found, error message, items kept) so the
caller can show "we tried these, here's what worked" instead of a silent failure.

Returns:
    DiscoveryResult
        .urls          : list[dict]  — combined, deduped by year, newest first
        .layer_status  : list[dict]  — per-layer { layer, urls_found, error }
"""

from __future__ import annotations

import logging
import re
import urllib.parse as _up
import urllib.request as _ur
from dataclasses import dataclass, field
from typing import Optional

from web.base import CompanyIdentifier

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    urls: list[dict] = field(default_factory=list)
    layer_status: list[dict] = field(default_factory=list)


# ── Curated company → website map for top NSE companies ─────────────────────
# Used as a HINT for the IR-page scraper. If a symbol isn't here we'll search.
# Generic — any Indian listed company can be added; not company-specific logic.
_COMPANY_SITES: dict[str, str] = {
    # Pharma
    "PPLPHARMA":   "https://www.piramalpharma.com",
    "PIRPHARMA":   "https://www.piramalpharma.com",
    "PIRAMAL":     "https://www.piramalpharma.com",
    "SUNPHARMA":   "https://sunpharma.com",
    "DRREDDY":     "https://www.drreddys.com",
    "CIPLA":       "https://www.cipla.com",
    "DIVISLAB":    "https://www.divislaboratories.com",
    "BIOCON":      "https://www.biocon.com",
    "LUPIN":       "https://www.lupin.com",
    "TORNTPHARM":  "https://www.torrentpharma.com",
    "AUROPHARMA":  "https://www.aurobindo.com",
    "ZYDUSLIFE":   "https://www.zyduslife.com",
    # Banks
    "HDFCBANK":    "https://www.hdfcbank.com",
    "ICICIBANK":   "https://www.icicibank.com",
    "SBIN":        "https://sbi.co.in",
    "AXISBANK":    "https://www.axisbank.com",
    "KOTAKBANK":   "https://www.kotak.com",
    "EQUITASBNK":  "https://www.equitasbank.com",
    # IT
    "TCS":         "https://www.tcs.com",
    "INFY":        "https://www.infosys.com",
    "WIPRO":       "https://www.wipro.com",
    "HCLTECH":     "https://www.hcltech.com",
    "TECHM":       "https://www.techmahindra.com",
    # FMCG / Consumer
    "HINDUNILVR":  "https://www.hul.co.in",
    "ITC":         "https://www.itcportal.com",
    "NESTLEIND":   "https://www.nestle.in",
    "BRITANNIA":   "https://www.britannia.co.in",
    # Manufacturing / Auto
    "TATASTEEL":   "https://www.tatasteel.com",
    "JSWSTEEL":    "https://www.jsw.in",
    "TATAMOTORS":  "https://www.tatamotors.com",
    "MARUTI":      "https://www.marutisuzuki.com",
    "M&M":         "https://www.mahindra.com",
    "RELIANCE":    "https://www.ril.com",
    # NBFCs
    "BAJFINANCE":  "https://www.bajajfinserv.in",
    "BAJAJFINSV":  "https://www.bajajfinserv.in",
    # Add more as we encounter them — heuristic search covers the long tail.
}


_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
_TIMEOUT = 15


# ── Playwright JS-rendering fallback ────────────────────────────────────────
# Many corporate IR pages are SPAs where the AR PDF links only appear after
# JavaScript runs (Piramal, ITC, several large pharma names). Static HTTP
# fetch sees a blank shell. Playwright renders the page; we then scrape the
# DOM for <a href="...pdf"> links + onclick handlers that point to PDFs.

def _playwright_render(url: str, timeout_ms: int = 15000,
                       wait_selector: Optional[str] = None) -> Optional[str]:
    """Render a URL with headless Chromium. Returns final HTML or None."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.debug("playwright not installed — JS fallback disabled "
                     "(run: python -m playwright install chromium --with-deps)")
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(user_agent=_USER_AGENT)
            page = ctx.new_page()
            page.goto(url, timeout=timeout_ms, wait_until="networkidle")
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=5000)
                except Exception:
                    pass
            # Try clicking common "Annual Reports" / "Quarterly Results" tabs
            # so the download links appear in the DOM. Generic — works on any
            # SPA tab pattern.
            for sel in ("text=/Annual.*Report/i", "text=/Quarterly.*Result/i",
                        "text=/Investor.*Presentation/i"):
                try:
                    el = page.locator(sel).first
                    if el.is_visible(timeout=1000):
                        el.click(timeout=1500)
                        page.wait_for_load_state("networkidle", timeout=3000)
                except Exception:
                    pass
            html = page.content()
            browser.close()
            return html
    except Exception as exc:
        logger.debug(f"playwright render failed for {url}: {exc}")
        return None


def _extract_pdfs_from_dom(html: str, base: str) -> list[tuple[str, str]]:
    """
    Parse rendered HTML for ANY pointer to a PDF: href, data-href, onclick,
    download attribute, etc. Returns list of (url, anchor_text) pairs.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    # Standard <a href="*.pdf">
    for m in _LINK_RE.finditer(html):
        href, text = m.group(1), (m.group(2) or "").strip()
        if not href:
            continue
        if href.lower().endswith(".pdf") or ".pdf?" in href.lower():
            full = _normalise_ar_url(href, base)
            if full not in seen:
                pairs.append((full, text)); seen.add(full)

    # data-href / data-url / data-file containing .pdf (common in SPA components)
    for m in re.finditer(
        r'data-(?:href|url|file|src)\s*=\s*"([^"]+\.pdf[^"]*)"',
        html, re.IGNORECASE,
    ):
        full = _normalise_ar_url(m.group(1), base)
        if full not in seen:
            pairs.append((full, "")); seen.add(full)

    # onclick="window.open('...pdf')" or onclick="...location='...pdf'"
    for m in re.finditer(
        r"""onclick\s*=\s*["'][^"']*?["'](?:[^"']+?\.pdf[^"']*)["']""",
        html, re.IGNORECASE,
    ):
        for u in re.findall(r"['\"]([^'\"]+\.pdf[^'\"]*)['\"]", m.group(0)):
            full = _normalise_ar_url(u, base)
            if full not in seen:
                pairs.append((full, "")); seen.add(full)

    # Bare PDF URLs anywhere in the rendered HTML (catches dynamically-built
    # download links that aren't in <a> tags)
    for m in re.finditer(r'(?:https?:)?//[^"\'\s<>]+\.pdf[^"\'\s<>]*', html):
        url = m.group(0)
        if url.startswith("//"):
            url = "https:" + url
        if url not in seen:
            pairs.append((url, "")); seen.add(url)

    return pairs


def _http_get(url: str, timeout: int = _TIMEOUT) -> Optional[str]:
    """Fetch a URL as text. Returns None on any failure."""
    try:
        req = _ur.Request(url, headers={"User-Agent": _USER_AGENT,
                                        "Accept-Language": "en-IN,en;q=0.9"})
        with _ur.urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type", "")
            data = resp.read()
            if "text/html" in ct or "text/plain" in ct or not ct:
                return data.decode("utf-8", errors="ignore")
            return None
    except Exception as exc:
        logger.debug(f"GET {url} failed: {exc}")
        return None


# ── Layer 4: Company website crawl (URL from data source, 1-level deep) ─────
# Strategy:
#   1. Get the company's website from yfinance / Screener / NSE
#      (no curated map — works for any company)
#   2. Fetch homepage HTML
#   3. Find every internal link whose URL or anchor text mentions
#      investor / annual / report / financial / results
#   4. Visit each candidate page (capped, deduped) and collect PDF links
#   5. For each PDF, infer the fiscal year and AR-vs-other status

# Keywords that mark a link as worth following
_FOLLOW_KEYWORDS = re.compile(
    r"(?:investor|annual|integrated\s*report|financial|result|filing|"
    r"shareholder|disclosure|sec-filing|earnings)",
    re.IGNORECASE,
)
_NON_INVESTOR_NEGATIVES = re.compile(
    r"(?:career|job|sustainability\b(?!.*report)|esg\b(?!.*report)|"
    r"contact|press-release|news|event|blog)",
    re.IGNORECASE,
)

_AR_LINK_RE = re.compile(
    r'<a[^>]+href="([^"]+\.pdf[^"]*)"[^>]*>\s*([^<]{0,200})</a>',
    re.IGNORECASE,
)
_AR_FILENAME_HINTS = re.compile(
    r"(?:annual[\s_\-]?report|integrated[\s_\-]?report|ar[\s_\-]?fy|"
    r"annual[\s_\-]?accounts)",
    re.IGNORECASE,
)
_YEAR_IN_TEXT_RE = re.compile(
    r"\b(?:F?Y?\s*)?(20\d{2})(?:[\s\-/]+(\d{2,4}))?\b",
)


def _site_root_for(company: CompanyIdentifier) -> Optional[str]:
    """
    Discover the company's website. Priority:
      1. Curated map (fast, ~30 top companies)
      2. yfinance .info['website'] — works for most NSE-listed names
      3. NSE company-info endpoint — fallback when yfinance lacks it
      4. DuckDuckGo search — last resort
    """
    sym = (company.nse_symbol or "").upper()

    # 1. Curated map (instant)
    if sym in _COMPANY_SITES:
        return _COMPANY_SITES[sym]

    # 2. yfinance — usually has the company URL
    if sym:
        try:
            import yfinance as _yf
            t = _yf.Ticker(f"{sym}.NS")
            url = (t.info or {}).get("website") or ""
            if url and url.startswith("http"):
                p = _up.urlparse(url)
                if p.scheme and p.netloc:
                    return f"{p.scheme}://{p.netloc}"
        except Exception as exc:
            logger.debug(f"_site_root_for: yfinance lookup failed: {exc}")

    # 3. NSE company-info endpoint
    if sym:
        try:
            import urllib.request as _ur, urllib.error as _ue
            import json as _json
            url = (f"https://www.nseindia.com/api/quote-equity"
                   f"?symbol={_up.quote(sym)}")
            req = _ur.Request(url, headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/json",
                "Referer": f"https://www.nseindia.com/get-quotes/equity?symbol={sym}",
            })
            with _ur.urlopen(req, timeout=8) as r:
                data = _json.loads(r.read().decode("utf-8", errors="ignore"))
            site = (data.get("info", {}) or {}).get("website") or ""
            if site and site.startswith("http"):
                p = _up.urlparse(site)
                if p.scheme and p.netloc:
                    return f"{p.scheme}://{p.netloc}"
        except Exception as exc:
            logger.debug(f"_site_root_for: NSE quote-equity failed: {exc}")

    # 4. DuckDuckGo last-resort
    name = company.name or sym
    if not name:
        return None
    q = _up.quote_plus(f"{name} official site")
    html = _http_get(f"https://duckduckgo.com/html/?q={q}")
    if not html:
        return None
    m = re.search(r'href="(https?://[^"]+)"', html)
    if m:
        candidate = m.group(1)
        if "duckduckgo.com" not in candidate and "/?u=" not in candidate:
            try:
                p = _up.urlparse(candidate)
                if p.scheme and p.netloc:
                    return f"{p.scheme}://{p.netloc}"
            except Exception:
                pass
    return None


def _normalise_ar_url(href: str, base: str) -> str:
    """Make absolute URL from a possibly-relative href."""
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return _up.urljoin(base, href)


def _infer_year(text: str, href: str) -> Optional[str]:
    """Try to infer F-year from link text or filename."""
    for hay in (text, href):
        if not hay:
            continue
        for m in _YEAR_IN_TEXT_RE.finditer(hay):
            try:
                y1 = int(m.group(1))
                y2 = m.group(2)
                if 2000 <= y1 <= 2050:
                    if y2:
                        # FY range "2023-24" → end year
                        y2i = int(y2) if len(y2) == 4 else 2000 + int(y2)
                        if 2000 <= y2i <= 2050 and y2i > y1:
                            return f"F{y2i}"
                    return f"F{y1}"
            except (ValueError, TypeError):
                pass
    return None


_LINK_RE = re.compile(
    r'<a\b[^>]*\bhref="([^"]+)"[^>]*>([^<]{0,200})</a>',
    re.IGNORECASE,
)


def _same_origin(url: str, root: str) -> bool:
    try:
        a, b = _up.urlparse(url), _up.urlparse(root)
        # Same registrable domain (allow subdomains)
        return (a.netloc.endswith(b.netloc) or b.netloc.endswith(a.netloc)
                or a.netloc == b.netloc)
    except Exception:
        return False


def discover_via_company_site(company: CompanyIdentifier,
                              years: list[str]) -> dict:
    """
    Crawl the company's homepage 1-level deep for AR PDF links — no fixed
    paths, no curated keywords beyond generic 'investor/annual/report/...'.
    Works for any site layout.
    """
    status = {"layer": "company_site", "urls_found": 0, "error": None,
              "site": None, "pages_visited": [], "items": []}

    site = _site_root_for(company)
    if not site:
        status["error"] = "could not determine company website"
        return status
    status["site"] = site

    year_set = set(years)
    found: dict[str, dict] = {}

    # Step 1: Fetch the homepage
    home_html = _http_get(site, timeout=12)
    status["pages_visited"].append({"url": site, "ok": bool(home_html)})
    if not home_html:
        status["error"] = f"homepage fetch failed for {site}"
        return status

    # Step 2: Find investor-relevant links from the homepage
    candidates: list[str] = []
    seen_links: set[str] = set()
    for m in _LINK_RE.finditer(home_html):
        href, text = m.group(1), (m.group(2) or "").strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        full = _normalise_ar_url(href, site)
        if not _same_origin(full, site):
            continue
        if full in seen_links:
            continue
        seen_links.add(full)
        haystack = f"{href} {text}".lower()
        if _NON_INVESTOR_NEGATIVES.search(haystack):
            continue
        # Direct PDF on homepage
        if full.lower().endswith(".pdf") or ".pdf?" in full.lower():
            if _AR_FILENAME_HINTS.search(haystack):
                yr = _infer_year(text, full)
                if yr and yr in year_set and yr not in found:
                    found[yr] = {
                        "year": yr, "url": full,
                        "filename": full.split("/")[-1].split("?")[0],
                        "size_mb": 0, "source": "company_site",
                    }
            continue
        # Otherwise queue as candidate IR-section page
        if _FOLLOW_KEYWORDS.search(haystack):
            candidates.append(full)

    # Step 3: Visit candidate pages (capped) and collect PDFs
    visited = set([site])
    for cand in candidates[:10]:
        if cand in visited:
            continue
        visited.add(cand)
        page_html = _http_get(cand, timeout=10)
        status["pages_visited"].append({"url": cand, "ok": bool(page_html)})
        if not page_html:
            continue
        for m in _LINK_RE.finditer(page_html):
            href, text = m.group(1), (m.group(2) or "").strip()
            full = _normalise_ar_url(href, cand)
            haystack = f"{href} {text}".lower()
            if not (full.lower().endswith(".pdf") or ".pdf?" in full.lower()):
                continue
            if not _AR_FILENAME_HINTS.search(haystack):
                continue
            yr = _infer_year(text, full)
            if yr and yr in year_set and yr not in found:
                found[yr] = {
                    "year": yr, "url": full,
                    "filename": full.split("/")[-1].split("?")[0],
                    "size_mb": 0, "source": "company_site",
                }
        if len(found) >= len(year_set):
            break

    # ── Step 4 (NEW): If static crawl found 0 PDFs, retry with Playwright ───
    # Many company IR pages (Piramal, ITC, large pharma) are JS-rendered SPAs
    # — the <a href=*.pdf> links only appear after JS executes.
    if not found:
        # Pick the most-promising candidates to render: explicit /financial-reports
        # / /investor-relations endpoints + the homepage
        js_candidates = [
            f"{site.rstrip('/')}/financial-reports",
            f"{site.rstrip('/')}/investor-relations",
            f"{site.rstrip('/')}/investors",
            site,
        ]
        for cand in js_candidates[:3]:
            html = _playwright_render(cand, timeout_ms=20000,
                                       wait_selector="a, button")
            status["pages_visited"].append({
                "url": cand, "ok": bool(html), "via": "playwright",
            })
            if not html:
                continue
            for pdf_url, text in _extract_pdfs_from_dom(html, cand):
                if not _AR_FILENAME_HINTS.search(f"{pdf_url} {text}"):
                    continue
                yr = _infer_year(text, pdf_url)
                if yr and yr in year_set and yr not in found:
                    found[yr] = {
                        "year": yr, "url": pdf_url,
                        "filename": pdf_url.split("/")[-1].split("?")[0],
                        "size_mb": 0, "source": "company_site_js",
                    }
            if len(found) >= len(year_set):
                break
        if found and not status.get("error"):
            status["error"] = None  # JS rendering succeeded — clear any prior

    status["urls_found"] = len(found)
    status["items"] = sorted(found.values(), key=lambda r: r["year"], reverse=True)
    return status


# ── Layer 5: DuckDuckGo PDF search ──────────────────────────────────────────

def discover_via_search(company: CompanyIdentifier, years: list[str]) -> dict:
    """
    DuckDuckGo HTML search for "<company> annual report <year> filetype:pdf".
    Catches PDFs hosted anywhere — IR pages, news sites, BSE archives.
    """
    status = {"layer": "search_engine", "urls_found": 0, "error": None,
              "queries_tried": [], "items": []}

    name = company.name or company.nse_symbol or ""
    if not name:
        status["error"] = "no company name to search"
        return status

    year_set = set(years)
    found: dict[str, dict] = {}

    # Try one search per requested year — improves precision
    for yr in years:
        try:
            yr_int = int(yr.lstrip("Ff"))
        except ValueError:
            continue
        # FY-range form: F2025 → "2024-25" (Indian fiscal-year labelling)
        fy_range = f"{yr_int-1}-{str(yr_int)[-2:]}"
        queries = [
            f'"{name}" "annual report" "{fy_range}" filetype:pdf',
            f'"{name}" annual report {yr_int} filetype:pdf',
        ]
        for q in queries:
            url = f"https://html.duckduckgo.com/html/?q={_up.quote_plus(q)}"
            html = _http_get(url, timeout=10)
            status["queries_tried"].append({"query": q, "ok": bool(html)})
            if not html:
                continue
            # DuckDuckGo wraps result URLs in /l/?uddg=<url>; pull the encoded
            # destination, OR find direct .pdf hrefs.
            for m in re.finditer(
                r'href="(?:/l/\?uddg=)?(https?[^"&]+\.pdf[^"&]*)"', html,
            ):
                pdf_url = _up.unquote(m.group(1))
                if yr in found:
                    break
                # Sanity: link must mention the company OR end with relevant words
                lower_url = pdf_url.lower()
                if not (
                    name.lower().split()[0] in lower_url
                    or "annual" in lower_url
                    or "ar" in lower_url
                ):
                    continue
                found[yr] = {
                    "year":     yr,
                    "url":      pdf_url,
                    "filename": pdf_url.split("/")[-1].split("?")[0],
                    "size_mb":  0,
                    "source":   "search_engine",
                }
                break
            if yr in found:
                break

    status["urls_found"] = len(found)
    status["items"] = sorted(found.values(), key=lambda r: r["year"], reverse=True)
    return status


# ── Direct NSE annual-reports fetch ─────────────────────────────────────────
# The legacy web.pdf_fetcher._SimpleSession bootstrap is 403'd by NSE (it
# fingerprints the multi-page cookie warmup as a bot). The probe path with
# plain headers gets HTTP 200 + a valid JSON body — so just use that.

def _parse_nse_year(rec: dict) -> Optional[str]:
    """Extract F-year from an NSE annual-reports JSON record."""
    # Try multiple field names NSE uses
    for fld in ("toDate", "endDate", "to_date", "fromDate", "startDate"):
        v = rec.get(fld) or ""
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(v))
        if m:
            yr, mn = int(m.group(1)), int(m.group(2))
            # Indian fiscal year ends Mar 31 — March-end = that calendar year
            if mn <= 6:
                return f"F{yr}"
            else:
                return f"F{yr+1}"
    return None


def _is_annual_report(filename: str, link_text: str = "") -> bool:
    haystack = f"{filename} {link_text}".lower()
    if not haystack.strip():
        return True
    if any(k in haystack for k in ("annual report", "annual_report",
                                    "ar_fy", "annualreport", "integrated report")):
        return True
    if "annual" in haystack and any(k in haystack
                                    for k in ("report", "accounts", "filing")):
        return True
    return False


def _fetch_nse_direct(symbol: str, years: list[str]) -> list[dict]:
    """Direct urllib call to NSE — the path proven to work by _probe_nse()."""
    import urllib.request as _ur, urllib.parse as _up, urllib.error as _ue
    import json as _json

    url = (f"https://www.nseindia.com/api/annual-reports"
           f"?index=equities&symbol={_up.quote(symbol.upper())}")
    req = _ur.Request(url, headers={
        "User-Agent": _USER_AGENT,
        "Accept": "application/json",
        "Accept-Language": "en-IN,en;q=0.9",
        "Referer": "https://www.nseindia.com/companies-listing/"
                   "corporate-filings-annual-reports",
    })
    try:
        with _ur.urlopen(req, timeout=10) as r:
            body = r.read().decode("utf-8", errors="ignore")
    except (_ue.HTTPError, _ue.URLError, Exception) as exc:
        logger.debug(f"_fetch_nse_direct: {exc}")
        return []

    try:
        data = _json.loads(body)
    except _json.JSONDecodeError:
        return []

    records = data if isinstance(data, list) else data.get("data", [])
    if not isinstance(records, list):
        return []

    year_set = set(years)
    out: list[dict] = []
    for rec in records:
        link = (rec.get("pdfLink") or rec.get("fileName")
                or rec.get("downloadLink") or rec.get("attachmentURL") or "")
        if not link:
            continue
        if link.startswith("/"):
            link = f"https://www.nseindia.com{link}"
        elif not link.startswith("http"):
            link = f"https://www.nseindia.com/{link}"
        filename = link.split("/")[-1].split("?")[0] or f"{symbol}_AR.pdf"
        if not _is_annual_report(filename):
            continue
        yr = _parse_nse_year(rec)
        if not yr or yr not in year_set:
            continue
        try:
            size_mb = round(int(rec.get("fileSize", 0) or 0) / 1_048_576, 1)
        except (TypeError, ValueError):
            size_mb = 0
        out.append({
            "year":     yr,
            "filename": filename,
            "url":      link,
            "size_mb":  size_mb,
        })

    # Dedup by year — newest first
    seen = set()
    deduped = []
    for it in sorted(out, key=lambda r: r["year"], reverse=True):
        if it["year"] not in seen:
            deduped.append(it); seen.add(it["year"])
    return deduped


# ── Diagnostic probes for legacy connectors ─────────────────────────────────
# These hit the same endpoints fetch_*_report_list uses, but capture the raw
# HTTP status / exception so the user can see WHY a layer returned 0 items.

def _probe_nse(symbol: str) -> str:
    try:
        import urllib.request as _ur, urllib.parse as _up, urllib.error as _ue
        url = (f"https://www.nseindia.com/api/annual-reports"
               f"?index=equities&symbol={_up.quote(symbol.upper())}")
        req = _ur.Request(url, headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/companies-listing/"
                       "corporate-filings-annual-reports",
        })
        with _ur.urlopen(req, timeout=8) as r:
            status = r.status
            body = r.read(2048)
            n = len(body)
        return f"HTTP {status} · {n}B body"
    except _ue.HTTPError as e:                       # noqa: F821
        return f"HTTP {e.code} {e.reason}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _probe_bse(bse_code: str) -> str:
    try:
        import urllib.request as _ur, urllib.error as _ue
        url = (f"https://api.bseindia.com/BseIndiaAPI/api/AnnualReport_New/w"
               f"?scripcode={bse_code}")
        req = _ur.Request(url, headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
            "Referer": "https://www.bseindia.com/",
        })
        with _ur.urlopen(req, timeout=8) as r:
            status = r.status
            body = r.read(2048)
        return f"HTTP {status} · {len(body)}B"
    except _ue.HTTPError as e:                        # noqa: F821
        return f"HTTP {e.code} {e.reason}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _probe_screener(symbol: str) -> str:
    try:
        import urllib.request as _ur, urllib.error as _ue
        url = f"https://www.screener.in/company/{symbol.upper()}/consolidated/"
        req = _ur.Request(url, headers={"User-Agent": _USER_AGENT})
        with _ur.urlopen(req, timeout=8) as r:
            status = r.status
            body = r.read(4096)
        return f"HTTP {status} · {len(body)}B"
    except _ue.HTTPError as e:                        # noqa: F821
        return f"HTTP {e.code} {e.reason}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


# ── Top-level orchestration ─────────────────────────────────────────────────

def discover_all(company: CompanyIdentifier, years: list[str],
                 enable_layers: Optional[list[str]] = None) -> DiscoveryResult:
    """
    Run all enabled discovery layers and merge results.
    `enable_layers` defaults to ["nse", "bse", "screener", "company_site",
                                 "search_engine"]; pass a subset to skip layers.
    """
    enable = enable_layers or ["nse", "bse", "screener",
                               "company_site", "search_engine"]
    res = DiscoveryResult()

    # Layers 1–3: existing connectors with introspecting wrappers.
    # The legacy fetchers swallow errors and return []; we hit the underlying
    # endpoints ourselves so we can report the ACTUAL HTTP status / exception.

    if "nse" in enable and company.nse_symbol:
        # Direct fetch first (legacy _SimpleSession bootstrap gets 403'd by NSE).
        # The probe path is what works — replicate it.
        items_direct = _fetch_nse_direct(company.nse_symbol, years)
        if items_direct:
            for it in items_direct:
                res.urls.append({**it, "source": "nse"})
            res.layer_status.append({
                "layer": "nse", "urls_found": len(items_direct), "error": None,
            })
        else:
            # Direct returned empty — fall back to legacy path & capture diag
            diag = _probe_nse(company.nse_symbol)
            try:
                from web.pdf_fetcher import fetch_nse_report_list
                items = fetch_nse_report_list(company.nse_symbol) or []
                n_kept = 0
                for it in items:
                    if it.get("year") in set(years):
                        res.urls.append({**it, "source": "nse"}); n_kept += 1
                res.layer_status.append({
                    "layer": "nse", "urls_found": n_kept,
                    "error": diag if not items else None,
                })
            except Exception as exc:
                res.layer_status.append({"layer": "nse", "urls_found": 0,
                                         "error": f"{exc} (probe: {diag})"})
    elif "nse" in enable:
        res.layer_status.append({"layer": "nse", "urls_found": 0,
                                 "error": "no NSE symbol"})

    if "bse" in enable and company.bse_code:
        diag = _probe_bse(company.bse_code)
        try:
            from web.pdf_fetcher import fetch_bse_report_list
            items = fetch_bse_report_list(company.bse_code) or []
            n_kept = 0
            for it in items:
                if it.get("year") in set(years):
                    res.urls.append({**it, "source": "bse"}); n_kept += 1
            res.layer_status.append({
                "layer": "bse", "urls_found": n_kept,
                "error": diag if not items else None,
            })
        except Exception as exc:
            res.layer_status.append({"layer": "bse", "urls_found": 0,
                                     "error": f"{exc} (probe: {diag})"})

    if "screener" in enable and company.nse_symbol:
        diag = _probe_screener(company.nse_symbol)
        try:
            from web.pdf_fetcher import fetch_screener_report_links
            items = fetch_screener_report_links(company.nse_symbol) or []
            n_kept = 0
            for it in items:
                if it.get("year") in set(years):
                    res.urls.append({**it, "source": "screener"}); n_kept += 1
            res.layer_status.append({
                "layer": "screener", "urls_found": n_kept,
                "error": diag if not items else None,
            })
        except Exception as exc:
            res.layer_status.append({"layer": "screener", "urls_found": 0,
                                     "error": f"{exc} (probe: {diag})"})

    # Layer 4: company website
    if "company_site" in enable:
        st = discover_via_company_site(company, years)
        for it in st.get("items", []):
            res.urls.append(it)
        res.layer_status.append({k: v for k, v in st.items() if k != "items"})

    # Layer 5: search engine
    if "search_engine" in enable:
        st = discover_via_search(company, years)
        for it in st.get("items", []):
            res.urls.append(it)
        res.layer_status.append({k: v for k, v in st.items() if k != "items"})

    # Dedup by year — earlier layers (more authoritative) win
    seen_years: set[str] = set()
    deduped: list[dict] = []
    for it in res.urls:
        yr = it.get("year")
        if yr and yr not in seen_years:
            deduped.append(it)
            seen_years.add(yr)
    deduped.sort(key=lambda r: r.get("year") or "", reverse=True)
    res.urls = deduped

    return res


__all__ = ["discover_all", "DiscoveryResult",
           "discover_via_company_site", "discover_via_search"]
