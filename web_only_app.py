"""
Full-Auto Financial Template Populator
========================================
Enter NSE symbol + upload template — the app handles everything:

  1. Reads template  — exact field labels + year columns needed
  2. Fetches web data — Screener.in + yfinance (structured, fast)
  3. Auto-downloads annual report PDF from NSE/BSE (no manual upload)
  4. Extracts tables from the PDF using pdfplumber (granular line items)
  5. LLM reconciles web data + PDF data → maps to template labels
  6. Writes to template — download the populated Excel

Web sources cover 70–80 % of fields cleanly.
PDF extraction fills the rest (granular notes, sub-line items, etc.)
Together they get you very close to 100 % coverage.

Run:
    streamlit run web_only_app.py
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import subprocess
import time
import traceback
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))


# ── Auto-install dependencies ──────────────────────────────────────────────────
def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *pkgs])

try:
    import streamlit as st
except ImportError:
    _pip("streamlit>=1.35.0"); import streamlit as st

try:
    import openpyxl
except ImportError:
    _pip("openpyxl>=3.1.2"); import openpyxl

try:
    import pandas as pd
except ImportError:
    _pip("pandas>=2.2.0"); import pandas as pd

try:
    import requests as _requests_mod
except ImportError:
    _pip("requests>=2.31.0"); import requests as _requests_mod

try:
    from bs4 import BeautifulSoup as _BS4
except ImportError:
    _pip("beautifulsoup4>=4.12.0"); from bs4 import BeautifulSoup as _BS4

try:
    import yfinance as yf
except ImportError:
    _pip("yfinance>=0.2.38"); import yfinance as yf

try:
    from openai import OpenAI
except ImportError:
    _pip("openai>=1.30.0"); from openai import OpenAI


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto Financial Extractor",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 Full-Auto Financial Template Populator")
st.caption(
    "Enter NSE symbol → auto-downloads annual report PDF + scrapes web → "
    "LLM maps everything to your template."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Template structure reader
# ═══════════════════════════════════════════════════════════════════════════════

def read_template_structure(template_bytes: bytes) -> dict:
    from utils.helpers import normalise_year, clean_label
    wb  = openpyxl.load_workbook(io.BytesIO(template_bytes), data_only=True)
    out = {"sheets": {}}

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        year_cols:  dict[int, str] = {}
        header_row: Optional[int]  = None

        for ri in range(1, 26):
            for ci in range(1, ws.max_column + 1):
                v = ws.cell(row=ri, column=ci).value
                if v is None:
                    continue
                yr = normalise_year(str(v).strip())
                if yr:
                    year_cols[ci] = yr
                    header_row = ri
            if year_cols:
                break

        if not year_cols or header_row is None:
            continue

        labels: list[str] = []
        seen:   set[str]  = set()
        for ri in range(header_row + 1, ws.max_row + 1):
            v = ws.cell(row=ri, column=1).value
            if v is None:
                continue
            lbl = str(v).strip()
            if not lbl or lbl.startswith("#"):
                continue
            c = clean_label(lbl)
            if c and c not in seen:
                seen.add(c)
                labels.append(lbl)

        if labels:
            out["sheets"][sheet_name] = {
                "years":  sorted(set(year_cols.values())),
                "labels": labels,
            }

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Web data: Screener.in (raw, no pre-mapping)
# ═══════════════════════════════════════════════════════════════════════════════

_SCREENER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.screener.in/",
}
_SCREENER_SECTIONS = {
    "profit-loss":   "P&L",
    "balance-sheet": "Balance Sheet",
    "cash-flow":     "Cash Flow",
}


def _scr_year(text: str) -> Optional[str]:
    m = re.search(r"(\d{4})", text.strip())
    return f"F{m.group(1)}" if m else None


def _scr_val(text: str) -> Optional[float]:
    t = text.strip().replace(",", "").replace(" ", "").replace("%", "")
    if not t or t in ("-", "—", "N/A"):
        return None
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    try:
        v = float(t)
        return -v if neg else v
    except ValueError:
        return None


def fetch_screener_raw(symbol: str, consolidated: bool = True) -> dict:
    suffix = "/consolidated/" if consolidated else "/"
    url    = f"https://www.screener.in/company/{symbol.upper()}{suffix}"
    try:
        r = _requests_mod.get(url, headers=_SCREENER_HEADERS, timeout=25)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
    except Exception:
        return {}

    soup   = _BS4(r.text, "html.parser")
    result = {}

    for sec_id, sec_name in _SCREENER_SECTIONS.items():
        el = soup.find("section", {"id": sec_id}) or soup.find(id=sec_id)
        if not el:
            continue
        tbl = el.find("table")
        if not tbl:
            continue
        thead = tbl.find("thead")
        if not thead:
            continue
        ths   = thead.find_all("th")
        years = [_scr_year(th.get_text(strip=True)) for th in ths[1:]]

        tbody = tbl.find("tbody")
        if not tbody:
            continue

        data: dict[str, dict[str, float]] = {}
        for tr in tbody.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue
            lbl = re.sub(r"\s*[+\-]\s*$", "", cells[0].get_text(separator=" ", strip=True)).strip()
            if not lbl:
                continue
            row: dict[str, float] = {}
            for ci, yr in enumerate(years):
                if yr is None:
                    continue
                v = _scr_val(cells[ci + 1].get_text(strip=True)) if ci + 1 < len(cells) else None
                if v is not None:
                    row[yr] = v
            if row:
                data[lbl] = row

        if data:
            result[sec_name] = {
                "years": sorted({y for y in years if y}),
                "data":  data,
            }

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Web data: yfinance
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_yfinance_raw(symbol: str, years: list[str]) -> dict:
    try:
        tk = yf.Ticker(f"{symbol.upper()}.NS")
    except Exception:
        return {}

    year_set = set(years)
    _CRORE   = 1e7
    result   = {}

    for sec_name, attr in [
        ("Income Statement", "income_stmt"),
        ("Balance Sheet",    "balance_sheet"),
        ("Cash Flow",        "cashflow"),
    ]:
        try:
            df = getattr(tk, attr)
            if df is None or df.empty:
                continue
            sec: dict[str, dict[str, float]] = {}
            for col in df.columns:
                try:
                    yr_label = f"F{col.year if col.month <= 3 else col.year + 1}"
                except AttributeError:
                    continue
                if yr_label not in year_set:
                    continue
                vals = {}
                for metric in df.index:
                    try:
                        v = float(df.loc[metric, col])
                        if v == v:
                            vals[str(metric)] = round(v / _CRORE, 4)
                    except Exception:
                        pass
                if vals:
                    sec[yr_label] = vals
            if sec:
                result[sec_name] = sec
        except Exception:
            pass

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PDF fetcher: auto-download annual report from NSE / BSE
# ═══════════════════════════════════════════════════════════════════════════════

from web.pdf_fetcher import (
    fetch_nse_report_list,
    fetch_bse_report_list,
    fetch_screener_report_links,
    download_pdf,
)


def get_available_reports(symbol: str, bse_code: Optional[str] = None) -> list[dict]:
    """Return all available annual report records from NSE + BSE + Screener."""
    seen_urls: set[str] = set()
    all_reports: list[dict] = []

    for rep in fetch_nse_report_list(symbol):
        if rep["url"] not in seen_urls:
            rep["source"] = "NSE"
            all_reports.append(rep)
            seen_urls.add(rep["url"])

    if bse_code:
        for rep in fetch_bse_report_list(bse_code):
            if rep["url"] not in seen_urls:
                rep["source"] = "BSE"
                all_reports.append(rep)
                seen_urls.add(rep["url"])

    for rep in fetch_screener_report_links(symbol):
        if rep["url"] not in seen_urls:
            rep["source"] = "Screener"
            all_reports.append(rep)
            seen_urls.add(rep["url"])

    all_reports.sort(key=lambda r: r.get("year") or "", reverse=True)
    return all_reports


# ═══════════════════════════════════════════════════════════════════════════════
# PDF text extraction — raw text from financial pages for LLM context
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf_text_for_llm(pdf_bytes: bytes, max_chars_per_page: int = 3500) -> str:
    """
    Extract cleaned plain text from financial statement pages of the PDF.

    Unlike table extraction (which tries to parse structure), this simply
    reads text line-by-line.  The LLM then does the semantic understanding.

    Returns a single string ready to be pasted into an LLM prompt.
    Max total output: ~50 000 characters (keeps token cost reasonable).
    """
    try:
        import pdfplumber
    except ImportError:
        return ""

    from extractor.pdf_parser import parse_pdf
    from extractor.table_extractor import _is_non_financial_page, _classify_text
    from utils.helpers import clean_for_llm

    try:
        doc = parse_pdf(io.BytesIO(pdf_bytes))
    except Exception:
        return ""

    # Collect financial page numbers
    financial_pages: set[int] = set()
    for sec in doc.sections:
        financial_pages.update(sec.page_numbers)
    for pg_num, pg_text in doc.raw_pages.items():
        if _classify_text(pg_text):
            financial_pages.add(pg_num)

    if not financial_pages:
        return ""

    MAX_TOTAL = 50_000
    parts: list[str] = []
    total_chars = 0

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            n_pages = len(pdf.pages)
            for pg_num in sorted(financial_pages):
                if pg_num > n_pages:
                    continue
                if total_chars >= MAX_TOTAL:
                    break
                raw_text = pdf.pages[pg_num - 1].extract_text() or ""
                if not raw_text.strip():
                    continue
                if _is_non_financial_page(raw_text):
                    continue
                cleaned = clean_for_llm(raw_text)[:max_chars_per_page]
                parts.append(f"\n=== Page {pg_num} ===\n{cleaned}")
                total_chars += len(cleaned)
    except Exception:
        pass

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# PDF table extraction (pdfplumber)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf_data(pdf_bytes: bytes, years: list[str], statement_type: str = "consolidated") -> dict:
    """
    Run pdfplumber table extraction on a PDF and return structured data.

    Returns:
        {
          "P&L":            {"F2025": {"Revenue from Operations": 45000, ...}, ...},
          "Balance Sheet":  {...},
          "Cash Flow":      {...},
        }
    """
    from extractor.pdf_parser import parse_pdf
    from extractor.table_extractor import extract_tables_from_pdf

    try:
        doc = parse_pdf(io.BytesIO(pdf_bytes))
        doc._pdf_bytes = pdf_bytes
    except Exception as e:
        return {"error": str(e)}

    try:
        raw_results = extract_tables_from_pdf(
            pdf_source=pdf_bytes,
            parsed_doc=doc,
            statement_type=statement_type,
        )
    except Exception as e:
        return {"error": str(e)}

    year_set = set(years)
    _SECTION_MAP = {
        "Annual P&L":           "P&L",
        "Annual Balance Sheet":  "Balance Sheet",
        "Annual Cash Flow":      "Cash Flow",
    }

    combined: dict[str, dict[str, dict[str, float]]] = {}

    for result in raw_results:
        if result.year not in year_set:
            continue
        section = _SECTION_MAP.get(result.section, result.section)
        combined.setdefault(section, {}).setdefault(result.year, {})
        for label, value in result.data.items():
            if label not in combined[section][result.year]:
                combined[section][result.year][label] = value

    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# LLM: maps template labels → merged web + PDF data
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are a financial data extraction expert for Indian company annual reports.

You receive:
  1. EXACT row labels from an Excel template that need to be filled
  2. Structured data from Screener.in, Yahoo Finance, and PDF table parsing
  3. RAW PDF TEXT from the actual financial statement pages of the annual report

Your job: for every template label, find the correct value for each year.

Data source priority (use highest available):
  RAW PDF TEXT  >  PDF TABLE EXTRACTION  >  SCREENER.IN  >  YAHOO FINANCE

Semantic matching rules — these labels mean the same thing:
  "Net Profit" = "Profit After Tax" = "PAT" = "Net Income" = "Profit for the year"
  "Sales" = "Revenue from Operations" = "Net Sales" = "Total Revenue" = "Total Income"
  "Interest Income" = "Interest Earned" = "Net Interest Income" (for banks)
  "Finance Costs" = "Interest Expense" = "Interest Paid" = "Finance cost"
  "Depreciation" = "Depreciation and Amortisation" = "D&A"
  "Operating Profit" = "EBIT" = "EBITDA" (context-dependent)
  "EPS" = "Earnings per share" = "Basic EPS"

Reading RAW PDF TEXT tables:
  Rows usually appear as:  Label | [Note ref 1-200, SKIP] | Value_Year1 | Value_Year2
  The first year column is the most recent year; second is prior year.
  Values in brackets like (123.45) are NEGATIVE → use -123.45.
  Convert units: if text says "₹ in Crores" divide by 1; "₹ in Lakhs" divide by 100;
  "₹ in Thousands" or "000's" divide by 10,000.
   - "OPM %" or "Tax %" = ratio rows, skip (set null)
3. If multiple sources have a value for the same field/year, prefer:
   PDF extraction > Screener > yfinance (PDF is most granular)
4. If no match exists, set the value to null.
5. Values are in INR Crores.
6. Return ONLY valid JSON — no markdown, no explanations.

Output format:
{
  "Template Label 1": {"F2025": 1234.5, "F2024": 1100.0},
  "Template Label 2": {"F2025": null,   "F2024": 567.8},
  ...
}
"""


def llm_map_fields(
    sheet_name: str,
    template_labels: list[str],
    years: list[str],
    screener_raw: dict,
    yfinance_raw: dict,
    pdf_data: dict,
    client: OpenAI,
    model: str = "deepseek-chat",
    pdf_text: str = "",
) -> dict[str, dict[str, Optional[float]]]:

    sl = sheet_name.lower()
    if "p&l" in sl or "profit" in sl or "income" in sl:
        s_secs  = ["P&L"]
        yf_secs = ["Income Statement"]
        p_secs  = ["P&L"]
    elif "balance" in sl or "bs" in sl:
        s_secs  = ["Balance Sheet"]
        yf_secs = ["Balance Sheet"]
        p_secs  = ["Balance Sheet"]
    elif "cash" in sl or "cf" in sl:
        s_secs  = ["Cash Flow"]
        yf_secs = ["Cash Flow"]
        p_secs  = ["Cash Flow"]
    else:
        s_secs  = list(screener_raw.keys())
        yf_secs = list(yfinance_raw.keys())
        p_secs  = list(pdf_data.keys())

    lines = [
        f"EXCEL SHEET: '{sheet_name}'",
        f"YEARS NEEDED: {years}",
        "",
        "TEMPLATE LABELS (fill EVERY one of these — use null only if truly not found):",
        json.dumps(template_labels, indent=2),
        "",
        "=== STRUCTURED DATA (pre-parsed, INR Crores) ===",
        "",
    ]

    # PDF data — highest priority structured source
    for sec in p_secs:
        if sec not in pdf_data or not pdf_data[sec]:
            continue
        filtered = {yr: pdf_data[sec][yr] for yr in years if yr in pdf_data[sec]}
        if filtered:
            lines.append(f"PDF TABLE EXTRACTION ({sec}):")
            lines.append(json.dumps(filtered, indent=2))
            lines.append("")

    # Screener
    for sec in s_secs:
        if sec not in screener_raw:
            continue
        d = {
            lbl: {yr: v for yr, v in yr_vals.items() if yr in years}
            for lbl, yr_vals in screener_raw[sec].get("data", {}).items()
        }
        d = {k: v for k, v in d.items() if v}
        if d:
            lines.append(f"SCREENER.IN ({sec}):")
            lines.append(json.dumps(d, indent=2))
            lines.append("")

    # yfinance
    for sec in yf_secs:
        if sec not in yfinance_raw:
            continue
        d = {yr: yfinance_raw[sec][yr] for yr in years if yr in yfinance_raw[sec]}
        if d:
            lines.append(f"YAHOO FINANCE ({sec}):")
            lines.append(json.dumps(d, indent=2))
            lines.append("")

    # Raw PDF text — highest priority, lets LLM find anything the parser missed
    if pdf_text:
        if "p&l" in sl or "profit" in sl or "income" in sl:
            kw = ["profit", "loss", "income", "revenue", "interest earned",
                  "operating", "provision", "tax", "earnings per share"]
        elif "balance" in sl or "bs" in sl:
            kw = ["balance sheet", "assets", "liabilities", "capital", "reserves",
                  "deposits", "borrowings", "investments", "advances"]
        elif "cash" in sl or "cf" in sl:
            kw = ["cash flow", "operating activities", "investing", "financing",
                  "net cash", "profit before tax"]
        else:
            kw = []

        relevant_text = pdf_text
        if kw:
            kept = []
            for chunk in pdf_text.split("\n=== Page "):
                cl = chunk.lower()
                if any(k in cl for k in kw):
                    kept.append(chunk)
            if kept:
                relevant_text = "\n=== Page ".join(kept)

        MAX_PDF_TEXT = 14_000
        if len(relevant_text) > MAX_PDF_TEXT:
            relevant_text = relevant_text[:MAX_PDF_TEXT] + "\n[...truncated...]"

        lines.append("=== RAW PDF TEXT (financial statement pages — highest priority) ===")
        lines.append("Use this to find ANY field not covered by structured data above.")
        lines.append("Row format: Label | [note ref 1-200, skip] | value_recent_year | value_prior_year")
        lines.append("")
        lines.append(relevant_text)
        lines.append("")

    user_prompt = "\n".join(lines)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.M)
            raw = re.sub(r"\n?```$", "", raw, flags=re.M).strip()

            parsed = json.loads(raw)
            result: dict[str, dict[str, Optional[float]]] = {}
            for k, v in parsed.items():
                if isinstance(v, dict):
                    result[k] = {yr: (float(val) if val is not None else None) for yr, val in v.items()}
                else:
                    result[k] = {yr: None for yr in years}
            return result

        except json.JSONDecodeError:
            if attempt == 2:
                return {}
            time.sleep(1)
        except Exception:
            if attempt == 2:
                return {}
            time.sleep(2)

    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Excel writer
# ═══════════════════════════════════════════════════════════════════════════════

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if (sa and sb) else 0.0


def write_to_template(template_bytes: bytes, filled: dict, overwrite: bool = False) -> tuple[bytes, dict]:
    from utils.helpers import normalise_year, clean_label

    wb    = openpyxl.load_workbook(io.BytesIO(template_bytes))
    stats = {"written": 0, "skipped_filled": 0, "skipped_formula": 0, "not_found": 0, "audit": []}

    for sheet_name, label_year_vals in filled.items():
        if sheet_name not in wb.sheetnames:
            continue
        ws = wb[sheet_name]

        year_cols:  dict[str, int] = {}
        header_row: Optional[int]  = None
        for ri in range(1, 26):
            for ci in range(1, ws.max_column + 1):
                v = ws.cell(row=ri, column=ci).value
                if v is None:
                    continue
                yr = normalise_year(str(v).strip())
                if yr:
                    year_cols[yr] = ci
                    header_row = ri
            if year_cols:
                break
        if not year_cols or header_row is None:
            continue

        label_rows: dict[str, int] = {}
        for ri in range(header_row + 1, ws.max_row + 1):
            v = ws.cell(row=ri, column=1).value
            if v is None:
                continue
            c = clean_label(str(v).strip())
            if c:
                label_rows[c] = ri

        for tmpl_label, yr_vals in label_year_vals.items():
            ct = clean_label(tmpl_label)
            row = label_rows.get(ct)
            if row is None:
                for lbl, ri in label_rows.items():
                    if ct and lbl and (ct in lbl or lbl in ct or _jaccard(ct, lbl) >= 0.6):
                        row = ri
                        break
            if row is None:
                stats["not_found"] += 1
                continue

            for yr, val in yr_vals.items():
                if val is None:
                    continue
                col = year_cols.get(yr)
                if col is None:
                    continue
                cell = ws.cell(row=row, column=col)
                if cell.data_type == "f" or (isinstance(cell.value, str) and cell.value.startswith("=")):
                    stats["skipped_formula"] += 1
                    continue
                if not overwrite and cell.value not in (None, "", 0):
                    stats["skipped_filled"] += 1
                    continue
                cell.value = round(val, 2)
                stats["written"] += 1
                stats["audit"].append({"Sheet": sheet_name, "Label": tmpl_label, "Year": yr, "Value (Cr)": round(val, 2)})

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue(), stats


# ═══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("DeepSeek API Key", type="password")
    api_model = st.selectbox("LLM Model", ["deepseek-chat", "deepseek-reasoner"])

    st.markdown("---")
    st.header("📥 Inputs")
    template_file = st.file_uploader("Excel Template (.xlsx)", type=["xlsx"])
    nse_symbol    = st.text_input("NSE Symbol", placeholder="e.g. HDFCBANK, RELIANCE").strip().upper()
    bse_code      = st.text_input("BSE Code (optional)", placeholder="e.g. 500180").strip()

    stmt_type  = st.radio("Statement type", ["Consolidated", "Standalone"], horizontal=True)
    overwrite  = st.checkbox("Overwrite existing cell values", value=False)

    st.markdown("---")
    st.header("📄 Annual Report PDF")
    pdf_mode = st.radio(
        "PDF source",
        ["Auto-download from NSE/BSE", "Upload manually", "Skip PDF (web only)"],
        help="Auto-download fetches the official annual report for you.",
    )
    manual_pdf = None
    if pdf_mode == "Upload manually":
        manual_pdf = st.file_uploader("Upload Annual Report PDF", type=["pdf"])

    run_btn = st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)


# ── Main pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not template_file:
        st.error("Please upload an Excel template.")
        st.stop()
    if not nse_symbol:
        st.error("Please enter an NSE symbol.")
        st.stop()
    if not api_key:
        st.error("Please enter your DeepSeek API key.")
        st.stop()

    client     = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    consolidated = stmt_type == "Consolidated"
    progress   = st.progress(0, text="Starting…")
    status     = st.empty()

    try:
        # ── 1. Read template ──────────────────────────────────────────────────
        status.info("📋 Step 1 / 5 — Reading template…")
        progress.progress(4, text="Reading template…")

        template_file.seek(0)
        template_bytes = template_file.read()
        needs          = read_template_structure(template_bytes)
        sheets_data    = needs["sheets"]

        if not sheets_data:
            st.error("No year columns detected. Ensure columns are labelled F2025, F2024, etc.")
            st.stop()

        all_years = sorted({yr for sd in sheets_data.values() for yr in sd["years"]})
        st.success(f"✅ Template: {len(sheets_data)} sheet(s) · Years: {', '.join(all_years)}")

        with st.expander("📊 Template fields detected"):
            for sn, sd in sheets_data.items():
                st.markdown(f"**{sn}** — {len(sd['labels'])} labels · {sd['years']}")
                st.caption(", ".join(sd["labels"][:25]) + ("…" if len(sd["labels"]) > 25 else ""))

        # ── 2. Fetch web data (Screener + yfinance) ───────────────────────────
        status.info(f"🌐 Step 2 / 5 — Fetching web data for **{nse_symbol}**…")
        progress.progress(12, text="Fetching Screener.in…")

        screener_raw = fetch_screener_raw(nse_symbol, consolidated=consolidated)
        if not screener_raw and consolidated:
            screener_raw = fetch_screener_raw(nse_symbol, consolidated=False)

        progress.progress(22, text="Fetching yfinance…")
        yfinance_raw = fetch_yfinance_raw(nse_symbol, all_years)

        s_count = sum(len(s["data"]) for s in screener_raw.values() if isinstance(s, dict) and "data" in s)
        y_count = sum(len(v) for s in yfinance_raw.values() for v in (s.values() if isinstance(s, dict) else []))
        c1, c2 = st.columns(2)
        c1.metric("Screener fields", s_count)
        c2.metric("yfinance metrics", y_count)

        with st.expander("🔍 Raw Screener data"):
            for sn, sd in screener_raw.items():
                st.markdown(f"**{sn}**")
                rows = [{"Label": lbl, **{yr: v for yr, v in yv.items() if yr in all_years}}
                        for lbl, yv in sd.get("data", {}).items()]
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── 3. Get / download PDF ──────────────────────────────────────────────
        pdf_bytes: Optional[bytes] = None
        pdf_label = ""

        if pdf_mode == "Upload manually" and manual_pdf:
            manual_pdf.seek(0)
            pdf_bytes = manual_pdf.read()
            pdf_label = manual_pdf.name

        elif pdf_mode == "Auto-download from NSE/BSE":
            status.info("📄 Step 3 / 5 — Finding annual reports on NSE/BSE…")
            progress.progress(30, text="Fetching report list…")

            reports = get_available_reports(nse_symbol, bse_code or None)

            if not reports:
                st.warning("⚠️ No annual report links found on NSE/BSE. Continuing with web data only.")
            else:
                # Filter to relevant years
                relevant = [r for r in reports if r.get("year") in all_years]
                if not relevant:
                    relevant = reports[:3]   # take most recent if no year match

                st.success(f"📄 Found {len(reports)} annual reports on {reports[0].get('source', 'NSE/BSE')}")

                # Let user pick which year(s) to download
                report_options = {
                    f"{r.get('year', '?')} — {r.get('filename', 'report.pdf')} "
                    f"({r.get('size_mb', '?')} MB) [{r.get('source', '')}]": r
                    for r in relevant
                }
                selected_label = st.selectbox(
                    "Select annual report to use:",
                    list(report_options.keys()),
                )
                selected_report = report_options[selected_label]

                progress.progress(35, text=f"Downloading {selected_report['filename']}…")
                dl_status = st.empty()
                dl_status.info(f"⬇️ Downloading **{selected_report['filename']}**…")

                def _progress_cb(downloaded: int, total: int):
                    if total:
                        pct = min(int(downloaded / total * 100), 99)
                        dl_status.info(f"⬇️ Downloading… {pct}% ({downloaded // 1024:,} KB)")

                pdf_bytes = download_pdf(selected_report["url"], progress_cb=_progress_cb)
                if pdf_bytes:
                    pdf_label = selected_report["filename"]
                    size_mb   = len(pdf_bytes) / 1_048_576
                    dl_status.success(f"✅ Downloaded **{pdf_label}** ({size_mb:.1f} MB)")
                else:
                    dl_status.warning("⚠️ PDF download failed — continuing with web data only.")

        else:
            status.info("📄 Step 3 / 5 — Skipping PDF (web-only mode).")
            progress.progress(35)

        # ── 4. Extract PDF data (tables + raw text) ───────────────────────────
        pdf_data: dict = {}
        pdf_text: str  = ""
        pdf_field_count = 0

        if pdf_bytes:
            status.info(f"🔍 Step 4 / 5 — Extracting data from **{pdf_label}**…")
            progress.progress(42, text="Extracting PDF tables…")

            # 4a. Structured table extraction (pdfplumber)
            with st.spinner("Running pdfplumber table extraction…"):
                pdf_data = extract_pdf_data(
                    pdf_bytes,
                    years=all_years,
                    statement_type="consolidated" if consolidated else "standalone",
                )

            if "error" in pdf_data:
                st.warning(f"⚠️ PDF table extraction error: {pdf_data['error']}")
                pdf_data = {}
            else:
                pdf_field_count = sum(
                    len(yr_data)
                    for sec_data in pdf_data.values()
                    for yr_data in sec_data.values()
                )

            # 4b. Raw text extraction (sent to LLM for comprehensive field coverage)
            progress.progress(52, text="Extracting PDF text for LLM…")
            with st.spinner("Extracting raw text from financial pages…"):
                pdf_text = extract_pdf_text_for_llm(pdf_bytes)

            c1, c2 = st.columns(2)
            c1.metric("PDF structured values", pdf_field_count)
            c2.metric("PDF text chars", f"{len(pdf_text):,}")

            with st.expander("📑 PDF structured data (table extraction)"):
                for sec, yr_map in pdf_data.items():
                    st.markdown(f"**{sec}**")
                    rows = []
                    for yr, fvals in yr_map.items():
                        for lbl, val in fvals.items():
                            rows.append({"Year": yr, "Label": lbl, "Value (Cr)": val})
                    if rows:
                        st.dataframe(pd.DataFrame(rows).head(200), use_container_width=True, hide_index=True)

            with st.expander("📄 Raw PDF text (sent to LLM)"):
                st.text(pdf_text[:3000] + ("…" if len(pdf_text) > 3000 else ""))
        else:
            status.info("📄 Step 4 / 5 — No PDF available, using web data only.")
            progress.progress(60)

        # ── 5. LLM maps everything → template ─────────────────────────────────
        status.info("🧠 Step 5 / 5 — LLM matching all data to template labels…")
        progress.progress(65, text="LLM field mapping…")

        filled: dict[str, dict] = {}
        total_matched = 0

        for i, (sheet_name, sheet_data) in enumerate(sheets_data.items()):
            pct = 65 + int((i / len(sheets_data)) * 22)
            progress.progress(pct, text=f"LLM: mapping '{sheet_name}'…")

            sheet_filled = llm_map_fields(
                sheet_name=sheet_name,
                template_labels=sheet_data["labels"],
                years=sheet_data["years"],
                screener_raw=screener_raw,
                yfinance_raw=yfinance_raw,
                pdf_data=pdf_data,
                client=client,
                model=api_model,
                pdf_text=pdf_text,
            )

            n = sum(1 for yv in sheet_filled.values() for v in yv.values() if v is not None)
            total_matched += n
            filled[sheet_name] = sheet_filled
            st.info(f"  **{sheet_name}**: {n} values matched")

        with st.expander("🧠 LLM mapping preview"):
            for sn, sf in filled.items():
                st.markdown(f"**{sn}**")
                rows = []
                for lbl, yv in sf.items():
                    if any(v is not None for v in yv.values()):
                        row = {"Label": lbl}
                        row.update({yr: (f"{v:,.2f}" if v is not None else "—") for yr, v in yv.items()})
                        rows.append(row)
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("No values matched for this sheet.")

        # ── Write to template ──────────────────────────────────────────────────
        progress.progress(90, text="Writing to template…")
        output_bytes, stats = write_to_template(template_bytes, filled, overwrite=overwrite)
        progress.progress(100, text="Done ✓")

        # Sources banner
        sources = []
        if s_count:      sources.append("Screener.in")
        if y_count:      sources.append("yfinance")
        if pdf_field_count: sources.append(f"PDF ({pdf_label})")

        status.success(
            f"✅ Done! **{stats['written']}** cells written · "
            f"{stats['skipped_formula']} formulas preserved · "
            f"{stats['skipped_filled']} existing kept · "
            f"{stats['not_found']} labels unmatched\n\n"
            f"📡 Sources used: {' + '.join(sources) if sources else 'none'}"
        )

        st.download_button(
            label="⬇️ Download Populated Template",
            data=output_bytes,
            file_name=f"{nse_symbol}_populated.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

        audit = stats.get("audit", [])
        if audit:
            st.subheader("📋 Cells Written")
            df_audit = pd.DataFrame(audit)
            # Mark the source of each value
            st.dataframe(df_audit, use_container_width=True, hide_index=True)

    except Exception as e:
        status.error(f"❌ Pipeline failed: {e}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
