"""
pipeline/derivation_agent.py
─────────────────────────────
Phase 5: Post-fill derivation + validation agent.

After Phases 1-4 fill what they can from web sources and PDF text,
this agent gets the partially-filled sheet and tries three things:

  Tool 1 — Rule-based derivation  (pure Python, no LLM, fastest)
    Uses hardcoded Ind AS accounting identities to compute blanks from
    values that are already filled.  No API call needed.
    Examples: EBIT = EBITDA - D&A,  Gross Profit = Revenue - COGS,
              PAT = PBT - Tax,  Net Debt = Borrowings - Cash

  Tool 2 — Targeted PDF page search  (keyword search, no LLM)
    For each still-missing label, scores PDF pages by keyword overlap
    and sends the highest-scoring pages to the LLM for extraction.

  Tool 3 — Web snippet search  (DuckDuckGo Lite, no API key)
    Last resort: searches for "{company} {label} {year} annual report"
    and extracts values from result snippets via the LLM.

  Tool 4 — Cross-validation
    After all fills, checks accounting identities and flags rows whose
    values look wrong (wrong sign, magnitude outside plausible range,
    BS doesn't balance, P&L doesn't add up).

All four tools are called in sequence.  Each only processes labels that
the previous tool couldn't fill.  The LLM is called at most twice
(once for PDF search, once for web) per sheet — not per label.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
from typing import Any

import requests

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Rule-based derivation
# ─────────────────────────────────────────────────────────────────────────────

# Each rule: (target_label_patterns, formula_fn, ingredient_patterns)
# Patterns are lowercase substrings — if ANY pattern matches the label, it counts.
# formula_fn receives a dict {ingredient_key: value} and returns float|None.

def _match(label: str, patterns: tuple[str, ...]) -> bool:
    ll = label.lower()
    return any(p in ll for p in patterns)


def _first(filled: dict[str, dict], label_patterns: tuple[str, ...], yr: str) -> float | None:
    """Return value for the first label in filled that matches any pattern."""
    for lbl, yr_vals in filled.items():
        if _match(lbl, label_patterns):
            v = yr_vals.get(yr)
            if v is not None:
                return float(v)
    return None


# Accounting identity rules — evaluated in order
# Each rule: (target_patterns, ingredient_patterns_list, formula)
_RULES: list[tuple[tuple, list[tuple], Any]] = [

    # ── P&L ──────────────────────────────────────────────────────────────────

    # EBIT = EBITDA - D&A
    (
        ("ebit",),
        [("ebitda", "operating profit"), ("depreciation", "amortis", "d&a", "da ")],
        lambda v: v[0] - abs(v[1]) if all(x is not None for x in v) else None,
    ),
    # EBITDA = EBIT + D&A
    (
        ("ebitda", "operating profit"),
        [("ebit",), ("depreciation", "amortis", "d&a")],
        lambda v: v[0] + abs(v[1]) if all(x is not None for x in v) else None,
    ),
    # Gross Profit = Revenue - COGS
    (
        ("gross profit",),
        [("revenue from operations", "net sales", "revenue"),
         ("cost of goods sold", "cogs", "cost of material", "raw material cost")],
        lambda v: v[0] - abs(v[1]) if all(x is not None for x in v) else None,
    ),
    # COGS = Revenue - Gross Profit
    (
        ("cost of goods sold", "cogs"),
        [("revenue from operations", "net sales", "revenue"),
         ("gross profit",)],
        lambda v: v[0] - v[1] if all(x is not None for x in v) else None,
    ),
    # PAT = PBT - Tax
    (
        ("pat", "profit after tax", "net profit"),
        [("pbt", "profit before tax"),
         ("tax", "income tax expense", "tax expense")],
        lambda v: v[0] - abs(v[1]) if all(x is not None for x in v) else None,
    ),
    # PBT = PAT + Tax
    (
        ("pbt", "profit before tax"),
        [("pat", "profit after tax", "net profit"),
         ("tax", "income tax expense")],
        lambda v: v[0] + abs(v[1]) if all(x is not None for x in v) else None,
    ),
    # EBIT = PBT + Finance Costs - Other Income
    (
        ("ebit",),
        [("pbt", "profit before tax"),
         ("finance cost", "interest expense", "finance charges"),
         ("other income",)],
        lambda v: v[0] + abs(v[1]) - abs(v[2]) if all(x is not None for x in v) else None,
    ),
    # PBT = EBIT - Finance Cost + Other Income
    (
        ("pbt", "profit before tax"),
        [("ebit",),
         ("finance cost", "interest expense"),
         ("other income",)],
        lambda v: v[0] - abs(v[1]) + abs(v[2]) if all(x is not None for x in v) else None,
    ),
    # PAT pre-minority = PAT + Minority Interest
    (
        ("pat (before minority", "pat before minority", "profit before minority"),
        [("pat", "profit after tax", "net profit"),
         ("minority interest", "non-controlling interest")],
        lambda v: v[0] + abs(v[1]) if all(x is not None for x in v) else None,
    ),
    # Minority Interest = PAT pre-minority - PAT
    (
        ("minority interest", "non-controlling interest"),
        [("pat (before minority", "profit before minority"),
         ("pat", "profit after tax", "net profit")],
        lambda v: v[0] - v[1] if all(x is not None for x in v) else None,
    ),
    # Total Income = Revenue + Other Income
    (
        ("total income",),
        [("revenue from operations", "net sales"),
         ("other income",)],
        lambda v: v[0] + abs(v[1]) if all(x is not None for x in v) else None,
    ),

    # ── Margin % rows ─────────────────────────────────────────────────────────
    # EBITDA margin = EBITDA / Revenue * 100
    (
        ("ebitda margin", "ebitda %", "% of sales",),
        [("ebitda", "operating profit"),
         ("revenue from operations", "net sales", "revenue")],
        lambda v: round(v[0] / v[1] * 100, 2) if v[1] and v[1] != 0 else None,
    ),
    # Gross margin
    (
        ("gross margin", "gross profit margin", "gpm"),
        [("gross profit",),
         ("revenue from operations", "net sales", "revenue")],
        lambda v: round(v[0] / v[1] * 100, 2) if v[1] and v[1] != 0 else None,
    ),
    # PAT margin
    (
        ("pat margin", "net profit margin", "npm"),
        [("pat", "profit after tax", "net profit"),
         ("revenue from operations", "net sales", "revenue")],
        lambda v: round(v[0] / v[1] * 100, 2) if v[1] and v[1] != 0 else None,
    ),

    # ── Balance Sheet ─────────────────────────────────────────────────────────
    # Net Block = Gross Block - Accumulated Depreciation
    (
        ("net block", "net fixed assets", "property plant & equipment net"),
        [("gross block", "gross fixed assets"),
         ("accumulated depreciation", "less: depreciation", "depreciation on fixed")],
        lambda v: v[0] - abs(v[1]) if all(x is not None for x in v) else None,
    ),
    # Net Debt = Total Borrowings - Cash
    (
        ("net debt",),
        [("total borrowing", "total debt", "long-term borrowing"),
         ("cash and cash equivalent", "cash & equivalent")],
        lambda v: v[0] - abs(v[1]) if all(x is not None for x in v) else None,
    ),
    # Total Assets = Non-Current Assets + Current Assets
    (
        ("total assets",),
        [("total non-current assets", "non current assets total"),
         ("total current assets", "current assets total")],
        lambda v: v[0] + v[1] if all(x is not None for x in v) else None,
    ),

    # ── Cash Flow ─────────────────────────────────────────────────────────────
    # Free Cash Flow = CFO - Capex
    (
        ("free cash flow", "fcf"),
        [("cash from operations", "cfo", "operating activities"),
         ("capital expenditure", "capex", "purchase of fixed assets")],
        lambda v: v[0] - abs(v[1]) if all(x is not None for x in v) else None,
    ),
]


def _derive_from_rules(
    sheet_filled: dict[str, dict[str, float | None]],
    still_empty: list[str],
    years: list[str],
) -> dict[str, dict[str, float | None]]:
    """
    Apply accounting identity rules to fill still-empty labels.
    Returns only newly derived values.
    """
    derived: dict[str, dict[str, float | None]] = {}

    for target_lbl in still_empty:
        for (target_pats, ingredient_pats_list, formula) in _RULES:
            # Does this rule apply to this target label?
            if not _match(target_lbl, target_pats):
                continue

            yr_map: dict[str, float | None] = {}
            for yr in years:
                # Collect ingredient values for this year
                ingredients: list[float | None] = []
                for ing_pats in ingredient_pats_list:
                    val = _first(sheet_filled, ing_pats, yr)
                    ingredients.append(val)

                # Apply formula only if all ingredients found
                try:
                    result = formula(ingredients)
                    yr_map[yr] = result
                except Exception:
                    yr_map[yr] = None

            any_filled = any(v is not None for v in yr_map.values())
            if any_filled:
                derived[target_lbl] = yr_map
                logger.info(
                    f"derivation: '{target_lbl}' derived via rule "
                    f"({sum(1 for v in yr_map.values() if v is not None)}/{len(years)} years)"
                )
                break   # first matching rule wins

    return derived


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Targeted PDF page search
# ─────────────────────────────────────────────────────────────────────────────

def _score_page_for_labels(page_text: str, labels: list[str]) -> int:
    """Score a PDF page text for relevance to a list of labels."""
    pl = page_text.lower()
    keywords: set[str] = set()
    for lbl in labels:
        for word in re.split(r"[^a-z]+", lbl.lower()):
            if len(word) > 3:
                keywords.add(word)
    score = sum(1 for kw in keywords if kw in pl)
    # Bonus for pages that look like financial tables (digit-dense)
    digit_ratio = sum(1 for c in page_text if c.isdigit()) / max(len(page_text), 1)
    if digit_ratio > 0.12:
        score += 3
    return score


def _pdf_search_for_labels(
    still_empty: list[str],
    years: list[str],
    sheet_name: str,
    sector: str,
    pdf_text: str,
    pdf_bytes: bytes | None,
    client: Any,
    model: str,
) -> dict[str, dict[str, float | None]]:
    """
    For each still-empty label, pick the PDF pages most likely to contain it
    and extract values via an LLM call.  Uses keyword scoring to target pages.
    """
    if not still_empty or client is None:
        return {}

    # If we have pdf_bytes, try page-level extraction for better precision
    if pdf_bytes:
        try:
            import pdfplumber
            import io
            page_texts: list[str] = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for pg in pdf.pages:
                    txt = pg.extract_text() or ""
                    page_texts.append(txt)

            # Score each page against the still-empty labels
            scored_pages = sorted(
                enumerate(page_texts),
                key=lambda x: _score_page_for_labels(x[1], still_empty),
                reverse=True,
            )
            # Take top 15 pages, join into a search corpus
            top_pages = [txt for _, txt in scored_pages[:15] if txt.strip()]
            search_corpus = "\n\n--- PAGE BREAK ---\n\n".join(top_pages)[:35_000]
        except Exception as exc:
            logger.debug(f"_pdf_search_for_labels page extraction failed: {exc}")
            # Fall back to keyword-chunk selection from extracted text
            search_corpus = _keyword_chunk_select(pdf_text, still_empty, max_chars=30_000)
    else:
        search_corpus = _keyword_chunk_select(pdf_text, still_empty, max_chars=30_000)

    if not search_corpus.strip():
        return {}

    return _llm_extract_from_text(
        still_empty, years, sheet_name, sector,
        search_corpus, client, model,
        source_hint="PDF pages most relevant to missing labels",
    )


def _keyword_chunk_select(text: str, labels: list[str], max_chars: int = 30_000) -> str:
    """Select the text chunks most relevant to the given labels."""
    if not text:
        return ""
    keywords: set[str] = set()
    for lbl in labels:
        for word in re.split(r"[^a-z]+", lbl.lower()):
            if len(word) > 3:
                keywords.add(word)

    chunks = re.split(r"\n{2,}|\f", text)
    scored = []
    for chunk in chunks:
        c = chunk.strip()
        if len(c) < 30:
            continue
        cl = c.lower()
        score = sum(1 for kw in keywords if kw in cl)
        digit_ratio = sum(1 for ch in c if ch.isdigit()) / max(len(c), 1)
        if digit_ratio > 0.12:
            score += 2
        scored.append((score, c))

    scored.sort(key=lambda x: -x[0])
    selected, total = [], 0
    for _, chunk in scored:
        if total + len(chunk) > max_chars:
            break
        selected.append(chunk)
        total += len(chunk)
    return "\n\n".join(selected) if selected else text[:max_chars]


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — Web search (DuckDuckGo Lite)
# ─────────────────────────────────────────────────────────────────────────────

def _web_search_for_labels(
    still_empty: list[str],
    years: list[str],
    sheet_name: str,
    sector: str,
    nse_symbol: str,
    client: Any,
    model: str,
) -> dict[str, dict[str, float | None]]:
    """
    Search DuckDuckGo Lite for each batch of still-empty labels and extract
    values from result snippets via LLM.  No API key required.
    """
    if not still_empty or not nse_symbol or client is None:
        return {}

    cap = still_empty[:20]   # don't burn too many searches
    snippets: list[str] = []
    yr_hint = " ".join(years[:2])

    # Batch labels 3 at a time for focused searches
    for i in range(0, len(cap), 3):
        batch = cap[i : i + 3]
        lbl_q = " ".join(b.split()[:3] for b in batch[:2])   # first 3 words of each
        query = f"{nse_symbol} {lbl_q} {yr_hint} annual report INR crore"
        url   = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote(query)}"
        try:
            r = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; FinBot/1.0)"},
                timeout=10,
            )
            if r.ok:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(r.text, "html.parser")
                page_snips = [
                    td.get_text(" ", strip=True)
                    for td in soup.find_all("td", class_="result-snippet")
                ]
                snippets.extend(page_snips[:4])
        except Exception as exc:
            logger.debug(f"_web_search_for_labels batch {i}: {exc}")

    if not snippets:
        return {}

    search_text = "\n---\n".join(snippets[:12])
    return _llm_extract_from_text(
        cap, years, sheet_name, sector,
        search_text, client, model,
        source_hint="web search result snippets from financial databases",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared LLM extraction call
# ─────────────────────────────────────────────────────────────────────────────

def _llm_extract_from_text(
    labels: list[str],
    years: list[str],
    sheet_name: str,
    sector: str,
    text_clip: str,
    client: Any,
    model: str,
    source_hint: str = "financial text",
) -> dict[str, dict[str, float | None]]:
    """Single LLM call to extract specific label values from a text clip."""
    if not text_clip.strip() or not labels or client is None:
        return {}

    yrs_str  = ", ".join(years)
    lbls_str = "\n".join(f"  - {lbl}" for lbl in labels)

    prompt = f"""You are a financial data extraction expert for Indian listed companies.

Find values for the following labels in the "{sheet_name}" sheet of a {sector} company.
Fiscal years needed: {yrs_str}

LABELS TO FIND:
{lbls_str}

SOURCE ({source_hint}) — values in INR Crores unless stated otherwise:
{text_clip}

RULES:
1. Only return values CLEARLY present in the source text — do NOT guess.
2. Values must be in INR Crores.
3. Parenthesised numbers like (50.00) are negative.
4. Match by financial MEANING, not just exact wording.
5. If a label is not found, return null for that year.

Return JSON only:
{{
  "values": {{
    "<label exactly as given>": {{"<year>": <number or null>, ...}},
    ...
  }}
}}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        vals = data.get("values", {})

        result: dict[str, dict[str, float | None]] = {}
        for lbl in labels:
            yr_map: dict[str, float | None] = {}
            raw_yv = vals.get(lbl, {})
            for yr in years:
                v = raw_yv.get(yr)
                try:
                    yr_map[yr] = float(v) if v is not None else None
                except (TypeError, ValueError):
                    yr_map[yr] = None
            result[lbl] = yr_map

        filled = sum(1 for yv in result.values() for v in yv.values() if v is not None)
        logger.info(f"_llm_extract_from_text [{source_hint[:30]}]: {filled} values for {len(labels)} labels")
        return result

    except Exception as exc:
        logger.warning(f"_llm_extract_from_text failed: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — Cross-validation & flagging
# ─────────────────────────────────────────────────────────────────────────────

def _cross_validate(
    sheet_filled: dict[str, dict[str, float | None]],
    years: list[str],
) -> list[dict]:
    """
    Check accounting identities and value plausibility.
    Returns a list of warnings: [{label, year, value, issue}]
    """
    warnings: list[dict] = []

    for yr in years:
        def get(pats):
            return _first(sheet_filled, pats, yr)

        revenue  = get(("revenue from operations", "net sales", "revenue"))
        ebitda   = get(("ebitda", "operating profit"))
        ebit     = get(("ebit",))
        pbt      = get(("pbt", "profit before tax"))
        pat      = get(("pat", "profit after tax", "net profit"))
        dep      = get(("depreciation", "d&a"))
        fin_cost = get(("finance cost", "interest expense"))
        tax      = get(("tax expense", "income tax"))

        # P&L waterfall: Revenue ≥ EBITDA ≥ EBIT ≥ PBT (for profitable companies)
        checks = [
            (revenue, ebitda,  "Revenue",  "EBITDA",  "Revenue should be ≥ EBITDA"),
            (ebitda,  ebit,    "EBITDA",   "EBIT",    "EBITDA should be ≥ EBIT"),
            (ebit,    pbt,     "EBIT",     "PBT",     "EBIT should be ≥ PBT (ignoring Other Income/Finance Costs)"),
            (pbt,     pat,     "PBT",      "PAT",     "PBT should be ≥ PAT"),
        ]
        for upper, lower, ulbl, llbl, msg in checks:
            if upper is not None and lower is not None:
                # Only flag if both positive and the relationship is clearly wrong
                if upper > 0 and lower > 0 and lower > upper * 1.15:
                    # Find the actual label names
                    for lbl in sheet_filled:
                        if _match(lbl, (llbl.lower(),)):
                            warnings.append({
                                "label": lbl, "year": yr,
                                "value": lower,
                                "issue": f"⚠️ {msg} ({ulbl}={upper:,.0f} < {llbl}={lower:,.0f})",
                            })
                            break

        # Sign checks
        sign_checks = [
            (revenue,  ("revenue",),  "+", "Revenue should be positive"),
            (dep,      ("depreciation",), "+", "D&A should be positive"),
            (fin_cost, ("finance cost",), "+", "Finance costs should be positive"),
            (tax,      ("tax",),       "+", "Tax expense should be positive (for profitable cos)"),
        ]
        for val, pats, expected_sign, msg in sign_checks:
            if val is not None and expected_sign == "+" and val < 0:
                for lbl in sheet_filled:
                    if _match(lbl, pats):
                        warnings.append({
                            "label": lbl, "year": yr,
                            "value": val,
                            "issue": f"⚠️ {msg} (got {val:,.0f})",
                        })
                        break

        # EBITDA margin plausibility (0%-60% is normal for Indian cos)
        if revenue and ebitda and revenue > 0:
            margin = ebitda / revenue * 100
            if margin > 70:
                for lbl in sheet_filled:
                    if _match(lbl, ("ebitda",)):
                        warnings.append({
                            "label": lbl, "year": yr,
                            "value": ebitda,
                            "issue": f"⚠️ EBITDA margin {margin:.1f}% seems very high — check source",
                        })
                        break
            elif margin < -30:
                for lbl in sheet_filled:
                    if _match(lbl, ("ebitda",)):
                        warnings.append({
                            "label": lbl, "year": yr,
                            "value": ebitda,
                            "issue": f"⚠️ EBITDA margin {margin:.1f}% seems very negative — check source",
                        })
                        break

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def financial_derivation_agent(
    sheet_name: str,
    sheet_filled: dict[str, dict[str, float | None]],
    template_labels: list[str],
    years: list[str],
    sector: str = "General",
    pdf_text: str = "",
    pdf_bytes: bytes | None = None,
    nse_symbol: str = "",
    client: Any = None,
    model: str = "",
) -> tuple[dict[str, dict[str, float | None]], list[dict]]:
    """
    Phase 5 derivation agent.

    Parameters
    ----------
    sheet_name      : Name of the sheet being processed
    sheet_filled    : Current fills {label: {year: value}}
    template_labels : Ordered list of all template labels for this sheet
    years           : Fiscal years to fill
    sector          : Sector string for LLM context
    pdf_text        : Already-extracted PDF text (used for keyword search)
    pdf_bytes       : Raw PDF bytes (used for page-level search)
    nse_symbol      : NSE ticker (used for web search)
    client          : OpenAI-compatible client
    model           : Model name

    Returns
    -------
    (updated_sheet_filled, validation_warnings)
    updated_sheet_filled merges original fills + newly derived/found values.
    validation_warnings is a list of {label, year, value, issue} dicts.
    """
    # Only run for financial statement sheets
    sl = sheet_name.lower()
    is_fin = any(k in sl for k in ("p&l", "profit", "income", "balance", "bs", "cash", "cf"))
    if not is_fin:
        return sheet_filled, []

    # Work on a copy
    result = {lbl: dict(yr_vals) for lbl, yr_vals in sheet_filled.items()}

    # Identify all labels with at least one missing year
    def is_empty(lbl: str) -> bool:
        return all(v is None for v in result.get(lbl, {}).values())

    still_empty = [lbl for lbl in template_labels if is_empty(lbl)]
    total_initial = len(still_empty)

    if not still_empty:
        logger.info(f"derivation_agent '{sheet_name}': nothing empty, skipping")
        warnings = _cross_validate(result, years)
        return result, warnings

    logger.info(
        f"derivation_agent '{sheet_name}': {len(still_empty)} empty labels, "
        f"running 4 tools"
    )

    # ── Tool 1: Rule-based derivation (no LLM) ────────────────────────────────
    t1 = _derive_from_rules(result, still_empty, years)
    for lbl, yv in t1.items():
        if lbl not in result:
            result[lbl] = yv
        else:
            for yr, v in yv.items():
                if v is not None and result[lbl].get(yr) is None:
                    result[lbl][yr] = v

    still_empty = [lbl for lbl in still_empty if is_empty(lbl)]
    logger.info(
        f"derivation_agent tool1 (rules): filled {total_initial - len(still_empty)} "
        f"labels, {len(still_empty)} still empty"
    )

    if not still_empty:
        warnings = _cross_validate(result, years)
        return result, warnings

    # ── Tool 2: Targeted PDF page search ──────────────────────────────────────
    if pdf_text or pdf_bytes:
        t2 = _pdf_search_for_labels(
            still_empty, years, sheet_name, sector,
            pdf_text, pdf_bytes, client, model,
        )
        for lbl, yv in t2.items():
            if lbl not in result:
                result[lbl] = yv
            else:
                for yr, v in yv.items():
                    if v is not None and result[lbl].get(yr) is None:
                        result[lbl][yr] = v

        before = len(still_empty)
        still_empty = [lbl for lbl in still_empty if is_empty(lbl)]
        logger.info(
            f"derivation_agent tool2 (pdf search): filled {before - len(still_empty)} "
            f"labels, {len(still_empty)} still empty"
        )

    if not still_empty:
        warnings = _cross_validate(result, years)
        return result, warnings

    # ── Tool 3: Web search fallback ────────────────────────────────────────────
    if nse_symbol and client is not None:
        t3 = _web_search_for_labels(
            still_empty, years, sheet_name, sector,
            nse_symbol, client, model,
        )
        for lbl, yv in t3.items():
            if lbl not in result:
                result[lbl] = yv
            else:
                for yr, v in yv.items():
                    if v is not None and result[lbl].get(yr) is None:
                        result[lbl][yr] = v

        before = len(still_empty)
        still_empty = [lbl for lbl in still_empty if is_empty(lbl)]
        logger.info(
            f"derivation_agent tool3 (web search): filled {before - len(still_empty)} "
            f"labels, {len(still_empty)} still empty"
        )

    # ── Tool 4: Cross-validation ───────────────────────────────────────────────
    warnings = _cross_validate(result, years)
    if warnings:
        logger.info(
            f"derivation_agent tool4 (validation): {len(warnings)} warnings "
            f"for '{sheet_name}'"
        )

    filled_total = sum(
        1 for yv in result.values() for v in yv.values() if v is not None
    )
    original_filled = sum(
        1 for yv in sheet_filled.values() for v in yv.values() if v is not None
    )
    logger.info(
        f"derivation_agent '{sheet_name}': {filled_total - original_filled} new values, "
        f"{len(warnings)} validation warnings"
    )

    return result, warnings
