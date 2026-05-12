"""
pipeline/financial_agent.py
────────────────────────────
Phase 5: Real tool-calling financial agent.

The LLM drives the entire loop using OpenAI function-calling. It receives a
snapshot of the partially-filled sheet and decides WHICH tools to call and in
WHAT ORDER based on what it finds — just like a senior analyst reasoning step
by step.

Tools available to the LLM
────────────────────────────
  lookup_pool          — fetch values from the multi-source data pool
  search_pdf_pages     — keyword-search PDF text for a line item
  derive_values        — apply accounting identities (Gross Profit = Rev – COGS, etc.)
  search_web           — DuckDuckGo search for a metric (no API key needed)
  validate_and_correct — cross-row sanity checks and flag inconsistencies
  submit_values        — commit the final fills and end the loop

Loop logic
───────────
  1. Build initial messages: system brief + filled-sheet snapshot + empty labels
  2. Call LLM with all tool schemas
  3. If LLM calls a tool → execute it, append result, loop
  4. If LLM calls submit_values → extract fills, break
  5. Max MAX_ITER iterations to prevent infinite loops
  6. Apply fills (confidence ≥ MIN_CONF) back to sheet_filled
  7. Return (updated_sheet_filled, tool_call_log)
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_ITER     = 10        # hard cap on LLM calls per sheet
MIN_CONF     = 0.55      # minimum confidence to accept a fill
MAX_LABELS   = 15        # max empty labels per agent batch
TIME_BUDGET_SEC = 120.0  # max wall-clock per agent invocation
FORCE_SUBMIT_AT_ITER = 7 # inject mandatory-submit reminder at this iteration


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas (OpenAI function-calling format)
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_pool",
            "description": (
                "Look up values from the multi-source financial data pool "
                "(XBRL, NSE/BSE exchange filings, Screener.in, Tofler, MoneyControl). "
                "Use this first for any label that might exist under a slightly different name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of template labels (or close variants) to search for in the pool."
                    },
                    "years": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fiscal years needed, e.g. ['FY2023', 'FY2022']"
                    }
                },
                "required": ["query_labels", "years"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_pdf_pages",
            "description": (
                "Keyword-search the extracted PDF annual report text to find "
                "values for specific financial line items. Returns the most "
                "relevant PDF chunks with numbers. Use when pool lookup fails."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Label names to search for in the PDF text."
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional keywords to narrow the search (optional)."
                    }
                },
                "required": ["labels"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "derive_values",
            "description": (
                "Compute missing values using standard accounting identities. "
                "E.g. Gross Profit = Revenue - COGS; EBIT = EBITDA - D&A; "
                "PAT = PBT - Tax; Working Capital = Current Assets - Current Liabilities. "
                "Use when you have the component values but not the derived total."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels you want to derive."
                    },
                    "known_values": {
                        "type": "object",
                        "description": "Already-known values as {label: {year: value}}. Use the current sheet fills."
                    }
                },
                "required": ["target_labels", "known_values"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web (DuckDuckGo) for specific financial metrics. "
                "Use as last resort when pool and PDF both fail. "
                "Returns text snippets containing numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'Infosys FY2023 EBITDA INR Crores annual report'"
                    },
                    "metric": {
                        "type": "string",
                        "description": "The specific metric being searched for."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_and_correct",
            "description": (
                "Validate the current sheet fills for cross-row consistency. "
                "Checks: Revenue > EBITDA > EBIT > PBT > PAT, Balance Sheet balances, "
                "sign conventions, totals vs sub-items. Returns issues found."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "current_fills": {
                        "type": "object",
                        "description": "Current sheet fills as {label: {year: value}}."
                    },
                    "sheet_type": {
                        "type": "string",
                        "enum": ["income_statement", "balance_sheet", "cash_flow", "other"],
                        "description": "Type of financial statement."
                    }
                },
                "required": ["current_fills", "sheet_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_values",
            "description": (
                "Submit the final values to be written into the template. "
                "Call this when you are done filling all possible labels. "
                "This ends the agent loop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fills": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label":      {"type": "string"},
                                "year":       {"type": "string"},
                                "value":      {"type": "number"},
                                "confidence": {"type": "number"},
                                "source":     {"type": "string"}
                            },
                            "required": ["label", "year", "value", "confidence", "source"]
                        },
                        "description": "List of fills to apply. Only include values you are confident about (confidence ≥ 0.65)."
                    },
                    "corrections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label":      {"type": "string"},
                                "year":       {"type": "string"},
                                "old_value":  {"type": "number"},
                                "new_value":  {"type": "number"},
                                "reason":     {"type": "string"},
                                "confidence": {"type": "number"}
                            }
                        },
                        "description": "Corrections to existing values that are wrong."
                    }
                },
                "required": ["fills"]
            }
        }
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

def _tool_lookup_pool(
    query_labels: list[str],
    years: list[str],
    pool: dict[str, dict[str, float]],
) -> dict:
    """Fuzzy-search the pool for labels and return matching values."""
    from difflib import SequenceMatcher

    results: dict = {}
    no_data: list[str] = []   # found in pool but no values for requested years

    for ql in query_labels:
        ql_lower = ql.lower()
        best_match: str | None = None
        best_score = 0.0

        for pool_label in pool:
            pl_lower = pool_label.lower()
            if ql_lower == pl_lower:
                best_match = pool_label
                best_score = 1.0
                break
            if ql_lower in pl_lower or pl_lower in ql_lower:
                score = 0.85
            else:
                score = SequenceMatcher(None, ql_lower, pl_lower).ratio()
            if score > best_score and score >= 0.70:
                best_score = score
                best_match = pool_label

        if best_match:
            yr_vals = {
                yr: pool[best_match][yr]
                for yr in years
                if pool[best_match].get(yr) is not None
            }
            if yr_vals:
                results[ql] = {
                    "matched_label": best_match,
                    "similarity": round(best_score, 3),
                    "values": yr_vals,
                }
            else:
                # BUG FIX: label exists in pool but has no data for the requested years.
                # Previously this fell through to word-overlap and returned "not found".
                # Now we report it accurately so the LLM knows to try PDF/web instead.
                no_data.append(ql)
            continue

        # Word-overlap fallback when string similarity scored < 0.70
        ql_words = set(re.split(r"[^a-z]+", ql_lower)) - {"", "the", "of", "and", "to"}
        top_candidates = []
        for pool_label in pool:
            pl_words = set(re.split(r"[^a-z]+", pool_label.lower())) - {"", "the", "of", "and", "to"}
            overlap = len(ql_words & pl_words) / max(len(ql_words), 1)
            if overlap >= 0.5:
                yr_vals = {
                    yr: pool[pool_label][yr]
                    for yr in years
                    if pool[pool_label].get(yr) is not None
                }
                if yr_vals:
                    top_candidates.append((overlap, pool_label, yr_vals))

        if top_candidates:
            top_candidates.sort(key=lambda x: -x[0])
            overlap, pool_label, yr_vals = top_candidates[0]
            results[ql] = {
                "matched_label": pool_label,
                "similarity": round(overlap, 3),
                "values": yr_vals,
                "note": "word-overlap match",
            }

    not_found = [ql for ql in query_labels if ql not in results and ql not in no_data]
    return {
        "found": results,
        "not_found": not_found,
        "found_but_no_year_data": no_data,   # exists in pool, no data for requested years
        "pool_size": len(pool),
    }


def _tool_search_pdf_pages(
    labels: list[str],
    pdf_text: str,
    keywords: list[str] | None = None,
    max_chars: int = 12_000,
) -> dict:
    """Keyword-search PDF text and return relevant chunks."""
    if not pdf_text:
        return {"chunks": [], "note": "No PDF text available"}

    all_kws: set[str] = set(keywords or [])
    for lbl in labels:
        for w in re.split(r"[^a-z]+", lbl.lower()):
            if len(w) > 3:
                all_kws.add(w)

    chunks = re.split(r"\n{2,}|\f", pdf_text)
    scored = []
    for chunk in chunks:
        c = chunk.strip()
        if len(c) < 30:
            continue
        cl = c.lower()
        kw_hits = sum(1 for kw in all_kws if kw in cl)
        if kw_hits == 0:
            continue
        digit_ratio = sum(1 for ch in c if ch.isdigit()) / max(len(c), 1)
        score = kw_hits + (3 if digit_ratio > 0.12 else 0)
        scored.append((score, c))

    scored.sort(key=lambda x: -x[0])
    parts, total = [], 0
    for score, chunk in scored[:20]:
        if total + len(chunk) > max_chars:
            break
        parts.append({"score": score, "text": chunk[:2000]})
        total += len(chunk)

    return {
        "chunks_found": len(parts),
        "chunks": parts[:8],   # return top 8
        "searched_keywords": list(all_kws)[:20],
    }


# Accounting identity rules: (target, [(component, sign), ...])
# sign +1 = add, -1 = subtract
_IDENTITY_RULES: list[tuple[str, list[tuple[str, int]]]] = [
    # Income Statement
    ("Gross Profit", [("Revenue From Operations", 1), ("Cost of Goods Sold", -1)]),
    ("Gross Profit", [("Net Revenue", 1), ("Cost of Goods Sold", -1)]),
    ("Gross Profit", [("Revenue From Operations", 1), ("COGS", -1)]),
    ("EBITDA", [("Gross Profit", 1), ("Other Income", 1), ("Employee Benefit Expense", -1),
                ("Other Expenses", -1), ("Selling General And Admin Expenses", -1)]),
    ("EBIT", [("EBITDA", 1), ("Depreciation And Amortization", -1)]),
    ("EBIT", [("EBITDA", 1), ("Depreciation", -1)]),
    ("EBIT", [("Revenue From Operations", 1), ("Total Operating Expenses", -1)]),
    ("PBT", [("EBIT", 1), ("Finance Cost", -1), ("Other Income", 1)]),
    ("PBT", [("EBIT", 1), ("Interest Expense", -1), ("Other Income", 1)]),
    ("PAT", [("PBT", 1), ("Tax Expense", -1)]),
    ("PAT", [("PBT", 1), ("Current Tax", -1), ("Deferred Tax", -1)]),
    ("EBITDA Margin %", []),   # handled separately
    ("EBIT Margin %", []),
    ("PAT Margin %", []),
    ("Gross Profit Margin %", []),
    # Cash Flow
    ("Free Cash Flow", [("Cash Flow From Operations", 1), ("Capital Expenditure", -1)]),
    ("Free Cash Flow", [("CFO", 1), ("Capex", -1)]),
    # Balance Sheet
    ("Working Capital", [("Total Current Assets", 1), ("Total Current Liabilities", -1)]),
    ("Net Debt", [("Total Borrowings", 1), ("Cash And Cash Equivalents", -1)]),
    ("Net Worth", [("Share Capital", 1), ("Reserves And Surplus", 1)]),
    ("Net Worth", [("Total Shareholders Equity", 1)]),
    ("Capital Employed", [("Total Assets", 1), ("Total Current Liabilities", -1)]),
    ("ROCE %", []),   # ratio — handled separately
    ("ROE %", []),
    ("ROA %", []),
]

# Revenue synonyms tried in order when looking up the denominator for margin calculations.
# BUG FIX: previously hardcoded "Revenue From Operations" — sheets using "Net Revenue",
# "Total Revenue", "Revenue", etc. got None for all margins.
_REVENUE_ALIASES = [
    "Revenue From Operations",
    "Net Revenue",
    "Total Revenue",
    "Revenue",
    "Net Sales",
    "Total Income From Operations",
    "Total Net Revenue",
]

_MARGIN_RULES = {
    "EBITDA Margin %":       ("EBITDA",       None),   # None → resolve via _REVENUE_ALIASES
    "EBIT Margin %":         ("EBIT",         None),
    "PAT Margin %":          ("PAT",          None),
    "Gross Profit Margin %": ("Gross Profit", None),
    "Net Profit Margin %":   ("PAT",          None),
    "ROCE %":                ("EBIT",         "Capital Employed"),
    "ROE %":                 ("PAT",          "Net Worth"),
    "ROA %":                 ("PAT",          "Total Assets"),
}

_RATIO_RULES = {
    "Debt To Equity": ("Total Borrowings",        "Net Worth"),
    "Current Ratio":  ("Total Current Assets",    "Total Current Liabilities"),
    "Interest Coverage Ratio": ("EBIT",           "Finance Cost"),
    "EV/EBITDA":      ("Enterprise Value",        "EBITDA"),
    "P/E Ratio":      ("Market Capitalisation",   "PAT"),
}


def _tool_derive_values(
    target_labels: list[str],
    known_values: dict[str, dict[str, float | None]],
    years: list[str],
) -> dict:
    """Apply accounting identities to derive missing values."""
    derived = {}
    notes   = []

    def get_val(lbl: str, yr: str) -> float | None:
        """Case-insensitive label lookup."""
        # Exact match
        if lbl in known_values and yr in known_values[lbl]:
            v = known_values[lbl][yr]
            if v is not None:
                return float(v)
        # Case-insensitive match
        lbl_lower = lbl.lower()
        for k, yrv in known_values.items():
            if k.lower() == lbl_lower:
                v = yrv.get(yr)
                if v is not None:
                    return float(v)
        return None

    for target in target_labels:
        target_lower = target.lower()

        # ── Margin / ratio rules ──────────────────────────────────────────────
        matched_margin = None
        for rule_label, (num_lbl, denom_lbl) in _MARGIN_RULES.items():
            if rule_label.lower() == target_lower:
                matched_margin = (num_lbl, denom_lbl)
                break

        if matched_margin:
            num_lbl, denom_lbl = matched_margin

            # BUG FIX: resolve revenue denominator via aliases when denom_lbl is None.
            # Tries "Revenue From Operations", "Net Revenue", "Total Revenue", etc.
            if denom_lbl is None:
                denom_lbl = next(
                    (alias for alias in _REVENUE_ALIASES if get_val(alias, years[0]) is not None),
                    "Revenue From Operations",   # last-resort default for note display
                ) if years else "Revenue From Operations"

            yr_results = {}
            for yr in years:
                num   = get_val(num_lbl, yr)
                denom = get_val(denom_lbl, yr)
                if num is not None and denom and abs(denom) > 0.001:
                    pct = round(num / denom * 100, 2)
                    yr_results[yr] = pct
            if yr_results:
                derived[target] = yr_results
                notes.append(f"{target} = ({num_lbl}/{denom_lbl}) × 100")
            continue

        matched_ratio = None
        for rule_label, (num_lbl, denom_lbl) in _RATIO_RULES.items():
            if rule_label.lower() == target_lower:
                matched_ratio = (num_lbl, denom_lbl)
                break

        if matched_ratio:
            num_lbl, denom_lbl = matched_ratio
            yr_results = {}
            for yr in years:
                num   = get_val(num_lbl, yr)
                denom = get_val(denom_lbl, yr)
                if num is not None and denom and abs(denom) > 0.001:
                    ratio = round(num / denom, 4)
                    yr_results[yr] = ratio
            if yr_results:
                derived[target] = yr_results
                notes.append(f"{target} = {num_lbl}/{denom_lbl}")
            continue

        # ── Identity rules ────────────────────────────────────────────────────
        for rule_target, components in _IDENTITY_RULES:
            if rule_target.lower() != target_lower or not components:
                continue

            yr_results = {}
            for yr in years:
                total = 0.0
                all_found = True
                for comp_lbl, sign in components:
                    v = get_val(comp_lbl, yr)
                    if v is None:
                        all_found = False
                        break
                    total += sign * v

                if all_found:
                    yr_results[yr] = round(total, 2)

            if yr_results:
                parts = []
                for i, (c, s) in enumerate(components):
                    if i == 0:
                        parts.append(c)
                    else:
                        parts.append(f"+ {c}" if s == 1 else f"− {c}")
                derived[target] = yr_results
                notes.append(f"{target} = {' '.join(parts)}")
                break   # use first matching rule

    not_derived = [t for t in target_labels if t not in derived]
    return {
        "derived": derived,
        "not_derived": not_derived,
        "notes": notes,
    }


def _tool_search_web(
    query: str,
    metric: str | None,
    nse_symbol: str = "",
) -> dict:
    """DuckDuckGo Lite search — no API key."""
    try:
        import urllib.request
        import urllib.parse
        import html

        q = urllib.parse.quote_plus(query)
        url = f"https://lite.duckduckgo.com/lite/?q={q}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (financial research bot)"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")

        # Extract text snippets
        text = re.sub(r"<[^>]+>", " ", raw)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()

        # Find sentences with numbers
        sentences = re.split(r"[.!?\n]", text)
        relevant = []
        kw = (metric or "").lower()
        for s in sentences:
            s = s.strip()
            if len(s) < 20:
                continue
            has_num = bool(re.search(r"\d", s))
            has_kw  = kw in s.lower() if kw else True
            if has_num and has_kw:
                relevant.append(s[:300])
            if len(relevant) >= 8:
                break

        return {
            "query": query,
            "results": relevant,
            "count": len(relevant),
        }
    except Exception as exc:
        logger.warning(f"web search failed: {exc}")
        return {"query": query, "results": [], "error": str(exc)}


def _tool_validate_and_correct(
    current_fills: dict[str, dict[str, float | None]],
    sheet_type: str,
    years: list[str],
) -> dict:
    """Cross-row validation and sanity checks."""
    issues = []

    def get_val(lbl: str, yr: str) -> float | None:
        lbl_l = lbl.lower()
        for k, yrv in current_fills.items():
            if k.lower() == lbl_l:
                v = yrv.get(yr)
                return float(v) if v is not None else None
        return None

    if sheet_type in ("income_statement", "other"):
        for yr in years:
            rev    = get_val("Revenue From Operations", yr) or get_val("Net Revenue", yr)
            ebitda = get_val("EBITDA", yr)
            ebit   = get_val("EBIT", yr)
            pbt    = get_val("PBT", yr)
            pat    = get_val("PAT", yr)

            if rev and ebitda and ebitda > rev:
                issues.append({
                    "label": "EBITDA", "year": yr,
                    "issue": f"EBITDA ({ebitda:,.1f}) > Revenue ({rev:,.1f}) — impossible",
                    "severity": "high"
                })
            if ebitda and ebit and ebit > ebitda:
                issues.append({
                    "label": "EBIT", "year": yr,
                    "issue": f"EBIT ({ebit:,.1f}) > EBITDA ({ebitda:,.1f}) — D&A must be positive",
                    "severity": "high"
                })
            if ebit and pbt and abs(pbt) > abs(ebit) * 2:
                issues.append({
                    "label": "PBT", "year": yr,
                    "issue": f"PBT ({pbt:,.1f}) seems very different from EBIT ({ebit:,.1f})",
                    "severity": "medium"
                })
            if pbt and pat and pat is not None:
                if pbt > 0 and pat > pbt:
                    issues.append({
                        "label": "PAT", "year": yr,
                        "issue": f"PAT ({pat:,.1f}) > PBT ({pbt:,.1f}) — tax can't be negative (unusual)",
                        "severity": "medium"
                    })

    if sheet_type == "balance_sheet":
        for yr in years:
            total_assets  = get_val("Total Assets", yr)
            total_liab    = get_val("Total Liabilities", yr)
            equity        = (get_val("Total Shareholders Equity", yr)
                             or get_val("Net Worth", yr))

            if total_assets and total_liab and equity:
                diff = abs(total_assets - (total_liab + equity))
                if diff > total_assets * 0.02:
                    issues.append({
                        "label": "Balance Sheet",
                        "year": yr,
                        "issue": f"BS doesn't balance: Assets={total_assets:,.1f}, "
                                 f"Liab+Equity={total_liab+equity:,.1f}, diff={diff:,.1f}",
                        "severity": "high"
                    })

    return {
        "issues_found": len(issues),
        "issues": issues,
        "sheet_type": sheet_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch tool calls
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch_tool(
    tool_name: str,
    tool_args: dict,
    pool: dict[str, dict[str, float]],
    pdf_text: str,
    nse_symbol: str,
    years: list[str],
) -> str:
    """Execute a tool call and return JSON string result."""
    t0 = time.time()
    try:
        if tool_name == "lookup_pool":
            result = _tool_lookup_pool(
                query_labels=tool_args.get("query_labels", []),
                years=tool_args.get("years", years),
                pool=pool,
            )

        elif tool_name == "search_pdf_pages":
            result = _tool_search_pdf_pages(
                labels=tool_args.get("labels", []),
                pdf_text=pdf_text,
                keywords=tool_args.get("keywords"),
            )

        elif tool_name == "derive_values":
            result = _tool_derive_values(
                target_labels=tool_args.get("target_labels", []),
                known_values=tool_args.get("known_values", {}),
                years=years,
            )

        elif tool_name == "search_web":
            result = _tool_search_web(
                query=tool_args.get("query", ""),
                metric=tool_args.get("metric"),
                nse_symbol=nse_symbol,
            )

        elif tool_name == "validate_and_correct":
            result = _tool_validate_and_correct(
                current_fills=tool_args.get("current_fills", {}),
                sheet_type=tool_args.get("sheet_type", "other"),
                years=years,
            )

        elif tool_name == "submit_values":
            # Handled in main loop — shouldn't reach here
            result = {"status": "submitted"}

        else:
            result = {"error": f"Unknown tool: {tool_name}"}

    except Exception as exc:
        logger.warning(f"Tool '{tool_name}' raised: {exc}", exc_info=True)
        result = {"error": str(exc)}

    elapsed = time.time() - t0
    logger.debug(f"tool '{tool_name}' completed in {elapsed:.2f}s")
    return json.dumps(result, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Sheet context helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_sheet_snapshot(
    template_labels: list[str],
    sheet_filled: dict[str, dict[str, float | None]],
    years: list[str],
    max_years: int = 4,
) -> str:
    show_years = years[:max_years]
    header = f"{'Label':<50} " + "  ".join(f"{yr:>10}" for yr in show_years)
    rows = [header, "-" * len(header)]
    for lbl in template_labels:
        yr_vals = sheet_filled.get(lbl, {})
        cells = []
        for yr in show_years:
            v = yr_vals.get(yr)
            cells.append(f"{v:>10,.1f}" if v is not None else f"{'—':>10}")
        rows.append(f"{lbl:<50} " + "  ".join(cells))
    return "\n".join(rows)


def _classify_sheet_type(sheet_name: str) -> str:
    sl = sheet_name.lower()
    if any(k in sl for k in ("p&l", "profit", "income", "pnl")):
        return "income_statement"
    if any(k in sl for k in ("balance", " bs", "bs ")):
        return "balance_sheet"
    if any(k in sl for k in ("cash", " cf", "cf ")):
        return "cash_flow"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_financial_agent(
    sheet_name:      str,
    sheet_filled:    dict[str, dict[str, float | None]],
    template_labels: list[str],
    years:           list[str],
    sector:          str = "General",
    pool:            dict[str, dict[str, float]] | None = None,
    pdf_text:        str = "",
    pdf_bytes:       bytes | None = None,
    nse_symbol:      str = "",
    client:          Any = None,
    model:           str = "",
    dropped_collector: Optional[list] = None,
    row_context:     Optional[dict[str, str]] = None,  # cleaned_label → "Consolidated / ASSETS / Non-Current Assets"
) -> tuple[dict[str, dict[str, float | None]], list[dict]]:
    """
    Phase 5: Tool-calling financial agent.

    The LLM drives the loop — it sees the partially filled sheet and
    decides WHICH tools to call (pool lookup, PDF search, derivation,
    web search, validation) and in WHAT ORDER until it submits the final values.

    Parameters
    ----------
    sheet_name      : Sheet name (e.g. "P&L", "Balance Sheet")
    sheet_filled    : Current fills after Phases 1-4
    template_labels : Ordered list of all template labels
    years           : Fiscal years e.g. ['FY2024', 'FY2023', 'FY2022', 'FY2021']
    sector          : Company sector
    pool            : Aggregated pool {label: {year: value}}
    pdf_text        : Extracted PDF text (full or keyword-filtered)
    pdf_bytes       : Raw PDF bytes (unused in this version, reserved)
    nse_symbol      : NSE ticker symbol
    client          : OpenAI-compatible client
    model           : Model name

    Returns
    -------
    (updated_sheet_filled, agent_log)
    agent_log : [{iter, tool, args_summary, result_summary, timestamp}]
    """
    pool = pool or {}
    agent_log: list[dict] = []

    # Sheet type informs the prompt but no longer gates execution. The agent
    # now runs on Operating Metrics, Quarterly, and company-specific sheets too
    # — segment data and operational KPIs often need to be pulled from PDF text
    # rather than structured connectors.
    sheet_type = _classify_sheet_type(sheet_name)
    if client is None:
        logger.info(f"financial_agent: skipping '{sheet_name}' (no client)")
        return sheet_filled, []

    # Work on a copy
    result = {lbl: dict(yr_vals) for lbl, yr_vals in sheet_filled.items()}

    # ── Find labels with ANY missing year (not just fully-empty ones) ─────────
    # BUG FIX: previously used `all(v is None ...)` which skipped labels that were
    # partially filled (e.g. EBIT with FY2024 filled but FY2023/2022/2021 = None).
    # Now we catch every label that has at least one missing year, and tell the LLM
    # exactly which (label, year) pairs need filling.
    labels_with_gaps: list[str] = []
    gap_detail: dict[str, list[str]] = {}   # label → [years that are None]

    for lbl in template_labels:
        yr_vals = result.get(lbl, {})
        missing_yrs = [yr for yr in years if yr_vals.get(yr) is None]
        if missing_yrs:
            labels_with_gaps.append(lbl)
            gap_detail[lbl] = missing_yrs

    if not labels_with_gaps:
        logger.info(f"financial_agent: '{sheet_name}' already fully filled — skipping")
        return result, []

    # Cap to avoid context bloat — prioritise fully-empty labels first,
    # then partially-filled ones (they're harder to mess up)
    fully_empty   = [l for l in labels_with_gaps
                     if all(v is None for v in result.get(l, {}).values())]
    partially_empty = [l for l in labels_with_gaps if l not in fully_empty]
    priority_order  = fully_empty + partially_empty
    priority_order  = priority_order[:MAX_LABELS]

    logger.info(
        f"financial_agent: '{sheet_name}' starting — "
        f"{len(fully_empty)} fully empty + {len(partially_empty)} partial = "
        f"{len(labels_with_gaps)} labels with gaps, pool={len(pool)} labels"
    )

    # Build gap summary for LLM — show exactly which years are missing per label
    # PLUS the KB aliases (so the agent knows what variant words the PDF uses).
    # Critical: template says "Less: Cost of goods sold" but PDF says "Cost of
    # materials consumed". Without aliases the agent has to guess.
    try:
        from pipeline.kb_loader import kb_entry_for as _kb_lookup
    except Exception:
        _kb_lookup = lambda lbl, sec: None  # noqa: E731

    row_context = row_context or {}

    gap_lines = []
    for lbl in priority_order:
        missing = gap_detail.get(lbl, [])
        filled  = [yr for yr in years if result.get(lbl, {}).get(yr) is not None]
        # Pull KB aliases — what the PDF probably calls this concept
        try:
            entry = _kb_lookup(lbl, sector)
        except Exception:
            entry = None
        alias_hint = ""
        if entry:
            aliases = entry.get("aliases") or []
            flat: list[str] = []
            if isinstance(aliases, dict):
                for v in aliases.values():
                    if isinstance(v, list):
                        flat.extend(v)
            elif isinstance(aliases, list):
                flat = aliases
            flat = [a for a in flat if isinstance(a, str)]
            if flat:
                shown = flat[:4]
                alias_hint = f"  ⟶ PDF may say: {', '.join(repr(a) for a in shown)}"
        # NEW: section breadcrumb tells the agent WHERE in the PDF to look
        # ("Consolidated > ASSETS > Non-Current Assets" → consolidated balance
        # sheet's non-current assets section)
        ctx = row_context.get(lbl) or row_context.get(lbl.lower())
        ctx_hint = f"  ⟸ in [{ctx}]" if ctx else ""

        if filled:
            line = f"  - {lbl}  [have: {', '.join(filled)}  MISSING: {', '.join(missing)}]"
        else:
            line = f"  - {lbl}  [ALL years missing: {', '.join(missing)}]"
        if ctx_hint:
            line += ctx_hint
        if alias_hint:
            line += alias_hint
        gap_lines.append(line)

    # Build initial context
    snapshot = _build_sheet_snapshot(template_labels, result, years)
    gap_str  = "\n".join(gap_lines)
    yrs_str  = ", ".join(years[:4])

    pdf_hint = ("PDF text is available — search_pdf_pages is your strongest tool. "
                if pdf_text else
                "No PDF text available — rely on lookup_pool, derivation, and web search. ")

    system_prompt = f"""You are an Ind AS / IGAAP financial analyst filling the
"{sheet_name}" sheet for {nse_symbol or "the company"} (sector: {sector}).
All values are INR Crores. Fiscal years: {yrs_str}.

────────────────────────────────────────────────────────────────────────
LABEL-DRIVEN PROTOCOL (you MUST follow this)
────────────────────────────────────────────────────────────────────────
The user has given you a list of TEMPLATE LABELS that are missing values.
Your job is simple: walk EVERY label in that list, find its value, submit it.

For EACH label in the list:
 1. Read THREE pieces of context together:
       (a) the SHEET the label is on (e.g. "Annual Balance Sheet",
           "Annual P&L", "Quarterly", "Operating metrics")
       (b) the SECTION breadcrumb shown after "⟸ in [...]"
           (e.g. "Consolidated > ASSETS > Non-Current Assets")
       (c) the label itself ("Borrowings", "PAT", "EBITDA", "- CDMO")
    Together these tell you WHERE in the PDF to look. Example:
       Label: "Borrowings"  in [Consolidated / Liabilities / Current Liabilities]
       → search the consolidated balance sheet, current-liabilities section,
         for short-term borrowings (NOT long-term — that's a different row)
 2. Use search_pdf_pages with keywords drawn from the section breadcrumb +
    the label + (if shown) the "PDF may say" alias hints. The PDF's table of
    contents and section headers will narrow the page range fast.
 3. If the value isn't directly stated, REASON to compute it:
       - Total Non-Current Assets = sum of all NCA sub-items above the row
       - Gross Profit = Revenue − COGS
       - EBITDA = Gross Profit − Employee − Other Expenses
       - PAT = PBT − Tax
       - Margin/% rows: derive_values once components are filled
 4. Pick the BEST tool for THIS label:
    - {pdf_hint}
    - Already-known close synonym in pool? lookup_pool first.
    - Pure derivation (totals / margins / ratios)? derive_values.
    - Last resort: search_web.
 5. Note the value(s) you find for the missing year(s) of this label.

────────────────────────────────────────────────────────────────────────
TOOLS
────────────────────────────────────────────────────────────────────────
• lookup_pool          — fuzzy match in the multi-source data pool
• search_pdf_pages     — keyword-search the company's annual report PDF text
• derive_values        — apply accounting identities (Gross Profit = Rev − COGS,
                         EBIT = EBITDA − D&A, sub-totals, ratios, margins)
• search_web           — DuckDuckGo last-resort web search
• validate_and_correct — sanity check the values you've decided on
• submit_values        — commit ALL fills + END the session (REQUIRED)

────────────────────────────────────────────────────────────────────────
HARD RULES
────────────────────────────────────────────────────────────────────────
1. You MUST attempt EVERY label in the missing-list. Don't skip any.
2. Use the EXACT template label string (with "Less:" / "Add:" / "(Net)" etc.)
   when you submit_values. The system has fuzzy resolution but exact is best.
3. For PARTIAL labels, submit only the MISSING years. Don't repeat filled years.
4. Confidence ≥ 0.65 to be accepted. Be honest — if unsure, set lower.
5. Cross-row sanity: components ≤ totals; PAT < PBT < EBIT (usually);
   magnitudes consistent year-over-year (no 10× swings).
6. Call submit_values at the end with EVERY fill you decided on.
"""

    # Provide the EXACT template-label strings + valid years as a JSON contract.
    # The LLM tends to paraphrase otherwise ("Tax" vs "Less: Tax") and fills get
    # dropped by the orchestrator's resolver.
    label_contract = json.dumps(priority_order, ensure_ascii=False)
    year_contract  = json.dumps(years, ensure_ascii=False)

    user_message = f"""CURRENT SHEET SNAPSHOT — "{sheet_name}":
{snapshot}

LABELS YOU MUST FILL ({len(priority_order)} labels with missing year(s)):
{gap_str}

═══════════════════════════════════════════════════════════════════
LABEL CONTRACT — when you call submit_values, the "label" field MUST
be copied VERBATIM from this list (including any "Less:", "Add:",
"(Net)" prefixes/suffixes — copy them exactly):
{label_contract}

VALID YEARS (use these strings, no others):
{year_contract}
═══════════════════════════════════════════════════════════════════

EXECUTION (you have ~10 LLM turns total — budget tightly):
Step A: ONE tool call to gather all the data you need. Prefer search_pdf_pages
        with multiple labels at once (e.g. labels=["Tax","Finance cost","COGS"])
        not separate calls per label. If pool likely has it, lookup_pool with
        all labels at once.
Step B: At MOST 2 more tool calls if Step A wasn't enough. derive_values for
        any totals/margins.
Step C: submit_values WITH EVERYTHING YOU FOUND. Don't keep searching — submit
        partial results rather than nothing. An empty fills array helps NOBODY.

🚨 HARD RULE: by your 4th LLM turn, you MUST call submit_values. If you reach
turn 7 without submitting, the system will force-end with whatever you have.
ALWAYS submit even if you only found values for 5 of 15 labels. Partial fills
are infinitely better than zero fills.

EXAMPLE OF A CORRECT submit_values CALL:
{{"fills": [
  {{"label": "Less: Tax",                "year": "{years[0] if years else 'F2024'}",  "value": 100.0, "confidence": 0.85, "source": "PDF income statement"}},
  {{"label": "Add: Other Income (Net)",  "year": "{years[0] if years else 'F2024'}",  "value":  50.0, "confidence": 0.90, "source": "PDF other income note"}}
]}}

Walk every label in the contract. Submit your fills at the end."""

    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_message},
    ]

    final_fills       : list[dict] = []
    final_corrections : list[dict] = []
    goto_submit       = False

    # ── Agent loop (time-budgeted) ───────────────────────────────────────────
    loop_start = time.time()
    for iteration in range(MAX_ITER):
        elapsed = time.time() - loop_start
        if elapsed > TIME_BUDGET_SEC:
            logger.warning(
                f"financial_agent: TIME BUDGET EXCEEDED "
                f"({elapsed:.0f}s > {TIME_BUDGET_SEC:.0f}s) — stopping early"
            )
            break
        logger.info(
            f"financial_agent iteration {iteration+1}/{MAX_ITER} "
            f"(elapsed {elapsed:.0f}s)"
        )

        # ── Force-submit reminder near end of budget ──────────────────────────
        # Agent has a habit of searching for too long without submitting. When
        # we're approaching MAX_ITER, inject a system message that REQUIRES
        # submit_values on the next turn so its work isn't wasted.
        if iteration >= FORCE_SUBMIT_AT_ITER:
            remaining = MAX_ITER - iteration
            messages.append({
                "role": "user",
                "content": (
                    f"⚠️ ITERATION {iteration+1}/{MAX_ITER}: only {remaining} "
                    f"turns left. You MUST call submit_values on your NEXT "
                    f"response with EVERY value you've found so far. Do NOT "
                    f"call any other tool — submit what you have or you'll "
                    f"lose all the work you've done. Empty fills array is "
                    f"NOT acceptable; submit your best guesses with their "
                    f"confidence scores."
                ),
            })

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                temperature=0,
            )
        except Exception as exc:
            logger.warning(f"financial_agent: LLM call failed at iter {iteration}: {exc}")
            break

        msg = response.choices[0].message

        # Append assistant message
        messages.append({
            "role":       "assistant",
            "content":    msg.content or "",
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in (msg.tool_calls or [])
            ] or None,
        })

        # No tool calls → LLM is done (text-only response)
        if not msg.tool_calls:
            logger.info(f"financial_agent: no tool calls at iter {iteration} — stopping")
            break

        # Process each tool call
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            # Log entry — "summary" key matches UI display in web_only_app.py
            log_entry = {
                "iteration": iteration + 1,
                "tool":      tool_name,
                "args_summary": _summarise_args(tool_name, tool_args),
                "summary":   _summarise_args(tool_name, tool_args),  # UI compat
                "timestamp": time.strftime("%H:%M:%S"),
            }

            # ── submit_values terminates the loop ─────────────────────────────
            if tool_name == "submit_values":
                final_fills       = tool_args.get("fills", [])
                final_corrections = tool_args.get("corrections", [])
                submit_summary = (
                    f"Submitted {len(final_fills)} fills, "
                    f"{len(final_corrections)} corrections"
                )
                log_entry["result_summary"] = submit_summary
                log_entry["summary"] = submit_summary
                agent_log.append(log_entry)
                logger.info(
                    f"financial_agent: submit_values called — "
                    f"{len(final_fills)} fills, {len(final_corrections)} corrections"
                )
                # Append tool result for submit_values
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"status": "submitted", "fills_accepted": len(final_fills)}),
                })
                # BUG FIX: OpenAI requires a tool_result for EVERY tool_call in the
                # assistant message. If submit_values fires mid-batch (other tools follow
                # it in the same response), append stub results for all remaining calls
                # so the conversation stays valid if the loop ever continues.
                remaining_tcs = [
                    t for t in msg.tool_calls
                    if t.id != tc.id and not any(
                        m.get("tool_call_id") == t.id
                        for m in messages
                        if m.get("role") == "tool"
                    )
                ]
                for rtc in remaining_tcs:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": rtc.id,
                        "content": json.dumps({"status": "skipped", "reason": "submit_values already called"}),
                    })
                goto_submit = True
                break

            # ── Execute tool ──────────────────────────────────────────────────
            result_json = _dispatch_tool(
                tool_name=tool_name,
                tool_args=tool_args,
                pool=pool,
                pdf_text=pdf_text,
                nse_symbol=nse_symbol,
                years=years,
            )
            result_summary = _summarise_result(tool_name, result_json)
            log_entry["result_summary"] = result_summary
            log_entry["summary"] = f"{log_entry['args_summary']} → {result_summary}"
            agent_log.append(log_entry)

            # Append tool result message
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result_json,
            })

        if goto_submit:
            break

    # ── Resolve agent label → template label (operator-prefix tolerant) ──────
    # Build a fuzzy lookup: normalised template label → actual template label.
    # This catches the common case where the agent returns "Tax" but the
    # template uses "Less: Tax", or "Other Income" vs "Add: Other Income (Net)".
    def _norm_label(s: str) -> str:
        s = (s or "").strip().lower()
        # Strip operator prefixes
        for p in ("less:", "add:", "plus:", "minus:"):
            if s.startswith(p):
                s = s[len(p):].strip()
        # Strip noise suffixes
        for suf in ("(net)", "(gross)", "(consolidated)", "(standalone)"):
            if s.endswith(suf):
                s = s[:-len(suf)].strip()
        # Collapse non-alphanumeric to single spaces
        return re.sub(r"[^a-z0-9]+", " ", s).strip()

    norm_to_template: dict[str, str] = {}
    for tlbl in result.keys():
        n = _norm_label(tlbl)
        norm_to_template.setdefault(n, tlbl)

    # KB-canonical lookup — the most reliable resolver. "Tax expense" and
    # "Less: Tax" both resolve to canonical `tax_expense` so they match.
    try:
        from pipeline.kb_loader import kb_entry_for as _kb_lookup
        from pipeline.kb_loader import load_merged_kb as _kb_all
    except Exception:
        _kb_lookup = lambda lbl, sec: None  # noqa: E731
        _kb_all = lambda sec: {}            # noqa: E731

    # Pre-compute canonical for each TEMPLATE label (cheap, one-time)
    template_canon: dict[str, str] = {}    # canonical_key → template_label
    canon_to_label: dict[str, str] = {}    # used by agent
    for tlbl in result.keys():
        ent = _kb_lookup(tlbl, sector)
        if ent:
            kb = _kb_all(sector)
            for ck, info in kb.items():
                if info is ent:
                    canon_to_label.setdefault(ck, tlbl)
                    break

    def _resolve_agent_label(agent_lbl: str) -> Optional[str]:
        """Map an agent-submitted label to the template's actual label.

        Priority:
          1. Exact string match in template
          2. Operator-stripped exact match
          3. KB-canonical match (BIG ONE — "Tax expense" → tax_expense ↔
             template "Less: Tax" → tax_expense)
          4. Containment fallback
        """
        if not agent_lbl:
            return None
        if agent_lbl in result:
            return agent_lbl
        n = _norm_label(agent_lbl)
        if not n:
            return None
        if n in norm_to_template:
            return norm_to_template[n]

        # KB canonical — the resolver that actually fixes label drift
        agent_entry = _kb_lookup(agent_lbl, sector)
        if agent_entry is not None:
            kb_dict = _kb_all(sector)
            for ck, info in kb_dict.items():
                if info is agent_entry and ck in canon_to_label:
                    return canon_to_label[ck]

        # Containment fallback
        agent_tokens = set(n.split())
        if not agent_tokens:
            return None
        for tn, tlbl in norm_to_template.items():
            tt = set(tn.split())
            shared = agent_tokens & tt
            if not shared:
                continue
            # Substring match with at least one shared token
            if (n in tn or tn in n) and len(shared) >= 1:
                return tlbl
            # Strong overlap (≥2 shared, OR shared/min ≥0.5)
            if len(shared) >= 2 or (
                len(shared) >= 1 and len(shared) / max(min(len(agent_tokens), len(tt)), 1) >= 0.5
            ):
                return tlbl
        return None

    # ── Apply fills ───────────────────────────────────────────────────────────
    fills_applied = 0
    fills_dropped: list[tuple[str, str]] = []
    for fill in final_fills:
        lbl_raw = fill.get("label", "")
        yr      = fill.get("year", "")
        val     = fill.get("value")
        conf    = float(fill.get("confidence", 0))

        if not lbl_raw or not yr or val is None:
            fills_dropped.append((lbl_raw, "missing label/year/value"))
            continue
        if conf < MIN_CONF:
            fills_dropped.append((lbl_raw, f"low confidence {conf:.2f} < {MIN_CONF}"))
            continue
        lbl = _resolve_agent_label(lbl_raw)
        if not lbl:
            fills_dropped.append((lbl_raw, "no template-row match"))
            continue
        # Sign-flip rule: agent might submit cash-flow-style negatives that
        # belong as positive magnitudes on a BS/P&L row.
        try:
            fv = float(val)
        except (TypeError, ValueError):
            fills_dropped.append((lbl_raw, "value not numeric"))
            continue
        tlbl_l = (lbl or "").lower()
        # Allow legitimate negatives only for profit/loss rows (a loss IS negative)
        _is_loss_eligible = any(k in tlbl_l for k in (
            "pat", "pbt", "profit", "earnings", "eps", "diluted",
            "exceptional", "gain", "loss", "comprehensive income",
        ))
        if fv < 0 and abs(fv) > 0.5 and not _is_loss_eligible:
            fv = abs(fv)
        # Only fill empty cells
        if result[lbl].get(yr) is None:
            result[lbl][yr] = fv
            fills_applied += 1
        else:
            fills_dropped.append((lbl_raw, "cell already filled"))

    if fills_dropped:
        from collections import Counter
        reasons = Counter(r for _, r in fills_dropped)
        logger.info(
            f"financial_agent '{sheet_name}': dropped "
            f"{len(fills_dropped)} fills — "
            + ", ".join(f"{r}={n}" for r, n in reasons.most_common())
        )
        # Surface for the orchestrator to display in UI
        if dropped_collector is not None:
            for lbl_raw, reason in fills_dropped:
                # Find the year from the original fill if we can
                yr = ""
                for f in final_fills:
                    if f.get("label") == lbl_raw:
                        yr = f.get("year", "")
                        break
                dropped_collector.append({
                    "sheet":  sheet_name,
                    "label_submitted": lbl_raw,
                    "year":   yr,
                    "reason": reason,
                })

    # ── Apply corrections ─────────────────────────────────────────────────────
    corrections_applied = 0
    for corr in final_corrections:
        lbl_raw  = corr.get("label", "")
        yr       = corr.get("year", "")
        new_val  = corr.get("new_value")
        conf     = float(corr.get("confidence", 0))

        if not lbl_raw or not yr or new_val is None or conf < 0.80:
            continue
        lbl = _resolve_agent_label(lbl_raw)
        if not lbl:
            continue
        old_val = result[lbl].get(yr)
        if old_val is not None and abs(float(new_val) - float(old_val)) > 0.01:
            result[lbl][yr] = float(new_val)
            corrections_applied += 1
            logger.info(
                f"financial_agent correction: '{lbl}' {yr} "
                f"{old_val} → {float(new_val)}"
            )

    logger.info(
        f"financial_agent '{sheet_name}': "
        f"{fills_applied} fills applied, {corrections_applied} corrections, "
        f"{len(agent_log)} tool calls"
    )

    return result, agent_log


# ─────────────────────────────────────────────────────────────────────────────
# Log summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def _summarise_args(tool_name: str, args: dict) -> str:
    if tool_name == "lookup_pool":
        lbls = args.get("query_labels", [])
        return f"{len(lbls)} labels: {', '.join(lbls[:3])}{'…' if len(lbls)>3 else ''}"
    if tool_name == "search_pdf_pages":
        lbls = args.get("labels", [])
        return f"labels: {', '.join(lbls[:3])}"
    if tool_name == "derive_values":
        lbls = args.get("target_labels", [])
        return f"targets: {', '.join(lbls[:4])}"
    if tool_name == "search_web":
        return args.get("query", "")[:80]
    if tool_name == "validate_and_correct":
        return f"sheet_type={args.get('sheet_type','?')}"
    if tool_name == "submit_values":
        n = len(args.get("fills", []))
        c = len(args.get("corrections", []))
        return f"{n} fills, {c} corrections"
    return str(args)[:80]


def _summarise_result(tool_name: str, result_json: str) -> str:
    try:
        r = json.loads(result_json)
    except Exception:
        return result_json[:120]

    if tool_name == "lookup_pool":
        found = r.get("found", {})
        not_found = r.get("not_found", [])
        return f"{len(found)} found, {len(not_found)} not found"
    if tool_name == "search_pdf_pages":
        n = r.get("chunks_found", 0)
        return f"{n} relevant PDF chunks"
    if tool_name == "derive_values":
        d = r.get("derived", {})
        nd = r.get("not_derived", [])
        return f"{len(d)} derived: {', '.join(list(d.keys())[:3])}; {len(nd)} not derivable"
    if tool_name == "search_web":
        n = r.get("count", 0)
        return f"{n} relevant snippets"
    if tool_name == "validate_and_correct":
        n = r.get("issues_found", 0)
        return f"{n} issues found"
    return str(r)[:120]
