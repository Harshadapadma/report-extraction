"""
pipeline/page_batch_fill.py
───────────────────────────
Final automated extraction layer: per-source-page batched LLM calls.

Architecture:
  1. Pattern matcher runs first (free) — fills ~210 cells.
  2. For each empty (sheet, statement_type, year) section in the template,
     use the TOC to find THE matching source page in the AR/QR.
  3. Send ONE LLM call per (source page, template section): the page's
     text + the empty template rows. LLM returns all values at once.
  4. Write fills back. Cell-lock ensures pattern fills aren't overwritten.

Cost target: ~$0.05/run. ~12 LLM calls × ~5K tokens each.

DOES NOT touch llm_fill.py or rag_fill.py. Standalone module.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pipeline.llm_fill import (
    _pdf_to_markdown, _classify_pdf, _pattern_fill,
    _read_template_specs, _write_filled_excel, _build_toc,
    PDFClassification, TemplateRowSpec,
)

logger = logging.getLogger(__name__)


@dataclass
class PageBatchResult:
    output_path: str = ""
    fills: list[dict] = field(default_factory=list)
    cells_written: int = 0
    by_confidence: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    pdfs_used: list[dict] = field(default_factory=list)
    n_llm_calls: int = 0
    est_cost_usd: float = 0.0
    elapsed_sec: float = 0.0


_SYSTEM_PROMPT = """You are a senior equity research analyst extracting values from a SINGLE source page into a template section.

You will be given:
  • One source page from a PDF (e.g. "Standalone Balance Sheet" or "Consolidated P&L")
  • A list of template rows to fill, each with year(s) requested

Your job: read the page, find each template row's value(s) for the requested year(s), return JSON.

OUTPUT SCHEMA:
{
  "fills": [
    {"label": "<exact template label>", "year": "<F2024 / 1QF2025>", "value": <number>, "confidence": "HIGH|MEDIUM|LOW"},
    ...
  ]
}

LABEL MATCHING (template ≡ source):
  "Revenue from operations" ≡ "Net sales" ≡ "Sales" ≡ "Turnover"
  "Less: Cost of goods sold" = "Cost of materials" + "Purchases of stock" + "Changes in inventories"
  "Employee benefits expense" ≡ "Staff cost" ≡ "Personnel cost"
  "Less: Finance cost" ≡ "Finance costs" ≡ "Interest expense"
  "Less: Depreciation, amortization & impairment" ≡ "Depreciation and amortisation"
  "Less: Tax" = "Current tax" + "Deferred tax" (sum if components shown)
  "Add: Other Income (Net)" ≡ "Other income"
  "Property, Plant & Equipment" ≡ "PPE" ≡ "Tangible assets" (Net Block)
  "Right of Use Assets" ≡ "Right-of-use assets"

CRITICAL RULES:
  • Values in INR Crores. Lakhs ÷ 100, Millions ÷ 10.
  • Bracketed numbers are negative: (40.90) → -40.90.
  • DERIVE values when components are shown but the total isn't.
  • Skip template rows where the value isn't present on this page AND can't be derived from this page.
  • Year columns on the source page map to template years via the column headers.

Return ONLY the JSON. No prose."""


def _is_qtr(year: str) -> bool:
    return bool(re.match(r"^[1-4]QF\d{4}$", year or ""))


def _extract_pages_with_titles(md_text: str, cls: PDFClassification
                                ) -> dict[int, tuple[str, str]]:
    """Return {page_num_0indexed: (page_text, title_from_toc)}."""
    pages: dict[int, str] = {}
    chunks = re.split(r"(?m)^## Page (\d+)\n", md_text)
    for k in range(1, len(chunks) - 1, 2):
        try:
            pg_num = int(chunks[k]) - 1
            pages[pg_num] = chunks[k + 1]
        except ValueError:
            continue
    out = {}
    toc = getattr(cls, "toc", None) or []
    toc_lookup = {pn - 1: title for pn, title, _scope in toc}
    for pn, txt in pages.items():
        title = toc_lookup.get(pn, "")
        out[pn] = (txt, title)
    return out


def _route_template_section(sheet_name: str, spec_label: str
                              ) -> Optional[str]:
    """Decide which source-page TYPE this template row needs."""
    sn = sheet_name.lower()
    lbl = (spec_label or "").lower()
    if "cash flow" in sn or "cashflow" in sn:
        return "cash_flow"
    if "balance" in sn or "bs" in sn.split():
        return "balance_sheet"
    if "quarterly" in sn or "quarter" in sn:
        return "quarterly"
    if "operating" in sn or "metric" in sn or "segment" in sn or "ratio" in sn:
        return "operating_metrics"
    # Default to P&L for Annual P&L / Income Statement
    return "p_and_l"


def _page_matches_section(page_title: str, section_type: str) -> bool:
    pt = (page_title or "").lower()
    if section_type == "balance_sheet":
        return "balance sheet" in pt or "financial position" in pt
    if section_type == "p_and_l":
        return ("statement of p&l" in pt or "p & l" in pt
                or "statement of profit" in pt or "p&l" in pt)
    if section_type == "cash_flow":
        return "cash flow" in pt
    if section_type == "quarterly":
        return "quarter" in pt or "three months" in pt
    if section_type == "operating_metrics":
        return ("segment" in pt or "kpi" in pt or "operating" in pt
                or "ratio" in pt or "schedule" in pt)
    return False


def _pick_source_pages(pdf_data: list[tuple[PDFClassification, str]],
                       sheet_name: str, stmt_type: str,
                       years: list[str]) -> list[tuple[str, int, str]]:
    """For a template section (sheet/stmt/years), return list of
    (pdf_name, page_idx, page_text) that LIKELY contain the values."""
    section_type = _route_template_section(sheet_name, "")
    out = []
    target_stmt = (stmt_type or "").upper()
    # For each PDF, find pages whose TOC title matches the section type
    # AND whose scope matches the spec's statement_type
    for cls, md_text in pdf_data:
        pages_dict = _extract_pages_with_titles(md_text, cls)
        toc = getattr(cls, "toc", None) or []
        for page_num_1, title, scope in toc:
            page_idx = page_num_1 - 1
            if scope == "MD&A/Pre" or scope == "Post-Financials":
                continue
            # Filter by scope
            if target_stmt in ("STANDALONE", "CONSOLIDATED"):
                if scope == "UNKNOWN_SCOPE" or scope == target_stmt:
                    pass
                else:
                    continue
            # Filter by section type
            if not _page_matches_section(title, section_type):
                continue
            txt, _ = pages_dict.get(page_idx, ("", ""))
            if not txt.strip():
                continue
            out.append((cls.name, page_idx, txt))
    return out


def _extract_via_llm(llm_client, model: str,
                      source_pages: list[tuple[str, int, str]],
                      template_rows: list[TemplateRowSpec],
                      years: list[str]) -> list[dict]:
    """Send one LLM call: source page(s) + template rows → fills.
    If multiple source_pages, concatenate them but cap total at ~12K chars."""
    if not source_pages or not template_rows:
        return []
    # Build the user message
    parts = ["# SOURCE PAGES\n"]
    used_chars = 0
    for pdf_name, page_idx, txt in source_pages:
        if used_chars > 12_000:
            break
        snip = txt[:6000]
        parts.append(f"\n## {pdf_name} page {page_idx+1}\n{snip}\n")
        used_chars += len(snip)
    parts.append("\n# TEMPLATE ROWS TO FILL\n")
    parts.append(f"Years requested: {', '.join(years)}\n\n")
    for r in template_rows:
        parts.append(f"  - {r.label} [{r.statement_type or '-'}]\n")
    parts.append("\nReturn JSON: {\"fills\": [{label, year, value, confidence}]}\n")
    user_msg = "\n".join(parts)

    try:
        resp = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content or "{}"
        obj = json.loads(raw)
        return obj.get("fills", [])
    except Exception as exc:
        logger.warning(f"_extract_via_llm failed: {exc}")
        return []


def page_batch_fill_template(
    template_path: str,
    pdf_paths:     list[str],
    output_path:   str = "",
    api_key:       str = "",
    model:         str = "deepseek-chat",
    base_url:      str = "https://api.deepseek.com",
) -> PageBatchResult:
    t0 = time.time()
    res = PageBatchResult(output_path=output_path)

    # Step 1: classify + extract md
    pdf_data: list[tuple[PDFClassification, str]] = []
    for pp in pdf_paths:
        try:
            cls, _txt = _classify_pdf(pp)
            md = _pdf_to_markdown(pp)
            pdf_data.append((cls, md))
            res.pdfs_used.append({"name": cls.name, "stmt_type": cls.stmt_type,
                                   "period_type": cls.period_type,
                                   "pages": cls.pages_total})
        except Exception as exc:
            res.warnings.append(f"PDF prep failed for {pp}: {exc}")

    if not pdf_data:
        res.warnings.append("No usable PDFs.")
        res.elapsed_sec = time.time() - t0
        return res

    # Step 2: pattern matcher pre-pass (free)
    specs = _read_template_specs(template_path)
    pattern_fills = _pattern_fill(specs, pdf_data)
    res.fills.extend(pattern_fills)
    logger.info(f"page_batch_fill: pattern pre-pass = {len(pattern_fills)} fills")
    res.warnings.append(
        f"Pattern pre-pass filled {len(pattern_fills)} cells FREE"
    )
    # Track pattern-filled cells so LLM skips them
    def _key(f): return (f.get("sheet"), (f.get("label") or "").lower().strip(),
                          f.get("year"),
                          (f.get("statement_type") or "").lower())
    pattern_filled_keys = {_key(f) for f in pattern_fills}

    # Step 3: group remaining specs by (sheet, stmt_type), pick source pages,
    # batch-LLM-call per group
    from openai import OpenAI
    try:
        llm_client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as exc:
        res.warnings.append(f"Could not init LLM: {exc}")
        res.elapsed_sec = time.time() - t0
        return res

    # Group specs by (sheet, statement_type)
    by_section: dict[tuple[str, str], list[TemplateRowSpec]] = {}
    for spec in specs:
        if spec.is_formula:
            continue
        # Filter out years already pattern-filled
        remaining_years = [
            y for y in spec.years_needed
            if (spec.sheet, spec.label.lower().strip(), y,
                (spec.statement_type or "").lower()) not in pattern_filled_keys
        ]
        if not remaining_years:
            continue
        new_spec = TemplateRowSpec(
            sheet=spec.sheet, statement_type=spec.statement_type,
            section=spec.section, label=spec.label,
            years_needed=remaining_years, is_formula=False,
        )
        by_section.setdefault((spec.sheet, spec.statement_type), []).append(new_spec)

    n_calls = 0
    for (sheet, stmt), rows in by_section.items():
        # Gather all years across these rows
        all_years = sorted(set(y for r in rows for y in r.years_needed))
        # Pick source pages for this section
        source_pages = _pick_source_pages(pdf_data, sheet, stmt, all_years)
        if not source_pages:
            res.warnings.append(
                f"No source page found for {sheet}/{stmt} — skipping"
            )
            continue
        # Cap to top 3 pages (most-relevant) to keep prompt small
        source_pages = source_pages[:3]
        logger.info(f"page_batch_fill: LLM call for {sheet}/{stmt} "
                    f"({len(rows)} rows × {len(all_years)} years, "
                    f"{len(source_pages)} source pages)")
        fills = _extract_via_llm(llm_client, model, source_pages, rows, all_years)
        # Tag fills with sheet + statement_type
        for f in fills:
            f["sheet"] = sheet
            f["statement_type"] = stmt or f.get("statement_type", "")
            f["source"] = f"page_batch:{','.join(p[0] for p in source_pages)}"
            res.fills.append(f)
        n_calls += 1

    res.n_llm_calls = n_calls
    # Estimate cost (very rough): N calls × 5K input + 1K output
    res.est_cost_usd = n_calls * (5000 * 0.27 + 1000 * 1.10) / 1_000_000
    res.warnings.append(
        f"Page-batch LLM: {n_calls} calls, est cost ~${res.est_cost_usd:.3f}"
    )

    # Step 4: write Excel via existing writer (cell-lock prevents overwrite)
    if not output_path:
        output_path = str(Path(template_path).with_name(
            Path(template_path).stem + "_page_batch_filled.xlsx"
        ))
        res.output_path = output_path
    stats = _write_filled_excel(template_path, output_path, res.fills)
    res.cells_written = stats["written"]
    res.by_confidence = stats["by_confidence"]
    res.elapsed_sec = time.time() - t0
    return res


__all__ = ["page_batch_fill_template", "PageBatchResult"]
