"""
LLM Extractor — DeepSeek-powered financial data extraction.

Speed optimisations vs original
---------------------------------
1. Single combined LLM call per document instead of one call per section.
   All detected sections are concatenated and sent together, so DeepSeek
   extracts P&L + Balance Sheet + Cash Flow in ONE round-trip.
2. Text is pre-trimmed to the most financially dense pages only
   (already done by pdf_parser), so token counts are small.
3. Concurrent extraction across multiple PDFs via ThreadPoolExecutor.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# ── Guarded openai import ─────────────────────────────────────────────────────
def _require_openai():
    try:
        from openai import OpenAI as _C
        return _C
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "openai>=1.30.0"]
        )
        import importlib, site
        importlib.invalidate_caches()
        for sp in site.getsitepackages():
            if sp not in sys.path:
                sys.path.insert(0, sp)
        from openai import OpenAI as _C
        return _C

OpenAI = _require_openai()

from config.settings import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from utils.helpers import get_logger, chunk_text, clean_for_llm, normalise_year, parse_number

logger = get_logger(__name__)

# ── Debug: store raw LLM responses so the UI can surface them ─────────────────
# Cleared at the start of each extract_all_sections call.
_last_raw_responses: list[dict] = []

def get_debug_responses() -> list[dict]:
    """Return raw LLM response snapshots from the last extraction run."""
    return list(_last_raw_responses)

def clear_debug_responses():
    global _last_raw_responses
    _last_raw_responses.clear()


# ── Client ────────────────────────────────────────────────────────────────────

def _get_client(api_key: str | None = None) -> OpenAI:
    # Always read from env at call-time so Streamlit sidebar changes take effect
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. Please enter it in the sidebar or .env file."
        )
    base_url = os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL)
    return OpenAI(api_key=key, base_url=base_url)


# ── Result class ──────────────────────────────────────────────────────────────

class ExtractionResult:
    def __init__(self, section, statement_type, year, currency, data,
                 notes="", confidence=1.0):
        self.section = section
        self.statement_type = statement_type
        self.year = year
        self.currency = currency
        self.data: dict[str, float] = data
        self.notes = notes
        self.confidence = confidence

    def __repr__(self):
        return (f"ExtractionResult(section={self.section!r}, "
                f"type={self.statement_type!r}, year={self.year!r}, "
                f"fields={len(self.data)})")


# ── Master prompt (one call for ALL sections) ─────────────────────────────────

_SYSTEM_PROMPT = """\
You are a financial data extraction specialist for Indian company annual reports.

RULES:
1. Extract ONLY values explicitly stated in the text — never calculate or infer.
2. All monetary values must be in INR Crores. Convert if stated in other units.
3. For each reporting year found, create a SEPARATE entry in the output array.
4. Bracketed numbers like (123.45) are negative: use -123.45.
5. Identify whether data is Consolidated or Standalone.
6. Return ONLY a valid JSON array. No markdown fences, no explanation.
7. Use the EXACT label text from the source — do not rename, abbreviate, or canonicalise.
8. Extract EVERY row that has a numeric value — do not skip any line."""

_COMBINED_PROMPT_TEMPLATE = """\
<<<SECTOR_HINT>>>
Extract ALL financial data from the document text below.

CRITICAL — PDF LINE FORMAT:
The text is extracted line-by-line from PDF tables. Each balance sheet / P&L row appears as:

  <Row Label>
  <Note/Schedule number>   ← a small whole integer (1-200), NOT a financial value
  <Value for Year 1>       ← first year column (usually the more recent year)
  <Value for Year 2>       ← second year column (prior year)

Example:
  Property, Plant & Equipment
  3                        ← this is Note 3, IGNORE IT
  2,010.75                 ← F2025 value
  1,727.28                 ← F2024 value

RULE: Any whole integer from 1–200 that appears between a row label line and
decimal number lines is a note/schedule reference. Skip it. Do not record it as a value.

YEAR COLUMNS:
Look for year headers near the top (e.g. "As at March 31, 2025" and "As at March 31, 2024").
The first decimal value after skipping the note ref = Year 1 (left column).
The second decimal value = Year 2 (right column).
Create a SEPARATE JSON entry for EACH year found.

IMPORTANT — DO NOT CONVERT UNITS:
Extract values EXACTLY as they appear in the document (same magnitude).
Do NOT divide or multiply by 100, 1000, or any other factor.
The calling code handles all unit conversion separately.

OUTPUT — return a JSON array, each element:
{{
  "section": "Annual P&L" | "Annual Balance Sheet" | "Annual Cash Flow",
  "statement_type": "consolidated" | "standalone",
  "reporting_year": "F2025" for year ending March 2025, "F2024" for March 2024, etc.,
  "currency": "as reported in source",
  "data": {{ "<exact label from source>": <number>, ... }}
}}

RULES:
- Extract EVERY row that has a financial value — main rows AND all sub-items / schedule detail lines.
- Schedule or note pages appear separately from the main statement. Extract every line item on
  every page — do not skip sub-items, breakdowns, or detail lines just because they are indented
  or appear under a schedule heading.
- Use the EXACT label text from the source — do not rename anything.
- Negative / bracketed values like (123.45) → -123.45.
- Omit fields you are uncertain about.
- Assign schedule/note sub-items to the same section as the parent statement
  (e.g., note pages that detail balance sheet items → "Annual Balance Sheet").

TEXT:
<<<TEXT>>>"""


# ── Robust JSON extractor ─────────────────────────────────────────────────────

def _extract_json_from_response(text: str) -> Optional[str]:
    """
    Find a JSON array (or object) anywhere inside an LLM response.

    Tries three strategies in order:
    1. Strip markdown code fences and attempt direct parse.
    2. Walk the string looking for the outermost '[' ... ']' pair.
    3. Walk the string looking for the outermost '{' ... '}' pair.
    """
    # Strategy 1 — strip fences, parse directly
    cleaned = re.sub(r"```(?:json)?[\s\n]*", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    if cleaned.startswith("[") or cleaned.startswith("{"):
        return cleaned

    # Strategy 2 & 3 — bracket matching
    for start_ch, end_ch in [("[", "]"), ("{", "}")]:
        idx = text.find(start_ch)
        if idx == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(idx, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
            if not in_str:
                if ch == start_ch:
                    depth += 1
                elif ch == end_ch:
                    depth -= 1
                    if depth == 0:
                        return text[idx : i + 1]
    return None


# ── Core LLM call ─────────────────────────────────────────────────────────────

def _call_llm(
    text: str,
    client: OpenAI,
    max_retries: int = 3,
    sector_hint: str = "",
) -> list[ExtractionResult]:
    """Single LLM call → list of ExtractionResults."""
    global _last_raw_responses
    if not text.strip():
        return []

    # Inject sector hint and text — sentinel replaces avoid .format() KeyError on braces.
    hint_block = f"CONTEXT: {sector_hint}\n\n" if sector_hint else ""
    user_prompt = (
        _COMBINED_PROMPT_TEMPLATE
        .replace("<<<SECTOR_HINT>>>", hint_block)
        .replace("<<<TEXT>>>", text)
    )

    for attempt in range(max_retries):
        raw = None
        try:
            model = os.environ.get("DEEPSEEK_MODEL", DEEPSEEK_MODEL)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=8192,
            )
            raw = response.choices[0].message.content.strip()

            # Capture for UI debugging (keep last 5 per session)
            _last_raw_responses.append({
                "attempt": attempt + 1,
                "chars": len(raw),
                "snippet": raw[:3000],
            })
            if len(_last_raw_responses) > 10:
                _last_raw_responses.pop(0)

            logger.info(f"  Raw LLM response ({len(raw)} chars): {raw[:200]!r}…")

            json_str = _extract_json_from_response(raw)
            if not json_str:
                raise json.JSONDecodeError(
                    f"No JSON structure found in response ({len(raw)} chars)", raw, 0
                )

            parsed = json.loads(json_str)
            if not isinstance(parsed, list):
                parsed = [parsed]

            results = []
            for item in parsed:
                r = _parse_item(item)
                if r and r.data:
                    results.append(r)

            # ── Unit normalisation (Python-side ONLY) ────────────────────────
            # The LLM is explicitly told NOT to convert units.
            # We detect the unit declaration from the source text and apply the
            # divisor once here in Python — this is the single conversion step.
            from utils.helpers import detect_unit_divisor
            divisor = detect_unit_divisor(text)
            unit_label = {
                1.0:      "INR Crores (no conversion)",
                100.0:    "INR Lakhs → Crores (÷100)",
                10.0:     "INR Millions → Crores (÷10)",
                10_000.0: "INR Thousands → Crores (÷10,000)",
            }.get(divisor, f"÷{divisor:g}")
            logger.info(f"  Unit detected: {unit_label}")
            if divisor != 1.0:
                for r in results:
                    r.data = {k: round(v / divisor, 4) for k, v in r.data.items()}
                    # Stamp unit info into notes so UI can display it
                    r.notes = (r.notes or "") + f" | unit_divisor={divisor}"

            logger.info(f"  LLM returned {len(results)} result blocks")
            return results

        except json.JSONDecodeError as e:
            snippet = (raw or "")[:400]
            logger.warning(f"  JSON parse error attempt {attempt+1}: {e}")
            logger.warning(f"  Response snippet: {snippet!r}")
            if attempt == max_retries - 1:
                logger.error("  All retries exhausted — returning empty")
                return []
            time.sleep(1)

        except Exception as e:
            logger.error(f"  LLM call failed attempt {attempt+1}: {type(e).__name__}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    return []


def _parse_item(item: dict) -> Optional[ExtractionResult]:
    try:
        raw_year = item.get("reporting_year", "")
        year = normalise_year(str(raw_year)) if raw_year else None
        if not year:
            return None

        raw_data = item.get("data", {})
        clean_data: dict[str, float] = {}
        for label, val in raw_data.items():
            if val is None:
                continue
            if isinstance(val, (int, float)):
                clean_data[label] = float(val)
            else:
                n = parse_number(str(val))
                if n is not None:
                    clean_data[label] = n

        return ExtractionResult(
            section=item.get("section", "Unknown"),
            statement_type=item.get("statement_type", "consolidated").lower(),
            year=year,
            currency=item.get("currency", "INR Crores"),
            data=clean_data,
            notes=item.get("notes", ""),
        )
    except Exception as e:
        logger.error(f"  _parse_item error: {e}")
        return None


def _merge_results(results: list[ExtractionResult]) -> list[ExtractionResult]:
    """Deduplicate and merge results with the same (section, type, year)."""
    merged: dict[tuple, ExtractionResult] = {}
    for r in results:
        key = (r.section, r.statement_type, r.year)
        if key not in merged:
            merged[key] = r
        else:
            for k, v in r.data.items():
                if k not in merged[key].data:
                    merged[key].data[k] = v
    return list(merged.values())


# ── High-level API ────────────────────────────────────────────────────────────

def extract_all_sections(
    parsed_doc,
    client: Optional[OpenAI] = None,
    rule_based_threshold: float = 0.80,
) -> list[ExtractionResult]:
    """
    Extract from a ParsedDocument using a hybrid rule-based + LLM strategy.

    Per-section logic (not mean):
      - Rule-based coverage >= rule_based_threshold → rule-based only for that section.
      - Rule-based coverage <  rule_based_threshold → LLM runs for that section;
        LLM results are preferred (they use the document's own field names, which
        match generic templates better). Rule-based values fill any gaps LLM missed.

    Set rule_based_threshold=0.0  to always use LLM for every section.
    Set rule_based_threshold=1.01 to always use rule-based only (never LLM).

    Default 0.80: LLM runs unless rule-based is very confident (≥80% coverage),
    ensuring generic company templates get accurate, document-terminology extraction.
    """
    clear_debug_responses()

    if not parsed_doc.sections:
        logger.warning("No sections found — skipping extraction")
        return []

    # ── Pass 1: Rule-based (per section) ─────────────────────────────────────
    from extractor.rule_based_extractor import extract_all_rule_based, extract_section
    rb_results, mean_cov = extract_all_rule_based(parsed_doc)

    # Build per-section coverage map
    section_coverages: dict[str, float] = {}
    for section in parsed_doc.sections:
        _, cov = extract_section(section.text, section.section_type)
        section_coverages[section.section_type] = cov

    high_cov_types = {st for st, cov in section_coverages.items() if cov >= rule_based_threshold}
    low_cov_sections = [s for s in parsed_doc.sections if s.section_type not in high_cov_types]

    logger.info(
        f"Rule-based mean coverage {mean_cov:.0%} | "
        f"high-conf sections: {sorted(high_cov_types)} | "
        f"LLM needed for: {[s.section_type for s in low_cov_sections]}"
    )

    if not low_cov_sections or client is None:
        # No LLM needed (or no client) — return rule-based only
        if not low_cov_sections:
            logger.info(f"All sections above threshold — skipping LLM")
        return _merge_results(rb_results)

    # ── Pass 2: LLM for sections below threshold ──────────────────────────────
    combined_text = "\n\n---\n\n".join(
        f"[{s.section_type.upper()}]\n{s.text}"
        for s in low_cov_sections
    )
    combined_text = clean_for_llm(combined_text)
    chunks = chunk_text(combined_text, max_chars=12_000)
    logger.info(
        f"LLM extraction for {len(low_cov_sections)} section(s) "
        f"→ {len(chunks)} call(s) ({len(combined_text):,} chars)"
    )

    sector_hint = getattr(parsed_doc, "llm_hint", "")
    llm_results: list[ExtractionResult] = []
    for i, chunk in enumerate(chunks):
        logger.info(f"  LLM call {i+1}/{len(chunks)} ({len(chunk)} chars)…")
        llm_results.extend(_call_llm(chunk, client, sector_hint=sector_hint))

    # Merge strategy:
    # - For HIGH-coverage sections: rule-based values take priority (they're reliable)
    # - For LOW-coverage sections: LLM values take priority (uses document's own field
    #   names, which match generic templates better); rule-based fills any gaps
    high_cov_rb = [r for r in rb_results if r.section in high_cov_types]
    low_cov_rb  = [r for r in rb_results if r.section not in high_cov_types]

    # LLM first, then low-cov rule-based fills gaps
    all_results = high_cov_rb + llm_results + low_cov_rb
    merged = _merge_results(all_results)
    logger.info(
        f"Hybrid extraction complete: {len(merged)} result blocks, "
        f"{sum(len(r.data) for r in merged)} total fields"
    )
    return merged


def extract_from_multiple_docs(
    parsed_docs: list,
    client: Optional[OpenAI] = None,
    max_workers: int = 3,
) -> tuple[list[ExtractionResult], list[str]]:
    """
    Extract from multiple ParsedDocuments concurrently.
    Returns (results, errors) — errors is a list of error strings for UI display.
    """
    if client is None:
        client = _get_client()

    all_results: list[ExtractionResult] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_all_sections, doc, client): doc.filename
            for doc in parsed_docs
        }
        for fut in as_completed(futures):
            fname = futures[fut]
            try:
                results = fut.result()
                all_results.extend(results)
                logger.info(f"  '{fname}': {len(results)} result blocks")
            except Exception as e:
                import traceback
                err_msg = f"'{fname}': {type(e).__name__}: {e}"
                logger.error(f"  Extraction failed — {err_msg}")
                logger.error(traceback.format_exc())
                errors.append(err_msg)

    return _merge_results(all_results), errors


def extract_from_full_text(
    full_text: str,
    target_sections: list[str] | None = None,
    client: Optional[OpenAI] = None,
) -> list[ExtractionResult]:
    """Fallback: extract from raw text when section detection fails."""
    if client is None:
        client = _get_client()
    chunks = chunk_text(full_text, max_chars=12_000)
    all_results = []
    for chunk in chunks:
        all_results.extend(_call_llm(chunk, client))
    return _merge_results(all_results)


# ── Template-guided extraction ────────────────────────────────────────────────
#
# Instead of "extract everything → fuzzy-map to template", we send the
# template's own field names to the LLM and ask it to find those specific
# values.  The LLM returns {template_field: value} so no mapping step is
# needed — the keys are already template labels.
#
# This eliminates the single biggest source of errors in the pipeline:
# the fuzzy/LLM mapper making wrong connections between PDF labels and
# template labels.

_TEMPLATE_GUIDED_SYSTEM = """\
You are a financial data extraction specialist for Indian company annual reports.
Your job is precise: find specific values for a given list of field names.
Return ONLY valid JSON. No explanation, no markdown fences."""

_TEMPLATE_GUIDED_PROMPT = """\
Extract values for the EXACT template fields listed below from the Indian financial \
report text provided.

FIELDS TO FIND (use these EXACT strings as JSON keys — do not rename them):
{field_list}

YEARS TO EXTRACT: {years}

HOW PDF TABLES ARE FORMATTED:
- Each row: label on one line, then optionally a small integer (1–200) as a note \
reference (SKIP IT), then one or two numeric values for the year columns.
- Year columns are declared at the top: "As at March 31, 2025" → year F2025.
- The left column is usually the more recent year.

MATCHING RULES:
- Match each field by MEANING — the PDF may word things differently than the template.
  Example: template "Net Interest Income" = PDF "Interest earned less interest expended".
- If a field genuinely cannot be found, omit it rather than guessing.
- Bracketed values like (1,234.56) are NEGATIVE — return as -1234.56.

IMPORTANT — DO NOT CONVERT UNITS:
Extract values EXACTLY as they appear in the document (same magnitude, same scale).
Do NOT divide or multiply by 100, 1000, or any other factor regardless of what unit \
declaration you see (Lakhs, Thousands, etc.).
The calling code handles all unit conversion separately.

OUTPUT — valid JSON only:
{{
  "F2025": {{
    "<EXACT template field name>": <number>,
    ...
  }},
  "F2024": {{
    "<EXACT template field name>": <number>,
    ...
  }}
}}
Only include years and fields where you found an actual value.

TEXT:
{text}"""


def _call_template_guided_llm(
    section_text: str,
    template_fields: list[str],
    years: list[str],
    client: OpenAI,
    statement_type: str = "consolidated",
    section_name: str = "Unknown",
    sector_hint: str = "",
    max_retries: int = 3,
) -> list[ExtractionResult]:
    """
    Single template-guided LLM call for one section.
    Returns ExtractionResults whose data keys ARE template field names.
    """
    global _last_raw_responses
    if not section_text.strip() or not template_fields:
        return []

    field_list = "\n".join(f"  - {f}" for f in template_fields)
    years_str  = ", ".join(years)
    hint_block = f"CONTEXT: {sector_hint}\n\n" if sector_hint else ""

    user_prompt = (
        hint_block +
        _TEMPLATE_GUIDED_PROMPT
        .replace("{field_list}", field_list)
        .replace("{years}", years_str)
        .replace("{text}", section_text)
    )

    for attempt in range(max_retries):
        raw = None
        try:
            model = os.environ.get("DEEPSEEK_MODEL", DEEPSEEK_MODEL)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _TEMPLATE_GUIDED_SYSTEM},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content.strip()

            _last_raw_responses.append({
                "attempt": attempt + 1,
                "chars": len(raw),
                "snippet": raw[:2000],
                "mode": "template_guided",
                "section": section_name,
            })
            if len(_last_raw_responses) > 10:
                _last_raw_responses.pop(0)

            json_str = _extract_json_from_response(raw)
            if not json_str:
                raise json.JSONDecodeError("No JSON in template-guided response", raw, 0)

            parsed = json.loads(json_str)
            if not isinstance(parsed, dict):
                raise json.JSONDecodeError("Expected dict at top level", json_str, 0)

            # ── Unit normalisation (Python-side ONLY) ────────────────────────
            # The LLM is explicitly told NOT to convert units.
            # Detect the unit declaration from the source text and apply the
            # divisor once here in Python — this is the single conversion step.
            from utils.helpers import detect_unit_divisor
            divisor = detect_unit_divisor(section_text)
            unit_label = {
                1.0:      "INR Crores (no conversion)",
                100.0:    "INR Lakhs → Crores (÷100)",
                10.0:     "INR Millions → Crores (÷10)",
                10_000.0: "INR Thousands → Crores (÷10,000)",
            }.get(divisor, f"÷{divisor:g}")
            logger.info(f"  Template-guided unit: {unit_label}")

            results: list[ExtractionResult] = []
            for year_key, field_vals in parsed.items():
                year = normalise_year(str(year_key))
                if not year or not isinstance(field_vals, dict):
                    continue

                clean_data: dict[str, float] = {}
                for field_name, val in field_vals.items():
                    # Only keep fields that were actually requested
                    if field_name not in template_fields:
                        continue
                    if val is None:
                        continue
                    n = parse_number(str(val)) if not isinstance(val, (int, float)) else float(val)
                    if n is not None:
                        clean_data[field_name] = round(n / divisor, 4) if divisor != 1.0 else n

                if clean_data:
                    results.append(ExtractionResult(
                        section=section_name,
                        statement_type=statement_type,
                        year=year,
                        currency="INR Crores",
                        data=clean_data,
                        notes=f"mode=template_guided | unit_divisor={divisor}",
                        confidence=0.9,
                    ))

            logger.info(
                f"  Template-guided [{section_name}]: "
                f"{sum(len(r.data) for r in results)} values across "
                f"{len(results)} year(s) found"
                + (f" (÷{divisor:g})" if divisor != 1.0 else "")
            )
            return results

        except json.JSONDecodeError as e:
            logger.warning(f"  Template-guided JSON error attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                return []
            time.sleep(1)

        except Exception as e:
            logger.error(f"  Template-guided LLM failed attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    return []


# ── Sheet → section_name mapping ─────────────────────────────────────────────

_SHEET_TO_SECTION: dict[str, str] = {
    "pl":  "Annual P&L",
    "bs":  "Annual Balance Sheet",
    "cf":  "Annual Cash Flow",
}

def _classify_sheet(sheet_name: str) -> str:
    """Return 'pl', 'bs', or 'cf' based on sheet name keywords."""
    sn = sheet_name.lower()
    if any(k in sn for k in ("p&l", "profit", "loss", "income", "revenue")):
        return "pl"
    if any(k in sn for k in ("balance", "bs", "assets", "liab")):
        return "bs"
    if any(k in sn for k in ("cash", "flow", "cf")):
        return "cf"
    return "bs"   # default


def template_guided_extract_all(
    parsed_doc,
    template_model,
    client: OpenAI,
    statement_type: str = "consolidated",
) -> list[ExtractionResult]:
    """
    Template-guided extraction: for each sheet in the template, send its
    field names to the LLM with the relevant PDF section text.  The LLM
    returns {template_field: value} — no mapping step required.

    This replaces the generic extract → fuzzy-map pipeline for all sections
    where template field names are known in advance (which is always).

    Parameters
    ----------
    parsed_doc      : ParsedDocument from pdf_parser.parse_pdf()
    template_model  : TemplateModel from template_reader.read_template()
    client          : OpenAI client (DeepSeek)
    statement_type  : "consolidated" | "standalone"

    Returns
    -------
    list[ExtractionResult] where every data key is an exact template label.
    These can be written to the template via direct exact-match lookup —
    no fuzzy matching or LLM mapping needed.
    """
    sector_hint = getattr(parsed_doc, "llm_hint", "")
    all_results: list[ExtractionResult] = []

    # Determine which years the template expects (union across all sheets)
    template_years: list[str] = []
    for sheet in template_model.sheets.values():
        for yr in sheet.year_cols:
            if yr not in template_years:
                template_years.append(yr)
    template_years = sorted(template_years)

    if not template_years:
        logger.warning("template_guided_extract_all: no year columns found in template")
        return []

    logger.info(
        f"template_guided_extract_all: template years={template_years}, "
        f"statement_type={statement_type}"
    )

    # Build a lookup from section type → PageSection text
    section_text_map: dict[str, str] = {}
    for sec in parsed_doc.sections:
        stype = _classify_sheet(sec.section_type)
        # Concatenate multiple sections of the same type
        existing = section_text_map.get(stype, "")
        section_text_map[stype] = existing + "\n\n" + sec.text if existing else sec.text

    # For each sheet in the template, run a targeted LLM call
    for sheet_name, sheet_model in template_model.sheets.items():
        # Get the field names for the correct statement type (C vs S section)
        fields = _get_fields_for_statement_type(sheet_model, statement_type)
        if not fields:
            logger.info(f"  Sheet '{sheet_name}': no writable fields — skipping")
            continue

        # Find the PDF section that matches this sheet
        stype = _classify_sheet(sheet_name)
        section_text = section_text_map.get(stype, "")
        if not section_text.strip():
            logger.warning(
                f"  Sheet '{sheet_name}' (type={stype}): "
                "no PDF section text found — skipping"
            )
            continue

        # Determine which years this sheet actually has column for
        sheet_years = [y for y in template_years if y in sheet_model.year_cols]
        if not sheet_years:
            continue

        section_name = _SHEET_TO_SECTION.get(stype, sheet_name)
        logger.info(
            f"  Sheet '{sheet_name}': {len(fields)} fields, "
            f"years={sheet_years}, text={len(section_text):,} chars"
        )

        # Chunk if text is very long; always send fields + years
        text_clean = clean_for_llm(section_text)
        chunks = chunk_text(text_clean, max_chars=14_000)

        chunk_results: list[ExtractionResult] = []
        for i, chunk in enumerate(chunks):
            logger.info(f"    chunk {i+1}/{len(chunks)} ({len(chunk):,} chars)…")
            chunk_results.extend(
                _call_template_guided_llm(
                    section_text=chunk,
                    template_fields=fields,
                    years=sheet_years,
                    client=client,
                    statement_type=statement_type,
                    section_name=section_name,
                    sector_hint=sector_hint,
                )
            )

        # Merge chunks: later chunks fill gaps, earlier chunks take priority
        merged = _merge_results(chunk_results)
        all_results.extend(merged)

    total_fields = sum(len(r.data) for r in all_results)
    logger.info(
        f"template_guided_extract_all: "
        f"{len(all_results)} result block(s), {total_fields} total fields extracted"
    )
    return all_results


def _get_fields_for_statement_type(
    sheet_model,
    statement_type: str,
) -> list[str]:
    """
    Return the list of template field names for the given statement type,
    excluding formula cells and pre-filled cells.

    If the sheet has no section_offsets (no C/S split), return all fields.
    """
    stmt_key = "standalone" if statement_type == "standalone" else "consolidated"

    section_start = 1
    section_end   = 99999
    if sheet_model.section_offsets:
        section_start = sheet_model.section_offsets.get(stmt_key, 1)
        other_starts  = [
            v for k, v in sheet_model.section_offsets.items()
            if v > section_start
        ]
        section_end = min(other_starts) if other_starts else 99999

    fields: list[str] = []
    for label, row_nums in sheet_model.row_index.items():
        # Check at least one row for this label falls in the right section
        label_rows = [
            rn for rn in row_nums
            if section_start <= rn < section_end
        ]
        if not label_rows:
            # No section filtering — include all labels
            label_rows = row_nums

        # Check at least one year column is writable for this label
        writable = False
        for rn in label_rows:
            for col in sheet_model.year_cols.values():
                if (rn, col) not in sheet_model.formula_cells \
                   and (rn, col) not in sheet_model.filled_cells:
                    writable = True
                    break
            if writable:
                break

        if writable:
            fields.append(label)

    return fields
