"""
Field Mapper — resolves extracted field names to template row labels.

Uses a three-tier strategy:
  1. Exact match (after cleaning)
  2. Fuzzy string match (fuzzywuzzy)
  3. LLM-based semantic mapping (DeepSeek) for ambiguous cases
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

# ── Guarded fuzzywuzzy import ──────────────────────────────────────────────────
def _require_fuzzywuzzy():
    try:
        from fuzzywuzzy import fuzz as _fuzz, process as _proc
        return _fuzz, _proc
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "fuzzywuzzy>=0.18.0", "python-Levenshtein>=0.25.0"]
        )
        import importlib, site
        importlib.invalidate_caches()
        for sp in site.getsitepackages():
            if sp not in sys.path:
                sys.path.insert(0, sp)
        from fuzzywuzzy import fuzz as _fuzz, process as _proc
        return _fuzz, _proc

fuzz, fuzz_process = _require_fuzzywuzzy()

# ── Guarded openai import ──────────────────────────────────────────────────────
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

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, FIELD_SYNONYMS
from extractor.llm_extractor import ExtractionResult
from mapper.template_reader import TemplateModel, SheetModel
from utils.helpers import get_logger, clean_label

logger = get_logger(__name__)


FUZZY_THRESHOLD = 75       # minimum score (0–100) for fuzzy match
LLM_FALLBACK_THRESHOLD = 50  # below this score, escalate to LLM


@dataclass
class MappedField:
    """Represents a resolved mapping from extracted data to a template cell."""
    extracted_label: str       # Label from the PDF/LLM extraction
    template_label: str        # Matched label in the template
    sheet_name: str
    row: int
    col: int
    year: str
    value: float
    match_method: str          # "exact", "fuzzy", "synonym", "llm", "manual"
    match_score: float         # 0.0 – 1.0
    statement_type: str        # "consolidated" | "standalone"
    # Cross-validation tier (set after web validation)
    confidence_tier: str = "MEDIUM"   # "VERY_HIGH"|"HIGH"|"MEDIUM"|"FLAG"
    web_value: float = None           # value from web source (if any)
    web_source: str = ""              # "yfinance"|"bse_xbrl"|""
    flag_reason: str = ""             # human-readable reason if FLAG/MEDIUM


@dataclass
class MappingReport:
    """Full mapping report for one ExtractionResult."""
    extraction: ExtractionResult
    mapped: list[MappedField]
    unmapped: list[str]        # extracted labels that couldn't be mapped


# ── Synonym expansion ──────────────────────────────────────────────────────────

def _build_synonym_lookup(synonyms: dict) -> dict[str, str]:
    """Build a flat lookup: lowercase synonym → canonical template label."""
    lookup: dict[str, str] = {}
    for canonical, aliases in synonyms.items():
        for alias in aliases:
            lookup[alias.lower()] = canonical
    return lookup


_SYNONYM_LOOKUP = _build_synonym_lookup(FIELD_SYNONYMS)


# ── Core mapper ────────────────────────────────────────────────────────────────

class FieldMapper:
    """
    Maps extraction results to template cells using exact/fuzzy/LLM matching.
    """

    def __init__(self, template: TemplateModel, use_llm: bool = True):
        self.template = template
        self.use_llm = use_llm
        self._llm_client: Optional[OpenAI] = None
        self._llm_cache: dict[str, Optional[str]] = {}   # extracted_label → template_label
        # Cache sheet-type classification so LLM runs at most once per template
        # Maps section_type ("pl"|"bs"|"cf") → resolved sheet name
        self._sheet_type_cache: dict[str, Optional[str]] = {}

    def _get_llm(self) -> Optional[OpenAI]:
        if not self.use_llm:
            return None
        if self._llm_client is None and DEEPSEEK_API_KEY:
            self._llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        return self._llm_client

    # ── Main entry point ──────────────────────────────────────────────────────

    def map_results(self, results: list[ExtractionResult]) -> list[MappingReport]:
        """
        Map a list of ExtractionResults to template cells.

        Pass 1: exact / synonym / fuzzy matching for every field.
        Pass 2: one batched LLM call resolves ALL still-unmapped fields at once,
                reusing the same mapping for the same extracted label.
        """
        reports = []
        for result in results:
            report = self._map_single(result)
            reports.append(report)

        # ── Pass 2: batch LLM for everything still unmapped ──────────────────
        if self.use_llm and self._get_llm():
            self._batch_llm_remap(reports)

        for report in reports:
            logger.info(
                f"Mapped {len(report.mapped)} / "
                f"{len(report.extraction.data)} fields for "
                f"{report.extraction.section} {report.extraction.statement_type} {report.extraction.year}"
                + (f" | {len(report.unmapped)} still unmapped" if report.unmapped else "")
            )
            if report.unmapped:
                logger.debug(f"  Still unmapped: {report.unmapped}")

        return reports

    def _batch_llm_remap(self, reports: list[MappingReport]) -> None:
        """
        Collect every unmapped extracted label, ask the LLM once to match them
        all to template labels, then inject the results back into the reports.
        """
        # Build a deduplicated set of (extracted_label, sheet_name, statement_type)
        # to resolve.  Group by sheet so each LLM call has the right candidate list.
        from mapper.template_reader import _resolve_row

        # sheet_name → { extracted_label → [(report, extraction_result)] }
        sheet_unmapped: dict[str, dict[str, list]] = {}
        for report in reports:
            sheet_name = self._resolve_sheet_name(report.extraction.section)
            if not sheet_name or sheet_name not in self.template.sheets:
                continue
            sheet = self.template.sheets[sheet_name]
            col = sheet.year_cols.get(report.extraction.year)
            if not col:
                continue
            for label in list(report.unmapped):
                sheet_unmapped.setdefault(sheet_name, {}).setdefault(label, []).append(
                    (report, report.extraction)
                )

        if not sheet_unmapped:
            return

        for sheet_name, label_map in sheet_unmapped.items():
            if not label_map:
                continue
            sheet = self.template.sheets[sheet_name]
            all_template_labels = list(sheet.row_index.keys())
            extracted_labels = list(label_map.keys())

            # Ask LLM to match all extracted labels at once
            llm_mapping = self._llm_batch_match(extracted_labels, all_template_labels)
            if not llm_mapping:
                continue

            for ext_label, tmpl_label in llm_mapping.items():
                if not tmpl_label:
                    continue
                for report, extraction in label_map.get(ext_label, []):
                    stmt = extraction.statement_type
                    row = _resolve_row(sheet, tmpl_label, stmt)
                    col = sheet.year_cols.get(extraction.year)
                    if row is None or col is None:
                        continue
                    if not self.template.is_cell_writable(sheet_name, row, col):
                        continue
                    value = extraction.data.get(ext_label)
                    if value is None:
                        continue
                    report.mapped.append(MappedField(
                        extracted_label=ext_label,
                        template_label=tmpl_label,
                        sheet_name=sheet_name,
                        row=row,
                        col=col,
                        year=extraction.year,
                        value=value,
                        match_method="llm_batch",
                        match_score=0.85,
                        statement_type=stmt,
                    ))
                    if ext_label in report.unmapped:
                        report.unmapped.remove(ext_label)

    def _llm_batch_match(
        self,
        extracted_labels: list[str],
        template_labels: list[str],
    ) -> dict[str, Optional[str]]:
        """
        Send all unmapped extracted labels to the LLM in one call.
        Returns {extracted_label: matched_template_label_or_None}.
        """
        client = self._get_llm()
        if not client or not extracted_labels or not template_labels:
            return {}

        # Build cache key; skip labels already cached
        result: dict[str, Optional[str]] = {}
        to_resolve: list[str] = []
        tmpl_hash = hash(tuple(sorted(template_labels)))
        for lbl in extracted_labels:
            cache_key = f"{lbl}|||{tmpl_hash}"
            if cache_key in self._llm_cache:
                result[lbl] = self._llm_cache[cache_key]
            else:
                to_resolve.append(lbl)

        if not to_resolve:
            return result

        tmpl_block = "\n".join(f"- {t}" for t in template_labels)
        ext_block  = "\n".join(f'  "{e}"' for e in to_resolve)
        prompt = (
            "You are a financial data expert. Match each extracted field name "
            "to the most appropriate template field name.\n\n"
            f"Template fields:\n{tmpl_block}\n\n"
            f"Extracted field names to match:\n{ext_block}\n\n"
            "STRICT MATCHING RULES:\n"
            "1. Only match if both labels refer to the SAME financial concept. "
            "Partial word overlap is NOT enough — the labels must mean the same thing.\n"
            "2. Return null when the extracted label is clearly more specific or different "
            "in nature than any template field (e.g. a footnote, an opening/closing balance "
            "of a reserve, a count of shares, or a contingent/off-balance-sheet item).\n"
            "3. Prefer null over a wrong match — an empty cell is better than a wrong value.\n\n"
            "Return ONLY a JSON object mapping each extracted name to the best "
            "template field name, or null if no good match exists.\n"
            'Example: {"Net Interest Income": "Net interest income", "Tax expense": "Less: Tax", '
            '"Investment fluctuation reserve - Opening Balance": null}'
        )
        try:
            from config.settings import DEEPSEEK_MODEL
            model = os.environ.get("DEEPSEEK_MODEL", DEEPSEEK_MODEL)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=800,
            )
            raw = response.choices[0].message.content.strip()
            # Extract JSON object
            import re as _re
            m = _re.search(r"\{.*\}", raw, _re.DOTALL)
            if not m:
                return result
            mapping: dict = json.loads(m.group(0))
            tmpl_lower = {clean_label(t).lower(): t for t in template_labels}
            for ext_lbl, tmpl_lbl in mapping.items():
                cache_key = f"{ext_lbl}|||{tmpl_hash}"
                if tmpl_lbl is None or tmpl_lbl == "null":
                    self._llm_cache[cache_key] = None
                    result[ext_lbl] = None
                else:
                    # Validate against template labels
                    matched = tmpl_lower.get(clean_label(str(tmpl_lbl)).lower())
                    self._llm_cache[cache_key] = matched
                    result[ext_lbl] = matched
        except Exception as e:
            logger.warning(f"Batch LLM mapping failed: {e}")

        return result

    def _map_single(self, result: ExtractionResult) -> MappingReport:
        """Map all fields in one ExtractionResult."""
        sheet_name = self._resolve_sheet_name(result.section)
        if not sheet_name or sheet_name not in self.template.sheets:
            logger.warning(f"No matching sheet for section '{result.section}'")
            return MappingReport(extraction=result, mapped=[], unmapped=list(result.data.keys()))

        sheet = self.template.sheets[sheet_name]
        col = sheet.year_cols.get(result.year)
        if not col:
            logger.warning(f"Year {result.year!r} not found in sheet '{sheet_name}'")
            return MappingReport(extraction=result, mapped=[], unmapped=list(result.data.keys()))

        mapped: list[MappedField] = []
        unmapped: list[str] = []
        template_labels = list(sheet.row_index.keys())

        # Build section-scoped label list for the right statement type
        section_labels = self._get_section_labels(sheet, result.statement_type)

        for extracted_label, value in result.data.items():
            match_result = self._find_match(
                extracted_label, section_labels or template_labels, sheet, result.statement_type
            )

            if match_result:
                template_label, method, score = match_result
                # row_index now stores lists; use _resolve_row for correct section lookup
                from mapper.template_reader import _resolve_row
                row = _resolve_row(sheet, template_label, result.statement_type)
                if row is None:
                    unmapped.append(extracted_label)
                    continue

                # Check writeability
                if not self.template.is_cell_writable(sheet_name, row, col):
                    logger.debug(
                        f"  Cell ({row},{col}) in '{sheet_name}' is formula/filled — skipping"
                    )
                    continue

                mapped.append(MappedField(
                    extracted_label=extracted_label,
                    template_label=template_label,
                    sheet_name=sheet_name,
                    row=row,
                    col=col,
                    year=result.year,
                    value=value,
                    match_method=method,
                    match_score=score,
                    statement_type=result.statement_type,
                ))
            else:
                unmapped.append(extracted_label)

        return MappingReport(extraction=result, mapped=mapped, unmapped=unmapped)

    # ── Match resolution ──────────────────────────────────────────────────────

    def _find_match(
        self,
        extracted: str,
        candidates: list[str],
        sheet: SheetModel,
        statement_type: str,
    ) -> Optional[tuple[str, str, float]]:
        """
        Returns (matched_template_label, method, score) or None.
        """
        extracted_clean = clean_label(extracted).lower()

        # Tier 1: Exact match
        for candidate in candidates:
            if clean_label(candidate).lower() == extracted_clean:
                return (candidate, "exact", 1.0)

        # Tier 2: Synonym lookup
        if extracted_clean in _SYNONYM_LOOKUP:
            canonical = _SYNONYM_LOOKUP[extracted_clean]
            # Find canonical in candidates
            for candidate in candidates:
                if clean_label(candidate).lower() == canonical.lower():
                    return (candidate, "synonym", 0.95)

        # Tier 3: Fuzzy match — but guard against matching wildly different labels.
        # A very long extracted label (e.g. a contingent-liability footnote) should
        # never silently overwrite a short summary row (e.g. "Capital").
        if candidates:
            ext_len = len(extracted_clean)
            # Build a filtered candidate list: skip candidates where the length ratio
            # is extreme (extracted is >4× or <¼ the candidate length).
            filtered_candidates = [
                c for c in candidates
                if 0.25 <= ext_len / max(len(clean_label(c)), 1) <= 4.0
            ]
            pool = filtered_candidates if filtered_candidates else candidates

            best_match, best_score = fuzz_process.extractOne(
                extracted_clean,
                [clean_label(c).lower() for c in pool],
                scorer=fuzz.token_set_ratio,
            )
            if best_score >= FUZZY_THRESHOLD:
                for candidate in pool:
                    if clean_label(candidate).lower() == best_match:
                        return (candidate, "fuzzy", best_score / 100.0)

            # Fuzzy didn't reach threshold — try LLM only if length ratio is reasonable
            if filtered_candidates:
                llm_match = self._llm_match(extracted, filtered_candidates)
                if llm_match:
                    return (llm_match, "llm", 0.85)

        return None

    def _get_section_labels(self, sheet: SheetModel, statement_type: str) -> list[str]:
        """Return labels scoped to consolidated/standalone section."""
        if not sheet.section_offsets:
            return list(sheet.row_index.keys())

        section_key = "standalone" if statement_type == "standalone" else "consolidated"
        section_start = sheet.section_offsets.get(section_key, 1)

        other_starts = [
            v for k, v in sheet.section_offsets.items()
            if v > section_start
        ]
        section_end = min(other_starts) if other_starts else 99999

        # row_index now stores lists of rows per label
        result = []
        for label, rows in sheet.row_index.items():
            if any(section_start <= r < section_end for r in rows):
                result.append(label)
        return result

    # ── LLM fallback matching ─────────────────────────────────────────────────

    def _llm_match(self, extracted: str, candidates: list[str]) -> Optional[str]:
        """Ask DeepSeek to match an extracted label to the best template candidate."""
        if not self.use_llm:
            return None

        # Check cache
        cache_key = f"{extracted}|||{hash(tuple(candidates[:20]))}"
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        client = self._get_llm()
        if not client:
            return None

        candidates_str = "\n".join(f"- {c}" for c in candidates[:30])
        prompt = f"""You are a financial data expert. Match the extracted field name to the most
appropriate template field name from the list below.

Extracted field: "{extracted}"

Template fields (pick EXACTLY one, or respond with "NO_MATCH" if none fits):
{candidates_str}

Respond with ONLY the exact template field name, or "NO_MATCH". No explanation."""

        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            result = response.choices[0].message.content.strip()
            if result == "NO_MATCH" or not result:
                self._llm_cache[cache_key] = None
                return None
            # Verify the result is actually in candidates
            for candidate in candidates:
                if clean_label(candidate).lower() == clean_label(result).lower():
                    self._llm_cache[cache_key] = candidate
                    return candidate
            self._llm_cache[cache_key] = None
            return None
        except Exception as e:
            logger.warning(f"LLM matching failed: {e}")
            return None

    # ── Sheet name resolution ─────────────────────────────────────────────────

    # Keywords that identify each section type within a sheet name
    _PL_NAME_KW  = {"p&l", "profit", "loss", "income", "p & l", "p and l",
                    "revenue", "earnings", "statement of operations"}
    _BS_NAME_KW  = {"balance", "bs", "assets", "liabilities", "financial position"}
    _CF_NAME_KW  = {"cash", "flow", "cf", "cashflow"}

    # Row-label indicators for content-scan (Tier 3)
    _PL_ROW_IND  = {"revenue", "profit", "loss", "ebitda", "earnings",
                    "sales", "employee", "depreciation", "finance cost", "tax"}
    _BS_ROW_IND  = {"assets", "liabilities", "equity", "borrowings",
                    "receivables", "inventories", "goodwill", "investments"}
    _CF_ROW_IND  = {"cash flow", "operating activities", "investing",
                    "financing", "net cash"}

    def _section_type(self, section: str) -> Optional[str]:
        """Return 'pl', 'bs', or 'cf' from the extraction section name."""
        s = section.lower()
        if any(k in s for k in ("p&l", "profit", "loss", "income statement")):
            return "pl"
        if any(k in s for k in ("balance sheet", "balance")):
            return "bs"
        if "cash" in s:
            return "cf"
        return None

    def _resolve_sheet_name(self, section: str) -> Optional[str]:
        """
        Map an extraction section name to the best-matching sheet in the template.

        Four tiers (stops at first confident hit):
          1. Hardcoded exact match — zero-cost, handles standard template names.
          2. Keyword scan of actual sheet names — handles "P&L", "BS", "CF",
             "Income Statement", "Balance Sheet", etc. (case-insensitive).
          3. Row-label content scan — counts how many row labels in each sheet
             match financial indicator words.  Works for sheets with arbitrary
             names like "Financials" or "Summary".
          4. LLM — asked once per section type; result is cached for the lifetime
             of this FieldMapper instance so no extra API calls on repeat lookups.
        """
        section_lower = section.lower()
        available = list(self.template.sheets.keys())

        # ── Tier 1: hardcoded fast-path ───────────────────────────────────────
        EXACT_MAP: dict[str, str] = {
            "annual p&l":           "Annual P&L",
            "profit and loss":      "Annual P&L",
            "p&l":                  "Annual P&L",
            "annual balance sheet": "Annual Balance Sheet",
            "balance sheet":        "Annual Balance Sheet",
            "annual cash flow":     "Annual Cash Flow",
            "cash flow":            "Annual Cash Flow",
            "operating metrics":    "Operating metrics",
            "operating metric":     "Operating metrics",
            "quarterly":            "Quarterly",
        }
        for key, target in EXACT_MAP.items():
            if key in section_lower and target in available:
                return target

        stype = self._section_type(section)
        if stype is None:
            return section  # unrecognised section type

        # Return cached LLM result if we already resolved this type
        if stype in self._sheet_type_cache:
            return self._sheet_type_cache[stype]

        kw_map  = {"pl": self._PL_NAME_KW,  "bs": self._BS_NAME_KW,  "cf": self._CF_NAME_KW}
        ind_map = {"pl": self._PL_ROW_IND,  "bs": self._BS_ROW_IND,  "cf": self._CF_ROW_IND}

        # ── Tier 2: keyword scan on actual sheet names ────────────────────────
        for sheet_name in available:
            sn = sheet_name.lower()
            if any(kw in sn for kw in kw_map[stype]):
                logger.info(f"  Sheet name-match: '{section}' → '{sheet_name}' (Tier 2)")
                self._sheet_type_cache[stype] = sheet_name
                return sheet_name

        # ── Tier 3: row-label content scan ────────────────────────────────────
        indicators = ind_map[stype]
        best_sheet: Optional[str] = None
        best_score = 0
        for sheet_name, sm in self.template.sheets.items():
            score = sum(
                1 for label in sm.row_index.keys()
                if any(ind in label.lower() for ind in indicators)
            )
            if score > best_score:
                best_score = score
                best_sheet = sheet_name

        if best_sheet and best_score >= 3:
            logger.info(
                f"  Sheet content-scan: '{section}' → '{best_sheet}' "
                f"(score={best_score}, Tier 3)"
            )
            self._sheet_type_cache[stype] = best_sheet
            return best_sheet

        # ── Tier 4: LLM — ask once, cache result ──────────────────────────────
        llm_result = self._llm_resolve_sheet(stype, available)
        self._sheet_type_cache[stype] = llm_result
        if llm_result:
            logger.info(f"  Sheet LLM-resolved: '{section}' → '{llm_result}' (Tier 4)")
            return llm_result

        return section  # nothing worked — return as-is (will log a warning upstream)

    def _llm_resolve_sheet(self, section_type: str, sheet_names: list[str]) -> Optional[str]:
        """
        Ask the LLM to identify which sheet name corresponds to the given
        section type ('pl', 'bs', or 'cf'). Returns the sheet name or None.
        """
        if not self.use_llm:
            return None
        client = self._get_llm()
        if not client:
            return None

        human_name = {
            "pl": "Profit & Loss / Income Statement",
            "bs": "Balance Sheet / Financial Position",
            "cf": "Cash Flow Statement",
        }[section_type]

        # Include a few sample row labels from each sheet to help the LLM
        sheet_summaries = []
        for sname, sm in self.template.sheets.items():
            sample_labels = list(sm.row_index.keys())[:6]
            sheet_summaries.append(
                f'  "{sname}": [{", ".join(repr(l) for l in sample_labels)}]'
            )
        sheets_block = "\n".join(sheet_summaries)

        prompt = (
            f"You are helping map Excel template sheets to financial statement types.\n\n"
            f"Which of the following sheet names most likely contains the "
            f"**{human_name}**?\n\n"
            f"Sheets (name: first few row labels):\n{sheets_block}\n\n"
            f"Reply with ONLY the exact sheet name (copy-paste), or 'NONE' if none fits."
        )
        try:
            from config.settings import DEEPSEEK_MODEL
            model = os.environ.get("DEEPSEEK_MODEL", DEEPSEEK_MODEL)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60,
            )
            answer = response.choices[0].message.content.strip().strip('"')
            if answer == "NONE" or not answer:
                return None
            # Validate: must be an actual sheet name
            for sname in sheet_names:
                if sname.strip().lower() == answer.lower():
                    return sname
            return None
        except Exception as e:
            logger.warning(f"LLM sheet resolution failed: {e}")
            return None


# ── Direct mapper for template-guided results ─────────────────────────────────

def map_direct(
    results: list,             # list[ExtractionResult] from template_guided_extract_all
    template: TemplateModel,
) -> list[MappingReport]:
    """
    Direct mapping for template-guided ExtractionResults.

    In template-guided extraction the data keys ARE template field labels,
    so no fuzzy matching or LLM is needed.  This function does a straight
    exact-match lookup for each (field, year) pair and returns MappingReports
    ready for the Excel writer.

    Fields that cannot be resolved (e.g. wrong year, formula cell) are
    placed in the unmapped list as usual.
    """
    from mapper.template_reader import _resolve_row

    reports: list[MappingReport] = []

    for result in results:
        # Resolve sheet name from section label
        sheet_name = _direct_resolve_sheet(result.section, template)
        if not sheet_name or sheet_name not in template.sheets:
            logger.warning(
                f"map_direct: no sheet for section '{result.section}' — "
                "all fields unmapped"
            )
            reports.append(MappingReport(
                extraction=result,
                mapped=[],
                unmapped=list(result.data.keys()),
            ))
            continue

        sheet = template.sheets[sheet_name]
        col   = sheet.year_cols.get(result.year)
        if col is None:
            logger.warning(
                f"map_direct: year '{result.year}' not in sheet '{sheet_name}' — "
                "all fields unmapped"
            )
            reports.append(MappingReport(
                extraction=result,
                mapped=[],
                unmapped=list(result.data.keys()),
            ))
            continue

        mapped:   list[MappedField] = []
        unmapped: list[str]         = []

        for field_name, value in result.data.items():
            row = _resolve_row(sheet, field_name, result.statement_type)
            if row is None:
                unmapped.append(field_name)
                continue
            if not template.is_cell_writable(sheet_name, row, col):
                logger.debug(
                    f"  map_direct: ({row},{col}) in '{sheet_name}' "
                    "is formula/filled — skipping"
                )
                continue

            mapped.append(MappedField(
                extracted_label=field_name,
                template_label=field_name,    # same — it IS a template label
                sheet_name=sheet_name,
                row=row,
                col=col,
                year=result.year,
                value=value,
                match_method="template_guided",
                match_score=1.0,
                statement_type=result.statement_type,
                confidence_tier="MEDIUM",     # upgraded by stamp_validation if web runs
            ))

        reports.append(MappingReport(
            extraction=result,
            mapped=mapped,
            unmapped=unmapped,
        ))
        logger.info(
            f"map_direct: {result.section} {result.year} → "
            f"{len(mapped)} mapped, {len(unmapped)} unmapped"
        )

    return reports


def _direct_resolve_sheet(section: str, template: TemplateModel) -> Optional[str]:
    """
    Resolve section name → sheet name for template-guided results.
    Uses the same keyword tiers as FieldMapper._resolve_sheet_name but
    without the LLM tier (we don't need it for standard section names).
    """
    sec = section.lower()
    available = list(template.sheets.keys())

    # Tier 1: exact known mappings
    EXACT: dict[str, str] = {
        "annual p&l":           "Annual P&L",
        "annual balance sheet":  "Annual Balance Sheet",
        "annual cash flow":      "Annual Cash Flow",
    }
    for key, target in EXACT.items():
        if key in sec and target in available:
            return target

    # Tier 2: keyword scan on sheet names
    PL_KW = {"p&l", "profit", "loss", "income", "revenue"}
    BS_KW = {"balance", "bs", "assets", "liab"}
    CF_KW = {"cash", "flow", "cf"}

    kw_map = {}
    if any(k in sec for k in ("p&l", "profit", "loss", "income")):
        kw_map = PL_KW
    elif any(k in sec for k in ("balance",)):
        kw_map = BS_KW
    elif "cash" in sec:
        kw_map = CF_KW

    if kw_map:
        for sname in available:
            if any(kw in sname.lower() for kw in kw_map):
                return sname

    # Tier 3: return first sheet as last resort
    return available[0] if available else None


# ── Web validation stamping ────────────────────────────────────────────────────

def stamp_validation(
    reports: list[MappingReport],
    validation_results: list,           # list[CrossValidationResult]
) -> None:
    """
    Stamp cross-validation outcomes onto MappedField objects in-place.

    For each MappedField, find the matching CrossValidationResult by
    (template_label, year) and copy its confidence_tier, web_value,
    web_source, and flag_reason across.

    Fields with no matching CrossValidationResult keep confidence_tier="MEDIUM"
    (i.e. PDF-only, unconfirmed).
    """
    if not validation_results:
        return

    # Build lookup: (normalised_label, year) → CrossValidationResult
    # Use the same normaliser as the web collector for consistency.
    import re as _re

    def _norm(s: str) -> str:
        s = clean_label(s).lower()
        s = _re.sub(r"[^a-z0-9 ]", " ", s)
        return _re.sub(r"\s+", " ", s).strip()

    val_index: dict[tuple, object] = {}
    for vr in validation_results:
        key = (_norm(vr.template_field), vr.year)
        # Prefer higher-confidence results when there are duplicates
        existing = val_index.get(key)
        if existing is None:
            val_index[key] = vr
        else:
            _prio = {"VERY_HIGH": 0, "HIGH": 1, "MEDIUM": 2, "FLAG": 3}
            if _prio.get(vr.confidence_upgrade, 9) < _prio.get(existing.confidence_upgrade, 9):
                val_index[key] = vr

    stamped = 0
    for report in reports:
        for mf in report.mapped:
            key = (_norm(mf.template_label), mf.year)
            vr = val_index.get(key)
            if vr is None:
                # No web data for this field — leave as MEDIUM (PDF only)
                mf.confidence_tier = "MEDIUM"
                continue
            mf.confidence_tier = vr.confidence_upgrade
            mf.web_value       = vr.web_value
            mf.web_source      = vr.web_source or ""
            mf.flag_reason     = vr.flag_reason or ""
            stamped += 1

    logger.info(
        f"stamp_validation: {stamped} fields stamped with web confidence "
        f"out of {sum(len(r.mapped) for r in reports)} total mapped fields"
    )
