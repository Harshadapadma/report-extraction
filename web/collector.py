"""
WebCollector — orchestrates all web data sources and cross-validates
against PDF-extracted values.

Cross-validation rules
───────────────────────────────────────────────────────────────────
  |pdf - web| / |web|  ≤ 0.02  →  VERY_HIGH (auto-write eligible, no review)
  |pdf - web| / |web|  ≤ 0.10  →  HIGH      (auto-write eligible)
  |pdf - web| / |web|  ≤ 0.25  →  MEDIUM    (human review recommended)
  |pdf - web| / |web|  >  0.25 →  FLAG      (human review required)
  No web data found for this field →  confidence unchanged (pass-through)
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from utils.helpers import get_logger, clean_label
from web.base import CompanyIdentifier, CrossValidationResult, WebDataResult

logger = get_logger(__name__)


# ── Tolerance thresholds ───────────────────────────────────────────────────────

_VERY_HIGH_TOL = 0.02   # ≤ 2%  → VERY_HIGH confidence
_HIGH_TOL      = 0.10   # ≤ 10% → HIGH confidence
_MEDIUM_TOL    = 0.25   # ≤ 25% → MEDIUM confidence (flag if > 25%)


# ── Label normaliser for fuzzy matching ───────────────────────────────────────

def _norm(label: str) -> str:
    """Lower-case, strip punctuation/whitespace for fuzzy matching."""
    s = clean_label(label).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ── Main collector ────────────────────────────────────────────────────────────

class WebCollector:
    """
    Runs all configured web connectors in parallel and returns a merged
    list of WebDataResult objects.  Also provides cross_validate() to
    compare PDF-extracted values against web data.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        # Connectors are imported lazily so missing deps don't break anything
        self._connectors: list = []
        self._setup_connectors()

    def _setup_connectors(self):
        """Register available connectors (gracefully skips if deps missing)."""
        try:
            from web.yfinance_connector import YFinanceConnector
            self._connectors.append(("yfinance", YFinanceConnector()))
            logger.info("WebCollector: yfinance connector ready")
        except Exception as e:
            logger.warning(f"WebCollector: yfinance unavailable — {e}")

        try:
            from web.screener_connector import ScreenerConnector
            self._connectors.append(("screener", ScreenerConnector()))
            logger.info("WebCollector: Screener.in connector ready")
        except Exception as e:
            logger.warning(f"WebCollector: Screener connector unavailable — {e}")

        try:
            from web.tickertape_connector import TickertapeConnector
            self._connectors.append(("tickertape", TickertapeConnector()))
            logger.info("WebCollector: Tickertape connector ready")
        except Exception as e:
            logger.warning(f"WebCollector: Tickertape connector unavailable — {e}")

        try:
            from web.nse_connector import NSEConnector
            self._connectors.append(("nse", NSEConnector()))
            logger.info("WebCollector: NSE connector ready")
        except Exception as e:
            logger.warning(f"WebCollector: NSE connector unavailable — {e}")

        try:
            from web.bse_connector import BSEConnector
            self._connectors.append(("bse_xbrl", BSEConnector()))
            logger.info("WebCollector: BSE XBRL connector ready")
        except Exception as e:
            logger.warning(f"WebCollector: BSE connector unavailable — {e}")

        # Stubs — can be uncommented when implemented:
        # from web.mca_connector import MCAConnector
        # from web.rbi_connector import RBIDatabaseConnector

    def collect(
        self,
        company: CompanyIdentifier,
        years: list[str],
    ) -> list[WebDataResult]:
        """
        Run all connectors in parallel and return merged results.
        Each connector runs in its own thread; failures are isolated.
        """
        if not self._connectors:
            logger.warning("WebCollector: no connectors available")
            return []

        all_results: list[WebDataResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(conn.fetch, company, years): name
                for name, conn in self._connectors
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    results = fut.result()
                    all_results.extend(results)
                    logger.info(f"  {name}: {len(results)} data points collected")
                except Exception as e:
                    logger.error(f"  {name}: connector failed — {e}")

        logger.info(
            f"WebCollector: {len(all_results)} total data points "
            f"from {len(self._connectors)} source(s)"
        )
        return all_results

    # ── Cross-validation ──────────────────────────────────────────────────────

    def cross_validate(
        self,
        pdf_results,                      # list[ExtractionResult] from llm_extractor
        web_results: list[WebDataResult],
        tolerance_auto_write: float = _HIGH_TOL,
    ) -> list[CrossValidationResult]:
        """
        Compare each PDF-extracted field/year against web data.

        Returns one CrossValidationResult per (template_field, year) pair
        that has a PDF value.  Fields with no web counterpart are returned
        with confidence unchanged and web_value=None.
        """
        # Build lookup: normalised_template_field → [(year, web_result)]
        web_index: dict[str, list[WebDataResult]] = {}
        for wr in web_results:
            if not wr.template_field:
                continue
            key = _norm(wr.template_field)
            web_index.setdefault(key, []).append(wr)

        validation_results: list[CrossValidationResult] = []

        for pdf_result in pdf_results:
            year = pdf_result.year
            pdf_source = getattr(pdf_result, "source", "pdf")

            for template_field, pdf_value in pdf_result.data.items():
                if pdf_value is None:
                    continue

                # Find web candidates for this field/year
                web_match = self._find_web_match(
                    template_field, year, web_index
                )

                if web_match is None:
                    # No web data — pass through without touching confidence
                    validation_results.append(CrossValidationResult(
                        template_field=template_field,
                        year=year,
                        pdf_value=pdf_value,
                        pdf_source=pdf_source,
                        web_value=None,
                        web_source=None,
                        agreement=False,
                        tolerance_pct=0.0,
                        confidence_upgrade="MEDIUM",  # default
                        flag_reason=None,
                    ))
                    continue

                web_val  = web_match.value
                web_src  = web_match.source

                # Compute relative difference
                if web_val == 0:
                    tol = abs(pdf_value) / 1.0   # avoid division by zero
                else:
                    tol = abs(pdf_value - web_val) / abs(web_val)

                # Assign confidence tier
                if tol <= _VERY_HIGH_TOL:
                    confidence = "VERY_HIGH"
                    agreement  = True
                    flag       = None
                elif tol <= _HIGH_TOL:
                    confidence = "HIGH"
                    agreement  = True
                    flag       = None
                elif tol <= _MEDIUM_TOL:
                    confidence = "MEDIUM"
                    agreement  = False
                    flag = (
                        f"PDF ({pdf_value:.2f}) differs from web "
                        f"({web_val:.2f}) by {tol:.1%} — review recommended"
                    )
                else:
                    confidence = "FLAG"
                    agreement  = False
                    flag = (
                        f"PDF ({pdf_value:.2f}) vs web ({web_val:.2f}) "
                        f"differ by {tol:.1%} — large discrepancy, human review required"
                    )

                validation_results.append(CrossValidationResult(
                    template_field=template_field,
                    year=year,
                    pdf_value=pdf_value,
                    pdf_source=pdf_source,
                    web_value=web_val,
                    web_source=web_src,
                    agreement=agreement,
                    tolerance_pct=tol,
                    confidence_upgrade=confidence,
                    flag_reason=flag,
                ))

        # Log summary
        n_agree  = sum(1 for r in validation_results if r.agreement)
        n_flag   = sum(1 for r in validation_results if r.confidence_upgrade == "FLAG")
        n_noweb  = sum(1 for r in validation_results if r.web_value is None)
        logger.info(
            f"Cross-validation: {len(validation_results)} fields | "
            f"agreed={n_agree} | flagged={n_flag} | no-web-data={n_noweb}"
        )
        return validation_results

    def _find_web_match(
        self,
        template_field: str,
        year: str,
        web_index: dict[str, list[WebDataResult]],
    ) -> Optional[WebDataResult]:
        """
        Find the best matching WebDataResult for a given template field + year.
        Uses normalised label matching: exact first, then token-overlap fallback.
        Prefers higher-confidence sources (yfinance > bse_xbrl).
        """
        norm_field = _norm(template_field)
        candidates: list[WebDataResult] = []

        # Exact normalised match
        if norm_field in web_index:
            candidates = [w for w in web_index[norm_field] if w.year == year]

        # Token-overlap fallback: any key that shares ≥60% of tokens
        if not candidates:
            field_tokens = set(norm_field.split())
            for key, wrs in web_index.items():
                key_tokens = set(key.split())
                if not field_tokens or not key_tokens:
                    continue
                overlap = len(field_tokens & key_tokens) / len(field_tokens | key_tokens)
                if overlap >= 0.6:
                    candidates.extend(w for w in wrs if w.year == year)

        if not candidates:
            return None

        # Priority order (lower = preferred):
        #   yfinance   — annual, HIGH confidence
        #   screener   — annual, HIGH confidence, different data coverage
        #   tickertape — annual, HIGH confidence, independent source
        #   nse        — quarterly aggregated, MEDIUM confidence (official exchange)
        #   bse_xbrl   — quarterly aggregated, MEDIUM confidence
        _source_priority = {
            "yfinance":   0,
            "screener":   1,
            "tickertape": 2,
            "nse":        3,
            "bse_xbrl":   4,
            "mca_xbrl":   5,
            "rbi_dbie":   6,
        }
        candidates.sort(key=lambda w: _source_priority.get(w.source, 99))
        return candidates[0]

    # ── Gap-fill: web as primary source for fields PDF missed ─────────────────

    def fill_missing_from_web(
        self,
        pdf_results,                       # list[ExtractionResult]
        web_results: list[WebDataResult],
        template_model,                    # TemplateModel — to know which fields exist
    ) -> list:
        """
        For every template field that the PDF extraction DID NOT produce a value
        for, check if the web data has a value and, if so, create a synthetic
        ExtractionResult so that field can be written to the template.

        Returns a list of NEW ExtractionResult objects (one per year that has
        web-only data).  These are marked with source "web_primary" and
        confidence HIGH (yfinance) or MEDIUM (bse_xbrl).

        Callers should extend their existing pdf_results list with these.
        """
        from extractor.llm_extractor import ExtractionResult

        # Build set of (template_field, year) already covered by PDF
        pdf_covered: set[tuple] = set()
        for r in pdf_results:
            for field in r.data:
                pdf_covered.add((_norm(field), r.year))

        # Group web results by year → {template_field: value}
        year_web: dict[str, dict[str, WebDataResult]] = {}
        for wr in web_results:
            if not wr.template_field:
                continue
            key = (_norm(wr.template_field), wr.year)
            if key in pdf_covered:
                continue  # PDF already has this — don't create duplicate
            year_web.setdefault(wr.year, {})[wr.template_field] = wr

        if not year_web:
            return []

        # Verify that each template_field actually exists in the template
        all_template_labels: set[str] = set()
        for sheet in template_model.sheets.values():
            for label in sheet.row_index:
                all_template_labels.add(_norm(label))

        new_results: list[ExtractionResult] = []
        for year, field_map in year_web.items():
            data: dict[str, float] = {}
            for tmpl_field, wr in field_map.items():
                if _norm(tmpl_field) not in all_template_labels:
                    continue  # web mapped to a non-existent template field
                data[tmpl_field] = wr.value

            if not data:
                continue

            # Determine statement type — default consolidated
            r = ExtractionResult(
                section="Annual Balance Sheet",   # best guess; mapper re-resolves sheet
                statement_type="consolidated",
                year=year,
                currency="INR Crores",
                data=data,
                notes="source=web_primary",
                confidence=0.8,
            )
            new_results.append(r)
            logger.info(
                f"fill_missing_from_web: year={year} → "
                f"{len(data)} web-only field(s) added as primary source"
            )

        return new_results

    # ── Convenience: summary report ───────────────────────────────────────────

    def validation_summary(
        self, results: list[CrossValidationResult]
    ) -> dict:
        """Return a summary dict suitable for display or logging."""
        total = len(results)
        if total == 0:
            return {"total": 0}
        return {
            "total":      total,
            "very_high":  sum(1 for r in results if r.confidence_upgrade == "VERY_HIGH"),
            "high":       sum(1 for r in results if r.confidence_upgrade == "HIGH"),
            "medium":     sum(1 for r in results if r.confidence_upgrade == "MEDIUM"),
            "flagged":    sum(1 for r in results if r.confidence_upgrade == "FLAG"),
            "no_web":     sum(1 for r in results if r.web_value is None),
            "auto_write_eligible": sum(
                1 for r in results
                if r.confidence_upgrade in ("VERY_HIGH", "HIGH") and r.web_value is not None
            ),
        }
