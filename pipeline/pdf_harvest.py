"""
pipeline/pdf_harvest.py
───────────────────────
Harvest data from a company's annual report PDFs.

For each requested year:
  1. Locate PDFs (NSE → BSE → Screener fallback)
  2. Download (cached on disk for 7 days)
  3. Parse into sections (P&L / BS / CF) using extractor.pdf_parser
  4. Run pdfplumber table extraction → structured (label, year, value) cells
  5. Concatenate section text → returned to caller for the financial agent's
     search_pdf_pages tool

Returned by `harvest_pdfs()`:
  PDFHarvestResult
    .pool_items   : list[WebDataResult] — feeds the unified pool
    .pdf_text     : str                — concatenated text of all relevant pages
    .pdfs_used    : list[dict]         — manifest of PDFs we actually processed
    .errors       : list[str]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from web.base import CompanyIdentifier, WebDataResult

from pipeline import cache as cache_mod

logger = logging.getLogger(__name__)


@dataclass
class PDFHarvestResult:
    pool_items: list[WebDataResult] = field(default_factory=list)
    pdf_text:   str = ""
    pdfs_used:  list[dict] = field(default_factory=list)
    errors:     list[str] = field(default_factory=list)
    total_bytes: int = 0
    elapsed_sec: float = 0.0
    discovery_layers: list[dict] = field(default_factory=list)  # per-layer status

    def summary(self) -> str:
        return (
            f"{len(self.pdfs_used)} PDFs, "
            f"{len(self.pool_items)} structured cells, "
            f"{len(self.pdf_text)//1000}k chars text, "
            f"{self.total_bytes//(1024*1024)}MB · {self.elapsed_sec:.1f}s"
        )


# ── PDF discovery ───────────────────────────────────────────────────────────

def _discover_pdf_urls(company: CompanyIdentifier, years: list[str]
                       ) -> tuple[list[dict], list[dict]]:
    """
    Multi-tier discovery via pipeline.pdf_discovery.
    Returns (urls, layer_status) — layer_status describes what each source did.
    """
    key = ("pdf_urls_v2", company.nse_symbol or company.bse_code,
           tuple(sorted(years)))
    cached = cache_mod.get(key, ttl_sec=7 * 24 * 3600)
    if cached is not None:
        return cached

    from pipeline.pdf_discovery import discover_all
    dr = discover_all(company, years)
    out = (dr.urls, dr.layer_status)
    # Only cache if we found at least one URL — prevents sticky 0-result misses
    if dr.urls:
        cache_mod.put(key, out)
    return out


# ── Single-PDF processing ──────────────────────────────────────────────────

def _download_cached(url: str) -> Optional[bytes]:
    key = ("pdf_bytes", url)
    cached = cache_mod.get(key, ttl_sec=30 * 24 * 3600)  # PDFs change rarely
    if cached is not None:
        return cached
    try:
        from web.pdf_fetcher import download_pdf
        data = download_pdf(url)
    except Exception as exc:
        logger.warning(f"pdf_harvest: download failed for {url}: {exc}")
        return None
    if data:
        cache_mod.put(key, data)
    return data


def _process_pdf(
    pdf_bytes: bytes,
    year_label: str,
    statement_type: str = "consolidated",
    max_text_chars: int = 60_000,
    is_presentation: bool = False,
) -> tuple[list[WebDataResult], str]:
    """
    Parse a single PDF into:
      - structured (label, year, value) cells (via pdfplumber tables)
      - concatenated section text capped at max_text_chars

    `is_presentation` : True for investor presentations / quarterly update decks.
                        These are typically short (≤80 pages) and have no standard
                        P&L/BS sections — the parser finds nothing to index under
                        doc.sections.  When set, we always fall back to dumping
                        ALL raw page text so the agent can search the full deck.
    """
    pool_items: list[WebDataResult] = []
    text_parts: list[str] = []

    try:
        import io as _io
        from extractor.pdf_parser import parse_pdf
        from extractor.table_extractor import extract_tables_from_pdf

        doc = parse_pdf(_io.BytesIO(pdf_bytes))

        # Auto-detect presentations: ≤80 pages AND parser found no/few sections
        if not is_presentation:
            is_presentation = (
                doc.total_pages <= 80 and
                sum(len(s.text) for s in doc.sections) < 2000
            )

        # ── Primary: pdfplumber grid-table extraction (works on PDFs WITH lines)
        try:
            tbl_results = extract_tables_from_pdf(
                pdf_source=pdf_bytes,
                parsed_doc=doc,
                statement_type=statement_type,
            )
        except Exception as exc:
            tbl_results = []
            logger.warning(f"pdf_harvest: table extract failed: {exc}")

        for r in tbl_results:
            for label, val in (r.data or {}).items():
                if val is None:
                    continue
                try:
                    fv = float(val)
                except (TypeError, ValueError):
                    continue
                pool_items.append(WebDataResult(
                    source="pdf",
                    raw_field=label,
                    template_field="",
                    value=fv,
                    year=r.year or year_label,
                    confidence="HIGH",
                    unit_applied=f"INR Crores (PDF/{statement_type})",
                ))

        # ── Fallback: text-line extractor (for PDFs WITHOUT grid lines).
        # Many real annual reports (Bajaj, HDFC, Reliance, ITC) format tables
        # using whitespace alignment only — pdfplumber.extract_tables() finds
        # nothing, but text extraction is fine. Tested 100% accurate on Bajaj.
        # Only run the fallback if the grid path got us very few cells.
        if len(pool_items) < 30:
            try:
                from extractor.text_line_extractor import extract_lines_from_pdf
                # Restrict to financial-statement page range to keep it fast
                # (typical AR: statements live around p180-p280 in a 400-page doc)
                import io as _io2, pdfplumber as _pp
                with _pp.open(_io2.BytesIO(pdf_bytes)) as _pdf:
                    n_pages = len(_pdf.pages)
                # Heuristic: pick the middle 25% of the doc as candidate range
                if n_pages > 100:
                    start = max(1, int(n_pages * 0.45))
                    end   = min(n_pages, int(n_pages * 0.75))
                else:
                    start, end = 1, n_pages
                txt_rows = extract_lines_from_pdf(pdf_bytes,
                                                   page_range=(start, end),
                                                   max_pages=80)
                added = 0
                for tr in txt_rows:
                    for yr, v in tr.values.items():
                        try:
                            pool_items.append(WebDataResult(
                                source="pdf",
                                raw_field=tr.label,
                                template_field="",
                                value=float(v),
                                year=yr,
                                confidence="MEDIUM",   # less certain than grid path
                                unit_applied=f"text_line_extractor (PDF p{tr.page})",
                            ))
                            added += 1
                        except (TypeError, ValueError):
                            pass
                if added:
                    logger.info(f"pdf_harvest: text-line fallback added "
                                f"{added} cells from {len(txt_rows)} rows "
                                f"(pages {start}-{end})")
            except Exception as exc:
                logger.warning(f"pdf_harvest: text-line fallback failed: {exc}")

        # ── Text for agent search ─────────────────────────────────────────────
        # For standard annual reports: only indexed sections (P&L / BS / CF /
        # Operating Metrics) — keeps context focused and avoids noise.
        # For investor presentations (short decks, ≤80 pages, or where the
        # parser found no sections): dump ALL raw page text so the agent can
        # search every slide for segment data, KPIs, and business metrics.
        if is_presentation or not doc.sections:
            # Dump full text of every extracted raw page (all slides)
            all_text = doc.get_all_text()
            if all_text:
                text_parts.append(
                    f"\n[Investor Presentation {year_label}]\n{all_text}"
                )
            logger.info(
                f"pdf_harvest: presentation mode — "
                f"dumped {len(all_text)//1000}k chars from "
                f"{len(doc.raw_pages)} pages"
            )
        else:
            for sec in doc.sections:
                text_parts.append(f"\n[{sec.section_type} {year_label}]\n{sec.text}")
                if sum(len(t) for t in text_parts) > max_text_chars:
                    break

    except Exception as exc:
        logger.warning(f"pdf_harvest: parse failed: {exc}")

    text = "".join(text_parts)
    if len(text) > max_text_chars:
        text = text[:max_text_chars] + "\n[…truncated]"
    return pool_items, text


# ── Main entry ──────────────────────────────────────────────────────────────

def harvest_pdfs(
    company: CompanyIdentifier,
    years: list[str],
    statement_type: str = "consolidated",
    max_pdfs: int = 4,
    text_per_pdf: int = 60_000,
    user_pdf_paths: Optional[list[str]] = None,
    save_dir: Optional[str] = None,        # if set, save all PDFs here with readable names
    discover_only: bool = False,           # download but skip text extraction
) -> PDFHarvestResult:
    """
    Discover, download, parse, extract — all cached.

    `user_pdf_paths` : list of local PDF paths the user supplied directly.
                       These are processed FIRST, then auto-discovered PDFs
                       are added (deduped by year). Lets the pipeline succeed
                       even when NSE/BSE/Screener return nothing.
    """
    t0 = time.time()
    res = PDFHarvestResult()

    # ── Set up visible save folder (if requested) ────────────────────────────
    save_path: Optional[Path] = None
    if save_dir:
        try:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            res.errors.append(f"could not create save dir {save_dir}: {exc}")
            save_path = None

    def _save_visible(pdf_bytes: bytes, year: str, source: str,
                      filename: str) -> Optional[str]:
        """Save PDF bytes to the visible folder with a readable name."""
        if not save_path:
            return None
        # Build readable filename: <YEAR>_<SOURCE>_<original-name>.pdf
        safe_orig = "".join(c if c.isalnum() or c in "._-" else "_"
                            for c in (filename or "")) or "ar.pdf"
        if not safe_orig.lower().endswith(".pdf"):
            safe_orig += ".pdf"
        nice = f"{year or 'unknown'}_{source}_{safe_orig}"[:120]
        out = save_path / nice
        try:
            out.write_bytes(pdf_bytes)
            return str(out)
        except Exception as exc:
            res.errors.append(f"save failed for {nice}: {exc}")
            return None

    # ── User-supplied PDFs go first (always reliable) ────────────────────────
    user_pdf_paths = user_pdf_paths or []
    used_years_from_user: set[str] = set()
    for upath in user_pdf_paths:
        try:
            from pathlib import Path as _P
            p = _P(upath)
            if not p.exists():
                res.errors.append(f"user PDF not found: {upath}")
                continue
            with open(p, "rb") as f:
                pdf_bytes = f.read()
            res.total_bytes += len(pdf_bytes)
            saved_path = _save_visible(pdf_bytes, "user", "upload", p.name)
            if discover_only:
                pool_items, text = [], ""
            else:
                # User-uploaded PDFs may be investor presentations, quarterly
                # update decks, or annual reports — treat as presentation so
                # we always capture full text (the auto-detect inside
                # _process_pdf will also kick in for short decks).
                pool_items, text = _process_pdf(
                    pdf_bytes, year_label="",
                    statement_type=statement_type, max_text_chars=text_per_pdf,
                    is_presentation=True,
                )
            res.pool_items.extend(pool_items)
            if text:
                res.pdf_text += "\n" + text
            res.pdfs_used.append({
                "year": "(user-supplied)", "source": "user_upload",
                "filename": p.name, "size_mb": round(len(pdf_bytes)/(1024*1024), 1),
                "structured_cells": len(pool_items),
                "text_chars": len(text),
                "saved_to": saved_path,
            })
            # Track which years got data from this PDF (so auto-fetch can skip)
            for it in pool_items:
                if it.year:
                    used_years_from_user.add(it.year)
            logger.info(f"pdf_harvest: user PDF {p.name} → "
                        f"{len(pool_items)} cells, {len(text)//1000}k text chars")
        except Exception as exc:
            res.errors.append(f"user PDF {upath}: {exc}")

    # ── Auto-discovery for years not already covered by user PDFs ────────────
    urls, layer_status = _discover_pdf_urls(company, years)
    res.discovery_layers = layer_status
    if not urls and not user_pdf_paths:
        res.elapsed_sec = time.time() - t0
        return res

    # Skip auto-discovered PDFs for years the user already covered
    if used_years_from_user:
        urls = [u for u in urls if u.get("year") not in used_years_from_user]

    # Cap to the most recent N PDFs to keep time bounded
    urls = urls[:max_pdfs]

    for entry in urls:
        url = entry.get("url")
        yr  = entry.get("year") or ""
        if not url:
            continue
        try:
            pdf_bytes = _download_cached(url)
        except Exception as exc:
            res.errors.append(f"download {entry.get('filename')}: {exc}")
            continue
        if not pdf_bytes:
            res.errors.append(f"empty download: {entry.get('filename')}")
            continue
        res.total_bytes += len(pdf_bytes)

        saved_path = _save_visible(
            pdf_bytes, yr, entry.get("source") or "auto",
            entry.get("filename") or "ar.pdf",
        )
        if discover_only:
            pool_items, text = [], ""
        else:
            pool_items, text = _process_pdf(
                pdf_bytes, year_label=yr,
                statement_type=statement_type, max_text_chars=text_per_pdf,
            )
        res.pool_items.extend(pool_items)
        if text:
            res.pdf_text += "\n" + text
        res.pdfs_used.append({
            "year": yr, "source": entry.get("source"),
            "filename": entry.get("filename"),
            "size_mb": entry.get("size_mb"),
            "structured_cells": len(pool_items),
            "text_chars": len(text),
            "saved_to": saved_path,
            "url": url,
        })
        logger.info(
            f"pdf_harvest: {entry.get('filename')} "
            f"({yr}/{entry.get('source')}) → {len(pool_items)} cells, "
            f"{len(text)//1000}k chars"
        )

    res.elapsed_sec = time.time() - t0
    return res


__all__ = ["harvest_pdfs", "PDFHarvestResult"]
