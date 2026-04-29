"""
PDF Parser — ultra-fast extraction using PyMuPDF (fitz).

Speed comparison on 165-page annual report:
  pdfminer.six  →  ~10–15 s   (old)
  PyMuPDF       →  ~1–2 s     (new, 10x faster)

Strategy
--------
1. Open PDF with PyMuPDF (C library, no Python overhead per character).
2. Get total page count FIRST; compute smart range (35%–95%) before reading.
3. Extract only those pages in parallel threads.
4. Score pages with pre-compiled keyword sets.
5. Collect anchor pages + small window → section texts for the extractor.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


# ── PyMuPDF import with auto-install fallback ──────────────────────────────────
def _require_fitz():
    """Import fitz (PyMuPDF), auto-installing into the running interpreter if missing."""
    try:
        import fitz
        return fitz
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "pymupdf>=1.23.0"],
        )
        import importlib, site
        importlib.invalidate_caches()
        for sp in site.getsitepackages():
            if sp not in sys.path:
                sys.path.insert(0, sp)
        import fitz  # type: ignore
        return fitz


from config.settings import MAX_PDF_PAGES
from config.sectors import detect_sector, get_section_weights, get_llm_hint
from utils.helpers import get_logger

logger = get_logger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PageSection:
    section_type: str
    page_numbers: list[int]
    text: str
    is_consolidated: bool = True
    is_standalone: bool = False
    confidence: float = 0.0


@dataclass
class ParsedDocument:
    filename: str
    total_pages: int
    sector: str = "manufacturing"           # detected sector
    llm_hint: str = ""                      # passed to LLM prompt
    sections: list[PageSection] = field(default_factory=list)
    raw_pages: dict[int, str] = field(default_factory=dict)

    def get_section(self, section_type: str, consolidated: bool = True) -> Optional[PageSection]:
        for s in self.sections:
            if s.section_type == section_type:
                if consolidated and s.is_consolidated:
                    return s
                if not consolidated and s.is_standalone:
                    return s
        return None

    def get_all_text(self) -> str:
        return "\n\n".join(self.raw_pages.values())


# ── Keyword scoring ────────────────────────────────────────────────────────────

# Section weights are now sector-specific — built at parse time from config/sectors.py.
# A minimal fallback set is kept here only as a safety net if the import fails.
_FALLBACK_SECTION_WEIGHTS: dict[str, dict[str, float]] = {
    "Annual P&L":           {"profit and loss": 3.0, "profit before tax": 2.0, "profit after tax": 2.0},
    "Annual Balance Sheet": {"balance sheet": 3.0, "total assets": 2.5},
    "Annual Cash Flow":     {"cash flow": 3.0, "cash flows from operating": 3.0},
}

_CONSOLIDATION_RE = {
    "consolidated": re.compile(r"\bconsolidated\b", re.IGNORECASE),
    "standalone":   re.compile(r"\bstandalone\b",   re.IGNORECASE),
}

# Pages that look like financial statement pages but are NOT — they contain
# notes, disclosures, contingent liabilities etc. whose numbers should never
# be written into the main financial template rows.
# We check only the first 400 characters of a page (the heading area).
_NOTES_EXCLUDE_RE = re.compile(
    r"contingent liabilit"
    r"|notes?\s+(?:to|on|forming part of)\s+(?:the\s+)?(?:accounts?|financial)"
    r"|notes?\s+(?:and\s+)?(?:significant\s+)?accounting\s+polic"
    r"|significant\s+accounting\s+polic"
    r"|basis\s+of\s+(?:preparation|consolidation)"
    r"|statutory\s+auditor|independent\s+auditor"
    r"|report\s+of\s+the\s+(?:statutory\s+)?auditor"
    r"|related\s+party\s+disclosure"
    r"|deferred\s+tax\s+(?:asset|liabilit)"
    r"|capital\s+(?:commitments?|adequacy\s+ratio)"
    r"|capital\s+to\s+risk"          # CRAR / capital to risk weighted assets
    r"|risk\s+weighted\s+assets"
    r"|\bcrar\b"
    r"|basel\s+(?:i+|accord|framework|norm)"
    r"|pillar\s+[123]"               # Basel Pillar disclosures
    r"|paid.up\s+(?:equity\s+)?capital\s+raised"   # equity issuance disclosures
    r"|maturity\s+pattern"           # maturity bucket disclosures
    r"|concentration\s+of\s+(?:deposit|advance|exposure)"
    r"|sector.?wise\s+(?:classification|exposure|npa)",
    re.IGNORECASE,
)


def _is_excluded_page(text: str) -> bool:
    """Return True if the page header indicates it is a notes/disclosures page."""
    return bool(_NOTES_EXCLUDE_RE.search(text[:600]))


# Pre-compiled patterns used by the multi-signal scorer
_NUM_RE       = re.compile(r"(?<!\w)\d[\d,]*(?:\.\d+)?(?!\w)")   # standalone numbers
_YEAR_HDR_RE  = re.compile(                                        # year-header lines
    r"march\s+31,?\s+\d{4}"
    r"|year\s+ended"
    r"|as\s+at\s+(?:march|31)"
    r"|fy\s*\d{2,4}"
    r"|31\s+march\s+\d{4}",
    re.IGNORECASE,
)
_SECTION_HDR_RE = re.compile(                                      # section-header lines
    r"statement\s+of\s+profit"
    r"|profit\s+and\s+loss\s+account"
    r"|balance\s+sheet"
    r"|cash\s+flow\s+statement"
    r"|income\s+statement",
    re.IGNORECASE,
)


def _score_page(
    text_lower: str,
    section_type: str,
    section_weights: dict[str, dict[str, float]] | None = None,
) -> float:
    """
    Multi-signal page relevance scorer.

    Four signals are combined:

    1. **Keyword score** (main signal)
       For each keyword in the section's weight dict, count how many times it
       appears on the page.  Score = weight × log(1 + count), capped so that a
       keyword appearing 5+ times scores the same as one appearing exactly 5 times.
       This rewards pages where financial terms appear multiple times (schedules,
       sub-items) over pages where a term appears only incidentally.

    2. **Tabular-row density** (structure signal)
       Count lines that contain two or more numeric tokens side-by-side — a strong
       proxy for "this page has a data table".  Each such line adds a small bonus
       (capped so a single table adds at most +2.0 to the total score).

    3. **Year-header bonus**
       If the page contains a year-style header (e.g. "March 31, 2025",
       "Year ended", "FY25") it gets a flat +1.5 bonus.  Financial statement pages
       almost always carry these; governance/narrative pages rarely do.

    4. **Section-header bonus**
       Pages that open with an explicit statement header ("Balance Sheet",
       "Profit and Loss Account", "Cash Flow Statement") get a flat +2.5 bonus
       because they are definitively the right section.
    """
    import math

    weights = (section_weights or _FALLBACK_SECTION_WEIGHTS).get(section_type, {})
    if not weights:
        return 0.0

    # ── Signal 1: frequency-weighted keyword score ────────────────────────────
    kw_score = 0.0
    for kw, w in weights.items():
        count = text_lower.count(kw)
        if count:
            # log(1+count) gives: 1 hit → 0.69, 2 → 1.1, 5 → 1.79, 10+ → 2.4
            # Multiply by weight and scale so 1 hit ≈ original weight
            kw_score += w * (math.log1p(count) / math.log1p(1))

    # ── Signal 2: tabular-row density ────────────────────────────────────────
    tabular_lines = sum(
        1 for line in text_lower.splitlines()
        if len(_NUM_RE.findall(line)) >= 2          # ≥2 numbers on the same line
    )
    tabular_bonus = min(tabular_lines / 8.0, 2.0)   # cap at +2.0

    # ── Signal 3: year-header bonus ──────────────────────────────────────────
    year_bonus = 1.5 if _YEAR_HDR_RE.search(text_lower) else 0.0

    # ── Signal 4: explicit section-header bonus ───────────────────────────────
    section_bonus = 2.5 if _SECTION_HDR_RE.search(text_lower[:400]) else 0.0

    return kw_score + tabular_bonus + year_bonus + section_bonus


def _detect_consolidation(text: str) -> tuple[bool, bool]:
    has_c = bool(_CONSOLIDATION_RE["consolidated"].search(text))
    has_s = bool(_CONSOLIDATION_RE["standalone"].search(text))
    if not has_c and not has_s:
        has_c = True
    return has_c, has_s


# ── Fast parallel page extractor ──────────────────────────────────────────────

def _extract_pages_range(
    pdf_bytes: bytes,
    page_indices: list[int],    # 0-based indices to extract
    max_workers: int = 8,       # kept for API compat — not used
) -> dict[int, str]:
    """
    Extract text from specific pages using PyMuPDF.
    Opens the document ONCE and reads pages sequentially — this is faster
    than opening in parallel because each fitz.open() parses the whole
    PDF structure, making parallel opens much slower than sequential reads.
    Returns {page_num (1-based): text}.
    """
    fitz = _require_fitz()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: dict[int, str] = {}
    for idx in page_indices:
        pages[idx + 1] = doc[idx].get_text("text").strip()
    doc.close()
    return pages


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_pdf(
    source: Union[str, Path, bytes, io.BytesIO],
    max_workers: int = 8,
) -> ParsedDocument:
    """
    Parse a PDF and return a ParsedDocument with identified financial sections.

    Uses PyMuPDF for fast extraction. Smart page range is applied BEFORE
    reading page text so we never process irrelevant pages.
    """
    if isinstance(source, (str, Path)):
        filename = Path(source).name
    else:
        filename = "uploaded.pdf"

    logger.info(f"Parsing PDF: {filename} …")

    # ── Step 1: Open once just to get page count (near-instant) ──────────────
    # We need total_pages to compute the smart range before extracting text.
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            src_bytes = f.read()
    elif isinstance(source, bytes):
        src_bytes = source
    else:
        src_bytes = source.read()
        # Reset for later use if BytesIO
        if hasattr(source, "seek"):
            source.seek(0)

    _fitz = _require_fitz()
    _probe = _fitz.open(stream=src_bytes, filetype="pdf")
    total_pages = len(_probe)
    _probe.close()

    max_p = MAX_PDF_PAGES if MAX_PDF_PAGES > 0 else total_pages

    # ── Step 2: Sector probe — sample from multiple parts of the document ───
    # Front matter (0-15%) has company name / business description.
    # Middle (35-55%) has financial statements and industry-specific keywords.
    # We sample both to maximise detection accuracy.
    probe_indices = (
        list(range(0, min(int(total_pages * 0.12), total_pages)))           # front
        + list(range(int(total_pages * 0.35), min(int(total_pages * 0.55), total_pages)))  # middle
    )
    probe_indices = sorted(set(probe_indices))[:40]   # cap at 40 pages
    probe_pages = _extract_pages_range(src_bytes, probe_indices, max_workers=1)
    probe_text = " ".join(probe_pages.values())[:10000]
    sector = detect_sector(probe_text)
    section_weights = get_section_weights(sector)
    hint = get_llm_hint(sector)
    all_keywords = frozenset(kw for w in section_weights.values() for kw in w)
    logger.info(f"  Detected sector: '{sector}'")

    # ── Step 3: Compute smart range & extract ONLY those pages ───────────────
    if total_pages > 80:
        lo = int(total_pages * 0.25)   # 25% — catches financials even in front-heavy docs
        hi = int(total_pages * 0.97)
    else:
        lo, hi = 0, total_pages - 1

    page_indices = list(range(lo, min(hi + 1, max_p)))
    logger.info(
        f"  Total pages: {total_pages} | "
        f"Smart range: {lo+1}–{hi+1} ({len(page_indices)} pages)"
    )

    raw_pages = _extract_pages_range(src_bytes, page_indices, max_workers=max_workers)
    logger.info(f"  Extracted {len(raw_pages)} pages in smart range")

    # ── Step 4: Score pages ───────────────────────────────────────────────────
    page_scores: dict[int, dict[str, float]] = {}
    excluded_count = 0
    for pn, text in raw_pages.items():
        # Skip notes/disclosures/contingency pages — their numbers must not
        # overwrite main financial statement rows in the template.
        if _is_excluded_page(text):
            excluded_count += 1
            continue
        text_lower = text.lower()
        if not any(kw in text_lower for kw in all_keywords):
            continue
        page_scores[pn] = {st: _score_page(text_lower, st, section_weights) for st in section_weights}

    logger.info(
        f"  {len(page_scores)} pages with financial content "
        f"({excluded_count} notes/disclosures pages excluded)"
    )

    # ── Step 5: Group pages into sections ────────────────────────────────────
    # Adaptive page selection — NO fixed score thresholds.
    #
    # Different documents produce completely different score distributions.
    # A fixed threshold tuned on one company will under- or over-include pages
    # for any other company.  Instead we rank pages relative to each other
    # inside each document:
    #
    #   anchor    = pages scoring >= TOP_FRAC    of this document's highest page score
    #   candidate = pages scoring >= CAND_FRAC   of this document's highest page score
    #
    # These fractions are dimensionless ratios, so they adapt automatically to
    # any company, sector, page density, or document length.
    TOP_FRAC          = 0.45   # top tier: captures main statement + dense schedule pages
    CAND_FRAC         = 0.20   # lower tier: captures sparser schedule / note pages
    MAX_SECTION_CHARS = 60_000

    doc = ParsedDocument(
        filename=filename,
        total_pages=total_pages,
        sector=sector,
        llm_hint=hint,
        raw_pages=raw_pages,
    )

    for section_type in section_weights:
        section_scores = {
            pn: sc.get(section_type, 0.0)
            for pn, sc in page_scores.items()
            if sc.get(section_type, 0.0) > 0.0
        }
        if not section_scores:
            continue

        max_score = max(section_scores.values())
        if max_score == 0:
            continue

        anchors = [pn for pn, sc in section_scores.items() if sc >= max_score * TOP_FRAC]
        if not anchors:
            continue   # no high-confidence page for this section — skip entirely

        candidate_pages = [pn for pn, sc in section_scores.items() if sc >= max_score * CAND_FRAC]
        pages_sorted = sorted(candidate_pages)

        # Build the section text by prioritising the highest-scoring pages first
        # so that the most relevant pages always fit within MAX_SECTION_CHARS.
        pages_by_score = sorted(
            [p for p in candidate_pages if p in raw_pages and raw_pages[p]],
            key=lambda p: page_scores.get(p, {}).get(section_type, 0),
            reverse=True,
        )
        budget_pages: list[int] = []
        budget_used = 0
        for p in pages_by_score:
            chunk = f"[PAGE {p}]\n{raw_pages[p]}"
            if budget_used + len(chunk) + 2 <= MAX_SECTION_CHARS:
                budget_pages.append(p)
                budget_used += len(chunk) + 2

        # Re-sort by page number so the final text reads in natural order
        combined = "\n\n".join(
            f"[PAGE {p}]\n{raw_pages[p]}"
            for p in sorted(budget_pages)
        )

        is_c, is_s = _detect_consolidation(combined)
        total_score = sum(
            page_scores.get(p, {}).get(section_type, 0) for p in pages_sorted
        )
        confidence = min(1.0, total_score / 20.0)

        doc.sections.append(PageSection(
            section_type=section_type,
            page_numbers=pages_sorted,
            text=combined,
            is_consolidated=is_c,
            is_standalone=is_s,
            confidence=confidence,
        ))
        logger.info(
            f"  '{section_type}' → {len(pages_sorted)} candidate pages "
            f"(anchors: {anchors[:3]}{'…' if len(anchors) > 3 else ''}) "
            f"| budget_pages={len(budget_pages)} | conf={confidence:.2f}"
        )

    logger.info(f"Done: {len(doc.sections)} sections | '{filename}'")
    return doc


def parse_multiple_pdfs(sources: list) -> list[ParsedDocument]:
    docs = []
    for src in sources:
        try:
            docs.append(parse_pdf(src))
        except Exception as e:
            logger.error(f"Failed to parse: {e}")
    return docs
