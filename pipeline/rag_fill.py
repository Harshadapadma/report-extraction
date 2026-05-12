"""
pipeline/rag_fill.py
────────────────────
Standalone RAG-based template filler. Lives ALONGSIDE pipeline/llm_fill.py.
Touching this file should never break the existing pipeline.

Design:
  1. Reuse llm_fill's _pdf_to_markdown to get clean .md per PDF
  2. Chunk the .md into ~300-token pieces with metadata
     (file, page, scope: STD|CONS|UNK, period_type, years_mentioned)
  3. Build an in-memory vector index over chunks:
       - If sentence-transformers is installed → semantic embeddings (best)
       - Else fall back to TF-IDF cosine similarity (works, less smart)
  4. For each template cell (sheet, label, statement_type, year):
       a. Build a natural-language query
       b. Vector-search with metadata filter (scope, year)
       c. Pass top-k chunks to LLM (or pattern-extract directly if no LLM)
  5. Write fills via the existing _write_filled_excel function

Modes:
  • `rag_fill_template(..., mode="pattern")`  — free, no LLM. Uses
    label-vector match against the prose extractor's structured output.
  • `rag_fill_template(..., mode="llm")` — full RAG, one small LLM call
    per template cell. Cost ~$0.30 per company on DeepSeek (145 cells × ~3K
    tokens input + 50 tokens output).

API surface (mirror of llm_fill):
  rag_fill_template(
      template_path, pdf_paths=[...], output_path="",
      api_key="...", model="deepseek-chat", base_url="...",
      mode="pattern" | "llm",
  ) -> RAGFillResult
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# We REUSE these from llm_fill — they're battle-tested
from pipeline.llm_fill import (
    _pdf_to_markdown,
    _classify_pdf,
    _read_template_specs,
    _write_filled_excel,
    _extract_prose_pairs,
    PDFClassification,
    TemplateRowSpec,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Chunk + Index
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """One indexed text chunk from a PDF."""
    chunk_id:        str
    text:            str          # actual content for retrieval
    pdf_name:        str          # source PDF filename
    page:            int          # page index (0-based)
    scope:           str          # "STANDALONE" | "CONSOLIDATED" | "UNKNOWN"
    period_type:     str          # "ANNUAL" | "QUARTERLY" | "INVESTOR_PRESENTATION"
    pdf_stmt_type:   str          # PDF-level stmt_type
    years_mentioned: list[str] = field(default_factory=list)
    # Pre-computed: prose pairs extracted from this chunk
    pairs:           list[tuple] = field(default_factory=list)


@dataclass
class RAGFillResult:
    """Mirrors LLMFillResult."""
    output_path:   str = ""
    fills:         list[dict] = field(default_factory=list)
    cells_written: int = 0
    by_confidence: dict = field(default_factory=dict)
    warnings:      list[str] = field(default_factory=list)
    pdfs_used:     list[dict] = field(default_factory=list)
    chunks_indexed: int = 0
    elapsed_sec:   float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Year detection — reuse regex from prose extractor
# ──────────────────────────────────────────────────────────────────────────────

_MONTH_RE = (r"(?:January|February|March|April|May|June|July|August|September|"
             r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)")
_DATE_PATTERNS = [
    rf"{_MONTH_RE}\s+\d{{1,2}},?\s*(\d{{4}})",
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_RE},?\s*(\d{{4}})",
    r"\bF[Y]?\s*(\d{4})\b",
    r"\bF[Y]?\s*['‘’]?(\d{2})\b",
]
_DATE_REGEXES = [re.compile(p, re.I) for p in _DATE_PATTERNS]


def _extract_years_from_text(text: str) -> list[str]:
    """Return unique F-year labels mentioned in text, sorted MOST-RECENT
    FIRST. The order matters: financial statements usually print current
    year first, prior year(s) after. The number at position 0 in a data
    line corresponds to the FIRST year in this list."""
    years = set()
    for rx in _DATE_REGEXES:
        for m in rx.finditer(text):
            try:
                yr = int(m.group(1))
                if yr < 100:
                    yr = 2000 + yr if yr < 50 else 1900 + yr
                if 1990 <= yr <= 2050:
                    years.add(f"F{yr}")
            except (ValueError, IndexError):
                continue
    return sorted(years, reverse=True)   # F2024, F2023, F2022, ...


def _chunk_text(text: str, target_tokens: int = 300,
                 overlap_tokens: int = 50) -> list[str]:
    """Split text into ~target_tokens chunks with overlap. Uses word count
    as a proxy for tokens (1 word ≈ 1.3 tokens, so target_tokens=300 ≈
    230 words)."""
    words = text.split()
    target_words = max(50, int(target_tokens / 1.3))
    overlap_words = int(overlap_tokens / 1.3)
    if len(words) <= target_words:
        return [text]
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + target_words])
        chunks.append(chunk)
        i += target_words - overlap_words
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Vector backend — sentence-transformers preferred, else TF-IDF fallback
# ──────────────────────────────────────────────────────────────────────────────

class _VectorBackend:
    """Common interface: encode_one(text) → vector, encode_many(texts) → matrix,
    query(query, top_k, filter_fn) → list of (chunk_id, score, chunk)."""

    def __init__(self):
        self.chunks: dict[str, Chunk] = {}
        self.vectors = None     # numpy array, populated by build()
        self.chunk_ids: list[str] = []
        self.kind = "unknown"

    def add(self, chunk: Chunk):
        self.chunks[chunk.chunk_id] = chunk

    def build(self):
        """After all chunks added, compute vectors for all texts."""
        raise NotImplementedError

    def query(self, query_text: str, top_k: int = 5,
              filter_fn=None) -> list[tuple[str, float, Chunk]]:
        """Return [(chunk_id, score, chunk), ...] highest-similarity first.
        filter_fn(chunk) is called per chunk; if False, exclude from results."""
        raise NotImplementedError


class _SentenceTransformerBackend(_VectorBackend):
    """Semantic backend using sentence-transformers all-MiniLM-L6-v2."""

    def __init__(self):
        super().__init__()
        from sentence_transformers import SentenceTransformer   # noqa
        import numpy as np                                       # noqa
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.np = np
        self.kind = "sentence-transformers"

    def build(self):
        self.chunk_ids = list(self.chunks.keys())
        texts = [self.chunks[cid].text for cid in self.chunk_ids]
        if not texts:
            return
        logger.info(f"_SentenceTransformerBackend: encoding {len(texts)} chunks…")
        self.vectors = self.model.encode(
            texts, batch_size=64, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )

    def query(self, query_text, top_k=5, filter_fn=None):
        if self.vectors is None or len(self.chunk_ids) == 0:
            return []
        q_vec = self.model.encode(
            [query_text], convert_to_numpy=True, normalize_embeddings=True,
        )[0]
        sims = self.vectors @ q_vec   # cosine since both normalized
        # Apply filter
        results = []
        for idx in self.np.argsort(-sims):
            cid = self.chunk_ids[idx]
            chunk = self.chunks[cid]
            if filter_fn and not filter_fn(chunk):
                continue
            results.append((cid, float(sims[idx]), chunk))
            if len(results) >= top_k:
                break
        return results


class _TfidfBackend(_VectorBackend):
    """Fallback: TF-IDF cosine similarity. Handles keyword overlap; less
    smart for semantic synonyms but zero new deps beyond sklearn."""

    def __init__(self):
        super().__init__()
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
        except ImportError as e:
            raise RuntimeError(
                f"_TfidfBackend needs scikit-learn: {e}. "
                f"Run `pip install scikit-learn` or install sentence-transformers."
            )
        self.TfidfVectorizer = TfidfVectorizer
        self.cosine_similarity = cosine_similarity
        self.np = np
        self.vectorizer = None
        self.kind = "tf-idf"

    def build(self):
        self.chunk_ids = list(self.chunks.keys())
        texts = [self.chunks[cid].text for cid in self.chunk_ids]
        if not texts:
            return
        logger.info(f"_TfidfBackend: fitting TF-IDF over {len(texts)} chunks…")
        self.vectorizer = self.TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_df=0.95,
            stop_words="english",
        )
        self.vectors = self.vectorizer.fit_transform(texts)

    def query(self, query_text, top_k=5, filter_fn=None):
        if self.vectors is None or len(self.chunk_ids) == 0:
            return []
        q_vec = self.vectorizer.transform([query_text])
        sims = self.cosine_similarity(q_vec, self.vectors).flatten()
        results = []
        for idx in self.np.argsort(-sims):
            if sims[idx] <= 0:
                break
            cid = self.chunk_ids[idx]
            chunk = self.chunks[cid]
            if filter_fn and not filter_fn(chunk):
                continue
            results.append((cid, float(sims[idx]), chunk))
            if len(results) >= top_k:
                break
        return results


class _PurePythonBOWBackend(_VectorBackend):
    """Last-resort backend: pure-Python bag-of-words cosine similarity.
    Zero external dependencies. Handles keyword overlap; not semantic but
    works for label-match queries which often have direct keyword overlap."""

    _STOPWORDS = {
        "the", "a", "an", "of", "and", "or", "for", "to", "in", "on",
        "at", "by", "with", "is", "are", "as", "from", "be", "was", "were",
    }

    def __init__(self):
        super().__init__()
        self.kind = "pure-python-bow"
        self.token_vectors: dict[str, dict[str, float]] = {}
        self.token_norms: dict[str, float] = {}

    def _tokenize(self, text: str) -> dict[str, float]:
        """Return token-frequency dict."""
        import math
        from collections import Counter
        toks = re.findall(r"[a-zA-Z][a-zA-Z0-9]{1,}", text.lower())
        toks = [t for t in toks if t not in self._STOPWORDS and len(t) > 1]
        if not toks:
            return {}
        cnt = Counter(toks)
        # Length-normalise to a unit vector
        total = sum(v * v for v in cnt.values())
        norm = math.sqrt(total) if total > 0 else 1.0
        return {t: c / norm for t, c in cnt.items()}

    def build(self):
        self.chunk_ids = list(self.chunks.keys())
        for cid in self.chunk_ids:
            self.token_vectors[cid] = self._tokenize(self.chunks[cid].text)
        logger.info(f"_PurePythonBOWBackend: vectorised {len(self.chunk_ids)} chunks")

    def query(self, query_text, top_k=5, filter_fn=None):
        if not self.chunk_ids:
            return []
        q_vec = self._tokenize(query_text)
        if not q_vec:
            return []
        # Compute cosine sim with every chunk
        scored = []
        for cid in self.chunk_ids:
            chunk_vec = self.token_vectors.get(cid, {})
            if not chunk_vec:
                continue
            # Cosine similarity = dot product (both pre-normalised)
            sim = sum(q_vec.get(t, 0) * v for t, v in chunk_vec.items())
            if sim > 0:
                scored.append((cid, sim))
        scored.sort(key=lambda x: -x[1])
        results = []
        for cid, score in scored:
            chunk = self.chunks[cid]
            if filter_fn and not filter_fn(chunk):
                continue
            results.append((cid, score, chunk))
            if len(results) >= top_k:
                break
        return results


def _make_backend() -> _VectorBackend:
    """Prefer sentence-transformers (semantic), else TF-IDF, else pure-python."""
    try:
        return _SentenceTransformerBackend()
    except ImportError:
        logger.info("sentence-transformers not installed")
    except Exception as exc:
        logger.warning(f"sentence-transformers failed ({exc})")
    try:
        return _TfidfBackend()
    except (ImportError, RuntimeError):
        logger.info("scikit-learn not installed")
    except Exception as exc:
        logger.warning(f"TF-IDF failed ({exc})")
    logger.info("Using pure-Python bag-of-words backend (no extra deps)")
    return _PurePythonBOWBackend()


# ──────────────────────────────────────────────────────────────────────────────
# Indexing
# ──────────────────────────────────────────────────────────────────────────────

def _build_index(pdf_data: list[tuple[PDFClassification, str]]) -> _VectorBackend:
    """Chunk each PDF and add to vector backend."""
    backend = _make_backend()
    logger.info(f"_build_index: using backend = {backend.kind}")

    # First, detect Standalone/Consolidated boundaries per AR
    sa_markers = (
        "standalone balance sheet", "standalone statement of profit",
        "standalone cash flow", "audit of the standalone financial",
        "report on the audit of the standalone",
        "notes to the standalone financial",
    )
    co_markers = (
        "consolidated balance sheet", "consolidated statement of profit",
        "consolidated cash flow", "audit of the consolidated financial",
        "report on the audit of the consolidated",
        "notes to the consolidated financial",
    )

    n_chunks = 0
    for cls, md_text in pdf_data:
        # Split md into per-page chunks first
        page_chunks = re.split(r"(?=## Page \d+)", md_text)
        # Pass A: find boundaries
        std_start = cons_start = None
        for chunk in page_chunks:
            m = re.match(r"## Page (\d+)", chunk)
            if not m:
                continue
            pn = int(m.group(1)) - 1
            tl = chunk[:3000].lower()
            if std_start is None and any(k in tl for k in sa_markers):
                std_start = pn
            if cons_start is None and any(k in tl for k in co_markers):
                cons_start = pn

        # Pass B: chunk each page and tag with scope + metadata
        for page_chunk in page_chunks:
            m = re.match(r"## Page (\d+)", page_chunk)
            if not m:
                continue
            pn = int(m.group(1)) - 1
            # Decide scope based on AR boundary; for QR/IP use cls.stmt_type
            if cls.period_type == "ANNUAL" and std_start is not None and cons_start is not None:
                if std_start <= pn < cons_start:
                    scope = "STANDALONE"
                elif pn >= cons_start:
                    scope = "CONSOLIDATED"
                else:
                    scope = "UNKNOWN"
            elif cls.period_type == "ANNUAL" and std_start is not None:
                scope = "STANDALONE" if pn >= std_start else "UNKNOWN"
            elif cls.period_type == "ANNUAL" and cons_start is not None:
                scope = "CONSOLIDATED" if pn >= cons_start else "UNKNOWN"
            else:
                # QR / IP — use PDF-level stmt_type
                if cls.stmt_type in ("STANDALONE", "CONSOLIDATED"):
                    scope = cls.stmt_type
                else:
                    scope = "UNKNOWN"

            # Sub-chunk by paragraph (~300 tokens each)
            sub_chunks = _chunk_text(page_chunk, target_tokens=300, overlap_tokens=50)
            # Pre-detect years for the WHOLE page so we have header context
            # even when a single chunk doesn't contain a year-header line.
            page_years = _extract_years_from_text(page_chunk)
            for sub_idx, text in enumerate(sub_chunks):
                if not text.strip() or len(text) < 50:
                    continue
                # Inject SCOPE marker + synthetic year header so the prose
                # extractor knows which year each number column maps to.
                chunk_years = _extract_years_from_text(text) or page_years
                if chunk_years:
                    year_header_parts = ["Particulars"]
                    for yr in chunk_years:
                        # Convert F2024 → "March 31, 2024" for the prose extractor
                        m = re.match(r"^F(\d{4})$", yr)
                        if m:
                            year_header_parts.append(f"March 31, {m.group(1)}")
                        else:
                            year_header_parts.append(yr)
                    synthetic_header = " ".join(year_header_parts)
                else:
                    synthetic_header = ""
                tagged_text = f"[SCOPE={scope}]\n{synthetic_header}\n{text}"
                pairs = _extract_prose_pairs(tagged_text, cls)
                ch = Chunk(
                    chunk_id=f"{cls.name}#{pn}#{sub_idx}",
                    text=text,
                    pdf_name=cls.name,
                    page=pn,
                    scope=scope,
                    period_type=cls.period_type,
                    pdf_stmt_type=cls.stmt_type,
                    years_mentioned=chunk_years,
                    pairs=pairs,
                )
                backend.add(ch)
                n_chunks += 1

    backend.build()
    logger.info(f"_build_index: total chunks indexed = {n_chunks}")
    return backend


# ──────────────────────────────────────────────────────────────────────────────
# Query construction
# ──────────────────────────────────────────────────────────────────────────────

# Query-expansion aliases: for BOW retrieval to find the right chunk, the
# query needs to include synonyms the source PDF might use. Without this,
# template label "Less: Cost of goods sold" won't match a chunk that says
# "Cost of materials consumed".
_QUERY_ALIASES = {
    "revenue from operations": ["revenue", "net revenue", "sales",
                                  "income from operations", "turnover"],
    "cost of goods sold":     ["cost of materials consumed", "purchases of stock",
                                  "changes in inventories", "raw material cost"],
    "employee benefits expense": ["staff cost", "personnel cost", "salaries"],
    "finance cost":           ["finance costs", "interest expense", "borrowing cost"],
    "depreciation":           ["depreciation and amortisation",
                                  "depreciation amortization impairment"],
    "other income":           ["other operating income", "miscellaneous income",
                                  "interest income", "dividend income"],
    "tax":                    ["tax expense", "current tax", "deferred tax",
                                  "income tax expense"],
    "diluted eps":            ["diluted earnings per share", "earnings per share diluted"],
    "share of profit of associates": ["share of associates",
                                          "share of net profit of associates"],
    "property plant equipment": ["ppe", "fixed assets", "tangible assets"],
    "capital work in progress": ["cwip", "capital work-in-progress"],
    "intangible assets":      ["intangibles", "intangible asset"],
    "right of use":           ["rou", "rou assets", "right-of-use assets"],
    "inventories":            ["stock", "inventory"],
    "trade receivables":      ["debtors", "receivables"],
    "cash and cash equivalents": ["cash", "bank balances", "cash and bank"],
    "equity share capital":   ["share capital", "issued capital"],
    "other equity":           ["reserves and surplus", "retained earnings"],
    "borrowings":             ["loans", "debt", "long-term borrowings", "short-term borrowings"],
    "trade payables":         ["creditors", "payables", "sundry creditors"],
    "deferred tax":           ["deferred tax liability", "deferred tax asset"],
}


def _expand_label_query(label_norm: str) -> str:
    """Inject synonyms into the query string so BOW retrieval can find
    chunks that use different wording for the same concept."""
    extras = []
    for key, syns in _QUERY_ALIASES.items():
        if key in label_norm or any(syn in label_norm for syn in syns):
            extras.extend(syns)
            extras.append(key)
    if extras:
        return label_norm + " " + " ".join(set(extras))
    return label_norm


def _build_query(spec: TemplateRowSpec, year: str) -> str:
    """Build a natural-language query for vector search with synonym expansion."""
    section = (spec.section or "").strip()
    sheet = (spec.sheet or "").strip()
    stmt = (spec.statement_type or "").strip()
    label = (spec.label or "").strip()
    label_norm = label.lower().replace("less:", "").replace("add:", "").strip()
    label_norm = re.sub(r"[^a-z0-9 ]", " ", label_norm)
    label_norm = re.sub(r"\s+", " ", label_norm).strip()
    expanded_label = _expand_label_query(label_norm)

    # Translate F2024 → "year ended March 31, 2024" for better retrieval
    yr_clause = ""
    m = re.match(r"^F(\d{4})$", year)
    if m:
        yr_clause = f"year ended March 31, {m.group(1)} {year}"
    m = re.match(r"^([1-4])QF(\d{4})$", year)
    if m:
        q, yr = m.group(1), int(m.group(2))
        cal_yr = yr - 1
        q_dates = {"1": ("June 30", cal_yr), "2": ("September 30", cal_yr),
                   "3": ("December 31", cal_yr), "4": ("March 31", yr)}
        d, y = q_dates[q]
        yr_clause = f"three months ended {d}, {y} {year}"

    parts = [stmt, expanded_label, yr_clause, section, sheet]
    return " ".join(p for p in parts if p).strip()


def _filter_fn_for_spec(spec: TemplateRowSpec, year: str):
    """Return a filter callable. LENIENT: only blocks definitive scope
    contradictions. Year filter accepts target year OR adjacent FY (since
    a Standalone P&L printed in F23-24 AR has both F2023 and F2024 columns
    side by side)."""
    spec_stmt = (spec.statement_type or "").upper()
    m = re.match(r"^F(\d{4})$", year or "")
    target_fy = int(m.group(1)) if m else None
    m2 = re.match(r"^([1-4])QF(\d{4})$", year or "")
    target_qy_fy = int(m2.group(2)) if m2 else None

    def fn(chunk: Chunk) -> bool:
        # Scope: only block hard contradiction (chunk explicitly opposite scope)
        if spec_stmt in ("STANDALONE", "CONSOLIDATED"):
            if chunk.scope in ("STANDALONE", "CONSOLIDATED") and chunk.scope != spec_stmt:
                return False
            # UNKNOWN scope chunks pass — LLM judges from content
        # Year: allow target year OR adjacent FYs (±1 year) since comparatives
        # appear in adjacent-year ARs
        if target_fy is not None and chunk.years_mentioned:
            for ym in chunk.years_mentioned:
                m_ym = re.match(r"^F(\d{4})$", ym)
                if m_ym and abs(int(m_ym.group(1)) - target_fy) <= 1:
                    return True
            # Also accept if chunk has the relevant FY annual mention for QR
            return False
        if target_qy_fy is not None and chunk.years_mentioned:
            # Quarter request: accept chunks that mention the FY or adjacent FYs
            for ym in chunk.years_mentioned:
                m_ym = re.match(r"^F(\d{4})$", ym)
                if m_ym and abs(int(m_ym.group(1)) - target_qy_fy) <= 1:
                    return True
            return False
        return True

    return fn


# ──────────────────────────────────────────────────────────────────────────────
# Pattern-RAG: fill directly from retrieved chunks' pairs (no LLM)
# ──────────────────────────────────────────────────────────────────────────────

def _normalise(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_NUM_RE = re.compile(r"-?\(?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*\)?")


def _parse_number(s: str) -> Optional[float]:
    s = s.strip().replace(",", "").replace(" ", "")
    if not s:
        return None
    neg = "(" in s and ")" in s
    s = s.replace("(", "").replace(")", "")
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None


def _pattern_extract_from_chunks(spec: TemplateRowSpec, year: str,
                                  chunks: list[tuple[str, float, Chunk]]) -> Optional[dict]:
    """Walk top-k retrieved chunks; find lines containing the spec label.
    For each such line, extract all numbers and map them to chunk's
    years_mentioned. Return the number that corresponds to the requested
    year."""
    spec_norm = _normalise(spec.label)
    if not spec_norm or len(spec_norm) < 4:
        return None
    spec_stmt = (spec.statement_type or "").upper()

    # First: try the pre-computed prose pairs (fast path)
    for cid, score, chunk in chunks:
        for label_raw, pair_yr, pair_scope, pair_val, pair_kind in chunk.pairs:
            if pair_yr != year:
                continue
            if spec_stmt in ("STANDALONE", "CONSOLIDATED"):
                if pair_scope not in ("UNKNOWN", spec_stmt):
                    continue
            pair_norm = _normalise(label_raw)
            if not pair_norm:
                continue
            if not (pair_norm == spec_norm
                    or pair_norm in spec_norm
                    or spec_norm in pair_norm):
                continue
            if abs(len(pair_norm) - len(spec_norm)) > 25:
                continue
            return {
                "sheet": spec.sheet, "label": spec.label,
                "statement_type": spec.statement_type or "",
                "year": year, "value": pair_val,
                "confidence": "HIGH" if score > 0.5 else "MEDIUM",
                "source": f"rag:{chunk.pdf_name}#p{chunk.page}",
            }

    # Slow path: scan the raw chunk text for label-matching lines, then
    # map numbers to the chunk's years_mentioned.
    for cid, score, chunk in chunks:
        # Scope filter
        if spec_stmt in ("STANDALONE", "CONSOLIDATED"):
            if chunk.scope not in ("UNKNOWN", spec_stmt):
                continue
        # Chunk must mention this year for us to extract for it
        if year not in (chunk.years_mentioned or []):
            continue
        years_in_chunk = chunk.years_mentioned   # ordered: most-recent first
        if year not in years_in_chunk:
            continue
        year_idx = years_in_chunk.index(year)   # 0 = most-recent (current)

        for line in chunk.text.split("\n"):
            line = line.strip()
            if not line or line.startswith("|") or line.startswith("---"):
                continue
            line_norm = _normalise(line)
            if spec_norm not in line_norm:
                continue
            # Find all numbers in the line
            nums = _NUM_RE.findall(line)
            parsed = [_parse_number(n) for n in nums]
            parsed = [v for v in parsed if v is not None]
            # Filter out year-like, note-ref-like, and page-number-like nums.
            # Keep: anything with a decimal OR magnitude > 1000.
            real_nums = []
            for v in parsed:
                if 1990 < abs(v) < 2100 and v == int(v):
                    continue   # year
                if v == int(v) and abs(v) < 1000:
                    continue   # integer < 1000 = note ref / page num / counter
                real_nums.append(v)
            if len(real_nums) <= year_idx:
                continue   # not enough numbers
            val = real_nums[year_idx]
            return {
                "sheet": spec.sheet, "label": spec.label,
                "statement_type": spec.statement_type or "",
                "year": year, "value": val,
                "confidence": "MEDIUM",
                "source": f"rag-scan:{chunk.pdf_name}#p{chunk.page}",
            }
    return None


# ──────────────────────────────────────────────────────────────────────────────
# LLM-RAG: per-cell LLM extraction from retrieved chunks
# ──────────────────────────────────────────────────────────────────────────────

_LLM_RAG_SYSTEM = """You are a senior equity research analyst extracting a single value from a small set of source-document chunks.

Given:
- A target cell (sheet, label, statement_type, year)
- Up to 5 short chunks of source text retrieved from PDFs

Your job: return the EXACT numeric value for that cell, in INR Crores, as a JSON object:
  {"value": <number or null>, "confidence": "HIGH|MEDIUM|LOW", "source_note": "<one-line citation>"}

CRITICAL — ACTIVELY DERIVE VALUES when components are present:
  • COGS = "Cost of materials consumed" + "Purchases of stock-in-trade" + "Changes in inventories"
  • Tax  = "Current tax" + "Deferred tax" (+ "Earlier years tax adjustments" if shown)
  • Total Income = Revenue from operations + Other income
  • EBITDA = Total Income − COGS − Employee − Other Expenses
  • EBIT = EBITDA − Depreciation
  • PBT = EBIT − Finance + Other Income + Associates
  • PAT = PBT − Tax
  • Net Block PPE = Gross Block − Accumulated Depreciation
  • Total Assets = Non-Current Assets + Current Assets
  • Total Equity + Total Liabilities = Total Assets

LABEL EQUIVALENCES (template label ≡ source wording):
  "Revenue from operations" ≡ "Net Revenue from Operations" ≡ "Sales" ≡ "Turnover"
  "Employee benefits expense" ≡ "Staff Cost" ≡ "Personnel Cost"
  "Finance cost" ≡ "Finance costs" ≡ "Interest expense" ≡ "Borrowing costs"
  "Depreciation, amortisation & impairment" ≡ "Depreciation and amortisation expense"
  "Less: Tax" ≡ "Tax expense" ≡ "Total tax expense"
  "Other Income (Net)" ≡ "Other income"
  "Property, Plant & Equipment" ≡ "PPE" ≡ "Tangible assets"
  "Right of Use Assets" ≡ "ROU assets" ≡ "Right-of-use assets"

RULES:
- If chunks DIRECTLY print the value with matching year and scope, return it.
- If components are present but the total isn't, DERIVE using the formulas above.
- Convert units: Lakhs ÷ 100, Millions ÷ 10 → INR Crores. Bracketed numbers are negative.
- Honour scope: a Standalone spec should not get a Consolidated value.
- ONLY return null if the value AND its components are absent from all chunks.
- Use the year mentions in chunks to map columns to template years.

Return ONLY the JSON object, no surrounding text."""


def _llm_extract_from_chunks(spec: TemplateRowSpec, year: str,
                              chunks: list[tuple[str, float, Chunk]],
                              llm_client, model: str) -> Optional[dict]:
    """Make a small LLM call to extract the value from retrieved chunks."""
    if not chunks:
        return None
    parts = [f"TARGET CELL:"]
    parts.append(f"  Sheet: {spec.sheet}")
    parts.append(f"  Label: {spec.label}")
    parts.append(f"  Statement type: {spec.statement_type or '(unspecified)'}")
    parts.append(f"  Year: {year}")
    parts.append(f"  Section: {spec.section or '(none)'}")
    parts.append("")
    parts.append("RETRIEVED CHUNKS:")
    for i, (cid, score, chunk) in enumerate(chunks):
        parts.append(f"\n--- Chunk {i+1} (sim={score:.2f}, file={chunk.pdf_name}, "
                     f"page={chunk.page+1}, scope={chunk.scope}) ---")
        parts.append(chunk.text[:2000])
    user_msg = "\n".join(parts)

    try:
        resp = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_RAG_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content or "{}"
        obj = json.loads(raw)
        v = obj.get("value")
        if v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return {
            "sheet": spec.sheet, "label": spec.label,
            "statement_type": spec.statement_type or "",
            "year": year, "value": v,
            "confidence": obj.get("confidence", "MEDIUM"),
            "source": f"rag-llm:{obj.get('source_note', '')[:80]}",
        }
    except Exception as exc:
        logger.debug(f"_llm_extract_from_chunks failed for {spec.label}/{year}: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────────────────────────────────────

def rag_fill_template(
    template_path: str,
    pdf_paths:     list[str],
    output_path:   str = "",
    api_key:       str = "",
    model:         str = "deepseek-chat",
    base_url:      str = "https://api.deepseek.com",
    mode:          str = "pattern",     # "pattern" (free) or "llm" (paid)
    top_k:         int = 5,
) -> RAGFillResult:
    """Run RAG-based template fill.

    Modes:
      - "pattern": index + retrieve + pattern-extract from retrieved chunks.
                   Zero API cost. Works for verbatim-printed values.
      - "llm":     index + retrieve + LLM per cell. ~$0.30/run on DeepSeek.
                   Handles derived values (COGS sum, unit conversion).
    """
    t0 = time.time()
    res = RAGFillResult(output_path=output_path)

    # Step 1: classify + extract md for each PDF (reuse llm_fill code)
    pdf_data: list[tuple[PDFClassification, str]] = []
    for pp in pdf_paths:
        try:
            cls, _text = _classify_pdf(pp)
            # We want the FULL md, not the classifier's selected/tagged subset.
            # Use _pdf_to_markdown for richest content.
            full_md = _pdf_to_markdown(pp)
            pdf_data.append((cls, full_md))
            res.pdfs_used.append({"name": cls.name, "stmt_type": cls.stmt_type,
                                  "period_type": cls.period_type,
                                  "pages": cls.pages_total,
                                  "md_chars": len(full_md)})
        except Exception as exc:
            res.warnings.append(f"PDF prep failed for {pp}: {exc}")
            continue

    if not pdf_data:
        res.warnings.append("No usable PDFs.")
        res.elapsed_sec = time.time() - t0
        return res

    # Step 2: build vector index
    index = _build_index(pdf_data)
    res.chunks_indexed = sum(1 for _ in index.chunks)
    logger.info(f"rag_fill: indexed {res.chunks_indexed} chunks "
                f"using {index.kind}")
    res.warnings.append(
        f"Indexed {res.chunks_indexed} chunks "
        f"({index.kind}) from {len(pdf_data)} PDFs"
    )

    # Step 3: read template
    specs = _read_template_specs(template_path)
    n_specs = sum(1 for s in specs if not s.is_formula)
    logger.info(f"rag_fill: template has {n_specs} fillable rows")

    # Step 4: LLM client (only needed for mode="llm")
    llm_client = None
    if mode == "llm":
        try:
            from openai import OpenAI
            llm_client = OpenAI(api_key=api_key, base_url=base_url)
        except Exception as exc:
            res.warnings.append(f"Could not init LLM client: {exc}")
            mode = "pattern"   # fall back

    # ── PRE-PASS 1: pattern matcher (free, deterministic) ────────────────
    # Always runs first. Fills verbatim-printed values. Marks those cells
    # so the LLM call in LLM mode SKIPS them (huge cost saving — typically
    # 200+ of 1100 cells fill via pattern alone).
    fills: list[dict] = []
    pattern_filled: set[tuple] = set()   # (sheet, label_norm, year, stmt)
    n_pattern = 0
    for spec in specs:
        if spec.is_formula:
            continue
        for year in spec.years_needed:
            query = _build_query(spec, year)
            filter_fn = _filter_fn_for_spec(spec, year)
            chunks = index.query(query, top_k=top_k, filter_fn=filter_fn)
            if not chunks:
                continue
            fill = _pattern_extract_from_chunks(spec, year, chunks)
            if fill:
                fills.append(fill)
                pattern_filled.add(
                    (spec.sheet, _normalise(spec.label), year,
                     (spec.statement_type or "").lower())
                )
                n_pattern += 1
    logger.info(f"rag_fill[{mode}]: pre-pass pattern filled {n_pattern} cells (free)")

    # ── PRE-PASS 2: LLM only for cells pattern couldn't fill ─────────────
    n_attempted = 0
    n_filled = 0
    n_no_match = 0
    n_skipped_pattern = 0
    if mode == "llm" and llm_client is not None:
        for spec in specs:
            if spec.is_formula:
                continue
            for year in spec.years_needed:
                cell_key = (spec.sheet, _normalise(spec.label), year,
                            (spec.statement_type or "").lower())
                if cell_key in pattern_filled:
                    n_skipped_pattern += 1
                    continue   # pattern matcher already got this
                n_attempted += 1
                query = _build_query(spec, year)
                filter_fn = _filter_fn_for_spec(spec, year)
                chunks = index.query(query, top_k=top_k, filter_fn=filter_fn)
                if not chunks:
                    n_no_match += 1
                    continue
                fill = _llm_extract_from_chunks(
                    spec, year, chunks, llm_client, model
                )
                if fill:
                    fills.append(fill)
                    n_filled += 1

    res.fills = fills
    logger.info(f"rag_fill[{mode}]: pattern={n_pattern} (free), "
                f"llm_attempted={n_attempted}, llm_filled={n_filled}, "
                f"llm_no_match={n_no_match}, skipped_pattern={n_skipped_pattern}")
    res.warnings.append(
        f"RAG ({mode}): pattern filled {n_pattern} cells FREE, "
        f"LLM attempted {n_attempted} cells (skipped {n_skipped_pattern} "
        f"already-pattern-filled to save cost), LLM filled {n_filled}, "
        f"no-match {n_no_match}"
    )
    # Approx cost estimate (DeepSeek input pricing)
    est_input_tokens = n_attempted * 1200    # ~1.2K tokens per cell
    est_cost_usd = est_input_tokens * 0.27 / 1_000_000
    if mode == "llm":
        res.warnings.append(
            f"Estimated LLM cost: ~${est_cost_usd:.3f} "
            f"({n_attempted} cells × ~1.2K tokens each)"
        )

    # Step 6: write Excel via existing writer
    if not output_path:
        output_path = str(Path(template_path).with_name(
            Path(template_path).stem + "_rag_filled.xlsx"
        ))
        res.output_path = output_path
    stats = _write_filled_excel(template_path, output_path, fills)
    res.cells_written = stats["written"]
    res.by_confidence = stats["by_confidence"]
    if stats.get("per_source_fills"):
        per_src = stats["per_source_fills"]
        lines = [f"  {src}: {n}" for src, n in
                 sorted(per_src.items(), key=lambda kv: -kv[1])[:15]]
        res.warnings.append("Per-source RAG fills:\n" + "\n".join(lines))

    res.elapsed_sec = time.time() - t0
    return res


__all__ = ["rag_fill_template", "RAGFillResult"]
