"""
Base data classes for the web data collection module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Company identifier ─────────────────────────────────────────────────────────

@dataclass
class CompanyIdentifier:
    """
    Holds all known identifiers for a company.
    At least one of nse_symbol, bse_code, or cin must be provided.
    """
    name: str
    nse_symbol: Optional[str] = None   # e.g. "HDFCBANK"  → ticker "HDFCBANK.NS"
    bse_code: Optional[str] = None     # e.g. "500180"     → ticker "500180.BO"
    cin: Optional[str] = None          # e.g. "L65920MH1994PLC080618" (MCA)
    isin: Optional[str] = None         # e.g. "INE040A01034"
    sector: Optional[str] = None       # "banking" | "manufacturing" | etc.

    @property
    def yfinance_ticker(self) -> Optional[str]:
        """Returns the yfinance ticker string, preferring NSE over BSE."""
        if self.nse_symbol:
            return f"{self.nse_symbol}.NS"
        if self.bse_code:
            return f"{self.bse_code}.BO"
        return None

    def __str__(self) -> str:
        parts = [self.name]
        if self.nse_symbol:
            parts.append(f"NSE:{self.nse_symbol}")
        if self.bse_code:
            parts.append(f"BSE:{self.bse_code}")
        return " | ".join(parts)


# ── Web data result ────────────────────────────────────────────────────────────

@dataclass
class WebDataResult:
    """
    A single financial data point fetched from a web source.

    template_field is the label as it appears in the Excel template's row_index.
    If the source field could not be mapped to a template label, template_field
    is empty and the result can still be used for fuzzy cross-validation.
    """
    source: str                # "yfinance" | "bse_xbrl" | "mca_xbrl" | "rbi_dbie"
    raw_field: str             # field name as returned by the source (e.g. "Total Revenue")
    template_field: str        # mapped template row label (empty if not resolved)
    value: float               # in INR Crores
    year: str                  # "F2025", "F2024", etc.
    confidence: str            # "HIGH" | "MEDIUM" | "LOW"
    unit_applied: str          # human-readable conversion applied, e.g. "÷1e7 (INR→Cr)"

    def __post_init__(self):
        # Normalise year format to F20XX
        y = str(self.year).strip()
        if not y.startswith("F"):
            self.year = f"F{y}"


# ── Cross-validation result ────────────────────────────────────────────────────

@dataclass
class CrossValidationResult:
    """
    Outcome of comparing a PDF-extracted value against one or more web sources.
    """
    template_field: str
    year: str

    # PDF extraction side
    pdf_value: float
    pdf_source: str            # "rule_based" | "pdfplumber" | "llm"

    # Best-matching web value (None if no web data found for this field/year)
    web_value: Optional[float] = None
    web_source: Optional[str] = None

    # Validation outcome
    agreement: bool = False              # True if within tolerance
    tolerance_pct: float = 0.0          # |pdf - web| / web  (0.02 = 2%)
    confidence_upgrade: str = "MEDIUM"  # "VERY_HIGH" | "HIGH" | "MEDIUM" | "FLAG"
    flag_reason: Optional[str] = None   # human-readable reason for FLAG

    def summary(self) -> str:
        if self.web_value is None:
            return f"{self.template_field} [{self.year}]: no web data"
        diff = abs(self.pdf_value - self.web_value)
        return (
            f"{self.template_field} [{self.year}]: "
            f"PDF={self.pdf_value:.2f} | Web={self.web_value:.2f} | "
            f"diff={self.tolerance_pct:.1%} | {self.confidence_upgrade}"
        )
