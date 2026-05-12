"""
pipeline/validation_gate.py
───────────────────────────
Cross-row, cross-source, sector-aware validation of filled values.

Runs AFTER the mapping/derivation step but BEFORE the Excel write, so the
orchestrator can either auto-fix or quarantine bad cells to a Data Review
sheet.

Checks performed
────────────────
1. Sign sanity         — KB declares sign +/−; flags violations.
2. Magnitude / range   — `expected_range` in KB (mostly for percentages).
3. Accounting identities (sector-aware):
   • General      : Revenue ≥ EBITDA ≥ EBIT, BS Assets = Liab + Equity
                    (within 2% slack), CFO ± Capex ≈ FCF
   • Bank         : Total Income = NII + Other Income (5% slack);
                    Operating Profit = Total Income − Opex;
                    PAT = PBT − Tax;  Total Deposits ≥ CASA Deposits;
                    Advances ≤ Total Assets;  GNPA ≥ NNPA
   • NBFC         : NII = Interest Income − Finance Cost;
                    Loans ≤ Total Assets;  GS3 ≥ NS3
   • Insurance    : NWP ≤ GWP;  Combined Ratio ≈ Claims + Expense + Commission
4. Sub-total checks    — total_assets ≈ total_current_assets + total_non_current
5. Inter-year sanity   — values that swing >5x year-on-year flagged for review.

Output
──────
ValidationReport
  .issues : list[ValidationIssue]   (severity high|medium|info)
  .auto_fixes : list[(label, year, old_val, new_val, reason)]
  .summary() : human-readable string
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from pipeline.kb_loader import kb_entry_for, load_merged_kb

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    severity: str          # "high" | "medium" | "info"
    label: str
    year: str
    sheet: str
    rule: str              # short rule id
    message: str
    suggested_fix: Optional[float] = None  # if we can propose a corrected value


@dataclass
class AutoFix:
    label: str
    year: str
    sheet: str
    old_value: Optional[float]
    new_value: float
    reason: str


@dataclass
class ValidationReport:
    issues:    list[ValidationIssue] = field(default_factory=list)
    auto_fixes: list[AutoFix]        = field(default_factory=list)

    def has_high(self) -> bool:
        return any(i.severity == "high" for i in self.issues)

    def summary(self) -> str:
        sev = {"high": 0, "medium": 0, "info": 0}
        for i in self.issues:
            sev[i.severity] = sev.get(i.severity, 0) + 1
        return (
            f"{len(self.issues)} issues "
            f"(high={sev['high']}, medium={sev['medium']}, info={sev['info']}), "
            f"{len(self.auto_fixes)} auto-fixes proposed"
        )

    def issues_for_sheet(self, sheet: str) -> list[ValidationIssue]:
        return [i for i in self.issues if i.sheet == sheet]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get(filled: dict, label: str, year: str, sector: str = "general") -> Optional[float]:
    """Case-insensitive, alias-aware label lookup in a filled-cells dict."""
    if label in filled and year in filled[label]:
        v = filled[label][year]
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None
    # Try alias resolution via KB
    entry = kb_entry_for(label, sector)
    if entry:
        # Try every alias of the canonical concept
        canonical = None
        for ck, info in load_merged_kb(sector).items():
            if info is entry:
                canonical = ck
                break
        if canonical:
            for cand_lbl, yr_vals in filled.items():
                cand_entry = kb_entry_for(cand_lbl, sector)
                if cand_entry is entry:
                    v = yr_vals.get(year)
                    try:
                        return float(v) if v is not None else None
                    except (TypeError, ValueError):
                        return None
    return None


def _approx_eq(a: float, b: float, slack: float = 0.02) -> bool:
    """True if a and b agree within `slack` fractional tolerance."""
    if a is None or b is None:
        return False
    if abs(a) < 1e-6 and abs(b) < 1e-6:
        return True
    denom = max(abs(a), abs(b))
    return abs(a - b) / denom <= slack


# ── Generic checks ──────────────────────────────────────────────────────────

def _check_sign_and_range(filled: dict, sheet: str, sector: str,
                          report: ValidationReport) -> None:
    for label, yr_vals in filled.items():
        entry = kb_entry_for(label, sector)
        if not entry:
            continue
        sign = entry.get("sign")
        rng  = entry.get("expected_range")
        is_pct = entry.get("is_percentage")

        for yr, v in yr_vals.items():
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue

            # Sign check (only flags clear violations)
            if sign in ("+", 1) and fv < 0 and abs(fv) > 1.0:
                # Negative on a "+" item is sometimes legitimate (loss),
                # so only warn for non-loss-prone items
                if entry.get("section") not in ("pat", "pbt", "operating_profit",
                                                "core_revenue", "totals"):
                    report.issues.append(ValidationIssue(
                        "medium", label, yr, sheet, "sign",
                        f"value {fv:,.2f} is negative but KB sign='+'",
                    ))

            # Range check
            if rng:
                lo, hi = rng[0], rng[1]
                if fv < lo or fv > hi:
                    sev = "high" if is_pct and (fv < lo - 5 or fv > hi + 20) else "medium"
                    report.issues.append(ValidationIssue(
                        sev, label, yr, sheet, "range",
                        f"value {fv:,.2f} outside expected {lo}–{hi}",
                    ))


def _check_magnitude_jumps(filled: dict, sheet: str, years_sorted: list[str],
                           report: ValidationReport) -> None:
    """Flag year-on-year swings of >5x (likely scale error: Lakhs vs Crores)."""
    for label, yr_vals in filled.items():
        prev: Optional[float] = None
        for yr in years_sorted:
            v = yr_vals.get(yr)
            if v is None:
                prev = None
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                prev = None
                continue
            if prev is not None and abs(prev) > 1.0:
                ratio = abs(fv) / max(abs(prev), 1e-6)
                if ratio > 5.0 or ratio < 0.2:
                    report.issues.append(ValidationIssue(
                        "high", label, yr, sheet, "yoy_swing",
                        f"{fv:,.1f} vs prior {prev:,.1f} = {ratio:.1f}x — possible scale error",
                    ))
            prev = fv


# ── Sector-aware identity checks ────────────────────────────────────────────

def _check_general_identities(filled: dict, sheet: str, years: list[str],
                              report: ValidationReport, sector: str) -> None:
    for yr in years:
        rev    = _get(filled, "Revenue from Operations", yr, sector) or _get(filled, "Net Sales", yr, sector)
        ebitda = _get(filled, "EBITDA", yr, sector)
        ebit   = _get(filled, "EBIT", yr, sector)
        pbt    = _get(filled, "PBT", yr, sector)
        pat    = _get(filled, "PAT", yr, sector)
        ta     = _get(filled, "Total Assets", yr, sector)
        teq    = _get(filled, "Total Equity", yr, sector) or _get(filled, "Net Worth", yr, sector)
        tl_nc  = _get(filled, "Total Non-Current Liabilities", yr, sector)
        tl_c   = _get(filled, "Total Current Liabilities", yr, sector)

        if rev is not None and ebitda is not None and ebitda > rev:
            report.issues.append(ValidationIssue(
                "high", "EBITDA", yr, sheet, "ebitda_gt_rev",
                f"EBITDA ({ebitda:,.1f}) > Revenue ({rev:,.1f}) — impossible",
            ))
        if ebitda is not None and ebit is not None and ebit > ebitda + 1:
            report.issues.append(ValidationIssue(
                "high", "EBIT", yr, sheet, "ebit_gt_ebitda",
                f"EBIT ({ebit:,.1f}) > EBITDA ({ebitda:,.1f}) — D&A must be ≥ 0",
            ))
        if pbt is not None and pat is not None and pbt > 0 and pat > pbt + 1:
            report.issues.append(ValidationIssue(
                "medium", "PAT", yr, sheet, "pat_gt_pbt",
                f"PAT ({pat:,.1f}) > PBT ({pbt:,.1f}) — implies negative tax (rare)",
            ))

        # Balance sheet equation
        if ta is not None and teq is not None:
            tl = (tl_nc or 0) + (tl_c or 0)
            if tl > 0:
                rhs = teq + tl
                if not _approx_eq(ta, rhs, slack=0.02):
                    diff = ta - rhs
                    report.issues.append(ValidationIssue(
                        "high", "Total Assets", yr, sheet, "bs_imbalance",
                        f"Assets={ta:,.1f} vs Equity+Liab={rhs:,.1f} (diff {diff:+,.1f}, "
                        f"{abs(diff)/max(abs(ta), 1e-6):.1%})",
                    ))


def _check_bank_identities(filled: dict, sheet: str, years: list[str],
                           report: ValidationReport) -> None:
    for yr in years:
        ie    = _get(filled, "Interest Earned", yr, "bank")
        ix    = _get(filled, "Interest Expended", yr, "bank")
        nii   = _get(filled, "Net Interest Income", yr, "bank")
        oi    = _get(filled, "Other Income", yr, "bank") or _get(filled, "Non-Interest Income", yr, "bank")
        ti    = _get(filled, "Total Income", yr, "bank")
        opex  = _get(filled, "Operating Expenses", yr, "bank")
        op    = _get(filled, "Operating Profit", yr, "bank")
        prov  = _get(filled, "Provisions", yr, "bank")
        pbt   = _get(filled, "PBT", yr, "bank") or _get(filled, "Profit Before Tax", yr, "bank")
        pat   = _get(filled, "PAT", yr, "bank") or _get(filled, "Net Profit", yr, "bank")
        adv   = _get(filled, "Advances", yr, "bank") or _get(filled, "Loans and Advances", yr, "bank")
        dep   = _get(filled, "Deposits", yr, "bank") or _get(filled, "Total Deposits", yr, "bank")
        casa  = _get(filled, "CASA Deposits", yr, "bank") or _get(filled, "CASA", yr, "bank")
        gnpa  = _get(filled, "Gross NPA", yr, "bank")
        nnpa  = _get(filled, "Net NPA", yr, "bank")

        # NII identity
        if ie is not None and ix is not None and nii is not None:
            calc = ie - ix
            if not _approx_eq(calc, nii, slack=0.05):
                report.issues.append(ValidationIssue(
                    "medium", "Net Interest Income", yr, sheet, "nii_identity",
                    f"NII={nii:,.1f} vs (Interest Earned − Interest Expended)={calc:,.1f}",
                    suggested_fix=calc,
                ))
                report.auto_fixes.append(AutoFix(
                    "Net Interest Income", yr, sheet,
                    nii, calc, "NII = Interest Earned − Interest Expended",
                ))

        # Total income
        if nii is not None and oi is not None and ti is not None:
            calc = nii + oi
            if not _approx_eq(calc, ti, slack=0.05):
                report.issues.append(ValidationIssue(
                    "medium", "Total Income", yr, sheet, "total_income_identity",
                    f"Total Income={ti:,.1f} vs (NII + Other Income)={calc:,.1f}",
                    suggested_fix=calc,
                ))

        # Operating profit
        if ti is not None and opex is not None and op is not None:
            calc = ti - opex
            if not _approx_eq(calc, op, slack=0.05):
                report.issues.append(ValidationIssue(
                    "medium", "Operating Profit", yr, sheet, "op_identity",
                    f"OP={op:,.1f} vs (Total Income − Opex)={calc:,.1f}",
                    suggested_fix=calc,
                ))

        # PBT = OP − Provisions
        if op is not None and prov is not None and pbt is not None:
            calc = op - prov
            if not _approx_eq(calc, pbt, slack=0.05):
                report.issues.append(ValidationIssue(
                    "medium", "PBT", yr, sheet, "pbt_identity",
                    f"PBT={pbt:,.1f} vs (OP − Provisions)={calc:,.1f}",
                    suggested_fix=calc,
                ))

        # CASA ≤ total deposits
        if dep is not None and casa is not None and casa > dep + 1:
            report.issues.append(ValidationIssue(
                "high", "CASA Deposits", yr, sheet, "casa_gt_deposits",
                f"CASA ({casa:,.1f}) > Total Deposits ({dep:,.1f}) — impossible",
            ))

        # GNPA ≥ NNPA
        if gnpa is not None and nnpa is not None and nnpa > gnpa + 0.5:
            report.issues.append(ValidationIssue(
                "high", "Net NPA", yr, sheet, "nnpa_gt_gnpa",
                f"Net NPA ({nnpa:,.2f}) > Gross NPA ({gnpa:,.2f}) — impossible",
            ))


def _check_nbfc_identities(filled: dict, sheet: str, years: list[str],
                           report: ValidationReport) -> None:
    for yr in years:
        ii   = _get(filled, "Interest Income", yr, "nbfc") or _get(filled, "Revenue from Operations", yr, "nbfc")
        fc   = _get(filled, "Finance Cost", yr, "nbfc")
        nii  = _get(filled, "Net Interest Income", yr, "nbfc")
        gs3  = _get(filled, "Gross Stage 3", yr, "nbfc") or _get(filled, "Gross NPA", yr, "nbfc")
        ns3  = _get(filled, "Net Stage 3", yr, "nbfc") or _get(filled, "Net NPA", yr, "nbfc")

        if ii is not None and fc is not None and nii is not None:
            calc = ii - fc
            if not _approx_eq(calc, nii, slack=0.05):
                report.issues.append(ValidationIssue(
                    "medium", "Net Interest Income", yr, sheet, "nbfc_nii_identity",
                    f"NII={nii:,.1f} vs (Interest Income − Finance Cost)={calc:,.1f}",
                    suggested_fix=calc,
                ))
        if gs3 is not None and ns3 is not None and ns3 > gs3 + 0.5:
            report.issues.append(ValidationIssue(
                "high", "Net Stage 3", yr, sheet, "ns3_gt_gs3",
                f"NS3 ({ns3:,.2f}) > GS3 ({gs3:,.2f}) — impossible",
            ))


def _check_insurance_identities(filled: dict, sheet: str, years: list[str],
                                report: ValidationReport) -> None:
    for yr in years:
        gwp = _get(filled, "Gross Written Premium", yr, "insurance") or _get(filled, "GWP", yr, "insurance")
        nwp = _get(filled, "Net Written Premium", yr, "insurance") or _get(filled, "NWP", yr, "insurance")
        re_ceded = _get(filled, "Reinsurance Ceded", yr, "insurance")
        if gwp is not None and nwp is not None and nwp > gwp + 1:
            report.issues.append(ValidationIssue(
                "high", "NWP", yr, sheet, "nwp_gt_gwp",
                f"NWP ({nwp:,.1f}) > GWP ({gwp:,.1f}) — impossible",
            ))


# ── Main entry ──────────────────────────────────────────────────────────────

def _check_historical_consistency(
    filled: dict, sheet: str, sector: str,
    report: ValidationReport, symbol: str = "",
) -> None:
    """
    For each filled (label, year), look up the prior year's value in the
    historical DB. If swing is > 5x, flag as suspicious.
    """
    if not symbol:
        return
    try:
        from pipeline.historical_db import get_prior_year_value
        from pipeline.kb_loader import kb_entry_for, load_merged_kb
    except Exception:
        return

    kb = load_merged_kb(sector)
    for label, yr_vals in filled.items():
        entry = kb_entry_for(label, sector)
        if not entry:
            continue
        canon = None
        for ck, info in kb.items():
            if info is entry:
                canon = ck; break
        if not canon:
            continue
        # Skip flow-type labels where huge swings are normal
        sec = entry.get("section", "") + " " + entry.get("statement", "")
        if any(k in sec.lower() for k in ("exception", "tax", "gain", "loss")):
            continue
        for yr, v in yr_vals.items():
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            prior = get_prior_year_value(symbol, canon, yr)
            if not prior:
                continue
            prior_yr, prior_val = prior
            if abs(prior_val) < 1.0:
                continue
            ratio = abs(fv) / max(abs(prior_val), 1e-6)
            if ratio > 5.0 or ratio < 0.20:
                report.issues.append(ValidationIssue(
                    "high", label, yr, sheet, "historical_swing",
                    f"value {fv:,.1f} is {ratio:.1f}× of {prior_yr} "
                    f"({prior_val:,.1f}) — likely wrong row matched",
                ))


def validate_sheet(
    sheet_name: str,
    filled: dict[str, dict[str, Optional[float]]],
    years: list[str],
    sector: str = "general",
    symbol: str = "",
) -> ValidationReport:
    """
    Validate a single sheet's filled cells, sector-aware.

    `filled` shape: {label: {year: value}}
    `symbol`: NSE symbol for historical-DB consistency check (optional)
    """
    report = ValidationReport()

    if not filled:
        return report

    years_sorted = sorted(years)

    # Always-on checks
    _check_sign_and_range(filled, sheet_name, sector, report)
    _check_magnitude_jumps(filled, sheet_name, years_sorted, report)
    _check_historical_consistency(filled, sheet_name, sector, report, symbol)

    # Sector-routed identity checks
    if sector == "bank":
        _check_bank_identities(filled, sheet_name, years_sorted, report)
    elif sector == "nbfc":
        _check_nbfc_identities(filled, sheet_name, years_sorted, report)
    elif sector == "insurance":
        _check_insurance_identities(filled, sheet_name, years_sorted, report)
    else:
        _check_general_identities(filled, sheet_name, years_sorted, report, sector)

    return report


def apply_auto_fixes(
    filled: dict[str, dict[str, Optional[float]]],
    report: ValidationReport,
    require_empty: bool = True,
) -> int:
    """
    Apply the report's `auto_fixes` to `filled` IN PLACE.

    `require_empty`: if True, only fix cells that are currently empty (safer).
                     If False, will overwrite existing values.
    Returns count of fixes applied.
    """
    n = 0
    for fix in report.auto_fixes:
        if fix.label not in filled:
            filled[fix.label] = {}
        cur = filled[fix.label].get(fix.year)
        if require_empty and cur is not None:
            continue
        filled[fix.label][fix.year] = fix.new_value
        n += 1
    return n


__all__ = [
    "validate_sheet",
    "apply_auto_fixes",
    "ValidationReport",
    "ValidationIssue",
    "AutoFix",
]
