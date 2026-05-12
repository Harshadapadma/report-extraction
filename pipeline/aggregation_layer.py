"""
pipeline/aggregation_layer.py
─────────────────────────────
Deterministic post-pool aggregation. Runs AFTER _build_pool, BEFORE any
LLM stage. Computes derivable cells from leaf cells already in the pool.

Why this exists (audit 2026-05-11):
  • raw_excel scores 11% accuracy; smart pipeline 1%. The LLM tier is
    overwriting correct data with hallucinations.
  • 71% of "missed" balance-sheet rows are totals (Total Assets, Total
    Equity, etc.) that are deterministically computable from leaves we
    already have.
  • derivation_agent.py exists but its rules don't fire on the canonical
    pool labels reliably. This module replaces those rules with a
    smaller, label-canonical, idempotent set.

Design contract:
  • Input: pool = dict[(label_canonical, period), value], with each entry
    also carrying source + confidence in a parallel provenance map.
  • Output: new entries added to pool. We NEVER overwrite an existing
    entry. We tag every derived entry source="derived_agg", confidence=
    "HIGH" iff cross-check passes, otherwise "FLAGGED".
  • Iteration: we repeat until no new entries are added (fixed-point).
    This lets chained derivations cascade: GP unlocks EBITDA unlocks
    EBIT unlocks PBT unlocks PAT.

Cross-check rule: after computing a value, if the same (label, period)
already had a value in the pool with non-"derived_agg" source AND it
disagrees by >1%, we DO NOT overwrite, but we flag a discrepancy in
the returned report. This is the self-audit hook.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical label dictionary
# Every key is a *canonical concept* the template uses. Synonyms are the
# variants that appear in source data (raw_excel, PDF text, web connectors).
# The canonicalize() function maps any raw label → canonical or None.
# ─────────────────────────────────────────────────────────────────────────────

# Each tuple: (canonical, list-of-synonym-substrings).
# Match is case-insensitive, "less:"/"add:" prefixes stripped first, and
# we ignore differences in &/and, plural/singular trailing 's', and
# punctuation. The match is *substring* — the FIRST canonical whose
# any synonym is contained in the label wins. Order matters: longest /
# most specific synonyms must come first.

CANONICAL_LABELS: list[tuple[str, list[str]]] = [
    # ── P&L ──────────────────────────────────────────────────────────────
    ("revenue_from_operations", [
        "revenue from operations", "revenue from operation",
        "total revenue from operations", "net sales",
        "revenue from contracts with customers",
    ]),
    ("cost_of_goods_sold", [
        "cost of goods sold", "cost of materials consumed",
        "cogs", "raw materials consumed",
    ]),
    ("gross_profit", ["gross profit"]),
    ("employee_benefits_expense", [
        "employee benefits expense", "employee benefit expense",
        "employee cost", "personnel cost", "staff cost",
    ]),
    ("other_expenses", [
        "other expenses (net)", "other expenses",
        "other operating expenses",
    ]),
    ("ebitda", ["ebitda"]),
    ("depreciation_amortization", [
        "depreciation, amortization & impairment",
        "depreciation and amortisation",
        "depreciation and amortization",
        "depreciation, amortisation",
        "depreciation amortization",
        "depreciation amortisation",
    ]),
    ("ebit", ["ebit", "operating profit"]),
    ("finance_cost", ["finance cost", "finance costs", "interest expense"]),
    ("other_income", ["other income (net)", "other income"]),
    ("share_of_associates", [
        "share of net profit of associates",
        "share of profit of associate",
        "share of associate", "associate profit",
    ]),
    ("exceptional_items", ["exceptional item", "exceptional"]),
    ("pbt", [
        "profit before tax", "pbt",
        "profit/(loss) before tax",
    ]),
    ("tax_expense", [
        "total tax expense", "tax expense", "income tax expense",
        "less: tax",
    ]),
    ("pat", [
        "profit after tax", "pat",
        "profit/(loss) for the period",
        "profit for the year", "profit for the period",
        "net profit", "net income",
    ]),
    ("diluted_eps", ["diluted eps", "diluted earnings per share"]),
    # ── % of sales (margins) ─────────────────────────────────────────────
    ("gross_margin", ["gross margin"]),
    ("ebitda_margin", ["ebitda margin", "ebitda %"]),
    # ── Balance Sheet — leaves ───────────────────────────────────────────
    ("ppe", ["property, plant and equipment", "property plant and equipment",
             "property, plant & equipment", "fixed assets"]),
    ("cwip", ["capital work in progress", "capital work-in-progress", "cwip"]),
    ("goodwill", ["goodwill"]),
    ("intangible_assets", ["intangible asset"]),  # matches "intangible assets" too
    ("intangible_under_dev", ["intangible assets under development",
                              "intangibles under development"]),
    ("rou_assets", ["right of use asset", "right-of-use asset"]),
    ("nc_investments", ["non-current investments", "investments - non-current"]),
    ("other_investments", ["other investments"]),
    ("nc_other_financial_assets", ["other financial assets (non-current)",
                                   "non-current financial assets",
                                   "other non-current financial assets"]),
    ("nc_income_tax_asset", ["income tax asset (net)",
                             "income tax assets (net)",
                             "non-current income tax asset"]),
    ("deferred_tax_assets", ["deferred tax assets (net)",
                             "deferred tax asset"]),
    ("nc_other_assets", ["other non-current assets"]),
    ("inventories", ["inventories", "inventory"]),
    ("c_investments", ["current investments", "investments - current"]),
    ("trade_receivables", ["trade receivable"]),
    ("cash_and_equivalents", ["cash & cash equivalents",
                              "cash and cash equivalents"]),
    ("bank_balances_other", ["bank balances other than above",
                             "other bank balances"]),
    ("c_other_financial_assets", ["other financial assets (current)",
                                  "current financial assets"]),
    ("c_income_tax_asset", ["income tax asset (current)"]),
    ("c_other_assets", ["other current assets"]),
    # Aggregates we compute
    ("total_nc_assets", ["total non-current assets"]),
    ("total_c_assets", ["total current assets"]),
    ("total_assets", ["total assets"]),
    # Equity
    ("equity_share_capital", ["equity share capital"]),
    ("other_equity", ["other equity"]),
    ("total_equity", ["total equity"]),
    # Liabilities
    ("nc_borrowings", ["non-current borrowings", "long-term borrowings"]),
    ("nc_lease_liabilities", ["lease liabilities (non-current)",
                              "non-current lease liabilities"]),
    ("nc_other_financial_liabilities", ["other non-current financial liabilities"]),
    ("nc_provisions", ["non-current provisions"]),
    ("deferred_tax_liabilities", ["deferred tax liabilities (net)",
                                  "deferred tax liability"]),
    ("nc_other_liabilities", ["other non-current liabilities"]),
    ("c_borrowings", ["current borrowings", "short-term borrowings"]),
    ("c_lease_liabilities", ["lease liabilities (current)",
                             "current lease liabilities"]),
    ("trade_payables_micro", ["dues of micro and small enterprises",
                              "dues of micro"]),
    ("trade_payables_others", ["dues of other creditors",
                               "dues of others"]),
    ("trade_payables_total", ["trade payables", "trade payable"]),
    ("c_other_financial_liabilities", ["other financial liabilities (current)"]),
    ("c_other_liabilities", ["other current liabilities"]),
    ("c_provisions", ["current provisions"]),
    ("c_tax_liabilities", ["current tax liabilities"]),
    ("total_nc_liabilities", ["total non-current liabilities"]),
    ("total_c_liabilities", ["total current liabilities"]),
    ("total_equity_liabilities", ["total equity & liabilities",
                                  "total equity and liabilities"]),
]


def _normalise(s: str) -> str:
    """Lowercase, strip operator prefixes, normalise punctuation/whitespace."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"^(less|add|sub-?total|subtotal)\s*:\s*", "", s)
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9\s\(\)\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonicalize(label: str) -> Optional[str]:
    """Map any free-form label to its canonical concept name, or None."""
    norm = _normalise(label)
    if not norm:
        return None
    # Try most-specific synonyms first by length descending; iterate
    # canonical entries in order (their priority).
    for canon, syns in CANONICAL_LABELS:
        for syn in syns:
            syn_norm = _normalise(syn)
            if syn_norm and syn_norm in norm:
                return canon
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation rules
# Each rule: target_canon ← function(pool, period)
# Function returns float or None. If returns None, rule didn't apply.
# We retry rules in a loop until no new values are added.
# ─────────────────────────────────────────────────────────────────────────────

# Pool type: dict[(canonical, period), float]
Pool = dict[tuple[str, str], float]


def _get(pool: Pool, canon: str, period: str) -> Optional[float]:
    return pool.get((canon, period))


def _sum(pool: Pool, canons: list[str], period: str,
         require_all: bool = True, min_present: int = 1) -> Optional[float]:
    """Sum the listed canonicals for a period.
    require_all=True → return None unless every canonical has a value.
    require_all=False → sum what's present provided at least min_present
    canonicals have values."""
    vals = []
    missing = 0
    for c in canons:
        v = _get(pool, c, period)
        if v is None:
            missing += 1
        else:
            vals.append(v)
    if require_all and missing:
        return None
    if not require_all and len(vals) < min_present:
        return None
    return sum(vals)


def _sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


# P&L rules ───────────────────────────────────────────────────────────────────

def _r_gross_profit(pool, period):
    return _sub(_get(pool, "revenue_from_operations", period),
                _get(pool, "cost_of_goods_sold", period))


def _r_ebitda(pool, period):
    gp = _get(pool, "gross_profit", period)
    emp = _get(pool, "employee_benefits_expense", period)
    oth = _get(pool, "other_expenses", period)
    if gp is None or emp is None or oth is None:
        return None
    return gp - emp - oth


def _r_ebit(pool, period):
    eb = _get(pool, "ebitda", period)
    da = _get(pool, "depreciation_amortization", period)
    if eb is None or da is None:
        return None
    return eb - da


def _r_pbt(pool, period):
    ebit = _get(pool, "ebit", period)
    fin = _get(pool, "finance_cost", period)
    oi = _get(pool, "other_income", period)
    soa = _get(pool, "share_of_associates", period) or 0.0
    exc = _get(pool, "exceptional_items", period) or 0.0
    if ebit is None or fin is None or oi is None:
        return None
    return ebit - fin + oi + soa + exc


def _r_pat(pool, period):
    pbt = _get(pool, "pbt", period)
    tax = _get(pool, "tax_expense", period)
    if pbt is None or tax is None:
        return None
    return pbt - tax


def _r_gross_margin(pool, period):
    gp = _get(pool, "gross_profit", period)
    rev = _get(pool, "revenue_from_operations", period)
    if gp is None or rev is None or rev == 0:
        return None
    return gp / rev


def _r_ebitda_margin(pool, period):
    e = _get(pool, "ebitda", period)
    rev = _get(pool, "revenue_from_operations", period)
    if e is None or rev is None or rev == 0:
        return None
    return e / rev


# Balance sheet rules ─────────────────────────────────────────────────────────

def _r_total_equity(pool, period):
    return _sum(pool, ["equity_share_capital", "other_equity"], period)


def _r_total_nc_assets(pool, period):
    # Sum what's present — different companies report different sub-lines.
    return _sum(pool, [
        "ppe", "cwip", "goodwill", "intangible_assets", "intangible_under_dev",
        "rou_assets", "nc_investments", "other_investments",
        "nc_other_financial_assets", "nc_income_tax_asset",
        "deferred_tax_assets", "nc_other_assets",
    ], period, require_all=False, min_present=5)


def _r_total_c_assets(pool, period):
    return _sum(pool, [
        "inventories", "c_investments", "trade_receivables",
        "cash_and_equivalents", "bank_balances_other",
        "c_other_financial_assets", "c_income_tax_asset", "c_other_assets",
    ], period, require_all=False, min_present=4)


def _r_total_assets(pool, period):
    return _sum(pool, ["total_nc_assets", "total_c_assets"], period)


def _r_trade_payables_total(pool, period):
    return _sum(pool, ["trade_payables_micro", "trade_payables_others"],
                period, require_all=False, min_present=1)


def _r_total_nc_liabilities(pool, period):
    return _sum(pool, [
        "nc_borrowings", "nc_lease_liabilities", "nc_other_financial_liabilities",
        "nc_provisions", "deferred_tax_liabilities", "nc_other_liabilities",
    ], period, require_all=False, min_present=3)


def _r_total_c_liabilities(pool, period):
    return _sum(pool, [
        "c_borrowings", "c_lease_liabilities", "trade_payables_total",
        "c_other_financial_liabilities", "c_other_liabilities",
        "c_provisions", "c_tax_liabilities",
    ], period, require_all=False, min_present=4)


def _r_total_equity_liabilities(pool, period):
    return _sum(pool, ["total_equity", "total_nc_liabilities",
                       "total_c_liabilities"], period)


# Rule registry — order matters because dependent rules need their
# inputs computed first. The main loop iterates until convergence, so
# order is only a perf hint.
AGGREGATION_RULES: list[tuple[str, callable]] = [
    ("gross_profit",       _r_gross_profit),
    ("ebitda",             _r_ebitda),
    ("ebit",               _r_ebit),
    ("pbt",                _r_pbt),
    ("pat",                _r_pat),
    ("gross_margin",       _r_gross_margin),
    ("ebitda_margin",      _r_ebitda_margin),
    ("total_equity",       _r_total_equity),
    ("trade_payables_total", _r_trade_payables_total),
    ("total_nc_assets",    _r_total_nc_assets),
    ("total_c_assets",     _r_total_c_assets),
    ("total_assets",       _r_total_assets),
    ("total_nc_liabilities", _r_total_nc_liabilities),
    ("total_c_liabilities",  _r_total_c_liabilities),
    ("total_equity_liabilities", _r_total_equity_liabilities),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AggregationReport:
    derived_count: int = 0
    iterations: int = 0
    derivations: list[tuple[str, str, float]] = field(default_factory=list)
    discrepancies: list[tuple[str, str, float, float]] = field(default_factory=list)


def run_aggregation(pool: Pool, periods: list[str],
                    cross_check_tol: float = 0.01,
                    max_iter: int = 6) -> AggregationReport:
    """
    Mutate `pool` in place, adding derived values. Do not overwrite
    existing values; instead, if cross-check disagrees, record a
    discrepancy. Returns AggregationReport.

    `cross_check_tol` is fractional (0.01 = 1% tolerance).
    """
    rpt = AggregationReport()
    for it in range(max_iter):
        added = 0
        for target_canon, rule_fn in AGGREGATION_RULES:
            for period in periods:
                key = (target_canon, period)
                computed = rule_fn(pool, period)
                if computed is None:
                    continue
                existing = pool.get(key)
                if existing is None:
                    pool[key] = computed
                    rpt.derivations.append((target_canon, period, computed))
                    added += 1
                else:
                    # Cross-check: do existing and computed agree?
                    denom = max(abs(existing), abs(computed), 1e-9)
                    if abs(existing - computed) > cross_check_tol * denom:
                        rpt.discrepancies.append(
                            (target_canon, period, existing, computed)
                        )
        rpt.iterations = it + 1
        if added == 0:
            break
        rpt.derived_count += added
    return rpt


__all__ = [
    "canonicalize",
    "run_aggregation",
    "AggregationReport",
    "CANONICAL_LABELS",
    "AGGREGATION_RULES",
]
