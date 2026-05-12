"""
pipeline/stmt_type_resolver.py
──────────────────────────────
Decide whether to fetch / fill consolidated, standalone, or both for each
sheet, given the template structure and what's available from sources.

Logic
─────
1. Inspect TemplateModel.section_offsets per sheet:
     • If both 'consolidated' and 'standalone' present  → fill BOTH.
     • If only 'consolidated' present                   → fill consolidated.
     • If only 'standalone' present                     → fill standalone.
     • If neither present (no section header)            → default consolidated;
       fall back to standalone if data unavailable.
2. For sectors where a parent has no subsidiaries (typical for many SFBs,
   single-entity insurers), there is no consolidated filing. The orchestrator
   will detect "no data" from the source and fall back to standalone
   automatically — this resolver only encodes the *intent*.
3. The user can override per-sheet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StatementPlan:
    """Per-sheet plan: which statement types to fetch and which to write."""
    sheet_name: str
    fetch:    list[str] = field(default_factory=list)   # ["consolidated"], ["standalone"], or both
    write:    list[str] = field(default_factory=list)   # what the template expects
    fallback: list[str] = field(default_factory=list)   # tried in order if primary returns nothing

    def __str__(self) -> str:
        return (f"{self.sheet_name}: write={self.write}, "
                f"fetch={self.fetch}, fallback={self.fallback}")


def resolve_for_sheet(
    sheet_name: str,
    template_model,                                  # mapper.template_reader.TemplateModel
    user_override: Optional[str] = None,             # "consolidated" | "standalone" | "both" | None
) -> StatementPlan:
    """
    Inspect the template's section structure and compute a plan.
    """
    sheet = template_model.sheets.get(sheet_name)
    if not sheet:
        return StatementPlan(sheet_name, fetch=["consolidated"], write=["consolidated"],
                             fallback=["standalone"])

    section_offsets = sheet.section_offsets or {}
    has_cons = "consolidated" in section_offsets
    has_std  = "standalone" in section_offsets

    # User override takes precedence
    if user_override == "consolidated":
        return StatementPlan(sheet_name, ["consolidated"], ["consolidated"], ["standalone"])
    if user_override == "standalone":
        return StatementPlan(sheet_name, ["standalone"], ["standalone"], ["consolidated"])
    if user_override == "both":
        return StatementPlan(sheet_name, ["consolidated", "standalone"],
                             ["consolidated", "standalone"], [])

    # Auto-detection based on template structure
    if has_cons and has_std:
        return StatementPlan(
            sheet_name,
            fetch=["consolidated", "standalone"],
            write=["consolidated", "standalone"],
            fallback=[],
        )
    if has_cons and not has_std:
        return StatementPlan(
            sheet_name, ["consolidated"], ["consolidated"], ["standalone"],
        )
    if has_std and not has_cons:
        return StatementPlan(
            sheet_name, ["standalone"], ["standalone"], ["consolidated"],
        )

    # No section header at all — common for Quarterly / Operating Metrics sheets
    return StatementPlan(
        sheet_name, ["consolidated"], ["consolidated"], ["standalone"],
    )


def resolve_for_template(
    template_model,
    user_override: Optional[str] = None,
) -> dict[str, StatementPlan]:
    """Build a plan for every sheet in the template."""
    return {
        sn: resolve_for_sheet(sn, template_model, user_override)
        for sn in template_model.sheets
    }


__all__ = ["StatementPlan", "resolve_for_sheet", "resolve_for_template"]
