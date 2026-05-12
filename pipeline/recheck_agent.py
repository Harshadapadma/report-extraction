"""
pipeline/recheck_agent.py
──────────────────────────
Phase 6: Recheck — verify accuracy + final gap-fill with full context.

This runs AFTER all previous phases (1-5) and does two things:

  Step A — Accuracy recheck
    Gives the LLM the ENTIRE filled sheet at once so it can see every
    value in context.  Flags any value that looks wrong given its
    neighbours (e.g. EBITDA > Revenue, negative Revenue, COGS > 80%
    of Revenue when peers are ~40%).  For each flagged value, the LLM
    also suggests a correction and a confidence score.

  Step B — Final gap-fill
    For labels still completely empty after all previous phases, the
    recheck makes one final LLM call with maximum context:
      • The full filled sheet (neighbours give scale/magnitude context)
      • PDF text (keyword-selected pages)
      • Multi-source pool snapshot (top-5 candidates per label with values)
    The LLM uses the already-known values to infer what the missing ones
    should be (e.g. "Revenue=5000, EBITDA=850, so COGS+Opex ≈ 4150").

Why this works better than earlier phases:
    Earlier phases map labels one at a time without seeing the whole sheet.
    The recheck sees the complete picture and can use cross-row reasoning.

Returns (updated_sheet_filled, corrections_log, gaps_log)
  corrections_log : [{label, year, old_val, new_val, reason}]
  gaps_log        : [{label, year, value, confidence, source}]
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Minimum confidence the LLM must report before we accept a correction
_CORRECTION_MIN_CONF = 0.75
# Minimum confidence for a gap-fill value to be accepted
_GAP_FILL_MIN_CONF   = 0.65


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_sheet_snapshot(
    template_labels: list[str],
    sheet_filled:    dict[str, dict[str, float | None]],
    years:           list[str],
    max_years:       int = 4,
) -> str:
    """
    Build a compact human-readable table of the filled sheet.
    Empty cells are shown as '—' so the LLM can spot gaps easily.
    """
    show_years = years[:max_years]
    header = f"{'Label':<50} " + "  ".join(f"{yr:>10}" for yr in show_years)
    rows = [header, "-" * len(header)]

    for lbl in template_labels:
        yr_vals = sheet_filled.get(lbl, {})
        cells = []
        for yr in show_years:
            v = yr_vals.get(yr)
            cells.append(f"{v:>10,.1f}" if v is not None else f"{'—':>10}")
        rows.append(f"{lbl:<50} " + "  ".join(cells))

    return "\n".join(rows)


def _select_pdf_pages(pdf_text: str, labels: list[str], max_chars: int = 20_000) -> str:
    """Keyword-select the most relevant PDF chunks for a list of labels."""
    if not pdf_text or not labels:
        return ""
    keywords: set[str] = set()
    for lbl in labels:
        for w in re.split(r"[^a-z]+", lbl.lower()):
            if len(w) > 3:
                keywords.add(w)

    chunks = re.split(r"\n{2,}|\f", pdf_text)
    scored = []
    for chunk in chunks:
        c = chunk.strip()
        if len(c) < 30:
            continue
        cl = c.lower()
        score = sum(1 for kw in keywords if kw in cl)
        digit_ratio = sum(1 for ch in c if ch.isdigit()) / max(len(c), 1)
        if digit_ratio > 0.12:
            score += 2
        scored.append((score, c))

    scored.sort(key=lambda x: -x[0])
    parts, total = [], 0
    for _, chunk in scored:
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts) if parts else pdf_text[:max_chars]


def _build_pool_snapshot(
    still_empty:  list[str],
    pool:         dict[str, dict[str, float]],
    years:        list[str],
    top_k:        int = 5,
) -> str:
    """
    For each still-empty label, show top-k pool candidates with values.
    Uses simple token overlap to rank candidates.
    """
    if not pool or not still_empty:
        return ""

    lines = ["POOL CANDIDATES (alternative label names + their values):"]
    for lbl in still_empty[:20]:   # cap to avoid bloating context
        lbl_words = set(re.split(r"[^a-z]+", lbl.lower())) - {"", "the", "of", "and"}
        ranked = sorted(
            pool.items(),
            key=lambda kv: len(
                set(re.split(r"[^a-z]+", kv[0].lower())) & lbl_words
            ),
            reverse=True,
        )[:top_k]
        if not ranked:
            continue
        lines.append(f"\n  Missing: \"{lbl}\"")
        for pool_lbl, yr_vals in ranked:
            sample = ", ".join(
                f"{yr}={yr_vals[yr]:,.0f}"
                for yr in years[:2]
                if yr in yr_vals and yr_vals[yr] is not None
            )
            lines.append(f"    • {pool_lbl}" + (f"  [{sample}]" if sample else ""))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Step A — Accuracy recheck
# ─────────────────────────────────────────────────────────────────────────────

def _recheck_accuracy(
    sheet_name:     str,
    sheet_snapshot: str,
    template_labels: list[str],
    years:           list[str],
    sector:          str,
    client:          Any,
    model:           str,
) -> list[dict]:
    """
    Ask the LLM to review the full filled sheet and flag suspicious values.
    Returns list of {label, year, issue, suggested_value, confidence}.
    """
    if not sheet_snapshot or client is None:
        return []

    show_years = years[:4]
    prompt = f"""You are a senior financial analyst reviewing data for an Indian {sector} company.

Below is the "{sheet_name}" sheet with all filled values (INR Crores, '—' = missing).
Review EVERY row carefully and flag values that look WRONG given the context of other rows.

{sheet_snapshot}

WHAT TO FLAG:
1. Values that violate accounting logic (e.g. EBITDA > Revenue, PAT > PBT, negative Revenue)
2. Values that are implausible in magnitude relative to neighbours
   (e.g. Finance Cost = 50,000 when Revenue = 5,000 makes no sense)
3. Values with wrong sign (expenses showing as negative when they should be positive)
4. A "Total" row whose value doesn't equal the sum of visible sub-items
5. Identical values across all years when variation is expected

For each flagged issue, also suggest what the correct value should be (if you can infer it)
and give a confidence score 0.0–1.0 for your suggested correction.

ONLY flag issues you are CONFIDENT about. Do NOT flag values just because they look unusual —
only flag clear violations of accounting identities or impossible magnitudes.

If everything looks correct, return an empty list.

Return JSON only:
{{
  "issues": [
    {{
      "label":           "<exact label from sheet>",
      "year":            "<fiscal year>",
      "current_value":   <number>,
      "issue":           "<one sentence describing the problem>",
      "suggested_value": <corrected number or null>,
      "confidence":      <0.0-1.0>
    }},
    ...
  ]
}}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data   = json.loads(resp.choices[0].message.content)
        issues = data.get("issues", [])
        logger.info(
            f"recheck_accuracy '{sheet_name}': LLM flagged {len(issues)} issue(s)"
        )
        return issues
    except Exception as exc:
        logger.warning(f"_recheck_accuracy failed for '{sheet_name}': {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Step B — Final gap-fill
# ─────────────────────────────────────────────────────────────────────────────

def _recheck_gap_fill(
    sheet_name:      str,
    still_empty:     list[str],
    sheet_snapshot:  str,
    pool_snapshot:   str,
    pdf_clip:        str,
    years:           list[str],
    sector:          str,
    client:          Any,
    model:           str,
) -> list[dict]:
    """
    One final LLM call with full context: filled sheet + pool candidates + PDF.
    Returns list of {label, year, value, confidence}.
    """
    if not still_empty or client is None:
        return []

    lbls_str    = "\n".join(f"  - {lbl}" for lbl in still_empty[:25])
    yrs_str     = ", ".join(years[:4])
    pdf_section = (
        f"\nRELEVANT PDF PAGES (INR Crores):\n{pdf_clip}\n"
        if pdf_clip else ""
    )
    pool_section = f"\n{pool_snapshot}\n" if pool_snapshot else ""

    prompt = f"""You are a financial data expert for Indian {sector} companies (Ind AS).

The following labels in the "{sheet_name}" sheet are STILL EMPTY after extensive automated filling.
Use ALL available context — the filled sheet, pool candidates, and PDF text — to infer the values.

YEARS NEEDED: {yrs_str}

STILL EMPTY LABELS:
{lbls_str}

CURRENT FILLED SHEET (use neighbouring values to infer missing ones):
{sheet_snapshot}
{pool_section}{pdf_section}
INSTRUCTIONS:
1. Use the filled rows as context — e.g. if Revenue=5000 and EBITDA=850, you know total costs≈4150.
2. Use pool candidates if a matching label is shown.
3. Use PDF text if the value is explicitly mentioned.
4. Only return a value if you are reasonably confident (confidence ≥ 0.65).
5. Values must be in INR Crores.
6. Parenthesised numbers like (50.00) are negative.

Return JSON only:
{{
  "fills": [
    {{
      "label":      "<exact label from still-empty list>",
      "year":       "<fiscal year>",
      "value":      <number or null>,
      "confidence": <0.0-1.0>,
      "source":     "<derivation|pdf|pool|inference>"
    }},
    ...
  ]
}}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data  = json.loads(resp.choices[0].message.content)
        fills = data.get("fills", [])
        logger.info(
            f"recheck_gap_fill '{sheet_name}': LLM returned {len(fills)} fill candidate(s)"
        )
        return fills
    except Exception as exc:
        logger.warning(f"_recheck_gap_fill failed for '{sheet_name}': {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def recheck_phase(
    sheet_name:      str,
    sheet_filled:    dict[str, dict[str, float | None]],
    template_labels: list[str],
    years:           list[str],
    sector:          str = "General",
    pdf_text:        str = "",
    pool:            dict[str, dict[str, float]] | None = None,
    client:          Any = None,
    model:           str = "",
) -> tuple[dict[str, dict[str, float | None]], list[dict], list[dict]]:
    """
    Phase 6: Recheck — verify accuracy + final gap-fill.

    Parameters
    ----------
    sheet_name      : Name of the sheet
    sheet_filled    : Current fills after Phases 1-5
    template_labels : Ordered list of all template labels for this sheet
    years           : Fiscal years
    sector          : Sector string for LLM context
    pdf_text        : Full extracted PDF text
    pool            : Raw pool {label: {year: value}} from _pool_web_labels
    client          : OpenAI-compatible client
    model           : Model name

    Returns
    -------
    (updated_sheet_filled, corrections_log, gaps_log)
      corrections_log : [{label, year, old_val, new_val, reason, confidence}]
      gaps_log        : [{label, year, value, confidence, source}]
    """
    # Only run for financial statement sheets
    sl = sheet_name.lower()
    is_fin = any(k in sl for k in ("p&l", "profit", "income", "balance", "bs", "cash", "cf"))
    if not is_fin or client is None:
        return sheet_filled, [], []

    result          = {lbl: dict(yr_vals) for lbl, yr_vals in sheet_filled.items()}
    corrections_log : list[dict] = []
    gaps_log        : list[dict] = []
    pool            = pool or {}

    # Build the full sheet snapshot (used by both steps)
    snapshot = _build_sheet_snapshot(template_labels, result, years)

    # ── Step A: Accuracy recheck ──────────────────────────────────────────────
    issues = _recheck_accuracy(
        sheet_name=sheet_name,
        sheet_snapshot=snapshot,
        template_labels=template_labels,
        years=years,
        sector=sector,
        client=client,
        model=model,
    )

    for issue in issues:
        lbl      = issue.get("label", "")
        yr       = issue.get("year", "")
        conf     = float(issue.get("confidence", 0))
        new_val  = issue.get("suggested_value")
        old_val  = result.get(lbl, {}).get(yr)

        if not lbl or not yr or lbl not in result:
            continue

        # Only apply correction if LLM is confident AND value actually changes
        if (conf >= _CORRECTION_MIN_CONF
                and new_val is not None
                and old_val is not None
                and abs(float(new_val) - float(old_val)) > 0.01):
            result[lbl][yr] = float(new_val)
            corrections_log.append({
                "label":      lbl,
                "year":       yr,
                "old_val":    old_val,
                "new_val":    float(new_val),
                "reason":     issue.get("issue", ""),
                "confidence": conf,
            })
            logger.info(
                f"recheck correction: '{lbl}' {yr}  {old_val} → {float(new_val)}"
                f"  (conf={conf:.2f})"
            )

    # ── Step B: Final gap-fill ────────────────────────────────────────────────
    still_empty = [
        lbl for lbl in template_labels
        if all(v is None for v in result.get(lbl, {}).values())
    ]

    if still_empty:
        pdf_clip     = _select_pdf_pages(pdf_text, still_empty, max_chars=20_000)
        pool_snapshot = _build_pool_snapshot(still_empty, pool, years)

        # Rebuild snapshot with any corrections applied
        snapshot_updated = _build_sheet_snapshot(template_labels, result, years)

        fills = _recheck_gap_fill(
            sheet_name=sheet_name,
            still_empty=still_empty,
            sheet_snapshot=snapshot_updated,
            pool_snapshot=pool_snapshot,
            pdf_clip=pdf_clip,
            years=years,
            sector=sector,
            client=client,
            model=model,
        )

        for fill in fills:
            lbl  = fill.get("label", "")
            yr   = fill.get("year", "")
            val  = fill.get("value")
            conf = float(fill.get("confidence", 0))
            src  = fill.get("source", "inference")

            if not lbl or not yr or val is None:
                continue
            if conf < _GAP_FILL_MIN_CONF:
                continue
            if lbl not in result:
                continue

            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue

            # Only fill if still empty
            if result[lbl].get(yr) is None:
                result[lbl][yr] = fval
                gaps_log.append({
                    "label":      lbl,
                    "year":       yr,
                    "value":      fval,
                    "confidence": conf,
                    "source":     src,
                })
                logger.info(
                    f"recheck gap-fill: '{lbl}' {yr} = {fval}  "
                    f"(conf={conf:.2f}, src={src})"
                )

    # Summary
    logger.info(
        f"recheck_phase '{sheet_name}': "
        f"{len(corrections_log)} corrections, {len(gaps_log)} gaps filled"
    )
    return result, corrections_log, gaps_log
