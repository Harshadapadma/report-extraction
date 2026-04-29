"""
CLI runner — use without Streamlit for batch processing or testing.

Usage:
    python run_cli.py \
        --template "PPL_Template_F2024.xlsx" \
        --pdfs "Annual Report FY 2024-25.pdf" "Investor Presentation FY2025.pdf" \
        --output "PPL_Template_Populated.xlsx" \
        --api-key "sk-..." \
        [--overwrite]

Or with .env file:
    DEEPSEEK_API_KEY=sk-... python run_cli.py --template ... --pdfs ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Financial Data Extraction & Template Population CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--template", required=True, help="Path to the Excel template (.xlsx)")
    parser.add_argument("--pdfs", nargs="+", required=True, help="Path(s) to PDF files")
    parser.add_argument("--output", default=None, help="Output Excel path (default: <template>_populated.xlsx)")
    parser.add_argument("--api-key", default=None, help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing values in template")
    parser.add_argument("--no-llm-mapping", action="store_true", help="Disable LLM field mapping (use fuzzy only)")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Min extraction confidence (0.0–1.0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    # Web cross-validation identifiers (optional)
    parser.add_argument("--nse", default=None, metavar="SYMBOL",
                        help="NSE ticker symbol for web cross-validation (e.g. HDFCBANK)")
    parser.add_argument("--bse", default=None, metavar="CODE",
                        help="BSE scrip code for web cross-validation (e.g. 500180)")
    args = parser.parse_args()

    # ── API key ────────────────────────────────────────────────────────────────
    if args.api_key:
        os.environ["DEEPSEEK_API_KEY"] = args.api_key
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set. Use --api-key or DEEPSEEK_API_KEY env var.")
        sys.exit(1)

    # ── Logging ────────────────────────────────────────────────────────────────
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Validate files ─────────────────────────────────────────────────────────
    template_path = Path(args.template)
    if not template_path.exists():
        print(f"ERROR: Template not found: {template_path}")
        sys.exit(1)

    pdf_paths = []
    for p in args.pdfs:
        pp = Path(p)
        if not pp.exists():
            print(f"WARNING: PDF not found: {pp} — skipping")
        else:
            pdf_paths.append(pp)

    if not pdf_paths:
        print("ERROR: No valid PDF files provided.")
        sys.exit(1)

    # ── Run pipeline ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Financial Data Extraction System")
    print(f"{'='*60}")
    print(f"  Template : {template_path}")
    print(f"  PDFs     : {[p.name for p in pdf_paths]}")
    print(f"  Output   : {args.output or 'auto-generated'}")
    print(f"  Overwrite: {args.overwrite}")
    print(f"{'='*60}\n")

    # Step 1: Read template
    print("[1/4] Reading template structure...")
    from mapper.template_reader import read_template
    template_model = read_template(template_path)
    print(f"  → {len(template_model.sheets)} sheets loaded")

    # Step 2: Parse PDFs
    print("\n[2/4] Parsing PDFs...")
    from extractor.pdf_parser import parse_pdf
    parsed_docs = []
    for pdf_path in pdf_paths:
        print(f"  → Parsing: {pdf_path.name}")
        doc = parse_pdf(pdf_path)
        parsed_docs.append(doc)
        print(f"     {len(doc.sections)} sections detected")

    # Step 3: Extract data
    # Primary: pdfplumber direct table extraction (no LLM for values)
    # Fallback: LLM extraction if pdfplumber gets 0 values
    print("\n[3/4] Extracting financial data from PDF tables...")
    from extractor.table_extractor import extract_tables_from_pdf
    from extractor.llm_extractor import (
        extract_all_sections, extract_from_full_text, _merge_results,
    )
    from openai import OpenAI
    from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    all_results = []

    for pdf_path, doc in zip(pdf_paths, parsed_docs):
        stmt_type = getattr(doc, "_stmt_override", None) or "consolidated"
        print(f"  → {doc.filename}")

        # Primary: direct table extraction
        tbl_results = extract_tables_from_pdf(
            pdf_source=str(pdf_path),
            parsed_doc=doc,
            statement_type=stmt_type,
        )
        n_tbl = sum(len(r.data) for r in tbl_results)
        print(f"     Table extraction: {n_tbl} values")

        if n_tbl > 0:
            all_results.extend(tbl_results)
        else:
            # Fallback: LLM
            print(f"     ⚠ Table extraction got 0 values — falling back to LLM...")
            if doc.sections:
                llm_results = extract_all_sections(doc, client=client)
            else:
                llm_results = extract_from_full_text(doc.get_all_text(), client=client)
            n_llm = sum(len(r.data) for r in llm_results)
            print(f"     LLM fallback: {n_llm} values")
            all_results.extend(llm_results)

    all_results = _merge_results(all_results)
    all_results = [r for r in all_results if r.confidence >= args.min_confidence]

    if not all_results:
        print("\nWARNING: No data was extracted. Check your PDFs.")
        sys.exit(1)

    # Step 3b: Web cross-validation (optional)
    if args.nse or args.bse:
        print(f"\n[3b] Running web cross-validation (NSE={args.nse}, BSE={args.bse})...")
        try:
            from web.collector import WebCollector
            from web.base import CompanyIdentifier
            company = CompanyIdentifier(
                name=args.nse or args.bse,
                nse_symbol=args.nse,
                bse_code=args.bse,
            )
            years = sorted({r.year for r in all_results})
            collector = WebCollector()
            web_results = collector.collect(company=company, years=years)
            print(f"  → {len(web_results)} web data points retrieved")

            if web_results:
                validation = collector.cross_validate(all_results, web_results)
                summary = collector.validation_summary(validation)
                print(
                    f"  → Agreed (≤10%): {summary['very_high'] + summary['high']} | "
                    f"Flagged: {summary['flagged']} | No web data: {summary['no_web']}"
                )

                # Gap-fill: add web values for fields PDF missed
                web_fill = collector.fill_missing_from_web(
                    all_results, web_results, template_model
                )
                if web_fill:
                    n_fill = sum(len(r.data) for r in web_fill)
                    all_results.extend(web_fill)
                    print(f"  → Gap-fill: {n_fill} additional field(s) sourced from web")

                # Print flagged fields
                flagged = [r for r in validation if r.confidence_upgrade == "FLAG"]
                if flagged:
                    print(f"\n  ⚠️  {len(flagged)} field(s) flagged for review (→ Data Review sheet):")
                    for r in flagged:
                        print(f"     - {r.template_field} [{r.year}]: {r.flag_reason}")
        except Exception as e:
            print(f"  ⚠️  Web validation failed: {e} (pipeline continues)")
    else:
        print("\n[3b] Web cross-validation skipped (no --nse or --bse provided)")

    # Step 4: Map + write
    print("\n[4/4] Mapping fields and writing to template...")
    from mapper.field_mapper import FieldMapper, stamp_validation, map_direct
    from integrator.excel_writer import write_to_template

    # Use direct mapping if template-guided extraction produced the results
    is_template_guided = any(
        getattr(r, "notes", "") == "mode=template_guided"
        for r in all_results
    )
    if is_template_guided:
        print("  → Template-guided mode — direct label matching")
        reports = map_direct(all_results, template_model)
    else:
        mapper = FieldMapper(template_model, use_llm=not args.no_llm_mapping)
        reports = mapper.map_results(all_results)

    # Stamp web validation confidence onto mapped fields (if web was used)
    web_validated = False
    if (args.nse or args.bse) and "validation" in dir():
        stamp_validation(reports, validation)
        web_validated = True

    total_mapped = sum(len(r.mapped) for r in reports)
    total_unmapped = sum(len(r.unmapped) for r in reports)
    print(f"  → {total_mapped} fields mapped, {total_unmapped} could not be mapped")

    if total_unmapped > 0:
        print("  → Unmapped fields:")
        for report in reports:
            for label in report.unmapped:
                print(f"       - {label}")

    output_path = args.output or str(
        template_path.parent / (template_path.stem + "_populated" + template_path.suffix)
    )
    write_result = write_to_template(
        template_path=template_path,
        mapping_reports=reports,
        output_path=output_path,
        overwrite_existing=args.overwrite,
        web_validated=web_validated,
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✅ COMPLETE")
    print(f"{'='*60}")
    print(f"  Output file   : {write_result.output_path}")
    print(f"  Cells written : {write_result.cells_written}")
    print(f"  Formulas kept : {write_result.cells_skipped_formula}")
    print(f"  Existing kept : {write_result.cells_skipped_filled}")
    if write_result.warnings:
        print(f"  Warnings      : {len(write_result.warnings)}")
    print(f"{'='*60}\n")

    # Print audit log
    if write_result.audit_log:
        print("Audit log:")
        print(f"  {'Sheet':<25} {'Cell':<6} {'Label':<40} {'Year':<8} {'Type':<15} {'Value':>12}  Method")
        print("  " + "-" * 115)
        for rec in write_result.audit_log:
            print(
                f"  {rec.sheet:<25} {rec.cell_ref:<6} {rec.row_label:<40} "
                f"{rec.year:<8} {rec.statement_type:<15} {rec.value:>12.2f}  {rec.match_method}"
            )


if __name__ == "__main__":
    main()
