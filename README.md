# 📊 Universal Financial Data Extraction & Template Population System

AI-powered system that extracts financial data from unstructured PDFs (annual reports, investor
presentations) and populates a standardised Excel template — non-destructively, year-accurately,
and across multiple companies.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd financial_extractor
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and add your DeepSeek API key:
# DEEPSEEK_API_KEY=sk-...
```

### 3a. Run the Streamlit UI (easiest)

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### 3b. Run the CLI (batch / scripting)

```bash
python run_cli.py \
    --template "PPL_Template_F2024.xlsx" \
    --pdfs "Annual Report FY 2024-25.pdf" "Investor Presentation FY2025.pdf" \
    --output "PPL_Template_Populated.xlsx"
```

---

## 🏗️ Architecture

```
financial_extractor/
├── app.py                      ← Streamlit web UI
├── run_cli.py                  ← CLI batch runner
├── requirements.txt
├── .env.example
│
├── config/
│   └── settings.py             ← API config, year aliases, field synonyms
│
├── extractor/
│   ├── pdf_parser.py           ← PDF text extraction + section detection
│   └── llm_extractor.py        ← DeepSeek API structured data extraction
│
├── mapper/
│   ├── template_reader.py      ← Excel template structure parser
│   └── field_mapper.py         ← 3-tier field mapping (exact/fuzzy/LLM)
│
├── integrator/
│   └── excel_writer.py         ← Non-destructive openpyxl writer + audit log
│
└── utils/
    └── helpers.py              ← Year normalisation, number parsing, etc.
```

### Data Flow

```
PDF(s) + Excel Template
        │
        ▼
[pdf_parser] — Page-level text extraction + section identification
        │
        ▼
[llm_extractor] — DeepSeek API → structured JSON per section × year
        │
        ▼
[template_reader] — Excel row/column index + formula/fill tracking
        │
        ▼
[field_mapper] — Exact → Fuzzy → LLM semantic matching
        │
        ▼
[excel_writer] — Non-destructive cell writes + audit log
        │
        ▼
Populated Excel Template
```

---

## 🔑 Key Design Principles

| Principle | Implementation |
|---|---|
| **Year-specific accuracy** | Year normalisation across 20+ formats (FY2025, 2024-25, March 31 2025 → F2025) |
| **Template-constrained mapping** | Rows and columns are read directly from the template — no hardcoding |
| **Non-destructive integration** | Formula cells and pre-filled cells are never overwritten |
| **Cross-company generalisation** | No company-specific logic; all mapping is semantic via LLM |
| **Naming conflict resolution** | 3-tier matching: exact → synonym dict → DeepSeek semantic |
| **Partial data handling** | Only confidently extracted fields are written; missing = skip |
| **Audit trail** | Every write is logged with method, score, source label, and timestamp |

---

## ⚙️ Configuration

Edit `.env` or set environment variables:

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | required | Your DeepSeek API key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API endpoint |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model name |
| `MIN_CONFIDENCE` | `0.7` | Minimum section confidence threshold |
| `MAX_PDF_PAGES` | `0` (unlimited) | Max pages to parse per PDF |

---

## 📋 Supported Sheets

| Template Sheet | Extracted From |
|---|---|
| Annual P&L | Profit & Loss statements (consolidated + standalone) |
| Annual Balance Sheet | Balance sheet (consolidated + standalone) |
| Annual Cash Flow | Cash flow statement |
| Operating metrics | Segment revenue, CDMO/CHG/PCH breakdowns |
| Quarterly | Quarterly income statement data |

---

## 🧪 Testing

```bash
# Test with your own files (CLI mode)
python run_cli.py \
    --template path/to/template.xlsx \
    --pdfs path/to/report.pdf \
    --verbose

# Quick template structure check
python -c "
from mapper.template_reader import read_template
m = read_template('PPL_Template_F2024.xlsx')
for name, sheet in m.sheets.items():
    print(f'{name}: {len(sheet.row_index)} rows, years={list(sheet.year_cols.keys())}')
"
```

---

## 🔧 Extending for New Companies

1. **No code changes needed** — the system reads template structure dynamically.
2. To add new field synonyms, edit `config/settings.py → FIELD_SYNONYMS`.
3. To add new year formats, edit `config/settings.py → YEAR_ALIASES`.
4. The LLM mapper handles novel field names automatically.

---

## 📦 Dependencies

- `pdfplumber` — PDF text extraction
- `openpyxl` — Excel read/write
- `openai` — DeepSeek API client (OpenAI-compatible)
- `fuzzywuzzy` — Fuzzy string matching
- `streamlit` — Web UI
- `pandas` — Data display
- `python-dotenv` — Environment config
