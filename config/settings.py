"""
Central configuration for the Financial Extraction System.
Loads API keys and system settings from environment / .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── DeepSeek / OpenAI-compatible LLM ──────────────────────────────────────────
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ── Extraction behaviour ───────────────────────────────────────────────────────
MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.7"))
MAX_PDF_PAGES: int = int(os.getenv("MAX_PDF_PAGES", "0"))   # 0 = unlimited

# ── Year normalisation map (PDF label → template column header) ───────────────
YEAR_ALIASES: dict[str, str] = {
    # Full fiscal year labels
    "fy2025": "F2025", "fy 2025": "F2025", "fy25": "F2025",
    "2024-25": "F2025", "2024-2025": "F2025",
    "march 31, 2025": "F2025", "march 2025": "F2025",
    "fy2024": "F2024", "fy 2024": "F2024", "fy24": "F2024",
    "2023-24": "F2024", "2023-2024": "F2024",
    "march 31, 2024": "F2024", "march 2024": "F2024",
    "fy2023": "F2023", "fy 2023": "F2023", "fy23": "F2023",
    "2022-23": "F2023", "2022-2023": "F2023",
    "march 31, 2023": "F2023", "march 2023": "F2023",
    "fy2022": "F2022", "fy 2022": "F2022", "fy22": "F2022",
    "2021-22": "F2022", "2021-2022": "F2022",
    "march 31, 2022": "F2022", "march 2022": "F2022",
    "fy2026": "F2026", "fy 2026": "F2026", "fy26": "F2026",
    "2025-26": "F2026", "2025-2026": "F2026",
    "march 31, 2026": "F2026", "march 2026": "F2026",
}

# ── Quarterly year aliases ─────────────────────────────────────────────────────
QUARTER_ALIASES: dict[str, str] = {
    "q1fy25": "1QF2025", "1qfy25": "1QF2025", "q1 fy25": "1QF2025",
    "q2fy25": "2QF2025", "2qfy25": "2QF2025",
    "q3fy25": "3QF2025", "3qfy25": "3QF2025",
    "q4fy25": "4QF2025", "4qfy25": "4QF2025",
    "q1fy24": "1QF2024", "q2fy24": "2QF2024",
    "q3fy24": "3QF2024", "q4fy24": "4QF2024",
}

# ── Sheet-to-document section mapping ─────────────────────────────────────────
# Maps template sheet names to PDF section keywords used for identification
SHEET_SECTION_KEYWORDS: dict[str, list[str]] = {
    "Annual P&L": [
        "profit and loss", "statement of profit", "revenue from operations",
        "ebitda", "cost of goods sold", "gross profit",
    ],
    "Annual Balance Sheet": [
        "balance sheet", "total assets", "equity and liabilities",
        "non-current assets", "current assets",
    ],
    "Annual Cash Flow": [
        "cash flow", "cash flows from operating", "cash flows from investing",
        "cash flows from financing",
    ],
    "Operating metrics": [
        "cdmo", "chg", "operating metrics", "segment revenue",
        "business verticals", "revenue by segment",
    ],
    "Quarterly": [
        "quarterly results", "q1", "q2", "q3", "q4",
        "quarter ended",
    ],
}

# ── Financial field synonym dictionary ────────────────────────────────────────
# Used as hints for the LLM mapper; not exhaustive — the LLM handles the rest.
FIELD_SYNONYMS: dict[str, list[str]] = {
    "Revenue from operations": [
        "revenue from operations", "net revenue", "total revenue",
        "revenues", "net sales", "total net revenue",
    ],
    "Less: Cost of goods sold": [
        "cost of goods sold", "cogs", "cost of materials consumed",
        "cost of revenue", "material costs", "cost of sales",
        "purchases of stock-in-trade", "cost of products sold",
    ],
    "Employee benefits expense": [
        "employee benefits expense", "staff costs", "personnel expenses",
        "payroll", "employee cost", "human resources cost",
    ],
    "Other expenses (Net)": [
        "other expenses", "selling general and administrative",
        "sg&a", "operating expenses", "other operating expenses",
    ],
    "Less: Depreciation, amortization & impairment": [
        "depreciation", "amortization", "d&a", "depreciation and amortization",
        "depreciation amortization and impairment",
    ],
    "Less: Finance cost": [
        "finance costs", "interest expense", "finance charges",
        "borrowing costs", "interest on borrowings",
    ],
    "Add: Other Income (Net)": [
        "other income", "non-operating income", "interest income",
        "other income net", "miscellaneous income",
    ],
    "Add: Share of net profit of associates": [
        "share of profit of associates", "equity in earnings",
        "share of net profit", "profit from associates",
    ],
    "Less: Tax": [
        "income tax", "tax expense", "provision for tax",
        "current tax", "deferred tax", "total tax",
    ],
    "Diluted EPS": [
        "diluted eps", "earnings per share diluted", "diluted earnings per share",
        "eps diluted",
    ],
}
