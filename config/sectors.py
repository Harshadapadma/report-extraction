"""
Sector Definitions
==================
Each sector entry defines:
  - detection_keywords : words found in the PDF that signal this sector
  - pl_scoring         : keywords for PDF page scoring (P&L section)
  - bs_scoring         : keywords for PDF page scoring (Balance Sheet section)
  - cf_scoring         : keywords for PDF page scoring (Cash Flow section)
  - pl_fields          : (canonical_name, [regex_patterns]) for rule-based P&L extraction
  - bs_fields          : same for Balance Sheet
  - cf_fields          : same for Cash Flow
  - llm_hint           : short description passed to the LLM prompt so it knows the context

Adding a new sector: just add a new key to SECTORS dict below.
No other file changes needed.
"""

from __future__ import annotations

SECTORS: dict[str, dict] = {

    # ── Banking / Small Finance Banks / Co-operative Banks ────────────────────
    "banking": {
        "detection_keywords": [
            "scheduled bank", "banking company", "reserve bank of india",
            "net interest margin", "net npa", "gross npa", "capital adequacy ratio",
            "car ratio", "priority sector", "slr", "crar", "net interest income",
            "interest earned", "interest expended", "advances", "deposits",
            "balances with rbi", "balances with reserve bank",
        ],
        "detection_threshold": 3,   # how many keywords needed to confirm sector

        "pl_scoring": {
            "profit and loss": 3.0,
            "interest earned": 3.0,
            "interest expended": 3.0,
            "net interest income": 2.5,
            "non-interest income": 2.0,
            "provisions and contingencies": 2.5,
            "profit before tax": 2.0,
            "profit after tax": 2.0,
            # Schedule pages for P&L detail
            "interest/discount on advances": 1.5,
            "interest on advances": 1.5,
            "income on investments": 1.5,
            "interest on balances with reserve bank": 1.5,
            "commission, exchange": 1.5,
            "profit/loss on sale of investments": 1.5,
            "interest on deposits": 1.5,
            "interest on reserve bank": 1.5,
            "payments to employees": 1.0,
            "rent, taxes": 1.0,
            "depreciation on bank's property": 1.0,
        },
        "bs_scoring": {
            "balance sheet": 3.0,
            "capital and liabilities": 3.0,
            "capital & liabilities": 3.0,
            "deposits": 2.5,
            "advances": 2.5,
            "balances with rbi": 2.5,
            "balances with reserve bank": 2.5,
            "total assets": 2.0,
            # Schedule pages — each schedule header contains these terms
            "government securities": 1.5,
            "other approved securities": 1.5,
            "bills purchased": 1.5,
            "term loans": 1.5,
            "cash and balances": 1.5,
            "balances with banks": 1.5,
            "money at call": 1.5,
            "gross npa": 2.0,
            "net npa": 2.0,
            "gnpa": 2.0,
            "nnpa": 2.0,
            "sub-standard": 1.0,
            "doubtful": 1.0,
            "debentures and bonds": 1.0,
            "debentures & bonds": 1.0,
        },
        "cf_scoring": {
            "cash flow": 3.0,
            "cash flows from operating": 3.0,
            "cash flows from investing": 2.5,
            "cash flows from financing": 2.5,
        },

        "pl_fields": [
            # Main P&L lines
            ("Interest earned",              [r"interest earned\b", r"interest income\b"]),
            ("Interest expended",            [r"interest expended\b", r"interest expense\b"]),
            ("Net Interest Income",          [r"net interest income\b", r"\bNII\b"]),
            ("Other income",                 [r"other income\b", r"non.interest income\b"]),
            ("Net Total Income",             [r"net total income\b", r"total net income\b"]),
            ("Operating expenses",           [r"operating expenses?\b", r"operating costs?\b"]),
            ("Employee cost",                [r"employee(?:s)? cost\b", r"staff (?:cost|expense)", r"employee benefits? expense"]),
            ("Other operating expenses",     [r"other operating expenses?\b"]),
            ("Pre-provision profit",         [r"pre.provision (?:operating )?profit\b", r"ppop\b"]),
            ("Provisions & contingencies",   [r"provisions?\s+(?:and|&)\s+contingencies?\b", r"provisions? for (?:loan|npa)"]),
            ("Profit before tax",            [r"profit before (?:income\s+)?tax\b", r"\bpbt\b"]),
            ("Tax",                          [r"(?:income\s+)?tax\s*(?:expense)?\b", r"provision\s+for\s+tax\b"]),
            ("Profit after tax",             [r"profit after (?:income\s+)?tax\b", r"profit for the (?:year|period)\b", r"\bpat\b"]),
            ("Diluted EPS",                  [r"diluted\s+(?:eps|earnings per share)", r"earnings per equity share.*diluted"]),
            # Schedule 13 — Interest earned breakdown
            ("Interest/discount on advances",[r"interest.discount\s+on\s+advance", r"interest\s+on\s+advances?\b"]),
            ("Income on investments",        [r"income on investments?\b", r"interest on investments?\b"]),
            ("Int. on bal. with RBI",        [r"interest.*balances?\s+with\s+(?:rbi|reserve bank)", r"int(?:erest)?.*rbi.*interbank"]),
            # Schedule 14 — Other income breakdown
            ("Commission, exchange & brokerage", [r"commission.*exchange.*brokerage\b", r"commission,\s*exchange\b"]),
            ("Profit/loss on sale of investments", [r"profit.*loss.*sale.*investments?\b", r"profit/loss on sale of inv"]),
            ("Profit/loss on sale of assets",[r"profit.*loss.*sale.*assets?\b"]),
            ("Miscellaneous income",         [r"miscellaneous\s+income\b"]),
            # Schedule 15 — Interest expended breakdown
            ("Interest on deposits",         [r"interest\s+on\s+deposits?\b"]),
            ("Interest on RBI/inter-bank",   [r"interest.*(?:rbi|reserve bank).*borrow", r"interest.*inter.bank\b"]),
        ],

        "bs_fields": [
            # Capital & Liabilities
            ("Capital",                      [r"^capital\b(?!\s*(?:and|&|work|adequacy))"]),
            ("Reserves & surplus",           [r"reserves?\s+(?:and|&)\s+surplus\b", r"reserves?\s+and\s+surplus\b"]),
            ("Networth",                     [r"\bnetworth\b", r"\bnet\s+worth\b", r"total equity\b", r"shareholders.?\s*funds?\b"]),
            ("Deposits",                     [r"\bdeposits\b(?!\s*(?:from|with|in))"]),
            ("Borrowings",                   [r"\bborrowings?\b"]),
            ("Other liabilities & provision",[r"other liabilit.*provision", r"other liabilities\b"]),
            ("Total liabilities",            [r"total (?:capital\s+(?:and|&)\s+)?liabilit", r"total liabilities\b"]),
            # Assets (main)
            ("Cash & balance with RBI",      [r"cash\s+(?:and|&)\s+balance.*rbi\b", r"cash.*reserve bank"]),
            ("Balances with banks and money at call", [r"balances?\s+with\s+banks?\s+and\s+money", r"balances?\s+with\s+banks?\s+and\s+money\s+at\s+call"]),
            ("Investments",                  [r"\binvestments?\b(?!\s*(?:in\s+associates|made|by))"]),
            ("Advances",                     [r"\badvances?\b(?!\s+(?:from|against))"]),
            ("Fixed assets",                 [r"fixed\s+assets?\b", r"property,?\s+plant\b"]),
            ("Other assets",                 [r"other\s+assets?\b"]),
            ("Total Assets",                 [r"total\s+assets?\b"]),
            # Schedule 6 — Cash and Balances with RBI
            ("Cash",                         [r"^cash\b(?!\s+(?:and|&|flow|credit))"]),
            ("Balances with RBI",            [r"balances?\s+with\s+(?:rbi|reserve bank)"]),
            ("- Current account",            [r"[-–]\s*current\s+account\b", r"current\s+account\b"]),
            ("- Other account",              [r"[-–]\s*other\s+account\b", r"other\s+account\b"]),
            # Schedule 7 — Balances with banks
            ("Balances with banks",          [r"balances?\s+with\s+banks?\b(?!\s+and\s+(?:rbi|reserve|money))"]),
            ("Money at call",                [r"money\s+at\s+call\b", r"call\s+(?:and\s+short\s+notice)?\s+money"]),
            # Schedule 8 — Investments
            ("Government securities",        [r"government\s+securities?\b", r"g.?sec\b"]),
            ("Other approved securities",    [r"other\s+approved\s+securities?\b"]),
            ("Shares",                       [r"^shares?\b"]),
            ("Debentures & bonds",           [r"debentures?\s*(?:and|&)\s*bonds?\b"]),
            # Schedule 9 — Advances
            ("Bills purchased & discounted", [r"bills?\s+purchased\s+(?:and|&)\s+discounted\b", r"bills?\s+purchased\b"]),
            ("CC, OD & loans repayable",     [r"(?:cc|cash\s+credit).*(?:od|overdraft).*loans?\s+repayable", r"cc,\s*od\b"]),
            ("Term loans",                   [r"\bterm\s+loans?\b"]),
            # NPA
            ("GNPA",                         [r"\bgnpa\b", r"gross\s+npa\b", r"gross\s+non.performing"]),
            ("NNPA",                         [r"\bnnpa\b", r"net\s+npa\b", r"net\s+non.performing"]),
            ("Less: Provisions",             [r"less\s*:\s*provisions?\b(?!.*tax)", r"less\s+provisions?\s+for\s+npa"]),
        ],

        "cf_fields": [
            ("Net Cash from Operating",      [r"net cash.*operating\b", r"cash.*operating activities?"]),
            ("Net Cash from Investing",      [r"net cash.*investing\b", r"cash.*investing activities?"]),
            ("Net Cash from Financing",      [r"net cash.*financing\b", r"cash.*financing activities?"]),
        ],

        "llm_hint": (
            "This is an Indian BANKING company (bank or small finance bank). "
            "Key P&L items: Interest earned, Interest expended, Net Interest Income, "
            "Other income, Operating expenses, Provisions & contingencies, PAT. "
            "Key BS items: Capital, Reserves & surplus, Deposits, Borrowings, "
            "Cash & balance with RBI, Balances with banks, Investments, Advances, "
            "Fixed assets, GNPA, NNPA. Extract ALL rows including schedule sub-items."
        ),
    },

    # ── NBFC (Non-Banking Financial Companies) ────────────────────────────────
    "nbfc": {
        "detection_keywords": [
            "non-banking financial", "nbfc", "rbi registration", "assets under management",
            "aum", "net interest margin", "gross npa", "loan book",
            "disbursements", "microfinance", "housing finance",
        ],
        "detection_threshold": 2,

        "pl_scoring": {
            "profit and loss": 3.0,
            "interest income": 3.0,
            "interest expense": 2.5,
            "net interest income": 2.5,
            "provisions and contingencies": 2.5,
            "profit before tax": 2.0,
            "profit after tax": 2.0,
        },
        "bs_scoring": {
            "balance sheet": 3.0,
            "total assets": 2.5,
            "loans and advances": 3.0,
            "borrowings": 2.5,
            "equity and liabilities": 2.0,
        },
        "cf_scoring": {
            "cash flow": 3.0,
            "cash flows from operating": 3.0,
            "cash flows from investing": 2.5,
            "cash flows from financing": 2.5,
        },

        "pl_fields": [
            ("Interest income",              [r"interest income\b", r"interest earned\b"]),
            ("Interest expense",             [r"interest expense\b", r"interest expended\b", r"finance costs?\b"]),
            ("Net Interest Income",          [r"net interest income\b"]),
            ("Fee & other income",           [r"fee.*income\b", r"other income\b", r"non.interest income\b"]),
            ("Total income",                 [r"total income\b", r"total revenue\b"]),
            ("Operating expenses",           [r"operating expenses?\b"]),
            ("Employee benefits expense",    [r"employee benefits? expense\b", r"staff costs?\b"]),
            ("Depreciation",                 [r"depreciation\b", r"amortis?ation\b"]),
            ("Provisions & write-offs",      [r"provisions?\b.*(?:write.off|contingencies?|impairment)", r"provision for (?:loan|npa|doubtful)"]),
            ("Profit before tax",            [r"profit before (?:income\s+)?tax\b"]),
            ("Tax",                          [r"(?:income\s+)?tax\s*(?:expense)?\b", r"provision\s+for\s+tax\b"]),
            ("Profit after tax",             [r"profit after (?:income\s+)?tax\b", r"profit for the (?:year|period)\b"]),
            ("Diluted EPS",                  [r"diluted\s+(?:eps|earnings per share)"]),
        ],

        "bs_fields": [
            ("Equity Share capital",         [r"(?:equity\s+)?share capital\b", r"paid.up.*capital\b"]),
            ("Other Equity",                 [r"other equity\b", r"reserves?\s+and\s+surplus\b"]),
            ("Total Equity",                 [r"total equity\b", r"shareholders.?\s*funds?\b"]),
            ("Borrowings",                   [r"\bborrowings?\b"]),
            ("Debt securities",              [r"debt\s+securities?\b"]),
            ("Deposits",                     [r"\bdeposits?\b(?!\s*(?:from|with|in))"]),
            ("Other financial liabilities",  [r"other financial liabilit\b"]),
            ("Total Liabilities",            [r"total liabilit\b"]),
            ("Cash & equivalents",           [r"cash and cash equivalents?\b", r"cash\s*&\s*cash equivalents?\b"]),
            ("Loans & advances",             [r"loans?\s+(?:and\s+)?advances?\b", r"loan\s+book\b"]),
            ("Investments",                  [r"\binvestments?\b(?!\s+in\s+associates)"]),
            ("Fixed assets",                 [r"property,?\s+plant.*equipment\b", r"fixed\s+assets?\b"]),
            ("Other assets",                 [r"other assets?\b"]),
            ("Total Assets",                 [r"total assets?\b"]),
            ("GNPA",                         [r"\bgnpa\b", r"gross\s+npa\b"]),
            ("NNPA",                         [r"\bnnpa\b", r"net\s+npa\b"]),
        ],

        "cf_fields": [
            ("Net Cash from Operating",      [r"net cash.*operating\b"]),
            ("Net Cash from Investing",      [r"net cash.*investing\b"]),
            ("Net Cash from Financing",      [r"net cash.*financing\b"]),
        ],

        "llm_hint": (
            "This is an Indian NBFC (Non-Banking Financial Company). "
            "Key P&L: Interest income, Interest expense, Net Interest Income, "
            "Fee income, Provisions, PAT. "
            "Key BS: Share capital, Reserves, Borrowings, Loans & advances, Investments, "
            "GNPA, NNPA. Extract ALL rows including sub-items."
        ),
    },

    # ── Insurance ─────────────────────────────────────────────────────────────
    "insurance": {
        "detection_keywords": [
            "insurance", "premium earned", "claims paid", "reinsurance",
            "solvency margin", "policyholder", "underwriting", "actuarial",
            "irda", "irdai", "life insurance", "general insurance",
        ],
        "detection_threshold": 2,

        "pl_scoring": {
            "profit and loss": 3.0,
            "premium earned": 3.0,
            "claims incurred": 3.0,
            "underwriting profit": 2.5,
            "profit before tax": 2.0,
            "profit after tax": 2.0,
        },
        "bs_scoring": {
            "balance sheet": 3.0,
            "total assets": 2.5,
            "policyholders": 2.5,
            "investments": 2.0,
        },
        "cf_scoring": {
            "cash flow": 3.0,
            "cash flows from operating": 3.0,
        },

        "pl_fields": [
            ("Gross Premium",                [r"gross\s+premium\b"]),
            ("Premium earned",               [r"premium earned\b", r"net\s+premium\b"]),
            ("Claims incurred",              [r"claims?\s+incurred\b", r"claims?\s+paid\b"]),
            ("Commission",                   [r"\bcommission\b"]),
            ("Operating expenses",           [r"operating expenses?\b"]),
            ("Underwriting profit",          [r"underwriting (?:profit|result)\b"]),
            ("Investment income",            [r"investment income\b", r"income from investments?\b"]),
            ("Profit before tax",            [r"profit before (?:income\s+)?tax\b"]),
            ("Tax",                          [r"(?:income\s+)?tax\s*(?:expense)?\b"]),
            ("Profit after tax",             [r"profit after (?:income\s+)?tax\b"]),
            ("Diluted EPS",                  [r"diluted\s+(?:eps|earnings per share)"]),
        ],

        "bs_fields": [
            ("Share capital",                [r"(?:equity\s+)?share capital\b"]),
            ("Reserves & surplus",           [r"reserves?\s+and\s+surplus\b", r"other equity\b"]),
            ("Total equity",                 [r"total equity\b", r"shareholders.?\s*funds?\b"]),
            ("Policyholder liabilities",     [r"policyholder\b.*liabilit", r"insurance\s+contract\b"]),
            ("Total liabilities",            [r"total liabilit\b"]),
            ("Investments",                  [r"\binvestments?\b"]),
            ("Cash & equivalents",           [r"cash and cash equivalents?\b"]),
            ("Total Assets",                 [r"total assets?\b"]),
        ],

        "cf_fields": [
            ("Net Cash from Operating",      [r"net cash.*operating\b"]),
            ("Net Cash from Investing",      [r"net cash.*investing\b"]),
            ("Net Cash from Financing",      [r"net cash.*financing\b"]),
        ],

        "llm_hint": (
            "This is an Indian INSURANCE company. "
            "Key items: Gross Premium, Claims incurred, Underwriting profit, "
            "Investment income, Policyholder liabilities, Solvency margin. "
            "Extract ALL rows including sub-schedules."
        ),
    },

    # ── IT / Software / Technology ────────────────────────────────────────────
    "it": {
        "detection_keywords": [
            "software services", "information technology", "it services",
            "digital services", "technology solutions", "bpo", "ites",
            "saas", "cloud services", "nasscom", "onsite offshore",
        ],
        "detection_threshold": 2,

        "pl_scoring": {
            "profit and loss": 3.0,
            "statement of profit": 3.0,
            "revenue from software": 3.0,
            "revenue from operations": 2.5,
            "employee benefits expense": 2.0,
            "profit before tax": 2.0,
            "profit after tax": 2.0,
        },
        "bs_scoring": {
            "balance sheet": 3.0,
            "total assets": 2.5,
            "equity and liabilities": 2.5,
            "goodwill": 2.0,
            "intangible assets": 2.0,
        },
        "cf_scoring": {
            "cash flow": 3.0,
            "cash flows from operating": 3.0,
            "cash flows from investing": 2.5,
            "cash flows from financing": 2.5,
        },

        "pl_fields": [
            ("Revenue from operations",      [r"revenue from (?:software|it\s+)?(?:services?|operations?)\b", r"\bnet\s+revenue\b", r"\bnet\s+sales?\b"]),
            ("Employee benefits expense",    [r"employee benefits? expense\b", r"staff costs?\b", r"personnel (?:expense|cost)"]),
            ("Subcontracting costs",         [r"subcontract(?:ing)?\s+costs?\b", r"third.party\s+costs?\b"]),
            ("Other expenses",               [r"other expenses?\b(?!\s*income)", r"operating expenses?\b"]),
            ("EBITDA",                       [r"\bebitda\b"]),
            ("Depreciation",                 [r"depreciation.*amortis?ation\b", r"\bdepreciation\b"]),
            ("EBIT",                         [r"\bebit\b"]),
            ("Finance costs",                [r"finance costs?\b", r"interest expense\b"]),
            ("Other income",                 [r"other income\b"]),
            ("Profit before tax",            [r"profit before (?:income\s+)?tax\b"]),
            ("Tax",                          [r"(?:income\s+)?tax\s*(?:expense)?\b"]),
            ("Profit after tax",             [r"profit after (?:income\s+)?tax\b", r"profit for the (?:year|period)\b"]),
            ("Diluted EPS",                  [r"diluted\s+(?:eps|earnings per share)"]),
        ],

        "bs_fields": [
            ("Property, Plant & Equipment",  [r"property,?\s+plant.*equipment\b"]),
            ("Goodwill",                     [r"\bgoodwill\b"]),
            ("Intangible assets",            [r"intangible assets?\b(?!\s+under)"]),
            ("Right of Use Assets",          [r"right.of.use assets?\b"]),
            ("Investments",                  [r"\binvestments?\b"]),
            ("Trade Receivables",            [r"trade receivables?\b", r"accounts\s+receivable\b", r"sundry\s+debtors?\b"]),
            ("Cash & equivalents",           [r"cash and cash equivalents?\b"]),
            ("Other current assets",         [r"other current assets?\b"]),
            ("Total Assets",                 [r"total assets?\b"]),
            ("Equity Share capital",         [r"(?:equity\s+)?share capital\b"]),
            ("Other Equity",                 [r"other equity\b", r"reserves?\s+and\s+surplus\b"]),
            ("Total Equity",                 [r"total equity\b"]),
            ("Borrowings",                   [r"\bborrowings?\b"]),
            ("Total Liabilities",            [r"total liabilit\b"]),
            ("Total Equity & Liabilities",   [r"total equity\s*(?:and|&)\s*liabilit\b"]),
        ],

        "cf_fields": [
            ("Net Cash from Operating",      [r"net cash.*operating\b"]),
            ("Net Cash from Investing",      [r"net cash.*investing\b"]),
            ("Net Cash from Financing",      [r"net cash.*financing\b"]),
        ],

        "llm_hint": (
            "This is an Indian IT / Software / Technology company. "
            "Key P&L: Revenue from software/IT services, Employee costs (usually 50-70% of revenue), "
            "Subcontracting, Depreciation, Finance costs, Other income, PAT, EPS. "
            "Key BS: Goodwill, Intangibles, Trade Receivables, Cash, Total Equity. "
            "Extract ALL rows."
        ),
    },

    # ── Generic fallback — works for any sector not explicitly defined ────────
    # Uses only universal financial keywords that appear in ANY Indian company's
    # annual report. Rule-based fields are None so LLM handles all extraction.
    "generic": {
        "detection_keywords": [],       # never detected — only used as fallback
        "detection_threshold": 999,

        "pl_scoring": {
            "profit and loss": 3.0,
            "income statement": 3.0,
            "profit before tax": 2.5,
            "profit after tax": 2.5,
            "total income": 2.0,
            "total expenses": 2.0,
            "earnings per share": 1.5,
        },
        "bs_scoring": {
            "balance sheet": 3.0,
            "total assets": 3.0,
            "total liabilities": 2.5,
            "shareholders": 2.0,
            "equity": 2.0,
            "net worth": 2.0,
        },
        "cf_scoring": {
            "cash flow": 3.0,
            "cash flows from operating": 3.0,
            "cash flows from investing": 2.5,
            "cash flows from financing": 2.5,
        },

        # No rule-based fields — LLM handles everything
        "pl_fields": None,
        "bs_fields": None,
        "cf_fields": None,

        "llm_hint": (
            "This is an Indian company. Extract ALL financial data from the statements. "
            "Use the exact label text from the source document — do not rename anything. "
            "Extract every row with a numeric value, including sub-items and schedules."
        ),
    },

    # ── Manufacturing / Pharma / FMCG / Industrial (default) ─────────────────
    "manufacturing": {
        "detection_keywords": [
            # Ind AS / common terms in Indian non-financial annual reports
            "cost of materials consumed", "raw material", "manufacturing expenses",
            "cost of goods sold", "inventories", "inventory", "work in progress",
            "work-in-progress", "purchases of stock-in-trade", "plant and machinery",
            "property, plant and equipment", "capital work-in-progress",
            "revenue from operations", "pharmaceutical", "pharma",
            "consumer products", "consumer goods", "fmcg", "industrial",
            "manufacturing", "chemicals", "cement", "steel", "textile",
        ],
        "detection_threshold": 2,

        "pl_scoring": {
            "profit and loss": 3.0,
            "statement of profit": 3.0,
            "revenue from operations": 2.5,
            "cost of materials consumed": 2.0,
            "employee benefits expense": 1.5,
            "finance costs": 1.5,
            "profit before tax": 2.0,
            "profit after tax": 2.0,
            "diluted eps": 1.5,
        },
        "bs_scoring": {
            "balance sheet": 3.0,
            "total assets": 2.5,
            "equity and liabilities": 2.5,
            "non-current assets": 2.0,
            "property, plant": 1.5,
            "trade receivables": 1.5,
            "total equity": 2.0,
        },
        "cf_scoring": {
            "cash flow": 3.0,
            "cash flows from operating": 3.0,
            "cash flows from investing": 2.5,
            "cash flows from financing": 2.5,
        },

        # P&L and BS fields reuse the existing rule_based_extractor definitions
        # (set to None so rule_based_extractor uses its own _PNL_FIELDS/_BS_FIELDS)
        "pl_fields": None,
        "bs_fields": None,
        "cf_fields": None,

        "llm_hint": (
            "This is an Indian manufacturing / pharma / FMCG / industrial company. "
            "Key P&L: Revenue from operations, Cost of goods sold (materials + purchases + inventory change), "
            "Employee benefits, Depreciation, Finance costs, Other income, Tax, PAT, EPS. "
            "Key BS: Property/Plant/Equipment, Inventories, Trade Receivables, Cash, "
            "Total Equity, Borrowings, Total Assets. Extract ALL rows."
        ),
    },
}

# ── Sector detection ──────────────────────────────────────────────────────────

def detect_sector(text: str) -> str:
    """
    Scan document text for sector signals.
    Returns a sector key from SECTORS.

    Priority order:
    1. Score all named sectors (banking, nbfc, it, insurance, manufacturing)
    2. Return the highest-scoring sector that meets its threshold
    3. If nothing matches → return 'generic' (full LLM, no hardcoded fields)
    """
    text_lower = text.lower()

    scores: dict[str, int] = {}
    for sector, cfg in SECTORS.items():
        if sector in ("generic",):
            continue   # never auto-detected, only used as fallback
        kws = cfg.get("detection_keywords", [])
        threshold = cfg.get("detection_threshold", 2)
        if not kws:
            continue
        hits = sum(1 for kw in kws if kw in text_lower)
        if hits >= threshold:
            scores[sector] = hits

    if not scores:
        return "generic"   # unknown sector → pure LLM extraction

    return max(scores, key=scores.get)


def get_section_weights(sector: str) -> dict[str, dict[str, float]]:
    """Return PDF page scoring weights for the detected sector."""
    cfg = SECTORS.get(sector, SECTORS["manufacturing"])
    return {
        "Annual P&L":           cfg.get("pl_scoring", {}),
        "Annual Balance Sheet": cfg.get("bs_scoring", {}),
        "Annual Cash Flow":     cfg.get("cf_scoring", {}),
    }


def get_llm_hint(sector: str) -> str:
    """Return the LLM context hint for the detected sector."""
    return SECTORS.get(sector, SECTORS["manufacturing"]).get("llm_hint", "")


def get_sector_fields(sector: str) -> dict[str, list | None]:
    """Return sector-specific rule-based field patterns."""
    cfg = SECTORS.get(sector, SECTORS["manufacturing"])
    return {
        "pl":  cfg.get("pl_fields"),
        "bs":  cfg.get("bs_fields"),
        "cf":  cfg.get("cf_fields"),
    }
