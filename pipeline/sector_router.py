"""
pipeline/sector_router.py
─────────────────────────
Auto-detect a company's sector from its NSE symbol or name, so the
orchestrator can pick the right knowledge-base overlay (banking / NBFC /
insurance / general) and the right validation rules.

Detection sources (in priority order)
─────────────────────────────────────
1. Built-in lookup table for top ~250 NSE listed companies (Nifty 500-ish).
2. Substring heuristics on the name/symbol  ("BANK" → bank, "FIN" → nbfc,
   "INSURANCE" / "LIFE" / "GENERAL" → insurance, etc.).
3. NSE classification API as a fallback (best-effort, cached).
4. Final fallback: "general" (uses base Ind AS taxonomy).

Public API
──────────
detect_sector(identifier_or_symbol_or_name) -> Sector
SECTORS : tuple of valid sector strings
load_sector_kb_overlay(sector) -> dict   # returns the JSON KB overlay
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# ── Canonical sector list ────────────────────────────────────────────────────
SECTORS: tuple[str, ...] = (
    "bank",          # Scheduled commercial banks, small finance banks, payments banks
    "nbfc",          # Non-Banking Financial Companies (HFCs, gold loan, MFIs)
    "insurance",     # Life and general insurance
    "amc",           # Asset management companies
    "broker",        # Stock-broking and exchange businesses
    "it",            # IT services and products
    "pharma",        # Pharma and biotech
    "fmcg",          # Fast-moving consumer goods
    "auto",          # Auto OEM, auto components, tyres
    "metals",        # Steel, aluminium, mining
    "oil_gas",       # Upstream / downstream / pipelines
    "power",         # Generation, T&D
    "realty",        # Real estate, REITs, construction
    "telecom",       # Wireless / wireline / towers
    "media",         # Print, broadcasting, OTT
    "chemicals",     # Specialty / commodity chemicals, fertilisers
    "cement",        # Cement and building products
    "capital_goods", # Engineering, capital equipment
    "consumer_dur",  # Consumer durables, white goods
    "retail",        # Apparel, food, e-commerce
    "logistics",     # Shipping, ports, trucking
    "general",       # Default — pure Ind AS, no sector overlay
)


@dataclass(frozen=True)
class SectorDetection:
    sector: str
    confidence: float            # 0.0 – 1.0
    method: str                  # "lookup" | "heuristic" | "nse_api" | "default"
    reason: str = ""

    def __str__(self) -> str:
        return f"{self.sector} (conf={self.confidence:.0%}, via {self.method})"


# ── Built-in lookup: NSE symbol → sector ─────────────────────────────────────
# Curated for accuracy on the most-traded names. Not exhaustive — heuristics
# fill the long tail.
_NSE_SYMBOL_SECTOR: dict[str, str] = {
    # ─ Banks ─
    "HDFCBANK": "bank", "ICICIBANK": "bank", "SBIN": "bank", "AXISBANK": "bank",
    "KOTAKBANK": "bank", "INDUSINDBK": "bank", "IDFCFIRSTB": "bank", "FEDERALBNK": "bank",
    "BANKBARODA": "bank", "PNB": "bank", "CANBK": "bank", "UNIONBANK": "bank",
    "BANKINDIA": "bank", "IOB": "bank", "CENTRALBK": "bank", "INDIANB": "bank",
    "UCOBANK": "bank", "MAHABANK": "bank", "PSB": "bank", "RBLBANK": "bank",
    "YESBANK": "bank", "IDBI": "bank", "SOUTHBANK": "bank", "KARURVYSYA": "bank",
    "CITYUNIONBNK": "bank", "DCBBANK": "bank", "CSBBANK": "bank", "TMB": "bank",
    "EQUITASBNK": "bank", "AUBANK": "bank", "UJJIVANSFB": "bank",
    "ESAFSFB": "bank", "FINOPB": "bank", "JANASFB": "bank", "SURYODAY": "bank",
    "UTKARSHBNK": "bank",
    # ─ NBFCs / HFCs ─
    "BAJFINANCE": "nbfc", "BAJAJFINSV": "nbfc", "CHOLAFIN": "nbfc",
    "M&MFIN": "nbfc", "MUTHOOTFIN": "nbfc", "MANAPPURAM": "nbfc",
    "SHRIRAMFIN": "nbfc", "LICHSGFIN": "nbfc", "PNBHOUSING": "nbfc",
    "CANFINHOME": "nbfc", "HUDCO": "nbfc", "RECLTD": "nbfc", "PFC": "nbfc",
    "IRFC": "nbfc", "POONAWALLA": "nbfc", "PEL": "nbfc", "SUNDARMFIN": "nbfc",
    "L&TFH": "nbfc", "IIFL": "nbfc", "IIFLFIN": "nbfc", "JIOFIN": "nbfc",
    "ABCAPITAL": "nbfc", "ABFRL": "retail", "EDELWEISS": "nbfc",
    "MOTILALOFS": "broker", "ANGELONE": "broker", "ICICIGI": "insurance",
    "HDFCAMC": "amc", "NAM-INDIA": "amc", "UTIAMC": "amc", "ABSLAMC": "amc",
    "BSE": "broker", "MCX": "broker", "CDSL": "broker", "CAMS": "broker",
    # ─ Insurance ─
    "SBILIFE": "insurance", "HDFCLIFE": "insurance", "ICICIPRULI": "insurance",
    "MAXFIN": "insurance", "STARHEALTH": "insurance", "GICRE": "insurance",
    "NIACL": "insurance", "LICI": "insurance", "LIFEINS": "insurance",
    # ─ IT ─
    "TCS": "it", "INFY": "it", "WIPRO": "it", "HCLTECH": "it",
    "TECHM": "it", "LTIM": "it", "MPHASIS": "it", "PERSISTENT": "it",
    "COFORGE": "it", "MINDTREE": "it", "OFSS": "it", "TATAELXSI": "it",
    "LTTS": "it", "BSOFT": "it", "KPITTECH": "it", "CYIENT": "it",
    "ZENSARTECH": "it", "ECLERX": "it", "INTELLECT": "it", "RAMCOSYS": "it",
    "FSL": "it", "NIITLTD": "it", "ROUTE": "it", "TANLA": "it",
    # ─ Pharma ─
    "SUNPHARMA": "pharma", "DRREDDY": "pharma", "CIPLA": "pharma",
    "DIVISLAB": "pharma", "LUPIN": "pharma", "AUROPHARMA": "pharma",
    "BIOCON": "pharma", "TORNTPHARM": "pharma", "ALKEM": "pharma",
    "ZYDUSLIFE": "pharma", "GLAND": "pharma", "GLENMARK": "pharma",
    "ABBOTINDIA": "pharma", "PFIZER": "pharma", "GSKPHARMA": "pharma",
    "SANOFI": "pharma", "IPCALAB": "pharma", "MANKIND": "pharma",
    "LAURUSLABS": "pharma", "AJANTPHARM": "pharma", "NATCOPHARM": "pharma",
    "ERIS": "pharma", "JBCHEPHARM": "pharma", "FORTIS": "pharma",
    "APOLLOHOSP": "pharma", "MAXHEALTH": "pharma", "NH": "pharma",
    "PIRPHARMA": "pharma", "PIRAMALPHARMA": "pharma", "PPLPHARMA": "pharma",
    "GRANULES": "pharma", "STAR": "pharma", "CAPLIPOINT": "pharma",
    # ─ FMCG ─
    "HINDUNILVR": "fmcg", "ITC": "fmcg", "NESTLEIND": "fmcg",
    "BRITANNIA": "fmcg", "DABUR": "fmcg", "MARICO": "fmcg",
    "GODREJCP": "fmcg", "COLPAL": "fmcg", "EMAMILTD": "fmcg",
    "TATACONSUM": "fmcg", "VBL": "fmcg", "RADICO": "fmcg",
    "UBL": "fmcg", "MCDOWELL-N": "fmcg", "BAJAJCON": "fmcg",
    "JYOTHYLAB": "fmcg", "GILLETTE": "fmcg", "PATANJALI": "fmcg",
    # ─ Auto ─
    "MARUTI": "auto", "TATAMOTORS": "auto", "M&M": "auto",
    "HEROMOTOCO": "auto", "BAJAJ-AUTO": "auto", "EICHERMOT": "auto",
    "TVSMOTOR": "auto", "ASHOKLEY": "auto", "BHARATFORG": "auto",
    "BOSCHLTD": "auto", "MOTHERSON": "auto", "MRF": "auto",
    "BALKRISIND": "auto", "APOLLOTYRE": "auto", "CEATLTD": "auto",
    "JKTYRE": "auto", "EXIDEIND": "auto", "AMARAJABAT": "auto",
    "ENDURANCE": "auto", "SUNDARMFAST": "auto", "SCHAEFFLER": "auto",
    "TIINDIA": "auto", "MINDAIND": "auto", "UNOMINDA": "auto",
    "OLECTRA": "auto",
    # ─ Metals ─
    "TATASTEEL": "metals", "JSWSTEEL": "metals", "HINDALCO": "metals",
    "VEDL": "metals", "JINDALSTEL": "metals", "SAIL": "metals",
    "NMDC": "metals", "COALINDIA": "metals", "HINDZINC": "metals",
    "NALCO": "metals", "MOIL": "metals", "RATNAMANI": "metals",
    "APL APOLLO": "metals", "WELCORP": "metals", "JINDALSAW": "metals",
    # ─ Oil & Gas ─
    "RELIANCE": "oil_gas", "ONGC": "oil_gas", "IOC": "oil_gas",
    "BPCL": "oil_gas", "HPCL": "oil_gas", "GAIL": "oil_gas",
    "OIL": "oil_gas", "PETRONET": "oil_gas", "IGL": "oil_gas",
    "MGL": "oil_gas", "GSPL": "oil_gas", "GUJGASLTD": "oil_gas",
    "MRPL": "oil_gas", "CASTROLIND": "oil_gas",
    # ─ Power ─
    "NTPC": "power", "POWERGRID": "power", "TATAPOWER": "power",
    "ADANIPOWER": "power", "JSWENERGY": "power", "ADANIGREEN": "power",
    "SUZLON": "power", "TORNTPOWER": "power", "NHPC": "power",
    "SJVN": "power", "RPOWER": "power", "CESC": "power", "INOXWIND": "power",
    # ─ Realty ─
    "DLF": "realty", "GODREJPROP": "realty", "OBEROIRLTY": "realty",
    "PRESTIGE": "realty", "BRIGADE": "realty", "PHOENIXLTD": "realty",
    "LODHA": "realty", "MACROTECH": "realty", "SOBHA": "realty",
    "MAHLIFE": "realty", "SUNTECK": "realty", "EMBASSY": "realty",
    "MINDSPACE": "realty", "BIRLA": "realty",
    # ─ Telecom / Media ─
    "BHARTIARTL": "telecom", "IDEA": "telecom", "INDUSTOWER": "telecom",
    "TATACOMM": "telecom", "RAILTEL": "telecom",
    "ZEEL": "media", "SUNTV": "media", "PVRINOX": "media", "TIPSINDLTD": "media",
    "SAREGAMA": "media", "DISHTV": "media", "TV18BRDCST": "media",
    # ─ Chemicals / Fertilisers ─
    "PIDILITIND": "chemicals", "SRF": "chemicals", "DEEPAKNTR": "chemicals",
    "AARTIIND": "chemicals", "ATUL": "chemicals", "TATACHEM": "chemicals",
    "GNFC": "chemicals", "CHAMBLFERT": "chemicals", "COROMANDEL": "chemicals",
    "UPL": "chemicals", "PIIND": "chemicals", "BAYERCROP": "chemicals",
    "SUMICHEM": "chemicals", "SOLARINDS": "chemicals", "NAVINFLUOR": "chemicals",
    "FINEORG": "chemicals", "VINATIORGA": "chemicals", "ALKYLAMINE": "chemicals",
    "GUJALKALI": "chemicals", "BASF": "chemicals", "GHCL": "chemicals",
    "TATACOMMCL": "chemicals",
    # ─ Cement ─
    "ULTRACEMCO": "cement", "SHREECEM": "cement", "ACC": "cement",
    "AMBUJACEM": "cement", "DALBHARAT": "cement", "RAMCOCEM": "cement",
    "JKCEMENT": "cement", "INDIACEM": "cement", "BIRLACORPN": "cement",
    "HEIDELBERG": "cement", "STARCEMENT": "cement", "PRISMCEM": "cement",
    "JKLAKSHMI": "cement",
    # ─ Capital goods / Engineering ─
    "LT": "capital_goods", "SIEMENS": "capital_goods", "ABB": "capital_goods",
    "BHEL": "capital_goods", "CUMMINSIND": "capital_goods", "HAVELLS": "capital_goods",
    "VOLTAS": "capital_goods", "BEL": "capital_goods", "HAL": "capital_goods",
    "BEML": "capital_goods", "THERMAX": "capital_goods", "GRINDWELL": "capital_goods",
    "AIAENG": "capital_goods", "SCHNEIDER": "capital_goods", "TIMKEN": "capital_goods",
    "SKFINDIA": "capital_goods", "KSB": "capital_goods", "ELGIEQUIP": "capital_goods",
    "IRCON": "capital_goods", "RVNL": "capital_goods", "RITES": "capital_goods",
    "MAZDOCK": "capital_goods", "GRSE": "capital_goods", "COCHINSHIP": "capital_goods",
    "DATAPATTNS": "capital_goods", "PRAJIND": "capital_goods", "AZAD": "capital_goods",
    # ─ Consumer durables ─
    "TITAN": "consumer_dur", "WHIRLPOOL": "consumer_dur", "BLUESTARCO": "consumer_dur",
    "DIXON": "consumer_dur", "AMBER": "consumer_dur", "VIPIND": "consumer_dur",
    "RELAXO": "consumer_dur", "BATAINDIA": "consumer_dur", "CROMPTON": "consumer_dur",
    "ORIENT": "consumer_dur", "RAJESHEXPO": "consumer_dur", "KAJARIACER": "consumer_dur",
    "CERA": "consumer_dur", "SOMANYCERA": "consumer_dur", "POLYCAB": "consumer_dur",
    # ─ Retail ─
    "DMART": "retail", "AVENUE": "retail", "TRENT": "retail",
    "ABFRL": "retail", "SHOPPERSTOP": "retail", "VMART": "retail",
    "ZOMATO": "retail", "NYKAA": "retail", "PAYTM": "retail",
    "FSNECOM": "retail", "DEVYANI": "retail", "JUBLFOOD": "retail",
    "WESTLIFE": "retail", "SAPPHIRE": "retail",
    # ─ Logistics / Transport ─
    "ADANIPORTS": "logistics", "CONCOR": "logistics", "GMRINFRA": "logistics",
    "INTERGLOBE": "logistics", "INDIGO": "logistics", "SPICEJET": "logistics",
    "GATI": "logistics", "BLUEDART": "logistics", "TCI": "logistics",
    "VRLLOG": "logistics", "MAHLOG": "logistics", "ALLCARGO": "logistics",
    "DELHIVERY": "logistics",
    # ─ Asset Mgmt / Brokers extra ─
    "ZERODHA": "broker", "NUVAMA": "broker", "JMFINANCIL": "nbfc",
    # ─ Conglomerates → "general" so multi-segment data is handled with base KB ─
    "HDFCAMC": "amc", "TATAINVEST": "general", "PIDILITIND": "chemicals",
}


# ── Heuristic keyword rules (applied in order, first match wins) ─────────────
_HEURISTIC_RULES: list[tuple[re.Pattern, str]] = [
    # Banks
    (re.compile(r"\bBANK\b|BNK$|BNKLTD$|SFB|FINANCE BANK|PAYMENTSBANK", re.I), "bank"),
    # Insurance
    (re.compile(r"INSURANCE|INSUR\b|\bLIFE\b|LIFEINS|GENINS|GIC\b|REINSUR", re.I), "insurance"),
    # AMC
    (re.compile(r"\bAMC\b|MUTUAL\s*FUND|ASSET\s*MANAGE", re.I), "amc"),
    # Broker / Exchange
    (re.compile(r"BROKING|SECURITIES|EXCHANGE|DEPOSITORY|CDSL|NSDL", re.I), "broker"),
    # NBFC / HFC / Finance
    (re.compile(r"FINSERV|HOUSING\s*FIN|FINCORP|HSGFIN|HOMEFIN|CREDIT|CAPITAL|MICROFIN|GOLDFIN|FINANCE\b|FIN\s*LTD\b|\bFIN$|NBFC", re.I), "nbfc"),
    # IT
    (re.compile(r"\bINFOTECH|TECHNOLOG|SYSTEMS|SOFTWARE|TECH\b|INFOSYS|TCS|WIPRO|HCL", re.I), "it"),
    # Pharma
    (re.compile(r"PHARMA|DRUG|BIOTEC|HOSPITAL|HEALTHCARE|LAB(?:S|ORATORIES)|MEDIC", re.I), "pharma"),
    # FMCG
    (re.compile(r"FOODS|BEVERAGE|CONSUMER\s*PROD|HOUSEHOLD|PERSONAL\s*CARE", re.I), "fmcg"),
    # Auto
    (re.compile(r"MOTORS|AUTOMOB|AUTO\s*PARTS|TYRE|TYRES|CYCLES|VEHICLE", re.I), "auto"),
    # Metals
    (re.compile(r"STEEL|ALUMIN|COPPER|ZINC|METAL|MINING|MINERAL|IRON\s*ORE", re.I), "metals"),
    # Oil & Gas
    (re.compile(r"PETROLEUM|REFINER|OIL\b|GAS\b|LNG|HYDROCARB", re.I), "oil_gas"),
    # Power
    (re.compile(r"POWER|ENERGY|ELECTRIC|RENEW|GRID|HYDEL", re.I), "power"),
    # Realty
    (re.compile(r"REALTY|REAL\s*EST|DEVELOPERS|PROPERT|REIT|INFRA(?:STRUCT)?", re.I), "realty"),
    # Telecom
    (re.compile(r"TELECOM|TOWERS|COMMUNICAT|WIRELESS|TELESERV", re.I), "telecom"),
    # Media
    (re.compile(r"MEDIA|BROADCAST|ENTERTAIN|TELEVIS|MUSIC", re.I), "media"),
    # Chemicals / Fertilisers
    (re.compile(r"CHEM|FERTI|PESTICID|AGROCHEM|POLYMER|RESIN|DYE", re.I), "chemicals"),
    # Cement
    (re.compile(r"CEMENT|ASBESTOS", re.I), "cement"),
    # Capital goods
    (re.compile(r"ENGINEER|ELECTRIC|MACHINE\s*TOOL|FORGING|SHIPYARD|DEFENCE", re.I), "capital_goods"),
    # Logistics
    (re.compile(r"LOGISTIC|SHIPPING|PORT|AIRWAY|AIRLINE|FREIGHT|COURIER|TRANSPORT", re.I), "logistics"),
]


# ── Public API ───────────────────────────────────────────────────────────────

def detect_sector(
    nse_symbol: Optional[str] = None,
    name: Optional[str] = None,
    explicit_sector: Optional[str] = None,
) -> SectorDetection:
    """
    Determine the company's sector for KB-overlay routing.

    Parameters
    ----------
    nse_symbol      : preferred — NSE ticker symbol
    name            : free-text company name (used if symbol not in lookup)
    explicit_sector : if user passes one explicitly, trust it

    Returns
    -------
    SectorDetection with sector, confidence, method, reason
    """
    # 0. Explicit override
    if explicit_sector:
        s = explicit_sector.lower().strip()
        if s in SECTORS:
            return SectorDetection(s, 1.0, "explicit", "user-supplied")

    # 1. Built-in symbol lookup
    if nse_symbol:
        sym = nse_symbol.upper().strip()
        if sym in _NSE_SYMBOL_SECTOR:
            return SectorDetection(
                _NSE_SYMBOL_SECTOR[sym], 1.0, "lookup",
                f"matched {sym} in built-in NSE table",
            )

    # 2. Heuristic on symbol + name
    haystack = " ".join(filter(None, [nse_symbol or "", name or ""])).upper()
    if haystack:
        for pat, sector in _HEURISTIC_RULES:
            m = pat.search(haystack)
            if m:
                return SectorDetection(
                    sector, 0.75, "heuristic",
                    f"keyword '{m.group(0)}' → {sector}",
                )

    # 3. Default
    return SectorDetection("general", 0.5, "default", "no match — using base Ind AS taxonomy")


def is_financial_services(sector: str) -> bool:
    """True for banks, NBFCs, insurance, AMC, brokers — they share
    a fundamentally different statement structure (no COGS, no working capital
    in the usual sense). Used by the orchestrator to switch validation rules."""
    return sector in ("bank", "nbfc", "insurance", "amc", "broker")


def overlay_path(sector: str) -> Optional[Path]:
    """Return the path to the sector-specific KB overlay JSON, if one exists."""
    base = Path(__file__).parent.parent / "web"
    candidates = {
        "bank":      base / "financial_kb_banks.json",
        "nbfc":      base / "financial_kb_nbfc.json",
        "insurance": base / "financial_kb_insurance.json",
    }
    p = candidates.get(sector)
    return p if p and p.exists() else None


def load_sector_kb_overlay(sector: str) -> dict:
    """Load the sector-specific KB overlay (returns {} if none)."""
    p = overlay_path(sector)
    if not p:
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to load sector overlay '{p}': {exc}")
        return {}


__all__ = [
    "SECTORS",
    "SectorDetection",
    "detect_sector",
    "is_financial_services",
    "load_sector_kb_overlay",
    "overlay_path",
]
