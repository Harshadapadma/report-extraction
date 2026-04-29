"""
Company Configuration Loader
-----------------------------
Loads per-company YAML files from config/companies/ and exposes a simple
get/set API so the rest of the codebase never imports YAML directly.

Usage
-----
    from config.company_loader import set_active_company, get_active_config

    # At app startup (or when the user picks a company in the UI):
    set_active_company("piramal")

    # In pdf_parser / rule_based_extractor / llm_extractor:
    cfg = get_active_config()
    scorer_keywords = cfg.operating_metrics_scorer   # dict[str, float]
    fields          = cfg.operating_fields           # list[OperatingField]
    llm_names       = cfg.llm_operating_field_names  # list[str]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pyyaml"])
    import yaml  # type: ignore

_COMPANIES_DIR = Path(__file__).parent / "companies"


@dataclass
class OperatingField:
    """One row in the Operating Metrics extraction table."""
    name: str
    patterns: list[re.Pattern]


@dataclass
class CompanyConfig:
    """Parsed, ready-to-use company configuration."""
    key: str                                          # e.g. "piramal"
    name: str                                         # e.g. "Piramal Pharma Ltd"
    display_name: str                                 # shown in UI
    operating_metrics_scorer: dict[str, float]        # keyword → weight
    operating_fields: list[OperatingField]            # compiled patterns
    llm_operating_field_names: list[str]              # for LLM prompt injection


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _parse_config(key: str, raw: dict) -> CompanyConfig:
    scorer = {str(k): float(v) for k, v in (raw.get("operating_metrics_scorer") or {}).items()}

    fields: list[OperatingField] = []
    for entry in raw.get("operating_fields") or []:
        compiled = [re.compile(p, re.IGNORECASE) for p in (entry.get("patterns") or [])]
        fields.append(OperatingField(name=entry["name"], patterns=compiled))

    return CompanyConfig(
        key=key,
        name=raw.get("name", key),
        display_name=raw.get("display_name", key),
        operating_metrics_scorer=scorer,
        operating_fields=fields,
        llm_operating_field_names=list(raw.get("llm_operating_field_names") or []),
    )


# ── Registry ──────────────────────────────────────────────────────────────────

_registry: dict[str, CompanyConfig] = {}
_active_key: str = "piramal"   # default


def _ensure_loaded(key: str) -> CompanyConfig:
    if key in _registry:
        return _registry[key]
    path = _COMPANIES_DIR / f"{key}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No company config found for '{key}'. "
            f"Expected: {path}\n"
            f"Available: {list_available_companies()}"
        )
    raw = _load_yaml(path)
    cfg = _parse_config(key, raw)
    _registry[key] = cfg
    return cfg


def list_available_companies() -> list[str]:
    """Return the keys of all .yaml files in config/companies/."""
    return sorted(p.stem for p in _COMPANIES_DIR.glob("*.yaml"))


def list_company_display_names() -> dict[str, str]:
    """Return {key: display_name} for all available companies."""
    result = {}
    for key in list_available_companies():
        try:
            cfg = _ensure_loaded(key)
            result[key] = cfg.display_name
        except Exception:
            result[key] = key
    return result


def set_active_company(key: str) -> CompanyConfig:
    """Switch the global active company. Returns the loaded config."""
    global _active_key
    cfg = _ensure_loaded(key)
    _active_key = key
    return cfg


def get_active_config() -> CompanyConfig:
    """Return the currently active CompanyConfig (default: piramal)."""
    return _ensure_loaded(_active_key)


def get_active_key() -> str:
    return _active_key
