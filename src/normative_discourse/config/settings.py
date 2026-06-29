"""Application settings with pydantic-settings.

Priority: CLI args > environment variables (.env) > default_settings.toml
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[import]
    except ImportError:
        import tomli as tomllib  # type: ignore[import,no-redef]

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_SETTINGS_TOML = CONFIG_DIR / "default_settings.toml"


def load_toml_config(path: Path | None = None) -> dict[str, Any]:
    """Load and return the default_settings.toml as a dict."""
    toml_path = path or DEFAULT_SETTINGS_TOML
    if not toml_path.exists():
        return {}
    with open(toml_path, "rb") as f:
        return tomllib.load(f)


@lru_cache(maxsize=1)
def _cached_toml() -> dict[str, Any]:
    return load_toml_config()


class Settings(BaseSettings):
    """Project-wide settings loaded from .env and environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- PubMed / NCBI ---
    pubmed_email: str = Field(default="", description="Email for NCBI E-utilities")
    pubmed_api_key: Optional[str] = Field(
        default=None, description="NCBI API key (optional, increases rate limit)"
    )

    # --- Adobe PDF Extract API ---
    adobe_client_id: Optional[str] = Field(default=None)
    adobe_client_secret: Optional[str] = Field(default=None)
    # Additional credential sets for quota rotation (legacy hardcoded fields)
    adobe_client_id_2: Optional[str] = Field(default=None)
    adobe_client_secret_2: Optional[str] = Field(default=None)
    adobe_client_id_3: Optional[str] = Field(default=None)
    adobe_client_secret_3: Optional[str] = Field(default=None)
    # Fields 4+ are loaded dynamically by get_adobe_credentials_list()

    # --- Unpaywall ---
    unpaywall_email: str = Field(default="", description="Email for Unpaywall API")

    # --- Elsevier Full-Text API (Phase 2b, Tier 2-5: Lancet) ---
    elsevier_api_key: Optional[str] = Field(
        default=None, description="Elsevier API key for Full-Text Retrieval API"
    )
    elsevier_inst_token: Optional[str] = Field(
        default=None, description="Elsevier institutional token"
    )

    # --- Wiley TDM (Phase 2b, Tier 2-6: Med Educ) ---
    wiley_tdm_token: Optional[str] = Field(
        default=None, description="Wiley TDM token (via Crossref click-through)"
    )

    # --- Semantic Scholar (Tier 1 OA) ---
    semantic_scholar_api_key: Optional[str] = Field(
        default=None, description="Semantic Scholar API key (optional, increases rate limit)"
    )

    # --- CORE (Tier 1 OA) ---
    core_api_key: Optional[str] = Field(
        default=None, description="CORE API key for repository search"
    )

    # --- Gemini (abstract screening) ---
    gemini_api_key: Optional[str] = Field(
        default=None, description="Gemini API key for abstract screening"
    )

    # --- General ---
    log_level: str = Field(default="INFO")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        v = v.upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in valid:
            raise ValueError(f"log_level must be one of {valid}, got '{v}'")
        return v

    def validate_required(self, phase: str = "all") -> list[str]:
        """Check that required env vars are set for a given phase.

        Returns a list of error messages (empty if all OK).
        """
        errors: list[str] = []

        if phase in ("all", "pubmed", "phase1"):
            if not self.pubmed_email:
                errors.append(
                    "PUBMED_EMAIL is not set. "
                    "Required for PubMed queries (Phase 1)."
                )

        if phase in ("all", "pdf", "phase2"):
            if not self.unpaywall_email:
                errors.append(
                    "UNPAYWALL_EMAIL is not set. "
                    "Required for PDF retrieval via Unpaywall (Phase 2)."
                )

        if phase in ("all", "adobe", "phase3"):
            if not self.adobe_client_id:
                errors.append(
                    "ADOBE_CLIENT_ID is not set. "
                    "Required for Adobe PDF Extract API (Phase 3)."
                )
            if not self.adobe_client_secret:
                errors.append(
                    "ADOBE_CLIENT_SECRET is not set. "
                    "Required for Adobe PDF Extract API (Phase 3)."
                )

        if phase in ("all", "screening"):
            if not self.gemini_api_key:
                errors.append(
                    "GEMINI_API_KEY is not set. "
                    "Required for abstract screening."
                )

        if phase in ("phase2b_elsevier", "phase2b_all"):
            if not self.elsevier_api_key:
                errors.append(
                    "ELSEVIER_API_KEY is not set. "
                    "Required for Elsevier Full-Text API (Tier 2-5: Lancet)."
                )
            if not self.elsevier_inst_token:
                errors.append(
                    "ELSEVIER_INST_TOKEN is not set. "
                    "Required for Elsevier Full-Text API (Tier 2-5: Lancet)."
                )

        return errors

    def get_adobe_credentials_list(self) -> list[tuple[str, str]]:
        """Return all configured Adobe credential pairs.

        Scans .env and environment variables for ADOBE_CLIENT_ID /
        ADOBE_CLIENT_SECRET (primary) and ADOBE_CLIENT_ID_N /
        ADOBE_CLIENT_SECRET_N (N=2,3,4,...) dynamically, so any number
        of credential sets can be added without code changes.
        """
        import os

        from dotenv import dotenv_values

        env_file = dotenv_values(PROJECT_ROOT / ".env")

        def _get(key: str) -> str | None:
            """Environment variable > .env file."""
            return os.environ.get(key) or env_file.get(key)

        pairs: list[tuple[str, str]] = []

        # Primary pair
        cid = _get("ADOBE_CLIENT_ID")
        csec = _get("ADOBE_CLIENT_SECRET")
        if cid and csec:
            pairs.append((cid, csec))

        # Numbered pairs: _2, _3, _4, ...
        n = 2
        while True:
            cid = _get(f"ADOBE_CLIENT_ID_{n}")
            csec = _get(f"ADOBE_CLIENT_SECRET_{n}")
            if not cid or not csec:
                break
            pairs.append((cid, csec))
            n += 1

        return pairs


def get_settings() -> Settings:
    """Load and return the application settings."""
    return Settings()


def get_pubmed_config() -> dict[str, Any]:
    """Return the [pubmed] section from default_settings.toml."""
    return _cached_toml().get("pubmed", {})


def get_paths_config() -> dict[str, str]:
    """Return the [paths] section from default_settings.toml."""
    return _cached_toml().get("paths", {})


def get_pdf_download_config() -> dict[str, Any]:
    """Return the [pdf_download] section from default_settings.toml."""
    return _cached_toml().get("pdf_download", {})


def get_browser_use_config() -> dict[str, Any]:
    """Return the [browser_use] section from default_settings.toml."""
    return _cached_toml().get("browser_use", {})


def get_adobe_config() -> dict[str, Any]:
    """Return the [adobe_api] section from default_settings.toml."""
    return _cached_toml().get("adobe_api", {})


def get_remediation_config() -> dict[str, Any]:
    """Return the [remediation] section from default_settings.toml."""
    return _cached_toml().get("remediation", {})


def get_detection_config() -> dict[str, Any]:
    """Return the [detection] section from default_settings.toml."""
    return _cached_toml().get("detection", {})


def get_statistics_config() -> dict[str, Any]:
    """Return the [statistics] section from default_settings.toml."""
    return _cached_toml().get("statistics", {})


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure console + file dual-output logging (kimura-download pattern)."""
    handlers: list[logging.Handler] = []

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    handlers.append(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers, force=True)
