"""Safe LangSmith tracing bootstrap for LangChain/LangGraph apps."""

from __future__ import annotations

import os
import re

import requests

from .logs import get_logger

logger = get_logger(__name__)

_LANGSMITH_DEFAULT_ENDPOINT = "https://api.smith.langchain.com"


def configure_langsmith() -> None:
    """Normalize and guard LangSmith env vars.

    This keeps LangChain tracing enabled only when configuration looks valid,
    avoiding repeated 403/500 spam during runtime.
    """
    tracing_requested = (
        _is_truthy(os.getenv("LANGSMITH_TRACING"))
        or _is_truthy(os.getenv("LANGCHAIN_TRACING_V2"))
        or _is_truthy(os.getenv("LANGSMITH_TRACING_V2"))
    )

    # Safe default: no tracing unless explicitly requested.
    if not tracing_requested:
        _force_disable_tracing_flags()
        return

    api_key = (os.getenv("LANGSMITH_API_KEY") or "").strip()
    openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()

    if not api_key:
        _disable_tracing("LANGSMITH_API_KEY is missing")
        return

    if openrouter_key and api_key == openrouter_key:
        _disable_tracing(
            "LANGSMITH_API_KEY matches OPENROUTER_API_KEY. "
            "Use a dedicated LangSmith key."
        )
        return

    if not _looks_like_langsmith_key(api_key):
        _disable_tracing(
            "LANGSMITH_API_KEY format looks invalid. "
            "Expected prefixes like 'lsv2_pt_' or 'ls__'."
        )
        return

    endpoint = (os.getenv("LANGSMITH_ENDPOINT") or _LANGSMITH_DEFAULT_ENDPOINT).strip().rstrip("/")
    os.environ["LANGSMITH_ENDPOINT"] = endpoint

    if not _preflight_langsmith(api_key=api_key, endpoint=endpoint):
        _disable_tracing("LangSmith preflight failed (auth/endpoint/service issue)")
        return

    # Keep both modern and legacy env aliases consistent.
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    # Valid values include: hybrid, langsmith, otel
    os.environ["LANGSMITH_TRACING_MODE"] = "langsmith"

    logger.info("LangSmith tracing enabled (endpoint=%s)", endpoint)


def _disable_tracing(reason: str) -> None:
    _force_disable_tracing_flags()
    logger.warning("LangSmith tracing disabled: %s", reason)


def _force_disable_tracing_flags() -> None:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING_V2"] = "false"
    # Remove mode entirely when tracing is disabled to avoid tracer init warnings.
    os.environ.pop("LANGSMITH_TRACING_MODE", None)


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _looks_like_langsmith_key(value: str) -> bool:
    # Accept modern keys like lsv2_pt_<id>_<suffix> and legacy ls__* keys.
    return bool(re.match(r"^(lsv2_[a-z]{2}_[A-Za-z0-9_\-]+|ls__[A-Za-z0-9_\-]+)$", value))


def _preflight_langsmith(api_key: str, endpoint: str) -> bool:
    """Best-effort credential/endpoint check to prevent runtime tracing spam."""
    headers = {"x-api-key": api_key}
    probes = [
        f"{endpoint}/info",
        f"{endpoint}/sessions?limit=1",
        f"{endpoint}/datasets?limit=1",
    ]

    for url in probes:
        try:
            resp = requests.get(url, headers=headers, timeout=6)
            if resp.status_code < 400:
                return True
            if resp.status_code in {401, 403}:
                logger.warning("LangSmith preflight auth failed at %s (HTTP %s)", url, resp.status_code)
                return False
        except requests.RequestException:
            continue

    logger.warning("LangSmith preflight failed for all probe endpoints")
    return False
