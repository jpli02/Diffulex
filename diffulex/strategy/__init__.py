"""Diffulex strategy package that imports built-in strategies to trigger registration."""
from __future__ import annotations

# Import built-in strategies so their registrations run at import time.
from . import d2f  # noqa: F401

__all__ = ["d2f"]

DECODING_STRATEGY = None

def fetch_decoding_strategy() -> str | None:
    return DECODING_STRATEGY

def set_decoding_strategy(strategy: str) -> None:
    global DECODING_STRATEGY
    DECODING_STRATEGY = strategy

def reset_decoding_strategy() -> None:
    global DECODING_STRATEGY
    DECODING_STRATEGY = None