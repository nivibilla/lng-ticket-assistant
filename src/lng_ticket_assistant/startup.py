"""Timed startup checks with safe diagnostics for the local server."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

LOGGER = logging.getLogger(__name__)


class StartupError(RuntimeError):
    """A required startup check failed; the server must not accept chats."""


@contextmanager
def startup_check(stage: str) -> Iterator[None]:
    """Report progress without logging provider errors or their credentials."""
    started = perf_counter()
    LOGGER.info("Startup: %s…", stage)
    try:
        yield
    except StartupError:
        raise  # A nested check already identified and logged its failed stage.
    except Exception as exc:  # noqa: BLE001 - sanitize failures at the startup boundary
        message = f"Startup check failed: {stage} ({type(exc).__name__})."
        LOGGER.error("%s Elapsed: %.1fs", message, perf_counter() - started)
        # Uvicorn prints startup exceptions. Suppress the original exception,
        # which can contain provider request URLs, keys, or local file paths.
        raise StartupError(message) from None
    LOGGER.info("Startup: %s passed (%.1fs)", stage, perf_counter() - started)
