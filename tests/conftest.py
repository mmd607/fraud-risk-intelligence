"""Shared pytest fixtures."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

import pytest

from fraud_risk.config.settings import ENV_PREFIX, get_settings


@pytest.fixture(autouse=True)
def _isolated_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[None]:
    """Isolate every test from the developer's environment variables and `.env` file."""
    for key in list(os.environ):
        if key.startswith(ENV_PREFIX):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_package_logger() -> Iterator[None]:
    """Drop handlers bound to per-test streams (e.g. captured stderr) after each test."""
    yield
    logger = logging.getLogger("fraud_risk")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.propagate = True
