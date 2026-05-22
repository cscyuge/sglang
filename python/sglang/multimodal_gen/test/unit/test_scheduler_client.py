# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.scheduler_client import (
    _DEFAULT_GENERATION_TIMEOUT_MS,
    _GENERATION_TIMEOUT_ENV,
    _generation_recv_timeout_ms,
)


def test_generation_recv_timeout_defaults_to_long_generation_window(monkeypatch):
    monkeypatch.delenv(_GENERATION_TIMEOUT_ENV, raising=False)

    assert _generation_recv_timeout_ms() == _DEFAULT_GENERATION_TIMEOUT_MS


def test_generation_recv_timeout_can_be_overridden(monkeypatch):
    monkeypatch.setenv(_GENERATION_TIMEOUT_ENV, "12345")

    assert _generation_recv_timeout_ms() == 12345


def test_generation_recv_timeout_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv(_GENERATION_TIMEOUT_ENV, "invalid")

    assert _generation_recv_timeout_ms() == _DEFAULT_GENERATION_TIMEOUT_MS

    monkeypatch.setenv(_GENERATION_TIMEOUT_ENV, "-2")

    assert _generation_recv_timeout_ms() == _DEFAULT_GENERATION_TIMEOUT_MS


def test_generation_recv_timeout_allows_infinite_timeout(monkeypatch):
    monkeypatch.setenv(_GENERATION_TIMEOUT_ENV, "-1")

    assert _generation_recv_timeout_ms() == -1
