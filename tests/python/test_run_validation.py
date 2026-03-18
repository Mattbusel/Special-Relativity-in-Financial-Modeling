"""Tests for validation/run_validation.py orchestration helpers."""
from __future__ import annotations
import sys, os, subprocess, tempfile, json
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pytest
from validation.run_validation import (
    run_regime_validator,
    run_backtest_runner,
    BINARY_TIMEOUT_SECS,
)

def test_regime_validator_missing_binary_returns_false(tmp_path):
    _, ok, msg = run_regime_validator(
        binary_path="does-not-exist-xyz",
        ticker="AAPL",
        csv_path=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path),
        timeout=5,
    )
    assert not ok
    assert "not found" in msg.lower() or "does-not-exist" in msg.lower()

def test_backtest_runner_missing_binary_returns_false(tmp_path):
    _, ok, msg = run_backtest_runner(
        binary_path="does-not-exist-xyz",
        ticker="AAPL",
        regime_csv=str(tmp_path / "regime.csv"),
        output_dir=str(tmp_path),
        timeout=5,
    )
    assert not ok

def test_regime_validator_binary_exit_nonzero_returns_false(tmp_path):
    # Use 'false' (always exits 1) or 'cmd /c exit 1' on Windows
    import platform
    if platform.system() == "Windows":
        binary = "cmd"
    else:
        binary = "false"
    _, ok, _ = run_regime_validator(
        binary_path=binary,
        ticker="TEST",
        csv_path=str(tmp_path / "x.csv"),
        output_dir=str(tmp_path),
        timeout=5,
    )
    assert not ok

def test_binary_timeout_secs_is_positive():
    assert BINARY_TIMEOUT_SECS > 0
