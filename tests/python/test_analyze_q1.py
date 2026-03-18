"""Tests for validation/analyze_q1.py"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import pandas as pd
import pytest
from validation.analyze_q1 import (
    compute_cohens_d, compute_variance_ratio,
    run_bartlett_test, run_levene_test, _split_regimes,
    TIMELIKE_LABEL, SPACELIKE_LABEL, INTERVAL_COL, RETURN_COL, TICKER_COL,
)

def make_df(n_tl, n_sl, var_tl=1.0, var_sl=2.0, ticker="A", seed=42):
    rng = np.random.default_rng(seed)
    tl = rng.normal(0.0, np.sqrt(var_tl), size=n_tl)
    sl = rng.normal(0.0, np.sqrt(var_sl), size=n_sl)
    return pd.DataFrame({
        TICKER_COL: [ticker]*(n_tl+n_sl),
        INTERVAL_COL: [TIMELIKE_LABEL]*n_tl + [SPACELIKE_LABEL]*n_sl,
        RETURN_COL: np.concatenate([tl, sl]),
    })

def test_split_regimes_sizes():
    tl, sl = _split_regimes(make_df(100, 200))
    assert len(tl)==100 and len(sl)==200

def test_split_regimes_ticker_filter():
    df = pd.concat([make_df(100,50,ticker="A"),make_df(80,40,ticker="B")],ignore_index=True)
    tl, sl = _split_regimes(df, ticker="A")
    assert len(tl)==100 and len(sl)==50

def test_variance_ratio_sl_higher():
    assert compute_variance_ratio(make_df(500,500,var_tl=1.0,var_sl=4.0,seed=0)) > 1.0

def test_variance_ratio_insufficient_is_nan():
    assert np.isnan(compute_variance_ratio(make_df(1,1)))

def test_variance_ratio_known_approx():
    rng = np.random.default_rng(5)
    tl = rng.normal(0,1.0,5000); sl = rng.normal(0,2.0,5000)
    df = pd.DataFrame({TICKER_COL:["X"]*10000,
                       INTERVAL_COL:[TIMELIKE_LABEL]*5000+[SPACELIKE_LABEL]*5000,
                       RETURN_COL:np.concatenate([tl,sl])})
    assert pytest.approx(compute_variance_ratio(df), rel=0.1) == 4.0

def test_cohens_d_positive_when_sl_higher_mean():
    rng = np.random.default_rng(10)
    tl = rng.normal(0,1,500); sl = rng.normal(1,1,500)
    df = pd.DataFrame({TICKER_COL:["X"]*1000,
                       INTERVAL_COL:[TIMELIKE_LABEL]*500+[SPACELIKE_LABEL]*500,
                       RETURN_COL:np.concatenate([tl,sl])})
    assert compute_cohens_d(df) > 0.5

def test_cohens_d_nan_on_insufficient():
    assert np.isnan(compute_cohens_d(make_df(1,1)))

def test_bartlett_small_p_different_variances():
    result = run_bartlett_test(make_df(500,500,var_tl=0.01,var_sl=4.0))
    assert result["p_value"] < 0.001

def test_bartlett_has_keys():
    result = run_bartlett_test(make_df(100,100))
    assert "statistic" in result and "p_value" in result

def test_bartlett_nan_on_insufficient():
    assert np.isnan(run_bartlett_test(make_df(1,1))["p_value"])

def test_bartlett_1pt27x_variance_significant_large_n():
    result = run_bartlett_test(make_df(5000,5000,var_tl=1.0,var_sl=1.27,seed=2025))
    assert result["p_value"] < 0.05

def test_levene_counts_match():
    result = run_levene_test(make_df(150,300,ticker="B"), ticker="B")
    assert result["n_timelike"]==150 and result["n_spacelike"]==300

def test_levene_different_variances_significant():
    assert run_levene_test(make_df(500,500,var_tl=0.01,var_sl=4.0))["p_value"] < 0.001
