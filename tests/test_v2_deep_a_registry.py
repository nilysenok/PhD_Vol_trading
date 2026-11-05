"""A15: реестр trials и контрольная точка — волна A не тратит trials."""
import json

import pandas as pd
import pytest

from models_v2.s5 import common, registry


LIMITS = {"objectives_target": 35, "feature_engineering": 45, "har_boosting": 50, "rolling_recency": 35,
          "regime_aware": 50, "ensemble_calibration": 25}


def counts_of(df):
    return {k: int((df["направление"] == k).sum()) for k in LIMITS}


def check_registry(df, ck):
    """Инварианты: шапка, уникальные trial_id, лимиты п. 26, счётчики CHECKPOINT.json = число строк реестра по направлениям."""
    assert list(df.columns) == list(registry.TRIAL_COLS) and df["trial_id"].is_unique
    assert set(df["направление"]) <= set(LIMITS)
    assert ck["trials_limit"] == LIMITS and ck["trials_limit_total"] == 240
    assert ck["trials_used"] == counts_of(df) and sum(ck["trials_used"].values()) == len(df) <= 240
    assert all(ck["trials_used"][k] <= LIMITS[k] for k in LIMITS)


def test_realestr_unikalen_i_checkpoint_soglasovan():
    df = pd.read_csv(common.REP / "OPTUNA_TRIALS.csv", dtype=str).fillna("")
    ck = json.loads((common.REP / "CHECKPOINT.json").read_text())
    check_registry(df, ck)
    assert ck["waves"]["A"] == "done" and ck["cache_key_wave_a"]


def test_mut_dubl_id_i_raskhozhdenie_schetchika_lovyatsya():
    df = pd.DataFrame([["a", "B1", "har_boosting", "{}", "", "", "", "42", ""]], columns=list(registry.TRIAL_COLS))
    ck = {"trials_limit": LIMITS, "trials_limit_total": 240, "trials_used": {**{k: 0 for k in LIMITS}, "har_boosting": 1}}
    check_registry(df, ck)
    with pytest.raises(AssertionError):
        check_registry(pd.concat([df, df], ignore_index=True), ck)
    with pytest.raises(AssertionError):
        check_registry(df, {**ck, "trials_used": {k: 0 for k in LIMITS}})


def test_ensure_ne_perezapisyvaet(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "REP", tmp_path)
    registry.ensure("k1")
    (tmp_path / "OPTUNA_TRIALS.csv").write_text("x\n")
    registry.ensure("k2")
    assert (tmp_path / "OPTUNA_TRIALS.csv").read_text() == "x\n"
    assert json.loads((tmp_path / "CHECKPOINT.json").read_text())["cache_key_wave_a"] == "k1"
