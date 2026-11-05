"""A15: реестр trials и контрольная точка — волна A не тратит trials."""
import json

from models_v2.s5 import common, registry


def test_realestr_pust_i_checkpoint_nuli():
    t = (common.REP / "OPTUNA_TRIALS.csv").read_text(encoding="utf-8").strip().splitlines()
    assert t == [",".join(registry.TRIAL_COLS)]
    c = json.loads((common.REP / "CHECKPOINT.json").read_text())
    assert c["trials_limit"] == {"objectives_target": 35, "feature_engineering": 45, "har_boosting": 50, "rolling_recency": 35,
                                 "regime_aware": 50, "ensemble_calibration": 25} and c["trials_limit_total"] == 240
    assert set(c["trials_used"].values()) == {0} and c["waves"]["A"] == "done" and c["cache_key_wave_a"]


def test_ensure_ne_perezapisyvaet(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "REP", tmp_path)
    registry.ensure("k1")
    (tmp_path / "OPTUNA_TRIALS.csv").write_text("x\n")
    registry.ensure("k2")
    assert (tmp_path / "OPTUNA_TRIALS.csv").read_text() == "x\n"
    assert json.loads((tmp_path / "CHECKPOINT.json").read_text())["cache_key_wave_a"] == "k1"
