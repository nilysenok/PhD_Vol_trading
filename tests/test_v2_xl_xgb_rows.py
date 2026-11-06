"""Строка таблицы волны B (rowsx.row_of): метрики из ряда прогнозов, знак разностей, перспективность и beats_f4_ref, нестабильность. Известный ответ на синтетике."""
import numpy as np
import pandas as pd
import pytest

from models_v2 import metrics
from models_v2.s6.xgb import cfgx, rowsx

N = 2271
DATES = pd.bdate_range("2017-01-03", periods=N)


def mk_ctx(rv, rng):
    ctrl = pd.DataFrame({"d": DATES - pd.Timedelta(days=1), "target_date": DATES, "rv": rv,
                         "s2_b2": rv * np.exp(rng.normal(0, 0.5, N)), "s2_harj": rv * np.exp(rng.normal(0.1, 0.6, N))})
    f4 = rv * np.exp(rng.normal(0, 0.4, N))
    folds = pd.DataFrame([{"Y": Y, "y": y, "q": 0.2} for Y in range(2017, 2027) for y in range(2015, Y)])
    return ctrl, f4, folds


def build(sigma2, crit_pen_f4=0.2, crit_pen_b2=0.21):
    rng = np.random.default_rng(0)
    rv = np.exp(rng.normal(0, 0.5, N))
    ctrl, f4, folds = mk_ctx(rv, rng)
    ctx = {"ctrl": ctrl, "f4_s2": f4, "crit_b2": {"crit_pen": crit_pen_b2}, "crit_f4": {"crit_pen": crit_pen_f4}}
    fc = pd.DataFrame({"d": ctrl["d"], "target_date": ctrl["target_date"], "rv": rv, "sigma2": sigma2(rv, rng)})
    return fc, folds, ctx


def test_row_known_answer_better_than_everything():
    fc, folds, ctx = build(lambda rv, rng: rv * np.exp(rng.normal(0, 0.1, N)))
    r = rowsx.row_of(cfgx.NEW[0], "tid", fc, folds, ctx)
    assert r["qlike_all"] == pytest.approx(float(metrics.qlike(fc["rv"], fc["sigma2"]).mean()), abs=1e-15)
    assert r["d_b2"] < 0 and r["d_harj_no_240222"] < 0 and r["d_f4"] < 0                 # минус — строка лучше базы
    assert r["crit_pen"] == pytest.approx(0.2) and r["crit_pen_f4"] == 0.2 and r["status"] == rowsx.STATUS
    assert r["perspective"] is True and r["unstable"] is False
    assert r["beats_f4_ref"] is False                                                     # crit_pen = 0,2 не лучше 0,2 — ничья не прирост


def test_row_beats_f4_only_with_strictly_lower_crit():
    fc, folds, ctx = build(lambda rv, rng: rv * np.exp(rng.normal(0, 0.1, N)), crit_pen_f4=0.25)
    assert rowsx.row_of(cfgx.NEW[0], "tid", fc, folds, ctx)["beats_f4_ref"] is True


def test_row_worse_than_harj_is_not_perspective_and_diff_sign_positive():
    fc, folds, ctx = build(lambda rv, rng: rv * np.exp(rng.normal(0.0, 1.5, N)), crit_pen_b2=0.3)
    r = rowsx.row_of(cfgx.NEW[0], "tid", fc, folds, ctx)
    assert r["d_harj_no_240222"] > 0 and r["perspective"] is False                      # внутренний критерий лучше, но хуже HAR-J(б)


def test_unstable_forecast_marks_row_and_blocks_flags():
    fc, folds, ctx = build(lambda rv, rng: rv * np.exp(rng.normal(0, 0.1, N)), crit_pen_f4=0.25)
    fc.loc[5, "sigma2"] = fc.loc[5, "sigma2"] * 1e3                                       # прогноз в 1000 раз выше опоры
    r = rowsx.row_of(cfgx.NEW[0], "tid", fc, folds, ctx)
    assert r["unstable"] is True and r["perspective"] is False and r["beats_f4_ref"] is False


def test_trial_record_has_register_columns():
    fc, folds, ctx = build(lambda rv, rng: rv * np.exp(rng.normal(0, 0.1, N)))
    r = rowsx.row_of(cfgx.NEW[0], "tid", fc, folds, ctx)
    rec = rowsx.trial_record(cfgx.NEW[0], "tid", r, 12.3, "2026-11-06 12:00")
    assert list(rec) == cfgx.TRIAL_HEAD and rec["trial_id"] == "tid" and "crit_pen=" in rec["metric_inner"] and rec["seed"] == 42
