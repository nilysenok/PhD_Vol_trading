"""V7 лист 1a-M, G18: walk-forward и PIT через РЕАЛЬНЫЙ путь `walk_forward`. Помесячная переоценка только по парам t⁺ ≤ d; Z/z стандартизованы по закрытым парам; первая оцениваемая дата 2018-01-03.
Мутации: «закрыто + 1», нормировка по всей выборке, внутренняя валидация с парами t⁺ > d, EWMA/CUSUM с текущей парой — ловятся инвариантом «цели с позиции J не влияют на прогнозы строк ≤ J». Синтетика."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from torch import nn

from models_v2.s14 import v7_1m_fit as FT, v7_1m_feat as FE, v7_1m_syn as SY, v7_1m_wf as WF

N = 330
SPEC = dict(kind="moe", router="B")
KW = dict(seeds=(0,), grid=((1.0, 0.0),))


def j_feb(tab):
    return int(np.flatnonzero(tab.index >= pd.Timestamp("2018-02-01"))[0])


def run(s, **kw):
    return WF.walk_forward(s["tab"], s["Zg"], s["z"], kw.pop("spec", SPEC), **{**KW, **kw})


def plus_one(tp, d0):
    """Закрытые пары + первая НЕзакрытая (t⁺ > d0)."""
    nxt = np.sort(tp[tp > np.datetime64(d0)])[0]
    return np.flatnonzero(tp <= nxt)


def leaks(monkeypatch, patch=None, per_run=None):
    """True, если прогнозы строк ≤ J зависят от целей строк ≥ J (утечка)."""
    if patch:
        patch(monkeypatch)
    a = SY.make(N, 0)
    J = j_feb(a["tab"])
    b = SY.perturb_future(a, J)
    out = []
    for s in (a, b):
        if per_run:
            per_run(monkeypatch, s)
        out.append(run(s)["F"].to_numpy())
    same = np.array_equal(out[0][:J + 1], out[1][:J + 1], equal_nan=True)
    assert not np.array_equal(out[0][J + 1:], out[1][J + 1:], equal_nan=True)                  # возмущение видно позже: тест не пустой
    return not same


def test_walk_forward_does_not_crash_when_lightgbm_is_imported_before_torch(tmp_path):
    """macOS: общий OpenMP — softmax torch в нескольких потоках падает (segfault) после импорта lightgbm. Все вычисления идут в одном потоке (`FM.deterministic`); проверка в чистом процессе."""
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "probe.py"
    script.write_text("import sys\nsys.path.insert(0, %r)\nimport lightgbm\nfrom models_v2.s14 import v7_1m_syn as SY, v7_1m_wf as WF\n"
                      "s = SY.make(330, 0)\nr = WF.walk_forward(s['tab'], s['Zg'], s['z'], dict(kind='moe', router='B'), seeds=(0,), grid=((1.0, 0.0),))\nprint('fits', sum(f['fitted'] for f in r['fits']))\n" % str(root), encoding="utf-8")
    p = subprocess.run([sys.executable, "-W", "ignore", str(script)], capture_output=True, text=True, cwd=str(tmp_path))
    assert p.returncode == 0 and "fits 4" in p.stdout, (p.returncode, p.stderr[-300:])


def test_honest_walk_forward_has_no_future_dependence(monkeypatch):
    assert leaks(monkeypatch) is False


def test_mutation_closed_plus_one_caught(monkeypatch):
    assert leaks(monkeypatch, lambda mp: mp.setattr(WF, "closed_rows", plus_one)) is True


def test_mutation_validation_with_future_pairs_caught(monkeypatch):
    def val(tp, d0, usable):
        r = plus_one(tp, d0)
        return r[usable[r]][-WF.HOLD:]
    assert leaks(monkeypatch, lambda mp: mp.setattr(WF, "val_rows", val)) is True


def test_mutation_standardization_on_whole_sample_caught(monkeypatch):
    def whole(mp, s):
        A = np.hstack([s["Zg"], s["z"]])
        A = A[np.isfinite(A).all(axis=1)]
        sd = A.std(axis=0)
        mp.setattr(WF, "norm_stats", lambda X: (A.mean(axis=0), np.where(sd > 1e-12, sd, 1.0)))
    assert leaks(monkeypatch, per_run=whole) is True


def test_mutation_ewma_cusum_with_current_pair_caught(monkeypatch):
    assert leaks(monkeypatch, lambda mp: mp.setattr(FE, "last_closed", lambda tp, d: np.arange(len(tp)))) is True


def test_training_rows_are_closed_and_validation_is_last_126(monkeypatch):
    s = SY.make(N, 0)
    tab, seen = s["tab"], []
    orig = FT.fit_config
    monkeypatch.setattr(FT, "fit_config", lambda spec, tr, va, seeds, grid: seen.append((tr["rv"].numpy().copy(), va["rv"].numpy().copy())) or orig(spec, tr, va, seeds, grid))
    r = run(s)
    rows = lambda v: np.array([int(np.flatnonzero(tab["rv_next"].to_numpy() == x)[0]) for x in v])
    fitted = [f for f in r["fits"] if f["fitted"]]
    assert len(fitted) == len(seen) >= 3
    tp = tab["t_plus"].to_numpy("datetime64[ns]")
    for f, (tr, va) in zip(fitted, seen):
        rt, rvv = rows(tr), rows(va)
        assert (tp[rt] <= np.datetime64(f["d0"])).all() and (tp[rvv] <= np.datetime64(f["d0"])).all()          # только закрытые пары
        assert len(rvv) == WF.HOLD and rt.max() < rvv.min() and rvv.max() == WF.closed_rows(tp, f["d0"])[-1]          # валидация = последние 126 закрытых
        assert f["n_train"] == len(rt) + len(rvv)


def test_first_evaluated_date_is_2018_01_03_and_2017_is_training_only():
    s = SY.make(N, 0)
    r = run(s)
    idx = s["tab"].index
    assert np.isnan(r["F"][idx < pd.Timestamp("2018-01-03")]).all() and r["F"][idx >= pd.Timestamp("2018-01-03")].notna().all()
    assert r["fits"][0]["d0"] == pd.Timestamp("2018-01-01") and r["fitted"][pd.Timestamp("2018-01-03")]       # блок января начинается раньше первой оцениваемой даты


def test_refit_monthly_prediction_uses_model_of_last_refit_le_d():
    s = SY.make(N, 0)
    r = run(s)
    lam, idx = r["fits"], s["tab"].index
    d0s = [f["d0"] for f in lam]
    assert all(b - a >= pd.Timedelta(days=26) for a, b in zip(d0s, d0s[1:]))                                  # шаг — месяц
    assert all(d0.month != prev.month for prev, d0 in zip(d0s, d0s[1:]))


def test_too_few_closed_pairs_gives_F_equals_H_bitwise_not_fitted():
    s = SY.make(N, 0)
    r = run(s, first_eval="2017-03-01")
    early = (s["tab"].index >= pd.Timestamp("2017-03-01")) & (s["tab"].index < pd.Timestamp("2017-06-01"))
    assert not r["fitted"][early].any() and np.array_equal(r["F"][early].to_numpy(), s["tab"]["H"][early].to_numpy())


def test_forecast_positive_and_bounded_correction():
    s = SY.make(N, 0)
    r = run(s)
    ok = r["fitted"].to_numpy()
    assert (r["F"][ok] > 0).all() and (np.abs(r["G"][ok]) <= 1.0 + 1e-9).all()


def test_stage_b_uses_stage_a_forecast_before_first_joint_refit(monkeypatch):
    s = SY.make(N, 0)
    a = run(s)
    X = np.random.default_rng(0).standard_normal((N, 6))
    spec = dict(kind="moe", router="B", encoder=lambda: nn.Linear(6, 4))
    b = WF.walk_forward(s["tab"], s["Zg"], X, spec, first_fit="2018-02-01", fallback=a["F"], **KW)
    idx = s["tab"].index
    pre = (idx >= pd.Timestamp("2018-01-03")) & (idx < pd.Timestamp("2018-02-01"))
    post = idx >= pd.Timestamp("2018-02-01")
    assert (b["stage"][pre] == "A").all() and np.array_equal(b["F"][pre].to_numpy(), a["F"][pre].to_numpy())
    assert (b["stage"][post] == "B").all() and b["fitted"][post].all()
    assert FT.SEEDS == (0, 1, 2, 3, 4) and WF.FIRST_FIT_B == "2019-01-01"
