"""P17, G16 (Ж-220, У-19…У-21): сквозной боевой путь `hp_p17_hyb.Run17` (A1, B1, B2, C1, C2, D1, D2) на синтетической панели; «17» — бумаги A–D (TRAIN подменён), «E11» — E, F (сеть и данные проекта не читаются).
Инварианты: удаление бумаг вне 17 из панели не меняет прогнозы 17 побитно (обучение, выбор по году, индивидуальные части — только 17); прогноз для бумаг вне 17 есть (перенос);
обрезка будущего всей панели после d не меняет прогноз на d. Мутации (подмена крючков hp_p17): бумага E11 в обучении — ловится."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_hyb_core as C, hp_hyb_run as HR, hp_p17 as X, hp_p17_hyb as H17, hp_paths as P

HP = dict(lr=0.2, rounds=15, patience=5, min_leaf=10, l2=1.0, es_dates=20, min_train=100)
CFGS = ["A1", "B1", "B2", "C1", "C2", "D1", "D2"]
T17, T11 = ("A", "B", "C", "D"), ("E", "F")


def make_panel(seed: int = 0, n: int = 700, tickers=T17 + T11) -> pd.DataFrame:
    """Строка (бумага, дата): признаки z_*, цель RV_{t⁺}, база K1; бумаги вне 17 устроены иначе (знак связи обратный) — чтобы их присутствие в обучении было заметно."""
    dates = pd.bdate_range("2015-01-01", periods=n)
    parts = []
    for k, t in enumerate(tickers):
        rng = np.random.default_rng(seed * 100 + ord(t))
        z = {c: rng.normal(size=n - 1) for c in C.Z28}
        k1 = np.exp(rng.normal(0, 0.2, n - 1))
        sg = -1.0 if t in T11 else 1.0
        rv = k1 * np.exp(sg * 0.4 * np.sign(z["z_x_d_m"]) + 0.3 * rng.normal(size=n - 1))
        parts.append(pd.DataFrame({"ticker": t, "date": dates[:-1], "t_plus": dates[1:], "rv_tplus": rv, "sigma2_k1": k1, "mkt_rv_tplus": 0.5 * rv * np.exp(0.2 * rng.normal(size=n - 1)),
                                   "x_d_m": z["z_x_d_m"], "x_w_m": rng.normal(size=n - 1), **z}))
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def corrupt(p: pd.DataFrame, d0) -> pd.DataFrame:
    q, rng = p.copy(), np.random.default_rng(11)
    late, tgt = (q["date"] > d0).to_numpy(), (q["t_plus"] > d0).to_numpy()
    for c in C.Z28 + ["x_d_m", "x_w_m"]:
        q.loc[late, c] = rng.normal(size=late.sum()) * 3 + 2
    q.loc[late, "sigma2_k1"] = np.exp(rng.normal(size=late.sum()))
    for c in ("rv_tplus", "mkt_rv_tplus"):
        q.loc[tgt, c] = 20 * np.exp(rng.normal(size=tgt.sum()))
    return q


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "OUT", tmp_path)
    monkeypatch.setattr(P, "REP", tmp_path)
    monkeypatch.setattr(X, "TRAIN", list(T17))

    def parts(panel):
        k = panel["sigma2_k1"].to_numpy(float)
        return panel[C.KEY].assign(hm=0.3 * k, hu=0.7 * k, b2=0.6)
    monkeypatch.setattr(HR, "parts_c", parts)
    return tmp_path


def calc(panel: pd.DataFrame, module: str, cfgs=CFGS) -> dict:
    r = H17.Run17(module, 2, lambda m: None, panel=panel, hp=HP)
    for n in cfgs:
        r.calc(n)
    return r.sig


@pytest.mark.parametrize("module", ["xgb", "lgbm"])
def test_udalenie_bumag_vne_17_ne_menyaet_prognozy_17_pobitno(module, env):
    p6 = make_panel()
    p4 = p6[p6["ticker"].isin(T17)].reset_index(drop=True)
    a, b = calc(p6, module), calc(p4, module)
    m17 = p6["ticker"].isin(T17).to_numpy()
    for n in CFGS:
        x, y = a[n][m17], b[n]
        assert np.isfinite(y).sum() > 1000, n
        assert np.array_equal(x, y, equal_nan=True), n                                  # E11 не влияет на модель, выбор и индивидуальные части 17
        out = a[n][~m17]
        assert np.isfinite(out).sum() > 500 and (out[np.isfinite(out)] > 0).all(), n    # прогноз для E11 есть (перенос зафиксированной модели)


def test_mutaciya_E11_v_obuchenii_lovitsya(env, monkeypatch):
    p6 = make_panel()
    p4 = p6[p6["ticker"].isin(T17)].reset_index(drop=True)
    m17 = p6["ticker"].isin(T17).to_numpy()
    real = calc(p4, "xgb", ["A1"])["A1"]
    monkeypatch.setattr(X, "train_exclude", lambda tickers: [])
    bad = calc(p6, "xgb", ["A1"])["A1"][m17]
    assert not np.array_equal(real, bad, equal_nan=True)                                # мутант: прогноз 17 зависит от E11 — тест инвариантности его поймал бы


@pytest.mark.parametrize("module", ["xgb", "lgbm"])
def test_obrezka_budushchego_vseh_konfiguracij_na_17(module, env):
    p = make_panel()
    d0 = pd.Timestamp("2016-12-30")
    a, b = calc(p, module), calc(corrupt(p, d0), module)
    past = (p["date"] <= d0).to_numpy()
    for n in CFGS:
        ok = past & np.isfinite(a[n])
        assert ok.sum() > 300, n
        assert np.array_equal(a[n][ok], b[n][ok]), n
    assert any(not np.array_equal(a[n][~past & np.isfinite(a[n]) & np.isfinite(b[n])], b[n][~past & np.isfinite(a[n]) & np.isfinite(b[n])]) for n in CFGS)


def test_podmena_dannyh_E11_posle_d_ne_menyaet_prognoz_E11_do_d_i_17_voobshche(env):
    p = make_panel()
    d0 = pd.Timestamp("2016-12-30")
    q = p.copy()
    m11 = p["ticker"].isin(T11).to_numpy()
    q.loc[m11] = corrupt(p, d0).loc[m11]                                                       # портятся только бумаги вне 17, только после d0
    a, b = calc(p, "xgb", ["A1", "B1", "D1"]), calc(q, "xgb", ["A1", "B1", "D1"])
    past = (p["date"] <= d0).to_numpy()
    for n in ("A1", "B1", "D1"):
        assert np.array_equal(a[n][~m11], b[n][~m11], equal_nan=True), n                 # 17: E11 не влияет на модель ни до, ни после d0
        ok = m11 & past & np.isfinite(a[n])
        assert ok.sum() > 100 and np.array_equal(a[n][ok], b[n][ok]), n                  # E11 до d0: обрезка будущего
        assert not np.array_equal(a[n][m11 & ~past], b[n][m11 & ~past], equal_nan=True), n   # после d0 порча действует (его собственные признаки)


def test_kesh_prohodov_uchityvaet_sostav_obucheniya(env, monkeypatch):
    p = make_panel(n=300)
    k1 = p["sigma2_k1"].to_numpy(float)
    X.cached_runs("xgb", "t", p, ["z_x_d_m"], k1, "s", "qlike", HP, (2,), 2)
    files = lambda: sorted(f.name for f in env.glob("hyb_run_p17_*"))
    assert len(files()) == 1
    X.cached_runs("xgb", "t", p, ["z_x_d_m"], k1, "s", "qlike", HP, (2,), 2)
    assert len(files()) == 1                                                            # тот же вход и состав — попадание
    monkeypatch.setattr(X, "TRAIN", list(T17) + list(T11))
    X.cached_runs("xgb", "t", p, ["z_x_d_m"], k1, "s", "qlike", HP, (2,), 2)
    assert len(files()) == 2                                                            # другой состав обучения — промах (Д5)
