"""Разбор прохода 1: режимы без заглядывания в d, диагностика неопределённого mSR, плата за шорт. Синтетика, без сети.
Мутации: режимы по данным с днём d (IMOEX) и ставкой того же дня; плата за шорт со знаком «+» (должна только уменьшать ряд)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import prohod1_lib as L  # noqa: E402
import prohod1_rezhimy as R  # noqa: E402
from fastbt.tselevaya import msr_i_gody  # noqa: E402


def test_imoex_ma200_bez_dnya_d(tmp_path, monkeypatch):
    dts = pd.bdate_range("2018-01-01", periods=400)
    c = np.linspace(100.0, 200.0, 400)
    c[350] = 1.0                                              # обвал в день d = dts[350]
    p = tmp_path / "IMOEX.csv"
    pd.DataFrame({"TRADEDATE": dts, "CLOSE": c}).to_csv(p, index=False)
    monkeypatch.setattr(R, "IMOEX", p)
    vyshe, est = R.imoex_vyshe_ma200(dts[[350, 351]].to_numpy().astype("datetime64[D]"))
    assert est.all()
    assert vyshe[0], "в день d признак обязан быть по данным до d (обвал d ещё неизвестен)"
    assert not vyshe[1], "на следующий день обвал виден"


def test_imoex_net_ma_v_nachale(tmp_path, monkeypatch):
    dts = pd.bdate_range("2018-01-01", periods=250)
    p = tmp_path / "IMOEX.csv"
    pd.DataFrame({"TRADEDATE": dts, "CLOSE": np.linspace(100.0, 120.0, 250)}).to_csv(p, index=False)
    monkeypatch.setattr(R, "IMOEX", p)
    vyshe, est = R.imoex_vyshe_ma200(dts[[10, 240]].to_numpy().astype("datetime64[D]"))
    assert not est[0] and not vyshe[0] and est[1]


def test_stavka_v_sile_na_d_minus_1(monkeypatch):
    from fastbt import msp
    s = pd.Series([0.0775, 0.075], index=pd.to_datetime(["2017-12-18", "2018-02-12"]))
    monkeypatch.setattr(msp, "rate_2013", lambda: s)
    st = R.stavka_po_dnyam(np.array(["2018-02-12", "2018-02-13"], "datetime64[D]"))
    assert st[0] == 0.0775 and st[1] == 0.075, "решение вступает с 12.02, в этот день ещё действует прежняя ставка"


def _ryad(n_dney=1010, seed=1):
    rng = np.random.default_rng(seed)
    daty = (np.datetime64("2016-01-04") + np.arange(n_dney)).astype("datetime64[D]")
    return daty, rng


def test_diagnostika_nan_ploskij_god():
    daty = (np.datetime64("2016-01-01") + np.arange(1461)).astype("datetime64[D]")
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(0.002, 0.01, len(daty)), rng.normal(0.002, 0.01, len(daty))])
    X[(daty >= np.datetime64("2017-01-01")) & (daty < np.datetime64("2018-01-01")), 1] = 0.0     # столбец 1: год без движения
    _, py, ust, _ = msr_i_gody(X, daty, L.GODY)
    d = L.diagnostika_nan(X, py, daty)
    assert d["hotya_by_ploskij_god"] == 1 and d["srednee_pol_vse_gody"] == 1
    assert d["ust_esli_nan_pri_pol_srednem"] == 1 and int(ust.sum()) == 1, "плоский год при среднем 0 не считается устойчивым"


def test_plata_za_shorty_umenshaet_ryad():
    daty, rng = _ryad()
    M = rng.normal(0.001, 0.01, (len(daty), 3))
    Z = np.abs(rng.normal(0.0, 0.001, M.shape))
    m0 = msr_i_gody(M, daty, L.GODY)[0]
    m5 = msr_i_gody(M - L.STAVKA_SHORT * Z, daty, L.GODY)[0]
    assert (m5 < m0).all(), "плата со знаком «+» (мутация) подняла бы целевую"
    assert L.sverka_tselevoj(M, Z, daty, n=3) == 0
