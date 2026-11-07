"""G18 (У-24, Ж-220 п. 6–7): K2/K3 только для основных 17; готовые `k23_<T>` не пересчитываются; `--check` требует 17 основных + GCHE (k23.parquet из 28 не нужен). Синтетика, сеть и данные проекта не читаются."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_k23 as K, hp_paths as P


def test_nabor_check_17_osnovnyh_i_gche():
    assert K.K23_17 == sorted(P.MAIN17 + ["GCHE"]) and len(K.K23_17) == 18


def test_bumaga_e11_v_spiske_scheta_otkaz():
    with pytest.raises(ValueError):                              # мутация: бумага E11 попала в список счёта
        K.todo(["LSRG", "MRKC"])
    with pytest.raises(ValueError):
        K.todo(["GCHE"])                                         # GCHE готова и в E11: не считается


def test_gotovaya_ne_pereschityvaetsya_i_ne_perezapisyvaetsya(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "OUT", tmp_path)
    (tmp_path / "k23_SBER.parquet").write_bytes(b"old")
    calls = []
    done = K.compute_missing(["SBER", "LSRG"], pd.DataFrame(), pd.DataFrame(), [], run=lambda t, *a: calls.append(t))
    assert calls == ["LSRG"] == done                             # мутация: пересчёт готовой — calls == ["SBER", "LSRG"]
    assert (tmp_path / "k23_SBER.parquet").read_bytes() == b"old"


def _panel(tickers, n=6):
    d = pd.bdate_range("2018-01-01", periods=n)
    return pd.concat([pd.DataFrame({"ticker": t, "date": d, "t_plus": d + pd.offsets.BDay(1), "rv_tplus": np.arange(1.0, n + 1) + i}) for i, t in enumerate(tickers)], ignore_index=True)


def _k23(panel, t):
    g = panel[panel["ticker"] == t].reset_index(drop=True)
    return g.assign(sigma2_k2=g["rv_tplus"] * 1.1, sigma2_k3=g["rv_tplus"] * 0.9)


def test_check_lovit_otsutstvie_k23_osnovnoj_bumagi():
    tk = [t for t in K.K23_17 if t != "SBER"]
    panel = _panel(tk)
    ok = K.check_k23(tk, panel, pd.DataFrame(), lambda t: _k23(panel, t))
    assert ok == []
    miss = K.check_k23(tk, panel, pd.DataFrame(), lambda t: None if t == "LSRG" else _k23(panel, t))
    assert len(miss) == 1 and "LSRG" in miss[0]                  # мутация: проверка проходит без k23_LSRG


def test_check_lovit_sdvig_setki_rv_i_nekonechnye():
    panel = _panel(["LSRG"])
    bad = _k23(panel, "LSRG")
    rv = bad.copy()
    rv.loc[2, "rv_tplus"] += 1e-12
    assert K.check_k23(["LSRG"], panel, pd.DataFrame(), lambda t: rv)
    tp = bad.copy()
    tp["t_plus"] = tp["t_plus"] + pd.Timedelta(days=1)
    assert K.check_k23(["LSRG"], panel, pd.DataFrame(), lambda t: tp)
    nf = bad.copy()
    nf.loc[3, "sigma2_k3"] = np.inf
    assert K.check_k23(["LSRG"], panel, pd.DataFrame(), lambda t: nf)
    neg = bad.copy()
    neg.loc[3, "sigma2_k2"] = -1.0
    assert K.check_k23(["LSRG"], panel, pd.DataFrame(), lambda t: neg)
    empty = bad.assign(sigma2_k2=np.nan, sigma2_k3=np.nan)
    assert K.check_k23(["LSRG"], panel, pd.DataFrame(), lambda t: empty)       # нулевой объём прогнозов — не «ок»


def test_check_sber_pobitno_components():
    panel = _panel(["SBER"])
    k = _k23(panel, "SBER")
    comp = pd.DataFrame({"d": k["date"], "target_date": k["t_plus"], "rv": k["rv_tplus"], "XJ": k["sigma2_k2"], "LF": k["sigma2_k3"]})
    assert K.check_k23(["SBER"], panel, comp, lambda t: k) == []
    comp.loc[1, "XJ"] += 1e-12
    assert K.check_k23(["SBER"], panel, comp, lambda t: k)
