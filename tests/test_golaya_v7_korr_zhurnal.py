"""Л1.6 доработка #860: (1) сигналы на скорректированном ряду (mutations № 33), (2) журнал сделок Ж-150."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis")]
import golaya_v7 as gv  # noqa: E402
import golaya_v7_tablicy as gt  # noqa: E402
from fastbt import bars_agg, dividendy, impuls_v7 as imp  # noqa: E402
from test_golaya_v7 import TK, bary  # noqa: E402


def _s_otsechkoj():
    """Тикер с отсечками в диапазоне баров. ideal — ряд без выплат (случайные 10-минутки); raw — тот же ряд с падением на D в день
    гэпа, построенный так, что adjust_ohlc(raw) == ideal (множитель дня f из префикса). Размер D — до 8 % уровня."""
    raw, im = bary()
    for t in TK:
        b = raw[t]
        day = pd.DatetimeIndex(b["timestamp"].dt.normalize())
        days = day.unique()
        g = dividendy.gap_divs(t, days.values)
        if not g[30:].any():
            continue
        k, ideal = g.max() / 0.08 / 3000, b.copy()
        for c in ("open", "high", "low", "close"):
            ideal[c] = b[c] * k
        cq, f, fd = ideal.groupby(day)["close"].last().to_numpy(), 1.0, np.ones(len(days))
        for i in range(len(days)):
            f = f / (1 - g[i] / (cq[i - 1] / f)) if i and g[i] else f
            fd[i] = f
        kf = pd.Series(fd, days).reindex(day).to_numpy()
        raw[t] = ideal.assign(**{c: ideal[c] / kf for c in ("open", "high", "low", "close")})
        return raw, im, t, ideal
    pytest.skip("нет отсечки в диапазоне синтетики")


def _ideal_sig(ideal, s, D):
    return imp.signal_dnya(bars_agg.prep_10m(ideal), s, "18:30").reindex(D["daty"]).fillna(0).to_numpy()


def test_signaly_na_skorrektirovannom_ryadu_i_myutaciya(monkeypatch):
    """Гэп вниз на D не даёт ложного сигнала: сигнал по ряду с выплатами == сигнал по ряду без них; сигнал на реальном ряду — нет."""
    raw, im, t, ideal = _s_otsechkoj()
    D, j = gv.iz_barov(raw, im, "train4", TK), TK.index(t)
    for s in ("S10", "S11", "S12"):
        assert np.array_equal(gv.signaly(D, s)[:, j], _ideal_sig(ideal, s, D)), s
    assert any(_ideal_sig(ideal, s, D).any() for s in ("S10", "S11", "S12"))
    monkeypatch.setattr(gv, "_korr10", lambda b, f: b)                   # мутация: сигнал на реальном ряду (mutations № 33)
    D2 = gv.iz_barov(raw, im, "train4", TK)
    assert any(not np.array_equal(gv.signaly(D2, s)[:, j], _ideal_sig(ideal, s, D2)) for s in ("S10", "S11", "S12"))


def test_ispolnenie_na_realnyh_cenah_s_D():
    raw, im, t, ideal = _s_otsechkoj()
    D, j = gv.iz_barov(raw, im, "train4", TK), TK.index(t)
    c, cr = D["c"].iloc[:, j], D["cr"].iloc[:, j]
    assert D["div"].iloc[:, j].gt(0).any() and (c >= cr - 1e-9).all() and c.iloc[-1] > cr.iloc[-1]   # исполнение: реальный ряд + D
    assert np.allclose(c, ideal.groupby(pd.DatetimeIndex(ideal["timestamp"].dt.normalize()))["close"].last().reindex(c.index))


@pytest.fixture(scope="module")
def DD():
    raw, im = bary()
    return gv.iz_barov(raw, im, "train4", TK)


@pytest.mark.parametrize("sid", ["S7", "S8", "S9", "S10", "S11", "S12"])
def test_zhurnal_summa_ravna_ispolnitelyu(DD, sid):
    tk = gv.shag(sid)
    o = gv.schet(DD, gv.signaly(DD, sid), **tk)
    z = gt.zhurnal(DD, sid, o, "centr", tk)
    assert len(z) == int(o["vhody"].sum()) > 0
    assert abs(z["pnl_posle"].sum() - o["excess"].sum()) <= 1e-12
    assert np.allclose(z["pnl_posle"], z["pnl_do_izderzhek"] - z["komissiya"] - z["plata_za_short"] - z["stavka"], atol=1e-15)
    assert (z["uderzhanie_dn"] >= 0).all() and (z["komissiya"] > 0).all() and set(z["tip_vhoda"]) == {"рынок"}
    assert (z.loc[z["storona"] > 0, "plata_za_short"] == 0).all()
    if sid in gv.SECH:
        assert z["sila"].between(0.5, 1).all()
    else:
        assert z["tf"].iloc[0] and z["sila"].isna().all()


def test_tercil_bez_zaglyadyvaniya():
    r = np.random.default_rng(3)
    sl = r.uniform(0.5, 1, (400, 5))
    a = gt.tercil(sl, 300, 0.9)
    sl2 = sl.copy()
    sl2[300:] = 0.5                                                      # будущее (дни ≥ d) не должно влиять
    assert a == gt.tercil(sl2, 300, 0.9) == 3 and gt.tercil(sl, 300, 0.51) == 1 and gt.tercil(sl, 3, 0.9) == ""
    sl3 = sl.copy()
    sl3[299] = 0.5                                                       # день d−1 — в окне: терциль может измениться
    assert gt.tercil(sl3, 300, 0.7) in (1, 2, 3)


def test_tablicy_pishut_zhurnal(DD, tmp_path):
    gt.tablicy(DD, "train4", tmp_path, n=2)
    z = pd.read_csv(tmp_path / "sdelki_train4.csv", comment="#")
    assert {"sid", "tiker", "data_vhoda", "data_vyhoda", "uderzhanie_dn", "tip_vhoda", "tf", "vhod", "sila", "tercil_sily",
            "pnl_do_izderzhek", "komissiya", "plata_za_short", "pnl_posle"}.issubset(z.columns)
    assert {"S7", "S11"}.issubset(set(z["sid"])) and set(z.loc[z["sid"] == "S12", "tf"]) == {"согласие 1ч+2ч"}
