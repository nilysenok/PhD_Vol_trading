"""Д-5 (Ж-145): таблицы (4), (5) на синтетике с известным ответом и мутации (G5.1–G5.3). Сети и данных ≥ 2020 нет."""
from __future__ import annotations

import numpy as np
import pytest

from fastbt import diag_v6_t45 as t45

N = 60
DATY = np.busday_offset(np.datetime64("2016-01-04"), np.arange(N), roll="forward").astype("datetime64[D]")
_ALT = [0.01, 0.03] * 20


def _x() -> np.ndarray:
    """40 дней 0,01/0,03, 10 дней +1 (лучшие), 10 дней −1 (худшие), порядок перемешан фиксированно."""
    return np.random.default_rng(0).permutation(np.array(_ALT + [1.0] * 10 + [-1.0] * 10))


def _sh(v) -> float:
    v = np.asarray(v, float)
    return float(v.mean() / v.std(ddof=1) * np.sqrt(252))


def _sd(vh, vy, storona, pnl=0.02, kom=0.0) -> dict:
    return dict(tiker="A", d_sig=vh, vhod=vh, vyhod=vy, storona=storona, pnl_do_izderzhek=pnl, komissiya=kom)


def _tochka(zhurnal=(), x=None) -> dict:
    return dict(sid="S1", i=7, rang=1, prohod="s1", x=_x() if x is None else x, daty=DATY, tik_poz=np.full((2, N), 0.5), zhurnal=list(zhurnal))


def _proverit_5(row) -> None:
    bez_luchshih = _ALT + [-1.0] * 10      # вычеркнуты 10 дней +1
    bez_hudshih = _ALT + [1.0] * 10        # вычеркнуты 10 дней −1
    assert row["sharpe_dn_poln"] == pytest.approx(_sh(_ALT + [1.0] * 10 + [-1.0] * 10))
    assert row["sharpe_dn_bez_luchshih"] == pytest.approx(_sh(bez_luchshih))
    assert row["sharpe_dn_bez_hudshih"] == pytest.approx(_sh(bez_hudshih))


def test_g51_bez_10_luchshih_izvestnyj_sharpe():
    row = t45.tablica_5([_tochka()]).iloc[0]
    _proverit_5(row)
    assert row["sharpe_dn_bez_luchshih"] < 0 < row["sharpe_dn_bez_hudshih"]
    assert row["dni"] == N and bool(row["zavisit_ot_dnej"]) is True    # без 10 лучших Шарп < 0 → прибыль держалась на лучших днях


def test_g51_zavisit_ot_dnej_kogda_vse_v_10_luchshih():
    x = np.random.default_rng(1).permutation(np.array([0.0] * 49 + [0.5] * 10 + [-0.02]))
    row = t45.tablica_5([_tochka(x=x)]).iloc[0]
    assert row["sharpe_dn_poln"] > 0 and row["sharpe_dn_bez_luchshih"] < 0.5 * row["sharpe_dn_poln"]
    assert bool(row["zavisit_ot_dnej"]) is True


def test_g51_ned_tri_stolbca_est_i_diag_podpisany():
    row = t45.tablica_5([_tochka()]).iloc[0]
    for k in ("msr_dn", "msr_ned"):
        assert all(f"{k}_{p}" in row.index for p in ("poln", "bez_luchshih", "bez_hudshih"))
    assert {"sharpe_norm_diag", "sharpe_sdelok_diag", "cena_prostoev_diag"} <= set(row.index)
    assert row["sharpe_norm_diag"] == pytest.approx(row["sharpe_dn_poln"] / 1.0)   # 2 тикера по 0,5 → валовая экспозиция 1


def test_g52_mutaciya_ubrany_hudshie_vmesto_luchshih_lovitsya(monkeypatch):
    orig = t45._ubrat
    monkeypatch.setattr(t45, "_ubrat", lambda x, h, n: orig(x, {"luch": "hud", "hud": "luch"}.get(h), n))
    with pytest.raises(AssertionError):
        _proverit_5(t45.tablica_5([_tochka()]).iloc[0])


def test_g52_mutaciya_9_dnej_vmesto_10_lovitsya(monkeypatch):
    orig = t45._ubrat
    monkeypatch.setattr(t45, "_ubrat", lambda x, h, n: orig(x, h, n - 1))
    with pytest.raises(AssertionError):
        _proverit_5(t45.tablica_5([_tochka()]).iloc[0])


def _zhurnal4() -> list:
    return [_sd("2016-12-20", "2017-01-10", 1, 0.05), _sd("2017-12-28", "2018-02-01", -1, -0.03, 0.01),
            _sd("2017-03-01", "2017-04-01", 1, 0.02), _sd("2017-05-01", "2017-06-01", 1, 0.02), _sd("2019-12-01", None, -1, 0.01)]


def test_g53_god_vyhoda_a_ne_vhoda():
    zh = _zhurnal4()
    t = _tochka(zh)
    df = t45.tablica_4([t])
    last = int(str(DATY[-1])[:4])
    g = df.set_index("god_vyhoda")
    assert sorted(g.index) == sorted({2017, 2018, last})
    assert (g.loc[2017, "n_long"], g.loc[2017, "n_short"]) == (3, 0)           # сделка входа 2016 выходит в 2017
    assert (g.loc[2018, "n_long"], g.loc[2018, "n_short"]) == (0, 1)           # вход 2017, выход 2018
    assert g.loc[2017, "pnl_long"] == pytest.approx(0.09) and g.loc[2018, "pnl_short"] == pytest.approx(-0.04)
    assert g.loc[2017, "dolya_long_sdelok"] == 1.0 and bool(g.loc[2017, "odna_storona"])


def test_g53_slaboe_mesto_odna_storona_v_dvuh_godah():
    df = t45.tablica_4([_tochka(_zhurnal4())])
    assert (df["let_odna_storona"] == df["odna_storona"].sum()).all()
    assert df["odna_storona"].all() and bool(df["slaboe_mesto"].all())          # каждый год — одна сторона 100 % > 80 %


def test_g53_mutaciya_god_vhoda_lovitsya(monkeypatch):
    monkeypatch.setattr(t45, "god_sdelki", lambda tr, daty: int(tr["vhod"][:4]))
    g = t45.tablica_4([_tochka(_zhurnal4())]).set_index("god_vyhoda")
    with pytest.raises(AssertionError):
        assert sorted(g.index) == sorted({2017, 2018, int(str(DATY[-1])[:4])})


def test_pustaya_lenta_i_nulevaya_ekspoziciya_ne_padayut():
    t = _tochka([], x=np.zeros(N))
    t["tik_poz"] = np.zeros((2, N))
    row = t45.tablica_5([t]).iloc[0]
    assert np.isnan(row["sharpe_norm_diag"]) and "экспозиция 0" in row["prichina_nan_diag"] and "сделок < 2" in row["prichina_nan_diag"]
    assert len(t45.tablica_4([t])) == 0
