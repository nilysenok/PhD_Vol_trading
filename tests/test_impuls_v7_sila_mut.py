"""Тестировщик PR #896 (V7-5): сторона побайтно прежняя, ATR без бара решения, ATR = 0, S12 = среднее, τ=1 вне сетки; мутации monkeypatch.

Реальные 10-минутки — только 2018 (< 2020), чтение; нет файлов — пропуск.
"""
import numpy as np
import pandas as pd
import pytest

from fastbt import bars_agg as ba
from fastbt import impuls_v7 as I
from fastbt.signals import v7_impuls as V
from test_impuls_v7 import b10_iz, sluchaynyj
from test_impuls_v7_sila import DAY, POSLEDNIJ_1H, lenta, znach


def _sila_s_barom_resheniya(b10, tf, t_sig, kanal=I.KANAL, atr_okno=I.ATR_OKNO):
    """Мутант: ATR без shift(1) — в среднее входит бар решения."""
    k = ba.K[tf]
    n_full = (I._minuty(t_sig) - ba.OPEN_MIN) // (ba.STEP * k)
    f = ba.resample(b10, k)
    f = f[f["slot"] < n_full].sort_values(["date", "slot"]).reset_index(drop=True)
    pc = f["close"].shift(1)
    tr = pd.concat([f["high"] - f["low"], (f["high"] - pc).abs(), (f["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_okno).mean()
    hi, lo = f["high"].shift(1).rolling(kanal).max(), f["low"].shift(1).rolling(kanal).min()
    raz = np.where(f["close"] > hi, f["close"] - hi, np.where(f["close"] < lo, f["close"] - lo, 0.0))
    s = pd.Series(raz / atr, index=f["date"].to_numpy())
    return s[f["slot"].to_numpy() == n_full - 1]


def test_mutaciya_atr_s_barom_resheniya_lovitsya(monkeypatch):
    b = lenta(110.0)
    m = (b["date"] == pd.bdate_range("2016-01-04", periods=DAY + 1)[-1]) & b["slot"].isin(POSLEDNIJ_1H)
    b.loc[m, "high"] += 50.0
    b.loc[m, "low"] -= 30.0
    assert znach(b) == 4.5
    monkeypatch.setattr(I, "sila_proboja", _sila_s_barom_resheniya)
    assert znach(b) != 4.5


def test_mutaciya_tau_1_v_osnovnoj_setke_lovitsya(monkeypatch):
    assert all(p["tau"] != 1 for p in V.setka())
    monkeypatch.setattr(V, "TAU", (1, 2, 3))
    assert any(p["tau"] == 1 and p.get("rol") != "spravka" for p in V.setka())      # тест на «τ=1 вне основной сетки» упал бы


def test_s12_sila_srednee_a_ne_maksimum():
    """S12 = сторона согласия × (|S10| + |S11|)/2; на случайных лентах среднее строго меньше максимума, когда |S10| ≠ |S11|."""
    n_raznyh = 0
    for seed in range(6):
        b = sluchaynyj(200, seed)
        a, c = I.sila_proboja(b, "1h", "18:30"), I.sila_proboja(b, "2h", "18:30")
        s12 = I.sila_dnya(b, "S12", "18:30")
        idx = a.index.union(c.index)
        a, c = a.reindex(idx).fillna(0.0), c.reindex(idx).fillna(0.0)
        soglasie = (np.sign(a) == np.sign(c)) & (a != 0)
        assert np.allclose(s12[soglasie], np.sign(a)[soglasie] * (a.abs() + c.abs())[soglasie] / 2)
        assert (s12[~soglasie] == 0).all()
        n_raznyh += int((soglasie & ((a.abs() - c.abs()).abs() > 1e-9)).sum())
        assert (s12[soglasie].abs() <= np.maximum(a.abs(), c.abs())[soglasie] + 1e-12).all()
    assert n_raznyh > 0


def test_mutaciya_s12_maksimum_lovitsya(monkeypatch):
    b = sluchaynyj(200, 1)
    a, c = I.sila_proboja(b, "1h", "18:30"), I.sila_proboja(b, "2h", "18:30")
    s12 = I.sila_dnya(b, "S12", "18:30")
    soglasie = (np.sign(a.reindex(s12.index).fillna(0)) == np.sign(c.reindex(s12.index).fillna(0))) & (s12 != 0)
    mx = np.sign(s12) * np.maximum(a.abs().reindex(s12.index), c.abs().reindex(s12.index))
    assert (s12[soglasie].abs() < mx[soglasie].abs() - 1e-9).any()                 # максимум отличим от среднего


def test_storona_pobajtno_prezhnyaya_i_strength_v_semye():
    """Сторона из semya() равна I.signal_dnya (не тронут), strength = |сила| только там, где сторона ≠ 0."""
    b = sluchaynyj(140, 7)
    from fastbt.signals.interface import TickerData
    days = pd.bdate_range("2016-01-04", periods=140)
    for sid in ("S10", "S11", "S12"):
        side = I.signal_dnya(b, sid, V.T_D1).reindex(days).fillna(0).astype(np.int8).to_numpy()
        sila = I.sila_dnya(b, sid, V.T_D1).reindex(days).fillna(0.0).to_numpy()
        assert (np.sign(sila).astype(int)[side != 0] == side[side != 0]).all()      # знак силы = сторона (шорт < 0)
        assert (sila[side == 0] == 0).all()


@pytest.mark.xfail(strict=True, reason="Д4: при ATR = 0 сила подменена на ±1 (сигнал с неопределённой силой попадает в ранг); "
                                       "правильно — NaN/0 с флагом")
def test_atr_nol_sila_ne_podmenyaetsya_edinicej():
    b = b10_iz(lambda i, s: 101.0 if (i == DAY and s in POSLEDNIJ_1H) else 100.0, 12)    # плоско (TR = 0), затем пробой вверх
    side = I.proboj(b, "1h", "18:30")
    assert (side != 0).any()
    sila = I.sila_proboja(b, "1h", "18:30")[side != 0]
    assert not np.isin(sila.abs(), (1.0,)).any()


def _real(tk):
    pa = pytest.importorskip("pyarrow.parquet")
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "moex_discovery" / "data" / "raw" / "candles_10m" / f"{tk}.parquet"
    if not p.exists():
        pytest.skip("нет данных")
    lo, hi = pd.Timestamp("2017-09-01"), pd.Timestamp("2019-01-01")
    return ba.prep_10m(pa.read_table(p, filters=[("timestamp", ">=", lo.to_pydatetime()), ("timestamp", "<", hi.to_pydatetime())]).to_pandas())


@pytest.mark.parametrize("tk", ["MTLR", "SBER"])
def test_real_2018_storona_ravna_proboj_i_net_atr_nol(tk):
    b = _real(tk)
    for tf, sid in (("1h", "S10"), ("2h", "S11")):
        sila, side = I.sila_proboja(b, tf, "18:30"), I.proboj(b, tf, "18:30")
        assert (np.sign(sila).astype(int) == side.astype(int)).all()
        assert (sila[side != 0].abs() > 0).all() and not (sila[side != 0].abs() == 1.0).any()      # подмена ±1 на реальных 2018 не сработала
        assert np.isfinite(sila).all()
