"""Л1.2 fastbt/impuls_v7.py: сигналы S10 (1ч), S11 (2ч), S12 (согласие) из 10-минуток (Ж-146). Только синтетика.

Мутации (каждая ловится тестом): цена/бар после момента решения T в сигнале — test_mutaciya_bar_posle_T_*,
test_sig_ne_zavisit_ot_budushchih_barov; значение дня d+1 — test_mutaciya_d_plus_1_*, test_sig_ne_zavisit_ot_budushchih_barov;
S11 ≠ эталон golaya_tf_v5 — test_s11_pri_1830_raven_etalonu_pobajtno; чтение ≥ 2020 — test_chtenie_ne_dalshe_2020.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "server"))
import golaya_tf_v5 as G  # noqa: E402

from fastbt import bars_agg as ba  # noqa: E402
from fastbt import impuls_v7 as I  # noqa: E402


def b10_iz(cena, ndays, start="2016-01-04"):
    """10-минутки: cena(день, слот) → цена (open = high = low = close); слоты 0..52 (10:00..18:40)."""
    days = pd.bdate_range(start, periods=ndays)
    rows = [(d, s, cena(i, s)) for i, d in enumerate(days) for s in range(ba.SLOTS)]
    df = pd.DataFrame(rows, columns=["date", "slot", "p"])
    for k in ("open", "high", "low", "close"):
        df[k] = df["p"]
    df["volume"] = 100.0
    return df.drop(columns="p")


def cena_proboj(slots, day=8, vverh=True):
    """Дни 0–7 и после: 100; в день `day` на слотах из `slots` — 110 (вверх) или 90 (вниз)."""
    def f(i, s):
        return (110.0 if vverh else 90.0) if (i == day and s in slots) else 100.0
    return f


def sluchaynyj(ndays, seed):
    """Случайное блуждание с размахом внутри бара: для сверки с эталоном."""
    rng = np.random.default_rng(seed)
    p = 100 + np.cumsum(rng.normal(0, 0.3, ndays * ba.SLOTS))
    b = b10_iz(lambda i, s: 0.0, ndays)
    b["close"] = p
    b["open"] = np.r_[p[:1], p[:-1]]
    b["high"] = np.maximum(b["open"], b["close"]) + rng.uniform(0, 0.2, len(b))
    b["low"] = np.minimum(b["open"], b["close"]) - rng.uniform(0, 0.2, len(b))
    return b


DAYS = pd.bdate_range("2016-01-04", periods=12)
D8 = DAYS[8]
S1H = range(42, 48)      # 17:00–18:00
S16 = range(36, 48)      # 16:00–18:00
POSLE = range(48, 53)    # 18:00–18:50


def znach(b, sid, t, d=D8):
    return int(I.signal_dnya(b, sid, t).loc[d])


def test_g1_proboj_16_18_chasovoj_i_dvuhchasovoj():
    """Прорыв на 16:00–18:00: 2ч-бар 16:00–18:00 пробивает канал → S11 = +1; 1ч: прорыв был на баре 16:00–17:00, на
    последнем баре 17:00–18:00 канал уже включает его → S10 = 0; S12 (нужно согласие) = 0."""
    b = b10_iz(cena_proboj(S16), 12)
    assert (znach(b, "S11", "18:30"), znach(b, "S10", "18:30"), znach(b, "S12", "18:30")) == (1, 0, 0)


def test_g1_proboj_17_18_oba_srazu_i_soglasie():
    b = b10_iz(cena_proboj(S1H), 12)
    assert [znach(b, s, "18:30") for s in ("S10", "S11", "S12")] == [1, 1, 1]
    bd = b10_iz(cena_proboj(S1H, vverh=False), 12)
    assert [znach(bd, s, "18:30") for s in ("S10", "S11", "S12")] == [-1, -1, -1]


def test_s_ostalnye_dni_nol_i_indeks_po_dnyam():
    s = I.signal_dnya(b10_iz(cena_proboj(S1H), 12), "S10", "18:30")
    assert s.index[8] == D8 and s.loc[D8] == 1
    assert (s.drop(D8).iloc[:8] == 0).all()
    assert s.index.is_unique


def test_soglasie_protivopolozhnye_i_nol():
    i = pd.to_datetime(["2016-01-04", "2016-01-05", "2016-01-06", "2016-01-07"])
    a = pd.Series([1, 1, -1, 0], index=i)
    b = pd.Series([1, -1, -1, 1], index=i)
    assert I.soglasie(a, b).tolist() == [1, 0, -1, 0]


def test_t_sig_sdvigaet_polnye_bary():
    """T = 17:00: полных 1ч-баров 7 (последний 16:00–17:00), 2ч — 3 (последний 14:00–16:00)."""
    b = b10_iz(cena_proboj(range(36, 42)), 12)         # 110 на 16:00–17:00
    assert znach(b, "S10", "17:00") == 1                # 1ч-бар 16:00–17:00 полный и пробивает
    assert znach(b, "S11", "17:00") == 0                # последний полный 2ч-бар кончается в 16:00 → прорыва нет
    assert znach(b, "S10", "18:30") == 0                # к 18:30 бар 17:00–18:00 уже в канале с 16:00–17:00


def test_mutaciya_bar_posle_T_18_30_lovitsya():
    """Прорыв только на 18:00–18:50 (неполный/после T): ни один сигнал. Мутация «бар после T в сигнале» даёт +1."""
    b = b10_iz(cena_proboj(POSLE), 12)
    assert [znach(b, s, "18:30") for s in ("S10", "S11", "S12")] == [0, 0, 0]


def test_mutaciya_bar_posle_T_17_00_lovitsya():
    b = b10_iz(cena_proboj(S1H), 12)                    # прорыв 17:00–18:00 — после T = 17:00
    assert [znach(b, s, "17:00") for s in ("S10", "S11", "S12")] == [0, 0, 0]
    b2 = b10_iz(cena_proboj(range(42, 44)), 12)         # 17:00–17:20: бар 17:00–18:00 не закрыт к 17:30
    assert znach(b2, "S10", "17:30") == 0


def test_mutaciya_d_plus_1_lovitsya():
    """Прорыв только на следующий день (день 9): в день 8 сигнала нет, в день 9 есть. Мутация d+1 переносит его на день 8."""
    b = b10_iz(cena_proboj(S1H, day=9), 12)
    s = I.signal_dnya(b, "S10", "18:30")
    assert s.loc[DAYS[8]] == 0 and s.loc[DAYS[9]] == 1


@pytest.mark.parametrize("sid", ["S10", "S11", "S12"])
def test_sig_ne_zavisit_ot_budushchih_barov(sid):
    """Обрезка ленты после дня i не меняет сигнал дней ≤ i; дописанные после T бары дня d тоже не меняют сигнал дня d."""
    b = sluchaynyj(60, 7)
    full = I.signal_dnya(b, sid, "18:30")
    nz = np.flatnonzero(full.to_numpy() != 0)
    assert len(nz) >= 3
    for i in np.r_[nz[1:4], nz[1:4] - 1]:           # обрезка на дне сигнала и накануне: d+1 не должен просочиться
        d = full.index[i]
        cut = I.signal_dnya(b[b["date"] <= d], sid, "18:30")
        pd.testing.assert_series_equal(cut, full[full.index <= d])
    b2 = b.copy()
    m = b2["slot"] >= 48
    b2.loc[m, ["open", "high", "low", "close"]] *= 3.0
    pd.testing.assert_series_equal(I.signal_dnya(b2, sid, "18:30"), full)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_s11_pri_1830_raven_etalonu_pobajtno(seed):
    b = sluchaynyj(80, seed)
    ref = G.impuls_signal(b)
    got = I.signal_dnya(b, "S11", "18:30")
    assert ref.abs().sum() > 0                      # данные не вырожденные
    pd.testing.assert_series_equal(got, ref, check_exact=True)
    assert np.array_equal(got.to_numpy(), ref.to_numpy()) and got.index.equals(ref.index)


def test_pustaya_lenta_i_nepravilnye_argumenty():
    pusto = ba.prep_10m(pd.DataFrame({"timestamp": pd.to_datetime([]), "open": [], "high": [], "low": [], "close": [],
                                       "volume": []}))
    assert len(I.signal_dnya(pusto, "S10", "18:30")) == 0
    b = b10_iz(cena_proboj(S1H), 12)
    with pytest.raises(ValueError):
        I.signal_dnya(b, "S9", "18:30")
    with pytest.raises(ValueError):
        I.signal_dnya(b, "S10", "19:30")
    with pytest.raises(ValueError):
        I.signal_dnya(b, "S10", "18:xx")


def test_chtenie_ne_dalshe_2020():
    b = b10_iz(lambda i, s: 100.0, 3, start="2022-01-10")
    with pytest.raises(ValueError, match="2020"):
        I.signal_dnya(b, "S11", "18:30")
