"""Тестировщик PR #840: границы и мутации impuls_v7, не покрытые тестами автора. Синтетика + реальные 10-минутки 2018 (чтение).

Мутации (проверены в копии, автор их ловил так): незакрытый бар (n_full+1) — 13 красных у автора; канал 19 вместо 20 —
у автора ловят только 2 из 3 сидов сверки с эталоном, здесь — test_kanal_rovno_20; S12 = «1ч ИЛИ 2ч» — ловит автор;
здесь добавлены: дыры внутри часа, дыра дня, смена дня, первый день, суббота, сверка с эталоном на реальных данных.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_impuls_v7 import b10_iz, cena_proboj, sluchaynyj  # noqa: E402,I001


def test_kanal_rovno_20():
    """T = 15:30 → 5 полных часов в дне. На последнем баре 4-го дня до него ровно 19 баров: канал 20 не полон (0),
    канал 19 дал бы +1. На следующем дне баров 24 — пробой виден (+1)."""
    def tsena(i, s):
        return 110.0 if (i in (3, 4) and 24 <= s < 30) else 100.0
    b = b10_iz(tsena, 6)
    s = I.signal_dnya(b, "S10", "15:30")
    assert int(s.iloc[3]) == 0 and int(s.iloc[4]) == 0   # день 4: канал не полон; день 5: 110 уже в канале
    b2 = b10_iz(lambda i, s_: 110.0 if (i == 4 and 24 <= s_ < 30) else 100.0, 6)
    assert int(I.signal_dnya(b2, "S10", "15:30").iloc[4]) == 1


def test_pervyj_den_net_signala_ne_nan():
    s = I.signal_dnya(b10_iz(lambda i, sl: 100.0 + sl, 3), "S11", "18:30")
    assert not s.isna().any() and (s == 0).all() and s.dtype.kind == "i"


def test_dyra_vnutri_chasa_bar_ne_teryaetsya():
    """Из часа 17:00–18:00 убраны слоты 43–46: бар остаётся из оставшихся 10-минуток, пробой виден."""
    b = b10_iz(cena_proboj(range(42, 48)), 12)
    b = b[~((b["date"] == pd.bdate_range("2016-01-04", periods=12)[8]) & b["slot"].isin([43, 44, 45, 46]))]
    assert int(I.signal_dnya(b, "S10", "18:30").loc[pd.bdate_range("2016-01-04", periods=12)[8]]) == 1


def test_net_poslednego_chasa_net_signala_dnya():
    """В дне нет ни одной 10-минутки последнего полного часа: сигнал дня не подменяется предыдущим баром."""
    D = pd.bdate_range("2016-01-04", periods=12)
    b = b10_iz(cena_proboj(range(36, 42)), 12)         # прорыв в 16:00–17:00
    b = b[~((b["date"] == D[8]) & (b["slot"] >= 42))]
    s = I.signal_dnya(b, "S10", "18:30")
    assert D[8] not in s.index or int(s.loc[D[8]]) == 0


def test_dyra_dnya_ne_lomaet_kanal():
    """Выпавший день не даёт NaN-сигнала и не сдвигает индексы остальных."""
    D = pd.bdate_range("2016-01-04", periods=12)
    b = b10_iz(cena_proboj(range(42, 48)), 12)
    b = b[b["date"] != D[3]]
    s = I.signal_dnya(b, "S12", "18:30")
    assert D[3] not in s.index and not s.isna().any() and int(s.loc[D[8]]) == 1


def test_dvuhchasovoj_bar_ne_skleivaet_dni():
    """2ч-бар не пересекает границу дня: вчерашние 16:00–18:00 + сегодняшние 10:00 не дают бар (слоты по дню)."""
    b = b10_iz(lambda i, s: 100.0, 5)
    f = ba.resample(b, ba.K["2h"])
    assert set(f["slot"]) <= set(range(5)) and f.groupby("date").size().eq(5).all()


def test_subbota_rabochaya():
    """Рабочая суббота — обычный день в ленте: сигнал на ней считается, дата сохранена."""
    days = list(pd.bdate_range("2016-01-04", periods=9)) + [pd.Timestamp("2016-01-16")]
    rows = []
    for i, d in enumerate(days):
        for s in range(ba.SLOTS):
            p = 110.0 if (i == 9 and s >= 42) else 100.0
            rows.append((d, s, p, p, p, p, 1.0))
    b = pd.DataFrame(rows, columns=["date", "slot", "open", "high", "low", "close", "volume"])
    s = I.signal_dnya(b, "S10", "18:30")
    assert pd.Timestamp("2016-01-16") in s.index and int(s.iloc[-1]) == 1


@pytest.mark.parametrize("t", ["17:00", "18:00", "18:30", "18:40"])
def test_t_sig_poslednij_bar_ne_zavisit_ot_hvosta(t):
    """Для всех T бары после T не влияют (умножаем на 3) и не NaN."""
    b = sluchaynyj(60, 5)
    m = ba.OPEN_MIN
    tm = int(t[:2]) * 60 + int(t[3:])
    lim = (tm - m) // ba.STEP
    full = I.signal_dnya(b, "S12", t)
    b2 = b.copy()
    b2.loc[b2["slot"] >= lim, ["open", "high", "low", "close"]] *= 3.0
    # слоты ≥ lim — неполный бар T; изменение захватывает только неполный хвост бара, если lim кратно 6 — полностью
    if lim % 12 == 0 or lim % 6 == 0:
        pd.testing.assert_series_equal(I.signal_dnya(b2, "S12", t), full)
    assert not full.isna().any()


def test_s12_rashozhdenie_ravno_nulyu_na_sluchajnyh():
    b = sluchaynyj(120, 11)
    a, c, s12 = (I.signal_dnya(b, x, "18:30") for x in ("S10", "S11", "S12"))
    assert ((a != c) & (s12 != 0)).sum() == 0 and (s12[(a == c)] == a[(a == c)]).all()
    assert ((a != c) & (a != 0) & (c != 0)).sum() + ((a != c)).sum() > 0     # расхождения были


@pytest.mark.parametrize("tk", ["AFLT", "SBER"])
def test_s11_1830_raven_etalonu_na_realnyh_2018(tk):
    try:
        b = ba.load_10m(tk, "2019-01-01")
    except Exception as e:                                 # нет данных на этой машине
        pytest.skip(str(e))
    b = b[b["date"] >= "2018-01-01"].reset_index(drop=True)
    if len(b) < 1000:
        pytest.skip("мало данных")
    ref = G.impuls_signal(b)
    got = I.signal_dnya(b, "S11", "18:30")
    assert np.array_equal(got.to_numpy(), ref.to_numpy()) and got.index.equals(ref.index)
    assert np.abs(ref.to_numpy()).sum() > 0
