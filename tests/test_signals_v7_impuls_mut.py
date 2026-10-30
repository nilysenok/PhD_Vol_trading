"""Тестировщик #848: непойманные мутации Л1.5 и характеризация. Только синтетика, в сеть и на данные не ходит.

Найдено: мутация «T + 10 мин» неразличима (ТФ 1ч/2ч квантуют T по часам) — для S10 граница T=17:00 ловится
только баром 17:00–18:00 (здесь); мутация «бар, начинающийся в T, виден» — тест S10 ниже.
Характеризация: устаревший выход прежнего сигнала закрывает позицию более позднего (см. последний тест).
"""
import numpy as np
import pandas as pd

from fastbt import bars_agg as ba
from fastbt import impuls_v7 as I
from fastbt.signals import interface as IF
from fastbt.signals import v7_impuls as V
from fastbt.tf_ispolnenie_v5 import sostoyanie

ND = 260
START = "2016-01-04"
DAYS = pd.bdate_range(START, periods=ND)
DAY = 230
S_17_18 = range(42, 48)      # 17:00–18:00: ровно один 1ч-бар, начинающийся в T=17:00


def lenta(zadano):
    """zadano: {индекс дня: (слоты, цена)}; остальное — 100."""
    rows = []
    for i, d in enumerate(DAYS):
        for s in range(ba.SLOTS):
            p = 100.0
            if i in zadano and s in zadano[i][0]:
                p = zadano[i][1]
            rows.append((d, s, p, p, p, p, 100.0))
    return pd.DataFrame(rows, columns=["date", "slot", "open", "high", "low", "close", "volume"])


def td_iz(n=ND):
    d = DAYS[:n].to_numpy("datetime64[ns]")
    c = np.full(n, 100.0)
    return IF.TickerData("AAA", d, c, c, c, c, np.full(n, 1e3), np.zeros(n))


def gen(b, sid, pts, n=ND):
    return V.semya(sid, {"AAA": b})(td_iz(n), {}, pts)


def tk(vhod="T", T="17:00", tau=1):
    return {"tau": tau, "vhod": vhod, "T": T}


def test_s10_bar_nachinayushchijsya_v_T_ne_vidim():
    b = lenta({DAY: (S_17_18, 110.0)})
    assert gen(b, "S10", [tk("T", "17:00")]).sig[0, DAY] == 0
    assert gen(b, "S10", [tk("T", "18:00")]).sig[0, DAY] == 1


def test_mutaciya_T_plyus_chas_lovitsya(monkeypatch):
    """Мутант: сигнал считается на час позже T. Тест s10_bar_… на нём обязан дать иное."""
    b = lenta({DAY: (S_17_18, 110.0)})
    nastoyashchaya = I.signal_dnya

    def mutant(b10, sid, t_sig="18:30"):
        h, m = (int(x) for x in t_sig.split(":"))
        return nastoyashchaya(b10, sid, f"{min(h + 1, 18):02d}:{m:02d}" if h < 18 else t_sig)

    monkeypatch.setattr(V.I, "signal_dnya", mutant)
    assert gen(b, "S10", [tk("T", "17:00")]).sig[0, DAY] == 1      # мутант виден => основной тест упал бы


def test_exit_na_poslednem_bare_ryada():
    """Сигнал в ND-1-τ → exit ровно на последнем баре (граница включена)."""
    for tau in (1, 2, 3):
        b = lenta({ND - 1 - tau: (S_17_18, 110.0)})
        ss = gen(b, "S10", [tk("T", "18:00", tau)])
        assert np.flatnonzero(ss.exit_l[0]).tolist() == [ND - 1] and ss.sig[0, ND - 1 - tau] == 1


def test_strength_raven_modulyu_signala():
    b = lenta({DAY: (S_17_18, 90.0)})
    ss = gen(b, "S10", [tk("T", "18:00")])
    # плоская лента (ATR = 0): сила не определена → strength 0 и флаг sila_neopr (V7-5, Д4), сторона прежняя
    assert ss.sig[0, DAY] == -1 and ss.strength[0, DAY] == 0.0 and ss.strength.dtype == np.float64
    assert ss.extra["sila_neopr"][0, DAY] and not np.isnan(ss.strength).any()


def test_tochki_ne_vliyayut_drug_na_druga():
    """Порядок точек не меняет строки (кэш по t_sig не течёт между точками)."""
    b = lenta({DAY: (S_17_18, 110.0)})
    p = [tk("T", "17:00", 2), tk("T", "18:00", 2), tk("d1", "17:00", 3)]
    a, r = gen(b, "S10", p), gen(b, "S10", p[::-1])
    assert np.array_equal(a.sig, r.sig[::-1]) and np.array_equal(a.exit_l, r.exit_l[::-1])


def test_harakteristika_ustarevshij_vyhod_zakryvaet_pozdnyuyu_poziciyu():
    """Характеризация (не оценка): сигналы d и d+1, τ=2. Выход d+2 — от сигнала d; в автомате позиция, открытая
    сигналом d+1, закрывается на d+2, то есть держится 1 день вместо τ=2. Замечание для волны 5, а не блокер."""
    b = lenta({DAY: (S_17_18, 110.0), DAY + 1: (S_17_18, 120.0)})
    ss = gen(b, "S10", [tk("T", "18:00", 2)])
    assert ss.sig[0, DAY] == ss.sig[0, DAY + 1] == 1
    assert np.flatnonzero(ss.exit_l[0]).tolist() == [DAY + 2, DAY + 3]
    st = sostoyanie(ss.sig[0], ss.exit_l[0], ss.exit_s[0])
    assert st[DAY + 1] == 1 and st[DAY + 2] == 0          # позиция d+1 закрыта через 1 день
