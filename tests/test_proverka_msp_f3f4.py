"""Проверяющий #505/#506: тесты на мутации, которые тесты авторов не поймали. Без сети и реальных данных.

1. М-Ф10: терцили f1/f2 считаются по окну с днём d (убран shift(1) в `intraday_priznaki.tercile_cell`).
2. Нетто: срабатывание шага 8 при нетто между 30 % и 60 % валовой (порог срабатывания расходится с целью).
3. Самоотключение в `vesa_sk` по умолчанию — два квартала подряд (по одному — нельзя): путь `kniga_dnya` → `vesa_sk`.
"""
import numpy as np
import pandas as pd
import pytest

from fastbt import kniga_msp as km
from fastbt import reyting


def test_g_f_porog_terciliya_bez_dnya_d():
    # окно d−252…d−1: 0..251 по возрастанию; верхняя граница терциля 2/3 ≈ 167,3. x_d = 170 > hi (без d) -> верхний.
    # Если день d попал в окно, граница сдвигается вверх (окно 1..251,170: квантиль 2/3 ≈ 168,3 — всё ещё < 170),
    # поэтому берём x_d на самой границе: 167,5 — выше границы без d и ниже границы с d.
    n, d = 300, 260
    x = pd.Series(np.arange(n, dtype=float), index=pd.bdate_range("2016-01-04", periods=n))
    x.iloc[d - 252:d] = np.arange(252, dtype=float)
    hi_bez_d = x.iloc[d - 252:d].quantile(2 / 3)
    x.iloc[d] = hi_bez_d + 0.1
    s_hi = x.iloc[d - 251:d + 1].quantile(2 / 3)
    assert s_hi > x.iloc[d]                              # тест не пустой: с днём d граница выше значения
    lg = pd.Series(1.0, index=x.index)
    # f2 (znak=+1), сторона = −1: шорт против верхнего терциля -> 0,5; если терциль «средний» (день d в окне) -> 1,0
    assert reyting.g_f(-lg, x, +1).iloc[d] == 0.5


def test_netto_srabatyvaet_mezhdu_30_i_60_procentov():
    u = km.netto(np.array([2.0, -1.0]))                  # нетто 1/3 валовой > 30 %
    L, S = u[u > 0].sum(), -u[u < 0].sum()
    assert abs(L - S) == pytest.approx(0.3 * (L + S))
    assert S == pytest.approx(1.0)                       # уменьшается доминирующая сторона, малая не трогается


def _ex_sk_dva_stolbca(T=520):
    rng = np.random.default_rng(11)
    plohoj = -0.001 + rng.normal(0, 0.005, T)            # SR ≈ −3 на любом окне -> < −0,5 на каждом конце квартала
    horoshij = 0.001 + rng.normal(0, 0.005, T)
    return np.column_stack([plohoj, horoshij]), np.arange(T) // 63


def test_vesa_sk_otklyuchenie_tolko_posle_dvuh_kvartalov_po_umolchaniyu():
    ex, kv = _ex_sk_dva_stolbca()
    # первый конец квартала с готовым SR (252 строки): t = 315 -> sr[314]; второй: t = 378
    w_odin = km.vesa_sk(ex, kv, 320)                     # один плохой конец -> ×1
    w_dva = km.vesa_sk(ex, kv, 380)                      # два подряд -> ×0,25 с первого дня следующего квартала
    assert w_odin.sum() == pytest.approx(1.0)
    assert w_dva.sum() < 1.0 - 1e-9
