"""Тестировщик #714: ось lestnica (две ноги обычных вызовов run, eval_v6_ispoln._legs) против эталона на синтетике.
Нога = run_e1 с весом 1/2 и входом k0 | k0 + lest; ноги суммирует etap1_run.sum_rows. Быстрый путь (fast_etap1.run вместо exec_e.run)
обязан дать побайтно те же строки E1Rows и те же сделки, что медленный. Мутация «нога потеряна» обязана ломать сверку.
FASTBT_V6_LEST_N — число путей (по умолчанию 300). Данных не читает, в сеть не ходит."""
import dataclasses
import os

import numpy as np

from fastbt import exec_e, fast_etap1 as fe, fast_exec as fx
from fastbt.etap1_run import run_e1, sum_rows
from test_fast_v6_osi import lenta, tochka

N_PUT = int(os.environ.get("FASTBT_V6_LEST_N", "300"))
LEST = (0.5, 1.0)
KOMBI = ({}, dict(gap_propusk=True), dict(zhizn=12), dict(zhizn=36), dict(t_vyhod="1500"), dict(t_vyhod="1700"), dict(ed=True),
         dict(gap_propusk=True, zhizn=60), dict(zhizn=12, t_vyhod="1700"), dict(gap_propusk=True, t_vyhod="1500", ed=True))


def _lest_rows(run, s, teryat=0):
    """Строки лесенки пути s: sum_rows ног (вес 1/2, вход k0, k0 + lest); teryat = 1 — вторая нога потеряна. Ошибка ядра — строка."""
    kw = dict(KOMBI[s % len(KOMBI)])
    ed_on = kw.pop("ed", False)
    p0 = tochka(s, **kw)
    days, sig, sigma, close, div, ed = lenta(70_000 + s)
    w, legs = np.full(len(days), 0.5), []
    exec_e_run = exec_e.run
    try:
        exec_e.run = run                                            # run_e1 берёт exec_e.run по имени из модуля
        for k in (p0.k, p0.k + LEST[s % 2]):
            legs.append(run_e1("T", days, sig, sigma, dataclasses.replace(p0, k=k), close=close, w=w, div=div, ed=ed if ed_on else None))
    except ValueError as e:
        return str(e)
    finally:
        exec_e.run = exec_e_run
    return sum_rows(legs[:len(legs) - teryat])


def _eq_rows(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    for f in ("g", "sides", "expo", "cnt", "short"):
        if getattr(a, f).tobytes() != getattr(b, f).tobytes():
            return False
    k = ("n_orders", "n_fills", "x_orders", "x_fills", "n_bez_tejka")
    return ([getattr(a, x) for x in k] == [getattr(b, x) for x in k]
            and [dataclasses.astuple(t) for t in a.trades] == [dataclasses.astuple(t) for t in b.trades])


def _sverka(n, fast_run, teryat=0):
    """(плохие пути, число путей с ненулевым результатом, число путей, где вторая нога меняет результат против одной ноги)."""
    plokho, zhivye, deistvuet = [], 0, 0
    for s in range(n):
        ref = _lest_rows(exec_e.run, s)
        if not _eq_rows(ref, _lest_rows(fast_run, s, teryat)):
            plokho.append(s)
            continue
        if not isinstance(ref, str):
            zhivye += bool(ref.trades)
            deistvuet += not _eq_rows(ref, _lest_rows(exec_e.run, s, 1))
    return plokho, zhivye, deistvuet


def test_lestnica_yadro_ravno_etalonu_pobajtno():
    calls, orig = [], fx.run_v5
    fx.run_v5 = lambda *a, **k: calls.append(1) or orig(*a, **k)
    try:
        plokho, zhivye, deist = _sverka(N_PUT, fe.run)
    finally:
        fx.run_v5 = orig
    assert not plokho, plokho[:10]
    assert zhivye >= N_PUT // 2 and deist >= N_PUT // 3, (zhivye, deist)    # лента живая, ноги реально различаются
    assert len(calls) >= N_PUT, len(calls)                                   # ядро, а не откат на медленный путь


def test_mutatsiya_noga_lestnicy_poteryana_lovitsya():
    plokho, _, _ = _sverka(N_PUT, fe.run, teryat=1)
    assert len(plokho) >= N_PUT // 3, len(plokho)


def test_mutatsiya_os_poteryana_v_lestnice_lovitsya():
    """Ось gap_propusk не доходит до ядра внутри лестницы (ноги считаются без неё) — сверка обязана упасть."""
    bez_gap = lambda t, d, s, sg, p, **k: fe.run(t, d, s, sg, dataclasses.replace(p, gap_propusk=False), **k)   # noqa: E731
    plokho, _, _ = _sverka(N_PUT, bez_gap)
    assert len(plokho) >= 5, len(plokho)
