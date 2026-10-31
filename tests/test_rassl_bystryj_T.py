"""Расследование 31.10: быстрый слой v6 ≠ эталонный путь (v6bystr__S1__15: T Мак 1010, здесь None).
Причина: eval_v5.run_rows (b97c211c, лист A S8–S12) передавал в exec_e.run `cena_T=None` всегда, когда есть rang; подменённый
быстрым слоем `fast_etap1.run` параметра cena_T не знает -> TypeError -> status oshibka -> T = None. Без сети, без данных."""
import inspect

import numpy as np
import pytest

from fastbt import eval_v5, exec_e, fast_etap1
from fastbt.exec_e_types import Params


class _Stop(Exception):
    pass


def _shpion(sig_f, polucheno):
    def f(*a, **k):
        polucheno.append(k)
        inspect.signature(sig_f).bind(*a, **k)      # вызов обязан подходить под сигнатуру целевой функции
        raise _Stop
    return f


def _vyzov(monkeypatch, sig_f, **kw):
    polucheno = []
    monkeypatch.setattr(exec_e, "run", _shpion(sig_f, polucheno))
    n = 3
    with pytest.raises(_Stop):
        eval_v5.run_rows("T1", [None] * n, np.zeros(n), np.ones(n), Params(), rang=np.zeros(n), close=np.ones(n), **kw)
    return polucheno[0]


def test_rang_bez_cena_T_podhodit_pod_signaturu_bystrogo_sloya(monkeypatch):
    """rang есть, cena_T нет (все точки v6): вызов exec_e.run — ровно как до b97c211c, быстрый run его принимает."""
    k = _vyzov(monkeypatch, fast_etap1.run)
    assert "cena_T" not in k


def test_cena_T_peredayotsya_kogda_zadana(monkeypatch):
    """Вход в T (v7): cena_T доходит до exec_e.run (эталонный путь), массивом, а не теряется."""
    c = np.arange(3.0)
    k = _vyzov(monkeypatch, _zapasnoj_run, cena_T=c)
    assert np.array_equal(k["cena_T"], c)


def _zapasnoj_run(*a, cena_T=None, **k):
    pass
