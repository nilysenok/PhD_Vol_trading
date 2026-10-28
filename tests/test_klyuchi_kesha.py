# -*- coding: utf-8 -*-
"""Д5: комиссия в ключе ledger и в digest входов; numba-кэш по хэшу кода. Синтетика, данных и счёта нет.
Мутации: «комиссия не в ключе» (ledger, digest), «каталог numba-кэша без хэша кода» — каждая ловится проверкой."""
from __future__ import annotations

import pickle

import pandas as pd

from fastbt import cache
from fastbt import etap1_poisk_vorota as V
from fastbt import s9
from fastbt.etap1 import E1Inputs

P = {"k_s": 1.0, "R": 2.0, "tau": 5}


def _row(v: float) -> dict:
    return dict(msr_by_year={2018: v}, sdelok_y={2018: 40}, vyrozhdena=False, excess_d={2018: [v, -v]})


def _counting(comm: float):
    calls = []

    def evaluate(params, period):
        calls.append(period)
        return _row(comm)
    evaluate.comm = comm
    return evaluate, calls


def ledger_tracks_comm(cls) -> bool:
    """Один ledger-объект на комиссию; общий кэш-словарь: другая комиссия при тех же params обязана дать промах."""
    ev04, c04 = _counting(0.0004)
    ev035, c035 = _counting(0.00035)
    a, b = cls(ev04), cls(ev035)
    b.cache = a.cache
    a(P, "train")
    b(P, "train")
    return len(c04) == 1 and len(c035) == 1 and a.n_unique == 2


class _LedgerNoComm(V.Ledger):          # мутация: комиссия не в ключе
    def __init__(self, evaluate, comm=None):
        super().__init__(evaluate, 0.0004)


def test_ledger_commission_in_key():
    assert ledger_tracks_comm(V.Ledger)


def test_ledger_commission_mutation_caught():
    assert not ledger_tracks_comm(_LedgerNoComm)


def test_ledger_same_commission_hits():
    ev, calls = _counting(0.0004)
    led = V.Ledger(ev)
    led(P, "train")
    led(P, "train")
    assert len(calls) == 1


def test_old_ledger_reads_as_004_bytewise():
    """G3 (синтетика): запись старого формата (ключ без комиссии) читается как 0,04 %: попадание в кэш, итог прежний."""
    old = {(V.pkey(P), "train"): (V.canon(P), _row(0.7))}
    raw = pickle.dumps(old, protocol=4)
    ev, calls = _counting(0.0004)
    led = V.Ledger(ev)
    led.cache = pickle.loads(raw)
    assert led(P, "train") == _row(0.7) and calls == []
    assert pickle.dumps(led.cache, protocol=4) == raw
    ev2, calls2 = _counting(0.00035)
    led2 = V.Ledger(ev2)
    led2.cache = pickle.loads(raw)
    led2(P, "train")
    assert calls2 == ["train"] and led2.n_unique == 2         # строка 0,04 % не подменяет 0,035 %


def _inp(comm: float) -> E1Inputs:
    idx = pd.date_range("2018-01-01", periods=3)
    return E1Inputs(sk=None, sigma={}, bars={}, close_n6={}, rezhim={}, sobytiya={}, tick={"A": 0.01},
                    rate=pd.Series([0.1, 0.1, 0.1], index=idx),
                    meta={"A": dict(dates=idx.values)}, commission=comm)


def digest_tracks_comm(make) -> bool:
    return make(0.0004).digest() != make(0.00035).digest() and make(0.0004).digest() == make(0.0004).digest()


def test_digest_commission_in_key():
    assert digest_tracks_comm(_inp)


def test_digest_commission_mutation_caught():
    assert not digest_tracks_comm(lambda c: _inp(0.0004))       # мутация: комиссия не доходит до входов


def numba_dir_tracks_code(fn) -> bool:
    old = cache._CODE
    try:
        cache._CODE = "a" * 40
        d1 = fn()
        cache._CODE = "b" * 40
        d2 = fn()
    finally:
        cache._CODE = old
    return d1 != d2


def test_numba_cache_dir_tracks_code_hash():
    assert numba_dir_tracks_code(s9.numba_cache_dir)
    assert s9.numba_cache_dir().name == cache.code_hash()[:16]


def test_numba_cache_dir_mutation_caught():
    assert not numba_dir_tracks_code(lambda: s9.MAIN / "data" / "fastbt" / "numba_cache")      # каталог без хэша


def test_numba_cache_dir_changes_with_source(tmp_path, monkeypatch):
    """Правка исходника в fastbt/ → другой code_hash → другой каталог (на копии пакета, основное дерево не трогаем)."""
    (tmp_path / "cache.py").write_text("", encoding="utf-8")
    m = tmp_path / "m.py"
    m.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(cache, "__file__", str(tmp_path / "cache.py"))
    monkeypatch.setattr(cache, "_CODE", None)
    d1 = s9.numba_cache_dir()
    m.write_text("x = 2\n", encoding="utf-8")
    monkeypatch.setattr(cache, "_CODE", None)
    assert s9.numba_cache_dir() != d1
