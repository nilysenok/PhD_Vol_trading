"""evaluate_v2 на period="test1": кварталы своих годов 2022–2024, ledger не пишется; train4 с датой 2020 — ValueError.
Синтетика, данных и сети нет. python3 -m pytest -q tests/test_eval_v2_test1.py"""
import numpy as np
import pandas as pd
import pytest

from fastbt import etap1_eval_v2 as v2, ledger_v2, tejk_signal, zaem_v2

KEYS = ("sharpe", "msr", "sortino", "kalmar", "mdd", "T", "skew", "kurt", "prichiny", "by_year", "msr_by_year", "tselevaya",
        "sdelok_y", "storony_y", "dolya_ispolneniya_limita")


def ev(monkeypatch, dates):
    """EvaluatorV2 без данных: ряды подставлены, Pisatel (запись ledger) падает при любом обращении."""
    n = len(dates)
    x = np.random.default_rng(1).normal(0.0005, 0.01, n)
    e = object.__new__(v2.EvaluatorV2)
    e.comm, e.fdr, e._kray = 0.0004, None, False
    v1 = {"kod": "k", "vhody": "v", "vyrozhdena": False, "comm_pct": 0.04, **{k: 0.0 for k in KEYS}}
    monkeypatch.setattr(v2.EvaluatorV2, "evaluate", lambda self, pt, period: v1)
    monkeypatch.setattr(v2.EvaluatorV2, "_series", lambda self, period, comm: (x, np.zeros((17, n)), dates, {5: x, 10: x},
                                                                               np.ones(n), dates))
    monkeypatch.setattr(zaem_v2, "ryad", lambda ev_, period: None)
    monkeypatch.setattr(tejk_signal, "pole_otcheta", lambda ev_, yrs: None)

    def boom(*a, **k):
        raise AssertionError("запись в ledger")
    monkeypatch.setattr(ledger_v2.Pisatel, "__init__", boom)
    return e


def d2022_24():
    return pd.bdate_range("2022-01-03", "2024-12-27")


def test_test1_ne_padaet_i_ne_pishet(monkeypatch, tmp_path):
    row, x, tik = ev(monkeypatch, d2022_24()).evaluate_v2({}, "test1")
    assert row["status"] == "ok" and len(row["by_quarter"]) == 12
    assert set(row["by_quarter"]) == {f"{y}Q{q}" for y in (2022, 2023, 2024) for q in range(1, 5)}
    assert not np.isnan(list(row["by_quarter"].values())).any() and len(row["by_quarter_zaem"]["5"]) == 12
    assert list(tmp_path.iterdir()) == []


def prover_train4_2020(monkeypatch):
    dates = pd.bdate_range("2019-12-02", "2020-02-28")
    with pytest.raises(ValueError, match="2016–2019"):
        ev(monkeypatch, dates).evaluate_v2({}, "train4")


def test_train4_2020_po_prezhnemu_valueerror(monkeypatch):
    prover_train4_2020(monkeypatch)


def test_mutatsiya_zashchita_snyata_lovitsya(monkeypatch):
    """Мутация: GODY расширен (защита train4 снята) — проверка выше обязана упасть."""
    with monkeypatch.context() as m:
        m.setattr(v2, "kvartaly", lambda d, v: ledger_v2.kvartaly_gody(d, v, tuple(range(2016, 2025))))
        with pytest.raises(pytest.fail.Exception):
            m2 = pytest.MonkeyPatch()
            try:
                prover_train4_2020(m2)
            finally:
                m2.undo()


def test_kvartaly_po_umolchaniyu_prezhnyaya_zashchita():
    with pytest.raises(ValueError):
        ledger_v2.kvartaly(np.array(["2022-01-03"], "datetime64[D]"), np.array([0.1]))
