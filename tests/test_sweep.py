"""Заглушка-минимум для хука остановки (Ж-031 Э1-а): тестировщик заменяет файл своим набором.
Проверяет оценщик на синтетике: контракт ключей, периоды, закрытость validation, зависимость от параметра."""
import pytest

from fastbt import sweep_stat as ss
from fastbt.sweep import Sweep
from fastbt.sweep_spec import Spec
from fastbt.sweep_synth import SynthEvaluator


def mk(**kw):
    return Sweep(Spec(obj={"kind": "Synth"}, base={"a": 5, "w": 1.0}, **kw), SynthEvaluator())


def test_row_has_driver_contract_keys():
    r = mk().evaluate({}, "train")
    for k in ("sharpe", "by_year", "msr", "sortino", "mdd", "T", "skew", "kurt", "N", "comm_pct"):
        assert k in r
    assert set(r["by_year"]) == {2018, 2019}


def test_param_changes_result_and_commission_lowers_sharpe():
    s = mk()
    assert s.evaluate({"a": 3}, "train")["sharpe"] != s.evaluate({"a": 40}, "train")["sharpe"]
    assert s.evaluate({}, "train", 0.04)["sharpe"] < s.evaluate({}, "train", 0.035)["sharpe"]


def test_validation_closed_and_periods_disjoint():
    with pytest.raises(PermissionError):
        mk().evaluate({}, "validation")
    with pytest.raises(ValueError):
        ss.Periods(setup=(2018, 2020), validation=(2020, 2021))
