"""Д5-2: `dates` доходит от оценщиков до движка — синтетика с разрывом > 5 дней, заявка снимается.
Мутация «dates не передаются из оценщика» обязана падать: сквозные тесты (etap1_run, eval_v5.run_rows, TEvaluator) и AST-проверка вызовов
в оценщиках с данными (etap1_eval, _v2, _plus, _ansambl, eval_v5, sem_e, sem_e_strogiy), где сквозной прогон без данных невозможен.
Исключения (dates не нужны): fastbt/diag_*.py, scripts/analysis/* — разовые диагностики, считают на фиксированных окнах ≤ 2021 без разрывов > 5 дней."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import etap1_run, eval_v5, exec_e
from fastbt.exec_e_types import Params
from fastbt.t_eval import TEvaluator, TInput
from test_exec_e import D, F

ROOT = Path(__file__).resolve().parents[1] / "fastbt"
DAYS = [D(F(100.0)), D([(99, 99, 95, 96)]), D(F(100.0)), D(F(100.0))]


def _dates(gap):
    return pd.DatetimeIndex(["2022-02-25", pd.Timestamp("2022-02-25") + pd.Timedelta(days=gap), "2022-03-30", "2022-03-31"])


def _sig():
    return np.array([1, 0, 0, 0]), np.full(4, 0.02)


def test_etap1_run_dates_dohodyat_do_dvizhka():
    sig, sg = _sig()
    p = Params(k=1.0, comm=0.0)
    assert etap1_run.run_e1("T", DAYS, sig, sg, p, dates=_dates(26)).trades == []
    assert len(etap1_run.run_e1("T", DAYS, sig, sg, p, dates=_dates(3)).trades) == 1
    assert len(etap1_run.run_e1("T", DAYS, sig, sg, p).trades) == 1         # без dates — прежнее (мутация «не передано» выглядит так)


def test_eval_v5_run_rows_rang_put_dates_dohodyat(monkeypatch):
    seen = {}
    real = exec_e.run

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)
    monkeypatch.setattr(eval_v5.exec_e, "run", spy)
    sig, sg = _sig()
    eval_v5.run_rows("T", DAYS, sig, sg, Params(k=1.0, comm=0.0), rang=np.full(4, 0.5), dates=_dates(26))
    assert seen["dates"] is not None and len(seen["dates"]) == 4


def test_eval_v5_run_rows_bez_rang_dates_cherez_run_e1():
    sig, sg = _sig()
    r = eval_v5.run_rows("T", DAYS, sig, sg, Params(k=1.0, comm=0.0), dates=_dates(26))
    assert r.trades == []


def test_t_eval_peredayot_dates(monkeypatch):
    seen = {}
    real = etap1_run.run_e1
    import fastbt.t_eval as te

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)
    monkeypatch.setattr(te, "run_e1", spy)
    n = 6
    cl = np.full(n, 100.0)
    dts = pd.bdate_range("2022-02-14", periods=n)
    x = TInput([D(F(100.0)) for _ in range(n)], cl, np.full(n, 0.02), 0.01, 0, n - 1, dts)

    class Eng:
        pass
    TEvaluator({"X": x}, Eng(), 0.0004)._one("X", [(1, 1, 3.5)], 2, 3.0, None, False, 1)
    assert seen["dates"] is not None and len(seen["dates"]) == n


@pytest.mark.parametrize("f", ["etap1_eval", "etap1_eval_v2", "etap1_plus_eval", "etap1_ansambl_eval", "eval_v5", "sem_e", "sem_e_strogiy"])
def test_ocenshchiki_peredayut_dates_ast(f):
    tree = ast.parse((ROOT / f"{f}.py").read_text(encoding="utf-8"))
    ok = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and any(k.arg == "dates" for k in n.keywords)]
    assert ok, f"{f}: нет ни одного вызова с dates= (Д5-2: dates не передаются из оценщика)"
