"""Э1-а (Ж-034/035): оценщик движка (`fastbt.sweep.Sweep.evaluate`) на синтетике `fastbt.sweep_synth`, известный ответ.
Детекторы `ok_*` возвращают True, если свойство выполнено; их роняют мутации в `test_sweep_mutacii.py`.
Реальных данных не читает (≥ 2022 — тем более)."""
import json

import numpy as np
import pandas as pd
import pytest

from fastbt import sweep_stat as ss
from fastbt.sweep import Sweep
from fastbt.sweep_spec import Spec
from fastbt.sweep_synth import SynthEvaluator

BASE = {"a": 5, "w": 1.0}
P = dict(BASE)
A_GRID = [3, 5, 10, 20, 40]


def mk(ev=None, cache_dir=None, **kw):
    return Sweep(Spec(obj={"kind": "Synth"}, base=dict(BASE), **kw), ev or SynthEvaluator(), cache_dir)


class Tiny:
    """Нулевые доходности, позиция известна: +1 (дни 5–9), −1 (дни 20–24) -> 4 стороны сделок. excess = −сторон·comm."""
    sig_names = {"a"}

    def __init__(self, end_log=None):
        self.dates = pd.bdate_range("2018-01-01", "2021-12-31")
        self.end_log = end_log if end_log is not None else []

    def series_digest(self): return "t"
    def code_id(self): return "t"

    def signals(self, sp):
        s = np.zeros(len(self.dates))
        s[5:10], s[20:25] = 1.0, -1.0
        return s

    def pnl(self, sigs, p, comm, end):
        self.end_log.append(end)
        m = self.dates < end
        pos = sigs[m]
        d = np.abs(np.diff(pos, prepend=0.0))
        return pd.DataFrame(dict(excess=-d * comm, exposure=np.abs(pos), turnover=d, sides=(d > 0) * 1.0), index=self.dates[m])


# ---------- детекторы ----------
def ok_param_signal() -> bool:
    sw = mk()
    return sw.evaluate({"a": 3})["sharpe"] != sw.evaluate({"a": 40})["sharpe"]


def ok_param_pnl() -> bool:
    sw = mk()
    return sw.evaluate({"w": 0.5})["sharpe"] != sw.evaluate({"w": 2.0})["sharpe"]


def ok_validation_independent() -> bool:
    """Строка на train не зависит от доходностей 2020–2021 (late меняет только их)."""
    r = [mk(SynthEvaluator(late=k)).evaluate(P) for k in (1.0, 6.0)]
    return r[0]["sharpe"] == r[1]["sharpe"] and r[0]["by_year"] == r[1]["by_year"]


def ok_validation_closed() -> bool:
    try:
        mk().evaluate(P, "validation")
    except PermissionError:
        return True
    return False


def ok_2022_closed() -> bool:
    try:
        mk()._df(P, 0.0004, (2022,))
    except PermissionError:
        return True
    return False


def ok_end_cut() -> bool:
    """В оценщик на train запрашиваются данные строго до 01.01.2020."""
    log = []
    mk(Tiny(log)).evaluate(P)
    return bool(log) and max(log) == pd.Timestamp("2020-01-01")


def ok_periods_raise(**kw) -> bool:
    try:
        ss.Periods(**kw)
    except ValueError:
        return True
    return False


def ok_commission_two_sides() -> bool:
    sw = mk(Tiny())
    for pct in (0.035, 0.04):
        c = pct / 100
        if not np.isclose(sw._df(P, c, (2018, 2019))["excess"].sum(), -4 * c, atol=1e-12):
            return False
        if not np.isclose(sw.evaluate(P, "train", pct)["comm_pct"], pct):
            return False
    net0 = sw._df(P, sw.comm[0], (2018, 2019))["excess"].sum()      # комиссия по умолчанию: % -> доля движком
    return bool(np.isclose(net0, -4 * sw.spec.comm_pct[0] / 100, atol=1e-12))


def ok_commission_lowers_sharpe() -> bool:
    sw = mk()
    return all(sw.evaluate({"a": a}, "train", 0.04)["sharpe"] < sw.evaluate({"a": a}, "train", 0.035)["sharpe"] for a in A_GRID)


def ok_ddof1() -> bool:
    x = np.random.default_rng(0).normal(0.001, 0.01, 250)
    df = pd.DataFrame(dict(excess=x, exposure=1.0, turnover=0.0, sides=0.0), index=pd.bdate_range("2018-01-01", periods=250))
    want = x.mean() / x.std(ddof=1) * np.sqrt(252)
    return bool(np.isclose(ss.yearly_sharpes(df, (2018,))[2018], want, rtol=1e-12)
                and np.isclose(ss.metrics_row(df, (2018,))["sharpe"], want, rtol=1e-12))


def ok_reopen_doubles_n(tmp_path) -> bool:
    sw = mk(out=str(tmp_path))
    for a in A_GRID:
        sw.evaluate({"a": a})
    sw.save()
    sw.open_validation()
    first = sw.evaluate(P, "validation")
    again = mk(out=str(tmp_path), reopen_note="сменили порог")
    again.open_validation()
    return first["N"] == 5 and again.evaluate(P, "validation")["n_mult"] == 2 and again.evaluate(P, "validation")["N"] == 10


# ---------- тесты ----------
def test_eng_param_signal_changes_result():
    assert ok_param_signal()


def test_eng_param_pnl_changes_result():
    assert ok_param_pnl()


def test_eng_validation_zakryta_i_ne_vliyaet_na_tselevuyu():
    assert ok_validation_closed() and ok_validation_independent()


def test_eng_podmeshali_2020_v_train_menyaet_rezultat():
    per = ss.Periods(setup=(2018, 2020), control=(2019,), validation=(2021,))
    r = [mk(SynthEvaluator(late=k), periods=per).evaluate(P)["sharpe"] for k in (1.0, 6.0)]
    assert r[0] != r[1]


def test_eng_granitsy_periodov():
    p = ss.Periods()
    assert p.train == (2018, 2019) and p.validation == (2020, 2021) and p.years("train") == (2018, 2019)
    assert ok_periods_raise(control=(2018,)) and ok_periods_raise(validation=(2019, 2020))
    assert ok_periods_raise(validation=(2021, 2022)) and ok_periods_raise(setup=(2020,), validation=(2018,))
    assert ok_end_cut()
    assert set(mk().evaluate(P)["by_year"]) == {2018, 2019} and set(mk().evaluate(P, "setup")["by_year"]) == {2018}


def test_eng_2022_ne_chitaetsya():
    assert ok_2022_closed()


def test_eng_komissiya_obe_storony():
    assert ok_commission_two_sides() and ok_commission_lowers_sharpe()


def test_eng_ddof1():
    assert ok_ddof1()


def test_eng_N_po_raznym_tochkam_a_ne_vyzovam():
    sw = mk()
    for a in A_GRID:
        sw.evaluate({"a": a})
    sw.evaluate({"a": 3})                       # повтор точки — не новая проба
    assert sw.evaluate(P)["N"] == 5 and len(sw.trials_sr()) == 5


def test_eng_validation_ne_pishetsya_v_probi(tmp_path):
    sw = mk(out=str(tmp_path))
    sw.evaluate(P)
    sw.open_validation()
    sw.evaluate({"a": 10}, "validation")
    assert len(sw.ledger) == 1
    assert json.loads((tmp_path / "validation_flag.json").read_text())["opens"][0]["text"].startswith("validation открыта: ")


def test_eng_povtornoe_otkrytie_trebuet_zapisi_i_udvaivaet_N(tmp_path):
    sw = mk(out=str(tmp_path))
    sw.evaluate(P)
    sw.open_validation()
    with pytest.raises(PermissionError):
        mk(out=str(tmp_path)).open_validation()
    assert ok_reopen_doubles_n(tmp_path / "x")
