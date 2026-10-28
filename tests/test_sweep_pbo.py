"""Э1-г/д, PR 1: `excess_d`/`daty` в строке оценщика и в ledger, PBO по ledger (`fastbt.sweep_pbo`). Только синтетика.
Детекторы `ok_*` возвращают True, если свойство выполнено; мутации — `test_sweep_pbo_mut.py`."""
import numpy as np
import pandas as pd
import pytest

from fastbt import metriki_e as me
from fastbt import sweep_pbo
from fastbt.sweep import Sweep
from fastbt.sweep_spec import Spec
from fastbt.sweep_synth import SynthEvaluator

P = {"a": 5, "w": 1.0}


class Noise:
    """Проба `a` = независимый шум со своим зерном; проба `a == STRONG` имеет настоящий сдвиг среднего."""
    sig_names = {"a"}
    STRONG = 3

    def __init__(self, seed=0, strong=True):
        self.dates, self.seed, self.strong = pd.bdate_range("2018-01-01", "2021-12-31"), seed, strong

    def series_digest(self): return f"n{self.seed}"
    def code_id(self): return "n"
    def signals(self, sp): return int(sp["a"])

    def pnl(self, sigs, p, comm, end):
        m = self.dates < end
        x = np.random.default_rng(self.seed * 1000 + sigs).normal(0, 0.01, len(self.dates))
        if self.strong and sigs == self.STRONG:
            x = x + 0.004
        return pd.DataFrame(dict(excess=x[m], exposure=np.ones(m.sum()), turnover=np.zeros(m.sum()),
                                 sides=np.zeros(m.sum())), index=self.dates[m])


def mk(ev=None, out=None):
    return Sweep(Spec(obj={"kind": "Synth"}, base=dict(P), out=out), ev or SynthEvaluator())


def filled(seed, strong, n=20):
    sw = mk(Noise(seed, strong))
    for a in range(n):
        sw.evaluate({"a": a}, "setup")
    return sw


def ok_excess_d_matches_sharpe() -> bool:
    r = mk().evaluate(P, "setup")
    d = pd.DatetimeIndex(r["daty"])
    return bool(len(r["excess_d"]) == len(r["daty"]) == r["T"] and set(d.year) == {2018}
                and abs(me.sharpe(np.array(r["excess_d"])) - r["sharpe"]) < 1e-12)


def ok_ledger_excess_setup_only() -> bool:
    sw = mk()
    sw.evaluate(P, "control")
    sw.evaluate(P, "train")
    sw.evaluate(P, "setup", 0.035)
    if sw.excess:
        return False
    r = sw.evaluate(P, "setup")
    (v,) = sw.excess.values()
    return bool(len(sw.excess) == len(sw.ledger) == 1 and v["excess_d"] == r["excess_d"] and v["daty"] == r["daty"])


def pbo_mean(strong: bool, seeds=range(12)) -> float:
    return float(np.mean([sweep_pbo.pbo_sweep(filled(s, strong))["pbo"] for s in seeds]))


def ok_pbo_noise() -> bool:
    return 0.35 < pbo_mean(False) < 0.65


def ok_pbo_signal() -> bool:
    return pbo_mean(True) < 0.2


def ok_dates_guard() -> bool:
    ex = {k: dict(v) for k, v in filled(0, False, 4).excess.items()}
    k = sorted(ex)[0]
    ex[k]["daty"] = ex[k]["daty"][1:] + ["2018-12-31"]
    try:
        sweep_pbo.pbo(ex)
    except ValueError:
        return True
    return False


def test_excess_d_sovpadaet_s_sharpe_stroki():
    assert ok_excess_d_matches_sharpe()


def test_ledger_excess_tolko_setup_pri_otbore():
    assert ok_ledger_excess_setup_only()


def test_ledger_excess_perezhivaet_save_load(tmp_path):
    sw = mk(out=str(tmp_path))
    sw.evaluate(P, "setup")
    sw.save()
    assert mk(out=str(tmp_path)).excess == sw.excess and len(sw.excess) == 1


def test_pbo_shum_okolo_polovinu_cherez_ledger():
    assert ok_pbo_noise()


def test_pbo_signal_nizkiy_cherez_ledger():
    assert ok_pbo_signal()


def test_pbo_raznye_daty_otkaz():
    assert ok_dates_guard()


def test_pbo_pustoy_ledger_otkaz():
    with pytest.raises(ValueError):
        sweep_pbo.pbo({})


def test_pbo_odna_proba_nan():
    out = sweep_pbo.pbo_sweep(filled(0, False, 1))
    assert np.isnan(out["pbo"]) and out["n_prob"] == 1
