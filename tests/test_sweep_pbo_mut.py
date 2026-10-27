"""Мутации PR 1 (excess_d, ledger, PBO): подложенная ошибка -> детектор из `test_sweep_pbo.py` обязан упасть."""
from fastbt import sweep_drivers as sd
from fastbt import sweep_pbo
from fastbt.sweep import Sweep
from test_sweep_pbo import (ok_dates_guard, ok_excess_d_matches_sharpe, ok_ledger_excess_setup_only, ok_pbo_noise,
                            ok_pbo_signal)


def test_mut_1_excess_d_znak_perepyutan(monkeypatch):
    orig = Sweep.evaluate

    def ev(self, *a, **k):
        r = orig(self, *a, **k)
        r["excess_d"] = [-x for x in r["excess_d"]]
        return r
    monkeypatch.setattr(Sweep, "evaluate", ev)
    assert not ok_excess_d_matches_sharpe()


def test_mut_2_excess_pishetsya_na_lyubom_periode(monkeypatch):
    orig = Sweep.evaluate

    def ev(self, params, period="train", comm_pct=None):
        r = orig(self, params, period, comm_pct)
        self.excess["x" + period + str(comm_pct)] = dict(daty=r["daty"], excess_d=r["excess_d"])
        return r
    monkeypatch.setattr(Sweep, "evaluate", ev)
    assert not ok_ledger_excess_setup_only()


def test_mut_3_matritsa_bez_kontrolya_dat(monkeypatch):
    def perf(excess):
        keys = sorted(excess)
        cols = [excess[k]["excess_d"] for k in keys]
        return list(excess[keys[0]]["daty"]), [list(r) for r in zip(*cols)], keys
    monkeypatch.setattr(sweep_pbo, "perf_matrix", perf)
    assert not ok_dates_guard()


def test_mut_4_pbo_na_inverse_matritse(monkeypatch):
    """Перепутанный знак: лучшая на train выбирается как худшая -> на сигнале PBO высок, на шуме остаётся ~0,5."""
    orig = sd.pbo_cscv
    monkeypatch.setattr(sd, "pbo_cscv", lambda m, nb=8, **k: orig([[-x for x in r] for r in m], nb))
    assert not ok_pbo_signal()


def test_mut_5_pbo_ne_chitaet_ledger_a_stroit_shum(monkeypatch):
    """PBO по не тому: матрица постоянная -> PBO на шуме не около 0,5."""
    orig = sd.pbo_cscv
    monkeypatch.setattr(sd, "pbo_cscv", lambda m, nb=8, **k: orig([[0.0] * len(m[0]) for _ in m], nb))
    assert not ok_pbo_noise()
