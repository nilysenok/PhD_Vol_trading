"""Э1-а (Ж-034): тестировщик движка перебора. Ядро v1 на синтетике (`fastbt.sweep_synth`), известный ответ.

Детекторы `ok_*` возвращают True, если свойство выполнено; их же роняют мутации в `test_sweep_mutacii.py`.
Реальных данных не читает (≥ 2022 — тем более)."""
import numpy as np
import pandas as pd
import pytest

from fastbt import sweep_stat as ss
from fastbt.sweep import Spec, Sweep
from fastbt.sweep_synth import SynthEvaluator

GRIDS = {"a": [3, 5, 10, 20, 40], "w": [0.5, 1.0, 2.0]}
BASE = {"a": 5, "w": 1.0}
P = dict(BASE)


def mk(ev=None, grids=None, cache_dir=None, **kw):
    spec = Spec(obj={"kind": "Synth"}, base=dict(BASE), grids=grids or GRIDS, **kw)
    return Sweep(spec, ev or SynthEvaluator(), cache_dir)


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
    return sw.objective({"a": 3, "w": 1.0}) != sw.objective({"a": 40, "w": 1.0})


def ok_param_pnl() -> bool:
    sw = mk()
    return sw.objective({"a": 5, "w": 0.5}) != sw.objective({"a": 5, "w": 2.0})


def ok_validation_independent() -> bool:
    """Целевая на train не зависит от доходностей 2020–2021 (late меняет только их)."""
    objs = [mk(SynthEvaluator(late=k)).objective(P) for k in (1.0, 6.0)]
    return objs[0] == objs[1]


def ok_validation_closed() -> bool:
    try:
        mk()._df(P, 0.0004, (2018, 2020))
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
    mk(Tiny(log)).objective(P)
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
        got = sw.evaluate(P, comm_pct=pct)
        net = sw._df(P, c, (2018, 2019))["excess"].sum()
        if not np.isclose(net, -4 * c, atol=1e-12) or not np.isclose(got["comm_pct"], pct):
            return False
    net0 = sw._df(P, sw.comm[0], (2018, 2019))["excess"].sum()      # комиссия, пересчитанная движком из % в долю
    return bool(np.isclose(net0, -4 * sw.spec.comm_pct[0] / 100, atol=1e-12))


def ok_oat_count() -> bool:
    pts = mk().points()
    base = [p for p in pts if p == BASE]
    one_off = all(sum(p[k] != BASE[k] for k in BASE) <= 1 for p in pts)
    return len(pts) == sum(len(v) - 1 for v in GRIDS.values()) + 1 and len(base) == 1 and one_off


def ok_plateau() -> bool:
    sw = mk()
    row = sw.evaluate(P)
    sh = []
    for n, v in GRIDS.items():
        i = v.index(P[n])
        for j in (i - 1, i + 1):
            if 0 <= j < len(v):
                sh.append(ss.metrics_row(sw._df({**P, n: v[j]}, sw.comm[0], (2018, 2019)), (2018, 2019))["sharpe"])
    return row["n_sosedey"] == 4 and np.isclose(row["plateau"], np.mean(sh), atol=1e-12)


def ok_ddof1() -> bool:
    x = np.random.default_rng(0).normal(0.001, 0.01, 250)
    df = pd.DataFrame(dict(excess=x), index=pd.bdate_range("2018-01-01", periods=250))
    got = ss.yearly_sharpes(df, (2018,))[2018]
    return bool(np.isclose(got, x.mean() / x.std(ddof=1) * np.sqrt(252), rtol=1e-12))


def ok_objective_formula() -> bool:
    sw = mk()
    sh = ss.yearly_sharpes(sw._df(P, sw.comm[0], (2018, 2019)), (2018, 2019))
    v = np.array(list(sh.values()))
    want = v.mean() - 0.5 * v.std(ddof=1)
    return bool(np.isclose(sw.objective(P), want, rtol=1e-12) and np.isclose(ss.objective({2018: 1.0, 2019: 0.0}), 0.5 - 0.5 * np.sqrt(0.5)))


# ---------- тесты ----------
def test_eng_param_signal_changes_result():
    assert ok_param_signal()


def test_eng_param_pnl_changes_result():
    assert ok_param_pnl()


def test_eng_validation_zakryta_i_ne_vliyaet_na_tselevuyu():
    assert ok_validation_closed() and ok_validation_independent()


def test_eng_podmeshali_2020_v_train_menyaet_rezultat():
    per = ss.Periods(setup=(2018, 2020), control=(2019,), validation=(2021,))
    r = [mk(SynthEvaluator(late=k), periods=per).objective(P) for k in (1.0, 6.0)]
    assert r[0] != r[1]


def test_eng_granitsy_periodov():
    p = ss.Periods()
    assert p.train == (2018, 2019) and p.validation == (2020, 2021) and p.years("train") == (2018, 2019)
    assert ok_periods_raise(control=(2018,)) and ok_periods_raise(validation=(2019, 2020))
    assert ok_periods_raise(validation=(2021, 2022)) and ok_periods_raise(setup=(2020,), validation=(2018,))
    assert ok_end_cut()
    with pytest.raises(ValueError):
        Spec(obj={}, base={}, grids={}, period="validation", mode="oat")


def test_eng_2022_ne_chitaetsya():
    assert ok_2022_closed()
    assert mk(Tiny()).run()["table"] is not None


def test_eng_komissiya_obe_storony():
    assert ok_commission_two_sides()


def test_eng_oat_chislo_tochek():
    assert ok_oat_count()
    pts = mk(grids={"a": [3, 5, 10]}).points()
    assert len(pts) == 3          # (3−1)+1, базовая a=5 — узел сетки


def test_eng_plato_srednee_sosedey():
    assert ok_plateau()


def test_eng_ddof1():
    assert ok_ddof1()


def test_eng_tselevaya_srednee_minus_pol_sigmy():
    assert ok_objective_formula()
    assert np.isnan(ss.objective({2018: float("nan"), 2019: 1.0}))
