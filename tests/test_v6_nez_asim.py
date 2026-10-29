"""Независимые тесты тестировщика к V6-3 (ось asim, fastbt/signals/v6_obertka.py: uzly, porog_shorta, primenit_asim).

Известный ответ: подставная семья sig_fn (шорты зависят от порога известным образом), шпион на передаваемую точку,
границы сетки, конфликт знаков; на реальных семьях v5 - asim=0 == v5 побайтно, long == max(sig,0),
композиция «лонги базы + шорты сдвинутого прогона» пересчитана в тесте независимо, причинность.
"""
import pathlib

import numpy as np
import pytest
import yaml

from fastbt.indicators import compute_base
from fastbt.signals import v5_s12, v5_s34, v5_s56
from fastbt.signals import v6_obertka as W
from fastbt.signals.interface import TickerData

AX = {"S1": "z_entry", "S2": "bb_std", "S3": "okno", "S4": "mnozh", "S5": "uroven", "S6": "dev_mult"}
SIDS = list(AX)
DESIGNS = pathlib.Path(__file__).resolve().parents[1] / "fastbt" / "designs" / "v5"


def yaml_nodes(sid):
    d = yaml.safe_load((DESIGNS / f"kub_{sid}.yaml").read_text(encoding="utf-8"))
    return list(d["signal"][AX[sid]]["setka"])


# ---------------- узлы и порог шорта ----------------
@pytest.mark.parametrize("sid", SIDS)
def test_uzly_match_yaml_and_strictly_increasing(sid):
    nodes = W.uzly(sid)
    assert nodes == yaml_nodes(sid)
    assert len(nodes) >= 3 and all(a < b for a, b in zip(nodes, nodes[1:]))


@pytest.mark.parametrize("sid", SIDS)
def test_porog_shorta_shift_all_positions(sid):
    nodes = W.uzly(sid)
    ax, m = AX[sid], len(nodes)
    for i, v in enumerate(nodes):
        for n in (-2, -1, 0, 1, 2):
            p2, kray = W.porog_shorta(sid, {ax: v, "other": 7}, n)
            j = i + n
            assert p2[ax] == nodes[min(max(j, 0), m - 1)]
            assert kray == (j < 0 or j > m - 1)
            assert p2["other"] == 7


def test_porog_shorta_does_not_mutate_input_and_rejects_non_node():
    p = {"z_entry": 1.0, "ma_window": 20}
    snap = dict(p)
    W.porog_shorta("S1", p, 1)
    assert p == snap
    for bad in (0.9, 1.01, 0, -1.0):
        with pytest.raises(ValueError):
            W.porog_shorta("S1", {"z_entry": bad}, 0)
    with pytest.raises(ValueError):
        W.porog_shorta("S3", {"okno": 11}, 1)


def test_s1_hand_values():
    # узлы S1: 0.3 0.5 0.75 1.0 1.25 1.5 ... 4.0 5.0
    assert W.porog_shorta("S1", {"z_entry": 1.0}, 2) == ({"z_entry": 1.5}, False)
    assert W.porog_shorta("S1", {"z_entry": 1.0}, -1) == ({"z_entry": 0.75}, False)
    assert W.porog_shorta("S1", {"z_entry": 5.0}, 1) == ({"z_entry": 5.0}, True)
    assert W.porog_shorta("S1", {"z_entry": 4.0}, 2) == ({"z_entry": 5.0}, True)   # 4.0 -> 5.0 (+1), +2 за краем
    assert W.porog_shorta("S1", {"z_entry": 0.5}, -2) == ({"z_entry": 0.3}, True)
    assert W.porog_shorta("S1", {"z_entry": 0.3}, 0) == ({"z_entry": 0.3}, False)


# ---------------- подставная семья: известный ответ ----------------
T = 12
LONG_SIG = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], np.int8)


def make_family(spy):
    """Семья S1: при z_entry <= 1.0 шорты в днях {1,5,9}; при z_entry > 1.0 - только {5}. Лонги базы - LONG_SIG."""
    def fn(p):
        spy.append(p["z_entry"])
        s = np.zeros(T, np.int8)
        if p["z_entry"] <= 1.0:
            s[[1, 5, 9]] = -1
        else:
            s[[5]] = -1
        if p["z_entry"] == 1.0:
            s += LONG_SIG
            s[1] = 1          # в базе в дне 1 - лонг, во сдвинутом - шорт: конфликт
        return s
    return fn


def test_known_answer_shift_plus_one_and_conflict():
    spy = []
    fn = make_family(spy)
    out, kray = W.primenit_asim("S1", fn, {"z_entry": 1.0}, 1)     # шорты из z=1.25: {5}
    # база: лонги {0,1,4,8}(день 1 - лонг), сдвинутый: шорты {5}; день 5 в базе: 0 (лонга нет)
    assert out.tolist() == [1, 1, 0, 0, 1, -1, 0, 0, 1, 0, 0, 0]
    assert not kray
    assert spy == [1.0, 1.25]


def test_known_answer_shift_minus_one_conflict_zeroed():
    spy = []
    fn = make_family(spy)
    out, kray = W.primenit_asim("S1", fn, {"z_entry": 1.0}, -1)    # шорты из z=0.75: {1,5,9}
    # день 1: лонг (база) и шорт (сдвинутый) -> 0; день 5, 9: шорты
    assert out.tolist() == [1, 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0]
    assert spy == [1.0, 0.75]


def test_known_answer_zero_and_long_and_edge():
    spy = []
    fn = make_family(spy)
    base = fn({"z_entry": 1.0})
    spy.clear()
    out0, k0 = W.primenit_asim("S1", fn, {"z_entry": 1.0}, 0)
    assert out0.tobytes() == base.tobytes() and not k0
    outl, kl = W.primenit_asim("S1", fn, {"z_entry": 1.0}, "long")
    assert outl.tolist() == [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] and outl.dtype == base.dtype and not kl
    # край: z=5.0, +1 -> узел 5.0 (тот же), kray
    out, kray = W.primenit_asim("S1", fn, {"z_entry": 5.0}, 1)
    assert kray and spy[-1] == 5.0 and spy[-2] == 5.0


def test_invalid_asim_values():
    fn = make_family([])
    for bad in (3, -3, "short", None, "Long", 0.5, [1]):
        with pytest.raises((ValueError, TypeError)):
            W.primenit_asim("S1", fn, {"z_entry": 1.0}, bad)


def test_sig_fn_output_not_mutated_between_calls():
    """Результат не должен менять массив, возвращённый sig_fn (кэш семьи может его переиспользовать)."""
    store = {}

    def fn(p):
        key = p["z_entry"]
        if key not in store:
            s = np.zeros(T, np.int8)
            s[[2, 6]] = 1 if key == 1.0 else -1
            store[key] = s
        return store[key]
    snap = {k: v.copy() for k, v in ((1.0, fn({"z_entry": 1.0})), (1.25, fn({"z_entry": 1.25})))}
    for a in (0, 1, -1, "long"):
        W.primenit_asim("S1", fn, {"z_entry": 1.0}, a)
    for k, v in snap.items():
        assert store[k].tobytes() == v.tobytes()


# ---------------- реальные семьи v5 ----------------
RT = 1200
_R = np.random.default_rng(31)
_t = np.arange(RT)
C0 = 80 * np.exp(np.cumsum(_R.normal(0, 0.018, RT)) + 0.45 * np.sin(_t / 30))
H0 = C0 * (1 + _R.uniform(0.001, 0.016, RT))
L0 = C0 * (1 - _R.uniform(0.001, 0.016, RT))
O0 = np.r_[C0[0], C0[:-1]]
V0 = _R.uniform(1e3, 6e3, RT)
PT = {
    "S1": dict(ma_tip="sma", dev_tip="std", ma_window=18, z_entry=1.0, z_exit="net"),
    "S2": dict(ma_tip="sma", bb_window=18, bb_std=1.5, podtv="none"),
    "S3": dict(okno=10, podtv="none"),
    "S4": dict(period=8, mnozh=1.25),
    "S5": dict(pivot_type="classic", pivot_period="den", uroven=2, atr_filtr=0, filter_set="net"),
    "S6": dict(vwap_window=10, dev_mult=0.5, dir_bias="OFF", filter_set="net"),
}
GEN = {"S1": v5_s12.generate_s1, "S2": v5_s12.generate_s2, "S3": v5_s34.generate_s3_v5,
       "S4": v5_s34.generate_s4_v5, "S5": v5_s56.generate5, "S6": v5_s56.generate6}


def mk(c=C0, h=H0, lo=L0, o=O0, v=V0):
    lr = np.r_[np.log(c[1:] / c[:-1]), 0.0]
    td = TickerData("Q", np.arange(len(c)).astype("datetime64[D]").astype("datetime64[ns]"), o, h, lo, c, v, lr)
    ind = compute_base(c, h, lo, o, v)
    return lambda p, sid: GEN[sid](td, ind, [p]).sig[0]


@pytest.mark.parametrize("sid", SIDS)
def test_real_asim0_equals_v5_bytes(sid):
    f = mk()
    fn = lambda p: f(p, sid)        # noqa: E731
    s = fn(PT[sid])
    assert (s != 0).sum() > 5
    out, kray = W.primenit_asim(sid, fn, PT[sid], 0)
    assert out.dtype == s.dtype and out.tobytes() == s.tobytes() and not kray


@pytest.mark.parametrize("sid", SIDS)
def test_real_long_equals_max_sig_0(sid):
    f = mk()
    fn = lambda p: f(p, sid)        # noqa: E731
    s = fn(PT[sid])
    out, _ = W.primenit_asim(sid, fn, PT[sid], "long")
    assert out.dtype == s.dtype and out.tobytes() == np.clip(s, 0, None).astype(s.dtype).tobytes()


@pytest.mark.parametrize("n", [-2, -1, 1, 2])
@pytest.mark.parametrize("sid", SIDS)
def test_real_composition_recomputed_independently(sid, n):
    f = mk()
    fn = lambda p: f(p, sid)        # noqa: E731
    nodes = yaml_nodes(sid)
    i = nodes.index(PT[sid][AX[sid]])
    j = min(max(i + n, 0), len(nodes) - 1)
    base = fn(PT[sid])
    sh = fn({**PT[sid], AX[sid]: nodes[j]})
    exp = np.zeros(RT, np.int8)
    for t in range(RT):
        lg, st = base[t] > 0, sh[t] < 0
        exp[t] = 1 if lg and not st else (-1 if st and not lg else 0)
    out, kray = W.primenit_asim(sid, fn, PT[sid], n)
    assert out.tobytes() == exp.tobytes()
    assert kray == (i + n < 0 or i + n > len(nodes) - 1)
    # лонги выхода - только лонги базы; шорты - только шорты сдвинутого прогона
    assert np.all(base[out > 0] > 0) and np.all(sh[out < 0] < 0)


@pytest.mark.parametrize("sid", ["S1", "S2", "S6"])
def test_real_asim_causality(sid):
    K = 650
    f = mk()
    full, _ = W.primenit_asim(sid, lambda p: f(p, sid), PT[sid], 1)
    r = np.random.default_rng(9)
    c, h, lo, o, v = (a.copy() for a in (C0, H0, L0, O0, V0))
    fac = np.exp(r.normal(0, 0.3, RT - K))
    for a in (c, h, lo, o):
        a[K:] *= fac
    g = mk(c, h, lo, o, v)
    bad, _ = W.primenit_asim(sid, lambda p: g(p, sid), PT[sid], 1)
    assert np.array_equal(full[:K], bad[:K])
    assert (full[:K] != 0).sum() > 0
