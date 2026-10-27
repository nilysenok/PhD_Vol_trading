"""Шаг 3′ (D, B): scripts/analysis/s9_shag_common.py, s9_shag_db.py на синтетике; данные не читаются.
Обязательные мутации §6 (в памяти) роняют проверки: сдвиг входа σ / RVI на +1, снятие комиссии выхода на
границе, к2 со сдвигом 0, к3 с невыключенной S5; отдельно — RVI_d вместо RVI_{d−1}.
python3 -m pytest tests/test_s9_shag_db.py -q
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts" / "analysis")]
from test_s9_shag0 import _world  # noqa: E402

import s9_shag_common as C  # noqa: E402
import s9_shag_db as DB  # noqa: E402

from fastbt import fast_kontrol as fk  # noqa: E402
from fastbt import s9  # noqa: E402

COMM = C.COMM_MAIN


def _inf_gate(ns):
    """Информативная маска (знает знак дня): сдвиг её ломает; для проверки p к к2."""
    return (ns.A_row * ns.lr_row > 0).astype(float)

# ---------------- комиссия выхода на границе ----------------
def _port_no_exit(ns):
    def port(pos, comm=None, cap=1.0, wgt=None):
        dp = np.diff(pos, prepend=0.0)
        dp[ns.first] = pos[ns.first]
        di, nd = ns.r["date"], len(ns.dates)
        sides = np.abs(dp)                                   # мутация: выход на границе не платится
        net = pos * ns.lr_row - sides * comm
        return ns._finish(np.bincount(di, net, nd) / ns.cnt, np.bincount(di, pos * ns.lr_row, nd) / ns.cnt,
                          np.bincount(di, np.abs(pos), nd) / ns.cnt, np.bincount(di, sides, nd) / ns.cnt, cap)
    return port


def test_exit_fee_paid_and_mutation_caught():
    ns = _world()
    assert np.any(ns.A_row[np.r_[ns.first[1:], True]] != 0)       # на границе есть открытые позиции
    C.check_exit_fee(ns, ns.A_row, COMM)
    ns.port = _port_no_exit(ns)
    with pytest.raises(AssertionError, match="комиссия выхода"):
        C.check_exit_fee(ns, ns.A_row, COMM)

# ---------------- к2: сдвиг 0, лаги ----------------
def test_k2_lag0_reproduces_point_and_mutation_caught():
    ns = _world()
    g = _inf_gate(ns)
    K = fk.Kontrol(ns, COMM, n_shift=20)
    sh = s9.metrics(ns.port(ns.A_row * g, comm=COMM))["sharpe"]
    C.check_lag0(K, g, sh)
    with pytest.raises(AssertionError):
        C.check_lag0(K, np.roll(g, 1), sh)                       # маска не та — сдвиг 0 не воспроизводит точку


def test_k2_zero_shift_mutation_caught(monkeypatch):
    ns = _world()
    g = _inf_gate(ns)
    base = C.make_base(ns, COMM, 50)
    K = fk.Kontrol(ns, COMM, n_shift=100)
    C.check_lags(K.LAG, K.t_tick)
    ok = C.eval_point(ns, K, base, ns.A_row * g, g)
    assert ok["p_vs_k2"] < 0.05 and ok["usl3"] is True
    K.LAG[:] = 0                                                 # мутация: сдвиг 0 у всех повторов
    bad = C.eval_point(ns, K, base, ns.A_row * g, g)
    assert bad["p_vs_k2"] == 1.0 and bad["usl3"] is False and not bad["passed_v2"]
    with pytest.raises(AssertionError, match="лаг вне"):
        C.check_lags(K.LAG, K.t_tick)

# ---------------- к3 ----------------
def test_k3_off_s5_and_mutation_caught(monkeypatch):
    ns = _world()
    good = fk.k3_positions(ns.A_row, ns.r["pair"], ns.pairs)
    C.check_k3(ns, good)
    for left in (("S6",), ("S5",)):
        bad = fk.k3_positions(ns.A_row, ns.r["pair"], ns.pairs, sids=left)
        with pytest.raises(AssertionError, match="к3"):
            C.check_k3(ns, bad)
    monkeypatch.setattr(fk, "k3_positions", lambda A, pair, pairs: np.asarray(A, float))
    with pytest.raises(AssertionError, match="к3"):              # к3 = A (S5/S6 не выключены) — make_base падает
        C.make_base(ns, COMM, 50)

# ---------------- условия С8 v2 отдельно; B без условия 3 ----------------
def test_flags_d_vs_b():
    ns = _world()
    base, K = C.make_base(ns, COMM, 50), fk.Kontrol(ns, COMM, n_shift=100)
    g = _inf_gate(ns)
    d = C.eval_point(ns, K, base, ns.A_row * g, g)
    b = C.eval_point(ns, K, base, ns.A_row * g, g, scale=True)
    assert d["usl3"] is True and "p_vs_k2" in d
    assert b["usl3"] == C.NOT_APPLIED and "p_vs_k2" not in b and "p_vs_k2_masshtab" in b
    assert b["passed_v2"] == bool(b["usl1"] and b["usl2"])
    assert d["sharpe_k3"] == base["m3"]["sharpe"] and d["usl2"] == (d["sharpe"] > d["sharpe_k3"])
    for k in ("d_vs_A", "p_vs_A", "d_vs_k3", "p_vs_k3", "ce1_pct_y", "fko_nad_poroga_pct", "znak_2018"):
        assert k in d


def test_select_max_sharpe_among_passed_only():
    df = pd.DataFrame(dict(sharpe=[2.0, 1.0, 1.5], passed_v2=[False, True, True], point=list("abc")))
    assert C.select(df)["point"] == "c"
    assert C.select(df.assign(passed_v2=False)) is None

# ---------------- входы: сдвиг +1 σ / RVI, RVI_d вместо RVI_{d−1} ----------------
N, K0 = 700, 400


def _inputs(seed=5):
    rng = np.random.RandomState(seed)
    S = np.exp(rng.normal(-4, 0.4, (2, N)))
    RV = 18 + np.abs(rng.normal(0, 5, (2, N)))
    return S, RV


def _gates(S, RV, ignore_lag=False):
    def rvi(lag):
        R = RV if ignore_lag else np.c_[np.full((2, lag), np.nan), RV[:, :N - lag]]
        return R
    e = types.SimpleNamespace(sigma=lambda s: S, rvi=rvi, n_t=np.array([N, N - 50]))
    fams = s9.S9.d_points(types.SimpleNamespace(e=e), "x")
    return {f: np.stack([G for _, G, _ in pts]) for f, pts in fams.items()}


def _peek(fn, S, RV, which):
    """Какие семьи меняют затвор до дня k (σ: [:k]; RVI_k до дня k включительно не виден → [:k+1])."""
    a = fn(S, RV)
    S2, RV2 = S.copy(), RV.copy()
    (S2 if which == "sigma" else RV2)[:, K0] *= 1.9
    b = fn(S2, RV2)
    cut = K0 if which == "sigma" else K0 + 1
    return {f for f in a if not np.array_equal(a[f][..., :cut], b[f][..., :cut], equal_nan=True)}


def test_gates_use_no_future_and_shift_mutations_caught(monkeypatch):
    monkeypatch.setattr(s9, "V2", True)
    S, RV = _inputs()
    assert {f for f in _gates(S, RV)} >= {"D1", "D2", "D4", "D5", "D6lag1", "D7"}
    assert _peek(_gates, S, RV, "sigma") == set()                         # σ: день d виден только с d
    assert _peek(_gates, S, RV, "rvi") - {"D6"} == set()                  # RVI: сегодня не виден у D6lag1
    # мутация: σ сдвинут на +1 (σ_{d+1} в решении d)
    leaky = lambda s, r: _gates(np.roll(s, -1, axis=1), r)               # noqa: E731
    assert _peek(leaky, S, RV, "sigma") >= {"D1", "D2", "D4", "D5", "D7"}
    # мутация: RVI_d вместо RVI_{d−1} (лаг игнорируется) — ловится на D6lag1
    same_day = lambda s, r: _gates(s, r, ignore_lag=True)                # noqa: E731
    assert "D6lag1" in _peek(same_day, S, RV, "rvi")
    # мутация: RVI сдвинут на +1 при верном лаге
    leaky_r = lambda s, r: _gates(s, np.roll(r, -1, axis=1))             # noqa: E731
    assert "D6lag1" in _peek(leaky_r, S, RV, "rvi")


def test_b_weights_no_future_and_shift_caught():
    S, _ = _inputs()
    def w(s, _r):
        e = types.SimpleNamespace(sigma=lambda x: s, n_t=np.array([N, N - 50]))
        return {"B": np.stack([s9.S9.b_weights(types.SimpleNamespace(e=e), "x", c, b)
                               for c in (1.0, 2.0) for b in (0.0, 0.2)])}
    assert _peek(w, S, S, "sigma") == set()
    assert _peek(lambda s, r: w(np.roll(s, -1, axis=1), r), S, S, "sigma") == {"B"}

# ---------------- сетка, план, устойчивость, итог B ----------------
def test_grid_counts_match_yaml(monkeypatch):
    monkeypatch.setattr(s9, "V2", True)
    S, RV = _inputs()
    g = _gates(S, RV)
    assert {f: len(v) for f, v in g.items() if f != "D6"} == dict(D1=9, D2=18, D4=27, D5=3, D6lag1=3, D7=6)
    assert sum(len(v) for f, v in g.items() if f != "D6") == C.plan("D")["na_ryad"] == 66
    assert C.plan("D")["vsego"] == 396 and C.plan("B")["vsego"] == 54 and C.plan("B", "naive")["vsego"] == 18


def test_dry_run_reads_no_data(monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("dry-run читает данные")
    monkeypatch.setattr(pd, "read_parquet", boom)
    pl = DB.dry_run("B", "hyb")
    assert pl["vsego"] == 18 and "точек на ряд 9" in capsys.readouterr().out


def _stab(vals, passed, sel_pt):
    pts = ["W126_p70", "W252_p70", "W504_p70"]
    df = pd.DataFrame(dict(family="D1", sigma="x", point=pts, sharpe=vals, passed_v2=passed))
    return C.ustoychivost(df, df[df["point"] == sel_pt].iloc[0])


def test_ustoychivost_verdicts():
    assert _stab([1.0, 1.0, 1.0], [True] * 3, "W252_p70").iloc[0]["vyvod"] == "плато"
    assert _stab([1.0, 1.0, 1.0], [True] * 3, "W126_p70").iloc[0]["vyvod"] == "край"
    assert _stab([1.0, 0.5, 0.4], [True, False, False], "W252_p70").iloc[0]["vyvod"] == "пик"
    assert _stab([1.0, 1.0, 0.4], [True, True, False], "W252_p70").iloc[0]["vyvod"] == "частичное плато"


def test_b_itog_negative_only_when_all_three_sigmas_done():
    assert C.b_itog({"naive": 0, "harj": 0, "hyb": 0}).iloc[-1]["vyvod"] == C.B_NEGATIVE
    assert C.b_itog({"naive": 0, "harj": 0, "hyb": 1}).iloc[-1]["vyvod"] != C.B_NEGATIVE
    assert (C.b_itog({"naive": 0}).iloc[:, 0] != "итог").all()          # σ не досчитаны — вывода нет
