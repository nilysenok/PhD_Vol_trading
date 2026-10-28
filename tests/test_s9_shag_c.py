"""Шаги 1–3 главы 4 (v3): fastbt.shag_c / shag_c_tab / scripts/analysis/s9_shag_c.py на синтетике (данные не читаются).
Обязательные мутации §6 (v3) — в памяти, `str.replace` с assert на одну замену; каждая: проверка `chk_*` проходит на верном
коде и падает на подложенной ошибке: сдвиг входа σ на +1 день; снятие комиссии выхода на границе; к2 со сдвигом 0;
к3 с невыключенной S5; маска с текущим днём в ранге."""
import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from test_s9_kontrol import _engine  # noqa: E402

from fastbt import fast_kontrol as fk  # noqa: E402
from fastbt import s9, s9_stats  # noqa: E402
from fastbt import shag_c as C  # noqa: E402

COMM = 0.0004
CATS = s9.CATS


def world(seed=3, Ds=(1000, 1000)):
    ns = _engine(Ds=Ds, drift=0.003, seed=seed)
    ns.pairs = [(f"T{i // 6}", f"S{i % 6 + 1}") for i in range(2 * 6)]
    ns.tickers, ns.n_t = ["T0", "T1"], np.array(Ds)
    ns.r["tick"] = ns.r["pair"] // 6
    ns.r["bar"] = ns.r["date"]
    ns.r["cat"] = np.array([CATS[ns.pairs[p][1]] for p in ns.r["pair"]])
    rng = np.random.RandomState(seed)
    S = np.exp(np.cumsum(rng.normal(0, 0.05, (2, max(Ds))), axis=1) * 0.3 + rng.normal(0, 0.3, (2, max(Ds))))
    RV = S * np.exp(rng.normal(0, 0.1, S.shape))
    ns.sigma, ns.rvw = (lambda lab: S), (lambda: RV)
    ns.g_row = lambda G: s9.Engine.g_row(ns, G)
    ns.gate_stats = lambda g: s9.Engine.gate_stats(ns, g)
    return ns


def mini_grid():
    ch = [[50, 70]]
    return dict(okna=[126], min_periods={126: 50}, sigmy={"hyb": {"h": [1], "tochek": 5}},
                semyi={"C1": dict(signal="sigma", filtr=[50, 70]), "C2W": dict(signal="sigma", chastichnaya=ch),
                       "C*": dict(signal="sigma_over_rvw", filtr=[70], chastichnaya=ch)})


def _mutated(fn, old, new, env):
    src = textwrap.dedent(inspect.getsource(getattr(fn, "py_func", fn)))
    src = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("@"))
    assert src.count(old) == 1, old
    ns = dict(env)
    exec(src.replace(old, new), ns)
    return ns[fn.__name__ if hasattr(fn, "__name__") else fn.py_func.__name__]


# ---------- проверки (проходят на верном коде) ----------
def chk_rank_without_today():
    """σ_50 = 44,5 при окне 0…49: ровно 45 из 50 меньше → ранг 0,9; контртренд p90 «R < 0,9» закрыт (g = 0)."""
    S = np.r_[np.arange(50.0), 44.5, np.arange(51, 200.0)][None, :] * 1.0 + 1.0
    eng = type("E", (), dict(sigma=lambda s, lab: S, rvw=lambda s: None, n_t=np.array([200])))()
    G = C.gates(eng, "hyb_h1", dict(okna=[126], min_periods={126: 50}, semyi={
        "C1": dict(signal="sigma", filtr=[90])}), naive=True)[("C1", "W126_p90")]
    assert G[1, 0, 50] == 0.0


def chk_exit_fee():
    ns = world()
    pos = np.ones(len(ns.A_row))
    ref = np.zeros(len(ns.dates))
    for p in np.unique(ns.r["pair"]):
        i = np.flatnonzero(ns.r["pair"] == p)
        fee = np.zeros(len(i))
        fee[0] += COMM
        fee[-1] += COMM                                   # позиция открыта на последнем баре: платит выход
        np.add.at(ref, ns.r["date"][i], pos[i] * ns.lr_row[i] - fee)
    ref /= ns.cnt
    assert np.allclose(ns.port(pos, comm=COMM)["r"].values, ref, atol=1e-14)


def _noisy(ns, t):
    S1, R1 = ns.sigma("x").copy(), ns.rvw().copy()
    S1[:, t + 1:] = np.random.RandomState(1).lognormal(0, 1, S1[:, t + 1:].shape)
    R1[:, t + 1:] = np.random.RandomState(2).lognormal(0, 1, R1[:, t + 1:].shape)
    return type("E", (), dict(sigma=lambda s, lab: S1, rvw=lambda s: R1, n_t=ns.n_t))()


def chk_causal(t=600):
    """σ после t → шум: затворы на барах ≤ t не меняются (σ_{d+1} не используется в решении дня d)."""
    ns, grid = world(), mini_grid()
    a = C.gates(ns, "hyb_h1", grid, naive=False)
    b = C.gates(_noisy(ns, t), "hyb_h1", grid, naive=False)
    assert all(np.array_equal(a[k][..., :t + 1], b[k][..., :t + 1]) for k in a)


def chk_k2_lags():
    b = C.Base(world(), COMM, 100)
    assert b.kon.LAG.min() >= fk.MIN_LAG
    g = np.where(np.arange(len(b.A)) % 7 < 3, 0.0, 1.0)
    zero = b.kon.k2_sharpes(g, np.zeros_like(b.kon.LAG))
    pt = s9.sharpe(b.eng.port(b.A * g, comm=COMM)["excess"].values)
    assert np.allclose(zero, pt, atol=1e-9)                     # сдвиг 0 = исходная маска (поэтому лаги ≥ 21)
    assert fk.p_perm(pt, b.kon.k2_sharpes(g)) < 0.5             # настоящие сдвиги Шарп точки не копируют


def chk_k3_sid():
    ns = world()
    b = C.Base(ns, COMM, 20)
    ref = ns.A_row.copy()
    for p, (_, s) in enumerate(ns.pairs):
        if s in ("S5", "S6"):
            ref[ns.r["pair"] == p] = 0.0
    assert np.array_equal(b.k3, ref)


def _all_checks():
    return [chk_rank_without_today, chk_exit_fee, chk_causal, chk_k2_lags, chk_k3_sid]


@pytest.mark.parametrize("chk", _all_checks())
def test_checks_pass_on_correct_code(chk):
    chk()


def test_mutation_rank_with_today(monkeypatch):
    f = _mutated(s9_stats.past_rank, "for j in range(lo, t):", "for j in range(lo, t + 1):", dict(np=np))
    monkeypatch.setattr(s9_stats, "past_rank", f)
    with pytest.raises(AssertionError):
        chk_rank_without_today()


def test_mutation_no_exit_fee(monkeypatch):
    old = "sides = np.abs(dp) + np.where(last, np.abs(pos), 0.0)"
    monkeypatch.setattr(s9.Engine, "port", _mutated(s9.Engine.port, old, "sides = np.abs(dp)", s9.__dict__))
    with pytest.raises(AssertionError):
        chk_exit_fee()


def test_mutation_sigma_shift_plus_one_day(monkeypatch):
    new = "S = np.concatenate([eng.sigma(src)[:, 1:], np.full((2, 1), np.nan)], axis=1)"
    monkeypatch.setattr(C, "gates", _mutated(C.gates, "S = eng.sigma(src)", new, C.__dict__))
    with pytest.raises(AssertionError):
        chk_causal()


def test_mutation_k2_shift_zero(monkeypatch):
    monkeypatch.setattr(fk, "draw_lags", lambda t, n=1000, seed=0, min_lag=21: np.zeros((n, len(t)), np.int64))
    with pytest.raises((AssertionError, ValueError)):
        chk_k2_lags()


def test_mutation_k3_with_s5(monkeypatch):
    monkeypatch.setattr(fk, "k3_sids", lambda: ("S6",))
    with pytest.raises(AssertionError):
        chk_k3_sid()


# ---------- сетка, условия, отбор, разложение ----------
def test_plan_counts_and_naive_without_cstar():
    g = C.load_grid()
    n = {s: C.plan(s, g) for s in C.SHAG}
    assert (len(n["naive"]), len(n["harj"]), len(n["hyb"])) == (36, 144, 144)
    assert all(g["sigmy"][s]["tochek"] == len(n[s]) for s in n)
    assert not any(p["family"] == "C*" for p in n["naive"]) and {p["h"] for p in n["naive"]} == {1}
    assert {p["W"] for p in n["hyb"]} == {126, 252, 504} and min(p["p1"] for p in n["hyb"]) == 50


