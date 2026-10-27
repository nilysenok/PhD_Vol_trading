"""К (Ъ владельца 27.10, 00:10; контроли v3 — 01:30): сетка C + p50/p60 и контроли без прогноза
к2 (круговой сдвиг) и к3 (A без S5/S6) (fastbt.fast_kontrol).

(а) расширение сетки (setka_C.yaml: dop.rassh): число точек = yaml; формулы = C1/C2 (S9.c_points);
    детектор утечек — σ после t → шум не меняет затворы на барах ≤ t, сдвиг +1 ловится;
(б) к2: круговой сдвиг внутри пар, |k| ≥ 22; доля, уровни и число переключений (оборот) сохраняются;
    Шарп ядра на тождественном сдвиге = s9.Engine.port → metrics; p = (1 + #≥) / (N + 1);
(в) синтетика: маска C случайная (та же доля) → p к к2 равномерно; информативная маска → p мал;
(г) мутации: сдвиг без круга и неверная доля ловятся; к3 обнуляет ровно S5/S6 (не только S5).
Данные не читаются. python3 -m pytest tests/test_s9_c8v2.py -q
"""
import types

import numpy as np
import pytest
import yaml
from scipy import stats

from fastbt import fast_kontrol as fk
from fastbt import s9
from fastbt.s9_stats import minp, past_rank

DOP = yaml.safe_load(s9.C_DOP_YAML.read_text())["dop"]
RSPEC = dict(min_periods=DOP["min_periods"], semi=DOP["rassh"])
RNG = np.random.RandomState(27)
T, N = 3, 1300
n_t = np.array([N, N - 120, N - 400])
S0 = np.full((T, N), np.nan)
for i in range(T):
    S0[i, 25:n_t[i]] = np.exp(RNG.normal(-4, 0.3, n_t[i] - 25))


# ---------------- (а) расширение сетки ----------------

def _flat(S):
    return [(f, lab, G) for f, pts in s9.c_dop_gates(S, None, n_t, RSPEC) for lab, G in pts]


def test_rassh_point_count_matches_yaml():
    got = {f: len(p) for f, p in s9.c_dop_gates(S0, None, n_t, RSPEC)}
    want = {k: v for k, v in DOP["tochek_rassh_na_sigma"].items() if k != "vsego"}
    assert got == want and sum(got.values()) == DOP["tochek_rassh_na_sigma"]["vsego"] == 18
    assert DOP["tochek_rassh_vsego"] == 18 * 6
    labs = [lab for _, lab, _ in _flat(S0)]
    assert len(set(labs)) == len(labs)
    assert {"W126_p50", "W252_p60", "W504_p50", "W252_p50_60", "W126_p60_80", "W504_p50_70"} <= set(labs)


def test_rassh_formulas_equal_c1_c2():
    """Те же формулы, что C1/C2 (S9.c_points): p70 через путь расширения = C1 W252 p70 и т. п."""
    class E:
        pass
    e = E()
    e.n_t = n_t
    e.sigma = lambda src: S0
    f = E()
    f.e = e
    pts1, pts2 = s9.S9.c_points(f, "x")
    old = dict(pts1 + pts2)
    spec = dict(min_periods=DOP["min_periods"],
                semi={"C1": dict(signal="sigma", okna=[126, 252], filtr=[70, 90]),
                      "C2": dict(signal="sigma", okna=[252], chastichnaya=[[70, 90], [70, 80]])})
    for _, pts in s9.c_dop_gates(S0, None, n_t, spec):
        for lab, G in pts:
            assert np.array_equal(G, old[lab]), lab
    R = np.full((T, N), np.nan)
    for ti in range(T):
        R[ti, :n_t[ti]] = past_rank(S0[ti, :n_t[ti]].copy(), 252, minp(252))
    new = dict((lab, G) for _, lab, G in _flat(S0))
    nan = np.isnan(R)
    with np.errstate(invalid="ignore"):
        assert np.array_equal(new["W252_p60"][0], np.where(nan | (R > 0.4), 1.0, 0.0))
        assert np.array_equal(new["W252_p50"][2], np.where(nan | ((R >= 0.5) & (R <= 0.5)), 1.0, 0.0))


def _noise_after(A, t):
    rng = np.random.RandomState(1000 + t)
    B = A.copy()
    B[:, t + 1:] = np.exp(rng.normal(-3, 1.0, B[:, t + 1:].shape))
    return B


def _shift1(A):
    B = np.full_like(A, np.nan)
    B[:, :-1] = A[:, 1:]
    return B


@pytest.mark.parametrize("t", [400, 700, 880])
def test_rassh_future_noise(t):
    base, pert = _flat(S0), _flat(_noise_after(S0, t))
    assert len(base) == len(pert) == 18
    for (f, lab, a), (_, _, b) in zip(base, pert):
        assert np.array_equal(a[:, :, :t + 1], b[:, :, :t + 1]), (f, lab)


def test_rassh_shift_plus1_caught():
    base = _flat(_shift1(S0))
    caught = np.zeros(len(base), bool)
    for t in range(560, 860, 30):
        pert = _flat(_shift1(_noise_after(S0, t)))
        caught |= [not np.array_equal(a[:, :, :t + 1], b[:, :, :t + 1])
                   for (_, _, a), (_, _, b) in zip(base, pert)]
    assert caught.all(), [(f, lab) for (f, lab, _), c in zip(base, caught) if not c]


# ---------------- синтетический движок ----------------

def _engine(G=12, D=500, drift=0.0, seed=0):
    """Пары × дни, строки по парам; A — позиции ±1/0 с инерцией; доходности — шум (+ дрейф по A)."""
    rng = np.random.RandomState(seed)
    n = G * D
    A = np.zeros(n)
    for g in range(G):
        s = np.sign(rng.normal(size=D // 20)).repeat(20)[:D]
        on = rng.random_sample(D) < 0.8
        A[g * D:(g + 1) * D] = s * on
    lr = rng.normal(0, 0.01, n) + drift * A
    pair = np.repeat(np.arange(G), D)
    date = np.tile(np.arange(D), G)
    first = np.r_[True, pair[1:] != pair[:-1]]
    kr = np.full(D, 0.07)
    ns = types.SimpleNamespace(A_row=A, lr_row=lr, first=first, dates=np.arange(D),
                               cnt=np.full(D, float(G)), kr=kr, rf=kr / 365.0,
                               cal=rng.choice([1.0, 3.0], D, p=[0.8, 0.2]),
                               r=dict(pair=pair, date=date))
    import pandas as pd
    ns.dates = pd.date_range("2018-01-01", periods=D, freq="B")
    ns._finish = lambda *a: s9.Engine._finish(ns, *a)
    return ns


def _gate_like_C(ns, seed, share0=0.3, share_half=0.1):
    """Затвор с уровнями 1 / 0,5 / 0 (доли 0,6 / 0,1 / 0,3 по дням с A ≠ 0 в каждой паре)."""
    rng = np.random.RandomState(seed)
    g = np.ones(len(ns.A_row))
    for p in np.unique(ns.r["pair"]):
        idx = np.flatnonzero((ns.r["pair"] == p) & (ns.A_row != 0))
        k0, kh = int(round(share0 * len(idx))), int(round(share_half * len(idx)))
        sel = rng.permutation(idx)
        g[sel[:k0]] = 0.0
        g[sel[k0:k0 + kh]] = 0.5
    return g


def _blocky_gate(ns, seed, block=15):
    """Затвор с сериями (блоки по 15 дней, доля нулей ~ 0,3) — для проверки оборота."""
    rng = np.random.RandomState(seed)
    g = np.ones(len(ns.A_row))
    for p in np.unique(ns.r["pair"]):
        idx = np.flatnonzero((ns.r["pair"] == p) & (ns.A_row != 0))
        v = np.where(rng.random_sample(len(idx) // block + 1) < 0.3, 0.0, 1.0).repeat(block)
        g[idx] = v[:len(idx)]
    return g


def _switches(x, circular):
    n = int(np.sum(x[1:] != x[:-1]))
    return n + int(x[0] != x[-1]) if circular else n


def _linear_shift_P(groups, k=30):
    """Мутация: сдвиг без круга — хвост замещается первым элементом группы (края не заворачиваются)."""
    P = np.arange(len(groups), dtype=np.int32)[None, :].copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        P[0, idx] = idx[np.maximum(np.arange(len(idx)) - k, 0)]
    return P


# ---------------- (б) к2: круговой сдвиг ----------------

def test_shift_index_circular_within_groups():
    rng = np.random.RandomState(1)
    sizes = [7, 1, 300, 120, 50, 44]
    groups = np.repeat(np.arange(6), sizes)[rng.permutation(sum(sizes))]
    P = fk.shift_index(groups, 40, 3)
    assert P.shape == (40, len(groups))
    assert np.array_equal(P, fk.shift_index(groups, 40, 3))         # seed фиксирован
    assert not np.array_equal(P, fk.shift_index(groups, 40, 4))
    for b in range(40):
        assert np.array_equal(np.sort(P[b]), np.arange(len(groups)))
        assert np.array_equal(groups[P[b]], groups)                  # внутри своей пары
    for g, m in enumerate(sizes):
        idx = np.flatnonzero(groups == g)                            # время — по возрастанию индекса
        ks = set()
        for b in range(40):
            src = np.searchsorted(idx, P[b, idx])                    # позиция источника внутри группы
            k = int((0 - src[0]) % m)                                # src[j] = (j − k) mod m
            assert np.array_equal(src, (np.arange(m) - k) % m)       # именно круг, а не перестановка
            ks.add(k)
        if m >= 44:
            assert all(22 <= k <= m - 22 for k in ks), (m, sorted(ks)[:3])
            assert len(ks) > 1 or m == 44                             # m = 44: единственный сдвиг 22
        else:
            assert ks == {0}                                          # допустимых сдвигов нет — на месте


def test_shift_keeps_share_and_turnover():
    ns = _engine(seed=3, D=400)
    k = fk.Kontrol(ns, 0.0004, n_shift=60)
    g = _blocky_gate(ns, 4)
    pos = ns.A_row * g
    ga = k.gate_act(pos)
    fk.check_share(ga, k.P, k.groups)
    for b in range(60):
        sh = ga[k.P[b]]
        for p in np.unique(k.groups):
            m = k.groups == p
            assert np.sum(sh[m] == 1) == np.sum(ga[m] == 1)           # доля дней
            assert _switches(sh[m], True) == _switches(ga[m], True)   # число переключений на круге
            assert abs(_switches(sh[m], False) - _switches(ga[m], False)) <= 1


def test_linear_shift_mutation_caught():
    """Сдвиг без круга меняет долю дней — check_share падает."""
    ns = _engine(seed=3, D=400)
    k = fk.Kontrol(ns, 0.0004, n_shift=5)
    ga = k.gate_act(ns.A_row * _blocky_gate(ns, 4))
    fk.check_share(ga, k.P, k.groups)
    with pytest.raises(AssertionError):
        fk.check_share(ga, np.repeat(_linear_shift_P(k.groups), 3, axis=0), k.groups)


def test_identity_equals_engine_port():
    ns = _engine(seed=5)
    k = fk.Kontrol(ns, 0.0004, n_shift=20)
    pos = ns.A_row * _gate_like_C(ns, 9)
    ref = s9.metrics(s9.Engine.port(ns, pos, comm=0.0004))["sharpe"]
    assert abs(k.identity_sharpe(pos) - ref) < 1e-10
    assert fk.p_perm(1.0, np.array([0.5, 1.0, 2.0])) == 3 / 4


def _p_null(ns, k, reps, informative=False):
    ps = []
    for s in range(reps):
        pos = ns.A_row * (_regime_gate(ns) if informative else _gate_like_C(ns, 100 + s))
        ps.append(fk.p_perm(s9.metrics(s9.Engine.port(ns, pos, comm=k.comm))["sharpe"],
                            k.k2_sharpes(pos)))
    return np.array(ps)


def _regime_gate(ns, block=60):
    """Режим ±1 блоками по 60 дней, общий для всех пар; затвор = 1 в «хорошем» режиме."""
    D = len(ns.dates)
    reg = np.where(np.sign(np.sin(np.arange(D) // block * 2.3)) > 0, 1.0, 0.0)
    return reg[ns.r["date"]]


def test_k2_p_uniform_under_random_mask():
    """Маска C случайная той же доли: p к к2 ~ U(0, 1) (KS), доля p < 0,05 около 5 %."""
    ns = _engine(seed=11, drift=0.002)
    k = fk.Kontrol(ns, 0.0004, n_shift=199)
    ps = _p_null(ns, k, 300)
    assert stats.kstest(ps, "uniform").pvalue > 0.01
    assert 0.01 <= np.mean(ps < 0.05) <= 0.10


def test_k2_informative_gate_has_small_p():
    """Затвор, совпадающий с режимом доходности, сдвиг ломает → p мал (контроль имеет силу)."""
    ns = _engine(seed=11, D=600)
    ns.lr_row = ns.lr_row + 0.004 * ns.A_row * np.where(_regime_gate(ns) > 0, 1.0, -1.0)
    k = fk.Kontrol(ns, 0.0004, n_shift=199)
    assert _p_null(ns, k, 1, informative=True)[0] < 0.05


# ---------------- к3: A без S5/S6 ----------------

def _pairs_sids(G=12):
    return [(f"T{i // 6}", f"S{i % 6 + 1}") for i in range(G)]


def test_k3_zeroes_exactly_s5_s6():
    ns = _engine(seed=2)
    pairs = _pairs_sids()
    k3 = fk.k3_positions(ns.A_row, ns.r["pair"], pairs)
    for pid, (_, sid) in enumerate(pairs):
        m = ns.r["pair"] == pid
        if sid in ("S5", "S6"):
            assert np.all(k3[m] == 0)
        else:
            assert np.array_equal(k3[m], ns.A_row[m])
    assert np.sum(ns.A_row[np.isin(ns.r["pair"], [4, 5, 10, 11])] != 0) > 0     # было что обнулять


def test_k3_mutation_s6_left_caught():
    """k3, не выключающий S6, отличается от верного — и тест выше падает на нём."""
    ns = _engine(seed=2)
    pairs = _pairs_sids()
    bad = fk.k3_positions(ns.A_row, ns.r["pair"], pairs, sids=("S5",))
    good = fk.k3_positions(ns.A_row, ns.r["pair"], pairs)
    assert not np.array_equal(bad, good)
    m6 = np.isin(ns.r["pair"], [5, 11])
    assert np.any(bad[m6] != 0) and np.all(good[m6] == 0)


def test_k3_commission_and_rate_as_A():
    """k3 считается тем же Engine.port: при A без S5/S6 = A доходность и комиссия идентичны (контроль знака)."""
    ns = _engine(seed=7)
    ref = s9.Engine.port(ns, ns.A_row, comm=0.0004)["excess"].values
    same = s9.Engine.port(ns, fk.k3_positions(ns.A_row, ns.r["pair"], [("T", "S1")] * 12),
                          comm=0.0004)["excess"].values
    assert np.array_equal(ref, same)


def test_s9_kontrol_columns_and_rule(monkeypatch):
    """S9.kontrol на синтетическом движке: столбцы v3 есть, k1 нет, passed_v2 = С8 И Шарп > к3 И p < 0,05."""
    ns = _engine(seed=11, D=400)
    ns.pairs = _pairs_sids()
    ns.port = lambda pos, comm=None, cap=1.0, wgt=None: s9.Engine.port(ns, pos, comm=comm, cap=cap, wgt=wgt)
    monkeypatch.setattr(s9, "V2", True)
    monkeypatch.setattr(s9, "SCOPE", "dev")
    st = types.SimpleNamespace(e=ns, reps=50)
    st.delta = lambda *a: s9.S9.delta(st, *a)
    st.boot = lambda p1, p0: s9.S9.boot(st, p1, p0)
    pos = ns.A_row * _regime_gate(ns)
    pt = s9.metrics(ns.port(pos))
    r = dict(sharpe=pt["sharpe"], passed=True)
    s9.S9.kontrol(st, r, pos)
    for c in ("sharpe_k3", "d_vs_k3", "p_vs_k3", "p_vs_k2", "n_shift_k2", "passed_v2"):
        assert c in r, c
    assert not {"c_k1", "sharpe_k1", "d_vs_k1"} & set(r)
    assert r["n_shift_k2"] == fk.N_SHIFT
    k3 = s9.metrics(ns.port(fk.k3_positions(ns.A_row, ns.r["pair"], ns.pairs)))["sharpe"]
    assert r["sharpe_k3"] == k3
    assert r["passed_v2"] == bool(r["sharpe"] > k3 and r["p_vs_k2"] < 0.05)
    r2 = dict(sharpe=pt["sharpe"], passed=False)
    s9.S9.kontrol(st, r2, pos)
    assert r2["passed_v2"] is False


def _v2(monkeypatch, d_sharpe, p):
    """passed_v2 при Шарпе точки = Шарп к3 + d_sharpe и подставленном p_vs_k2 = p (граничные случаи)."""
    ns = _engine(seed=11, D=400)
    ns.pairs = _pairs_sids()
    ns.port = lambda pos, comm=None, cap=1.0, wgt=None: s9.Engine.port(ns, pos, comm=comm, cap=cap, wgt=wgt)
    monkeypatch.setattr(s9, "V2", True)
    monkeypatch.setattr(s9, "SCOPE", "dev")
    monkeypatch.setattr(fk, "p_perm", lambda *a: p)
    st = types.SimpleNamespace(e=ns, reps=50)
    st.delta = lambda *a: s9.S9.delta(st, *a)
    st.boot = lambda p1, p0: s9.S9.boot(st, p1, p0)
    k3 = s9.metrics(ns.port(fk.k3_positions(ns.A_row, ns.r["pair"], ns.pairs)))["sharpe"]
    r = dict(sharpe=k3 + d_sharpe, passed=True)
    s9.S9.kontrol(st, r, ns.A_row * _regime_gate(ns))
    assert r["sharpe_k3"] == k3
    return r["passed_v2"]


def test_c8v2_boundary_sharpe_equal_k3_fails(monkeypatch):
    """Шарп точки ровно равен Шарпу к3 — строгое «>»: не проходит (мутация «>=» краснеет)."""
    assert _v2(monkeypatch, 0.0, 0.001) is False
    assert _v2(monkeypatch, 1e-9, 0.001) is True


def test_c8v2_boundary_p_equal_alpha_fails(monkeypatch):
    """p_vs_k2 ровно 0,05 — строгое «<»: не проходит (мутация «<=» краснеет)."""
    assert s9.C8V2_ALPHA == 0.05
    assert _v2(monkeypatch, 1.0, 0.05) is False
    assert _v2(monkeypatch, 1.0, 0.0499) is True


def test_gate_outside_A_rejected():
    ns = _engine(seed=2)
    k = fk.Kontrol(ns, 0.0004, n_shift=5)
    pos = ns.A_row.copy()
    pos[np.flatnonzero(ns.A_row == 0)[0]] = 1.0
    with pytest.raises(AssertionError):
        k.gate_act(pos)

