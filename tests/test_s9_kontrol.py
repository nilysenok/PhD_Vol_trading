"""К3 (Ъ 27.10; к2 — постановка v3 §0.4): контроли к2 (круговой сдвиг маски по тикеру, лаг ∈ [21, T−21])
и к3 (A без S5/S6) — fastbt.fast_kontrol и S9.kontrol. Данные не читаются; синтетика.

Каждый тест ловит подложенную ошибку (мутации — в памяти, в тесте):
  лаг 0 → исходная маска; сдвиг без круга; лаг вне [21, T−21]; один лаг на все тикеры;
  разные лаги у стратегий одного тикера; к3 без S6; границы С8 v2 (Шарп = к3, p = 0,05).
python3 -m pytest tests/test_s9_kontrol.py -q
"""
import types

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fastbt import fast_kontrol as fk
from fastbt import s9

DS = (500, 450, 600)               # длина ряда тикера (торговых дней)


def _engine(Ds=DS, drift=0.0, seed=0):
    """3 тикера × 6 стратегий; строки по парам; A — позиции ±1/0 с инерцией и дырами; доходности — шум."""
    rng = np.random.RandomState(seed)
    pair, tick, date, A = [], [], [], []
    for ti, D in enumerate(Ds):
        for s in range(6):
            pair.append(np.full(D, ti * 6 + s))
            tick.append(np.full(D, ti))
            date.append(np.arange(D))
            sg = np.sign(rng.normal(size=D // 20 + 1)).repeat(20)[:D]
            A.append(sg * (rng.random_sample(D) < 0.8))
    pair, tick, date, A = (np.concatenate(x) for x in (pair, tick, date, A))
    n, Dm = len(A), max(Ds)
    first = np.r_[True, pair[1:] != pair[:-1]]
    kr = np.full(Dm, 0.07)
    ns = types.SimpleNamespace(
        A_row=A, lr_row=rng.normal(0, 0.01, n) + drift * A, first=first,
        cnt=np.bincount(date, minlength=Dm).astype(float), kr=kr, rf=kr / 365.0,
        cal=rng.choice([1.0, 3.0], Dm, p=[0.8, 0.2]), r=dict(pair=pair, tick=tick, date=date),
        dates=pd.date_range("2018-01-01", periods=Dm, freq="B"),
        pairs=[(f"T{i // 6}", f"S{i % 6 + 1}") for i in range(3 * 6)])
    ns._finish = lambda *a: s9.Engine._finish(ns, *a)
    ns.port = lambda pos, comm=None, cap=1.0, wgt=None: s9.Engine.port(ns, pos, comm=comm, cap=cap, wgt=wgt)
    return ns


def _gate_rand(ns, seed, share0=0.3, share_half=0.1):
    """Затвор 1 / 0,5 / 0 (доли 0,6 / 0,1 / 0,3) по ВСЕМ дням каждой пары, в т. ч. где A = 0."""
    rng = np.random.RandomState(seed)
    g = np.ones(len(ns.A_row))
    for p in np.unique(ns.r["pair"]):
        idx = np.flatnonzero(ns.r["pair"] == p)
        k0, kh = int(share0 * len(idx)), int(share_half * len(idx))
        sel = rng.permutation(idx)
        g[sel[:k0]] = 0.0
        g[sel[k0:k0 + kh]] = 0.5
    return g


def _gate_blocky(ns, seed, block=15):
    rng = np.random.RandomState(seed)
    g = np.ones(len(ns.A_row))
    for p in np.unique(ns.r["pair"]):
        idx = np.flatnonzero(ns.r["pair"] == p)
        v = np.where(rng.random_sample(len(idx) // block + 1) < 0.3, 0.0, 1.0).repeat(block)
        g[idx] = v[:len(idx)]
    return g


def _gate_regime(ns, block=60):
    """Режим блоками по 60 дней, общий для всех пар; затвор = 1 в «хорошем» режиме."""
    reg = np.where(np.sign(np.sin(np.arange(max(DS)) // block * 2.3)) > 0, 1.0, 0.0)
    return reg[ns.r["date"]]


def _runs(x):
    """Мультимножество длин серий на круге (поворот так, чтобы начать со смены значения)."""
    k = np.flatnonzero(x != np.roll(x, 1))
    if len(k) == 0:
        return [len(x)]
    return sorted(np.diff(np.r_[k, k[0] + len(x)]).tolist())


# ---------------- к2: лаг 0, сохранение свойств ----------------

def test_lag0_gives_original_mask_and_engine_sharpe():
    ns = _engine(seed=5)
    k = fk.Kontrol(ns, 0.0004, n_shift=20)
    g = _gate_rand(ns, 9)
    src0 = fk.src_rows(np.zeros(k.LAG.shape[1], np.int64), k.tick, k.pstart, k.plen, k.loc)
    assert np.array_equal(g[src0], g)                              # сдвиг 0 → исходная маска
    ref = s9.metrics(ns.port(ns.A_row * g, comm=0.0004))["sharpe"]
    assert abs(k.identity_sharpe(g) - ref) < 1e-10
    bad = fk.src_rows(np.ones(k.LAG.shape[1], np.int64), k.tick, k.pstart, k.plen, k.loc)
    assert not np.array_equal(g[bad], g)                           # ловит «лаг 0 ≠ тождество»
    assert fk.p_perm(1.0, np.array([0.5, 1.0, 2.0])) == 3 / 4


def test_shift_keeps_turnover_runs_and_share():
    ns = _engine(seed=3)
    k = fk.Kontrol(ns, 0.0004, n_shift=60)
    g = _gate_blocky(ns, 4)
    k.check(g)
    for b in range(60):
        s = g[k.src(b)]
        for p in np.unique(k.pair):
            m = k.pair == p
            assert sorted(s[m]) == sorted(g[m])                    # доля дней и уровни
            assert np.sum(s[m] != np.roll(s[m], 1)) == np.sum(g[m] != np.roll(g[m], 1))   # оборот
            assert _runs(s[m]) == _runs(g[m])                      # длины серий


def test_linear_shift_without_circle_caught():
    """Подложенная ошибка: сдвиг без круга (край замещается первым значением) меняет долю дней."""
    ns = _engine(seed=3)
    k = fk.Kontrol(ns, 0.0004, n_shift=5)
    g = _gate_blocky(ns, 4)
    lag = np.full(k.LAG.shape[1], 30)
    bad = k.pstart + np.maximum(k.loc - lag[k.tick], 0)
    with pytest.raises(AssertionError):
        fk.check_share(g, bad, k.pair)
    fk.check_share(g, fk.src_rows(lag, k.tick, k.pstart, k.plen, k.loc), k.pair)   # круг — проходит


# ---------------- к2: лаг, независимость по тикерам, один лаг на тикер ----------------

def _lag_ok(LAG, T, lo=21):
    return all(LAG[:, i].min() >= lo and LAG[:, i].max() <= T[i] - lo for i in range(len(T)))


def _independent(LAG):
    if all(np.array_equal(LAG[:, 0], LAG[:, i]) for i in range(1, LAG.shape[1])):
        return False
    c = np.corrcoef(LAG.T.astype(float))
    return np.max(np.abs(c - np.eye(len(c)))) < 0.2


def test_lag_in_21_to_T_minus_21():
    T = np.array(DS)
    LAG = fk.draw_lags(T, 1000)
    assert _lag_ok(LAG, T)
    assert LAG[:, 0].min() < 40 and LAG[:, 0].max() > DS[0] - 40   # края диапазона достигаются
    assert not _lag_ok(fk.draw_lags(T, 1000, min_lag=5), T)         # подложенная ошибка: [5, T−5]
    assert not _lag_ok(LAG + 1, T)                                  # и сдвиг границы
    assert (fk.draw_lags(np.array([41, 42]), 50)[:, 0] == 0).all()   # T < 42 — лага нет
    assert (fk.draw_lags(np.array([41, 42]), 50)[:, 1] == 21).all()  # T = 42 — единственный лаг 21
    assert np.array_equal(LAG, fk.draw_lags(T, 1000))                # seed фиксирован


def test_tickers_shifted_independently():
    LAG = fk.draw_lags(np.array(DS), 1000)
    assert _independent(LAG)
    same = np.repeat(LAG[:, :1], LAG.shape[1], axis=1)               # подложенная ошибка: один лаг на всех
    assert not _independent(same)


def test_all_strategies_of_ticker_share_one_lag():
    ns = _engine(Ds=(500, 500, 500), seed=1)
    k = fk.Kontrol(ns, 0.0004, n_shift=30)
    day = k.loc.astype(float)                                       # g = номер дня: сдвиг виден прямо
    for b in range(30):
        sh = day[k.src(b)]
        for ti in range(3):
            rows = [sh[k.pair == p] for p in range(ti * 6, ti * 6 + 6)]
            assert all(np.array_equal(rows[0], x) for x in rows[1:])  # одна маска у всех стратегий тикера
        assert not np.array_equal(sh[k.tick == 0][:500], sh[k.tick == 1][:500]) or k.LAG[b, 0] == k.LAG[b, 1]
    # подложенная ошибка: лаг по паре, а не по тикеру — стратегии тикера расходятся
    lag_pair = np.arange(18) + 21
    bad = k.pstart + (k.loc - lag_pair[k.pair]) % k.plen
    shb = day[bad]
    assert not np.array_equal(shb[k.pair == 0], shb[k.pair == 1])


def test_kernel_nonzero_lag_equals_engine_port():
    """shift_sharpe при ненулевом LAG = Шарп Engine.port(A · g[src(b)]) до 1e-12; лаг по паре в ядре
    дал бы другое число (подложенная ошибка ловится сравнением с вариантом «лаг по паре»)."""
    ns = _engine(seed=8)
    k = fk.Kontrol(ns, 0.0004, n_shift=6)
    g = _gate_blocky(ns, 5)
    assert (k.LAG > 0).all()
    sh = k.k2_sharpes(g)
    for b in range(6):
        ref = s9.metrics(ns.port(ns.A_row * g[k.src(b)], comm=0.0004))["sharpe"]
        assert abs(sh[b] - ref) < 1e-12, b
        lag_pair = k.LAG[b][k.tick[k.pstart]][np.arange(18)] + np.arange(18)     # лаг по паре
        bad = k.pstart + (k.loc - lag_pair[k.pair]) % k.plen
        bad_sh = s9.metrics(ns.port(ns.A_row * g[bad], comm=0.0004))["sharpe"]
        assert abs(sh[b] - bad_sh) > 1e-6, b


def test_seed_and_min_lag_match_declaration():
    """Константы кода = объявлению в setka_C.yaml (seed 20261027, лаг от 21, 1 000 повторов)."""
    import yaml
    kon = yaml.safe_load(s9.C_DOP_YAML.read_text())["dop"]["kontroli"]
    assert fk.SEED == 20261027 == kon["seed"]
    assert fk.MIN_LAG == 21 == kon["min_shift_dney"]
    assert fk.N_SHIFT == 1000 == kon["n_shift"]


# ---------------- к2: размер критерия и сила ----------------

def _p_list(ns, k, gates, comm=0.0004):
    return np.array([fk.p_perm(s9.metrics(ns.port(ns.A_row * g, comm=comm))["sharpe"], k.k2_sharpes(g))
                     for g in gates])


def test_k2_p_uniform_under_random_mask():
    """Маска случайная той же доли: p к к2 ~ U(0, 1) (KS), доля p < 0,05 около 5 %."""
    ns = _engine(seed=11, drift=0.002)
    k = fk.Kontrol(ns, 0.0004, n_shift=199)
    ps = _p_list(ns, k, [_gate_rand(ns, 100 + s) for s in range(250)])
    assert stats.kstest(ps, "uniform").pvalue > 0.01
    assert 0.01 <= np.mean(ps < 0.05) <= 0.10


def test_k2_informative_mask_has_small_p():
    """Маска совпадает с режимом доходности, сдвиг её ломает → p мал; сдвиг маски на другой режим — нет."""
    ns = _engine(seed=11)
    g = _gate_regime(ns)
    ns.lr_row = ns.lr_row + 0.004 * ns.A_row * np.where(g > 0, 1.0, -1.0)
    k = fk.Kontrol(ns, 0.0004, n_shift=199)
    assert _p_list(ns, k, [g])[0] < 0.05
    assert _p_list(ns, k, [1.0 - g])[0] > 0.5                       # перепутанный знак маски → p не мал


# ---------------- к3: A без S5/S6 ----------------

def test_k3_zeroes_exactly_s5_s6():
    ns = _engine(seed=2)
    k3 = fk.k3_positions(ns.A_row, ns.r["pair"], ns.pairs)
    for pid, (_, sid) in enumerate(ns.pairs):
        m = ns.r["pair"] == pid
        if sid in ("S5", "S6"):
            assert np.all(k3[m] == 0) and np.any(ns.A_row[m] != 0)
        else:
            assert np.array_equal(k3[m], ns.A_row[m])


def test_k3_mutation_s6_left_caught():
    ns = _engine(seed=2)
    bad = fk.k3_positions(ns.A_row, ns.r["pair"], ns.pairs, sids=("S5",))
    good = fk.k3_positions(ns.A_row, ns.r["pair"], ns.pairs)
    m6 = np.array([sid == "S6" for _, sid in ns.pairs])[ns.r["pair"]]
    assert np.any(bad[m6] != 0) and np.all(good[m6] == 0)


# ---------------- S9.kontrol: столбцы, правило, границы ----------------

def _stub(monkeypatch, ns):
    monkeypatch.setattr(s9, "V2", True)
    monkeypatch.setattr(s9, "SCOPE", "dev")
    st = types.SimpleNamespace(e=ns, reps=50)
    st.delta = lambda *a: s9.S9.delta(st, *a)
    st.boot = lambda p1, p0: s9.S9.boot(st, p1, p0)
    return st


def test_s9_kontrol_columns_and_rule(monkeypatch):
    ns = _engine(seed=11)
    st = _stub(monkeypatch, ns)
    g = _gate_regime(ns)
    pos = ns.A_row * g
    sh = s9.metrics(ns.port(pos))["sharpe"]
    r = dict(sharpe=sh, passed=True)
    s9.S9.kontrol(st, r, pos, g)
    for c in ("sharpe_k3", "d_vs_k3", "p_vs_k3", "p_vs_k2", "n_shift_k2", "passed_v2"):
        assert c in r, c
    assert not {"c_k1", "sharpe_k1", "d_vs_k1"} & set(r)
    k3 = s9.metrics(ns.port(fk.k3_positions(ns.A_row, ns.r["pair"], ns.pairs)))["sharpe"]
    assert r["sharpe_k3"] == k3 and r["n_shift_k2"] == fk.N_SHIFT
    assert r["passed_v2"] == bool(sh > k3 and r["p_vs_k2"] < 0.05)
    r2 = dict(sharpe=sh, passed=False)
    s9.S9.kontrol(st, r2, pos, g)
    assert r2["passed_v2"] is False
    r3 = dict(sharpe=sh, passed=True)                                # u02: маски нет — к2 не определён
    s9.S9.kontrol(st, r3, pos)
    assert r3["passed_v2"] is False and r3["n_shift_k2"] == 0
    with pytest.raises(AssertionError):                              # позиция ≠ A × g
        s9.S9.kontrol(st, dict(sharpe=sh, passed=True), pos * 0.5, g)


def _v2(monkeypatch, d_sharpe, p):
    ns = _engine(seed=11)
    st = _stub(monkeypatch, ns)
    monkeypatch.setattr(fk, "p_perm", lambda *a: p)
    k3 = s9.metrics(ns.port(fk.k3_positions(ns.A_row, ns.r["pair"], ns.pairs)))["sharpe"]
    r = dict(sharpe=k3 + d_sharpe, passed=True)
    g = _gate_regime(ns)
    s9.S9.kontrol(st, r, ns.A_row * g, g)
    assert r["sharpe_k3"] == k3
    return r["passed_v2"]


def test_c8v2_boundary_sharpe_equal_k3_fails(monkeypatch):
    """Шарп точки ровно равен Шарпу к3 — строгое «>» (мутация «>=» краснеет)."""
    assert _v2(monkeypatch, 0.0, 0.001) is False
    assert _v2(monkeypatch, 1e-9, 0.001) is True


def test_c8v2_boundary_p_equal_alpha_fails(monkeypatch):
    """p_vs_k2 ровно 0,05 — строгое «<» (мутация «<=» краснеет)."""
    assert s9.C8V2_ALPHA == 0.05
    assert _v2(monkeypatch, 1.0, 0.05) is False
    assert _v2(monkeypatch, 1.0, 0.0499) is True
