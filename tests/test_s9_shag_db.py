"""Шаг 3′ (D, B): scripts/analysis/s9_shag_common.py, s9_shag_db.py на синтетике; данные не читаются.
Мутации §6 (в памяти) роняют проверки: комиссия выхода на границе, к2 со сдвигом 0, к3 с невыключенной S5,
условие 3 для B пропущено. Мутации входов (σ / RVI) — tests/test_s9_shag_db_mut.py; константы и run() — _run.py.
python3 -m pytest tests/test_s9_shag_db.py -q
"""
import sys
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


def _inf_scale(ns):
    """Информативный множитель масштаба (B, D7): 1 в дни с верным знаком, 0,3 в прочие."""
    return np.where(ns.A_row * ns.lr_row > 0, 1.0, 0.3)


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


def test_check_shift_lag0_and_lags():
    ns = _world()
    g = _inf_gate(ns)
    K = fk.Kontrol(ns, COMM, n_shift=20)
    sh = s9.metrics(ns.port(ns.A_row * g, comm=COMM))["sharpe"]
    C.check_shift(K, g, sh)
    with pytest.raises(AssertionError, match="сдвиг 0"):
        C.check_shift(K, np.roll(g, 1), sh)                      # маска не та — сдвиг 0 не воспроизводит точку
    K.LAG[0, 0] = 0
    with pytest.raises(AssertionError, match="лаг вне"):
        C.check_shift(K)


def test_k2_zero_shift_mutation_caught():
    ns = _world()
    g = _inf_gate(ns)
    base = C.make_base(ns, COMM, 50)
    K = fk.Kontrol(ns, COMM, n_shift=100)
    ok = C.eval_point(ns, K, base, ns.A_row * g, g)
    assert ok["p_vs_k2"] < 0.05 and ok["usl3"] is True
    K.LAG[:] = 0                                                 # мутация: сдвиг 0 у всех повторов
    with pytest.raises(AssertionError, match="лаг вне"):         # eval_point зовёт check_shift сам
        C.eval_point(ns, K, base, ns.A_row * g, g)


def test_run_grid_calls_check_shift(monkeypatch):
    ns = _world()
    ns.g_row = lambda G: G
    calls = []
    orig = C.check_shift
    monkeypatch.setattr(C, "check_shift", lambda *a, **k: (calls.append(a), orig(*a, **k))[1])
    monkeypatch.setattr(DB, "points", lambda *a: [("D1", "p", _inf_gate(ns), False, 0.3, True)])
    DB.run_grid("D", ns, s9, ("x",), COMM, 50)
    assert len(calls) == 2 and calls[0][1:] == () and calls[1][1] is not None   # лаги + сдвиг 0 точки


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


def test_flags_condition3_applies_to_scale_too(monkeypatch):
    ns = _world()
    base, K = C.make_base(ns, COMM, 50), fk.Kontrol(ns, COMM, n_shift=100)
    for g in (_inf_gate(ns), _inf_scale(ns)):
        r = C.eval_point(ns, K, base, ns.A_row * g, g)
        assert r["usl3"] is True and r["p_vs_k2"] < 0.05 and "p_vs_k2_masshtab" not in r
        assert r["passed_v2"] == bool(r["usl1"] and r["usl2"] and r["usl3"])
        assert r["sharpe_k3"] == base["m3"]["sharpe"] and r["usl2"] == (r["sharpe"] > r["sharpe_k3"])
    for k in ("d_vs_A", "p_vs_A", "d_vs_k3", "p_vs_k3", "ce1_pct_y", "fko_nad_poroga_pct", "znak_2018"):
        assert k in r
    assert _noise_scale_blocked(ns, K, base)
    monkeypatch.setattr(C.fk, "p_perm", lambda *a: 0.0)              # мутация: условие 3 для B пропущено
    assert not _noise_scale_blocked(ns, K, base)                     # …шум проходит ложно — проверка выше упала бы


def _noise_scale_blocked(ns, K, base) -> bool:
    """Масштаб без информации (перемешан) условием 3 останавливается: p ≥ α, usl3 False, не прошёл."""
    rnd = np.random.RandomState(1).permutation(_inf_scale(ns))
    r = C.eval_point(ns, K, base, ns.A_row * rnd, rnd)
    return r["usl3"] is False and not r["passed_v2"] and r["p_vs_k2"] >= C.ALPHA


def test_k2_scale_shift_invariants():
    """Сдвиг ряда масштаба: сдвиг 0 — исходный ряд; распределение масштаба и число серий (±1 на шве круга) сохранены; длины серий не сравниваются."""
    ns = _world()
    K = fk.Kontrol(ns, COMM, n_shift=20)
    g =np.concatenate([np.repeat(np.random.RandomState(p).choice([1.0, 0.6, 0.3], 60), 20)[:np.sum(ns.r["pair"] == p)]
                        for p in np.unique(ns.r["pair"])])
    assert np.array_equal(g[fk.src_rows(np.zeros(len(K.t_tick), int), K.tick, K.pstart, K.plen, K.loc)], g)
    K.check(g)                                                      # мультимножество уровней и число переключений
    src = K.src(3)
    for p in np.unique(ns.r["pair"]):
        m = ns.r["pair"] == p
        n_runs = lambda x: int(np.sum(x[1:] != x[:-1])) + 1   # noqa: E731  число серий
        a, s = g[m], g[src][m]
        assert sorted(a) == sorted(s) and abs(n_runs(a) - n_runs(s)) <= 1


def test_select_max_sharpe_among_passed_only():
    df = pd.DataFrame(dict(sharpe=[2.0, 1.0, 1.5], passed_v2=[False, True, True], point=list("abc")))
    assert C.select(df)["point"] == "c"
    assert C.select(df.assign(passed_v2=False)) is None


def test_grid_counts_match_yaml(monkeypatch):
    assert C.plan("D")["na_ryad"] == 66 and C.plan("B")["na_ryad"] == 9
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
