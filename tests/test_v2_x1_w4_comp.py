"""W4-B2, ядро ступени 1 (W4_SPEC 2.1–2.3, гейт B.G4): при ρ ≡ 0 M⁺ = M побитно; ρ — argmin QLIKE с ничьей в пользу меньшего; K_a — ln mean(RV/M); первая переоценка — первый месяц с ≥ 312 закрытыми парами;
до неё g = 0, ρ = 0; PIT: будущие цели не меняют прошлые месяцы. Синтетика через тот же код (`walk`), короткий отрезок; реальные таблицы только читаются. Подмены — в `test_v2_x1_w4_comp_mut.py`."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w4_comp as C

ROOT = Path(C.__file__).resolve().parents[2]
REP = ROOT / "docs/otchety/harj_x1"
REAL = [REP / "ALL_TRIALS.csv", REP / "CHECKPOINT.json", REP / "W4_STAGE1.csv", REP / "W4_STAGE1_REPORT.md", ROOT / "data/regen/harj_x1/W4/mplus_S1.parquet"]


def real_state() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in REAL}


@pytest.fixture(autouse=True)
def real_files_untouched():
    before = real_state()
    yield
    assert real_state() == before


def synth(n: int = 520, seed: int = 3) -> tuple:
    """n рабочих дней с 2017-01-03; t⁺ = следующий день; e = 0,4·tanh(x0) + шум 0,3 (сигнал есть)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-01-03", periods=n + 1)
    X = rng.normal(size=(n, 3))
    M = 0.0004 * np.exp(rng.normal(0, 0.2, n))
    e = 0.4 * np.tanh(X[:, 0]) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"t_plus": idx[1:], "rv": M * np.exp(e), "M": M}, index=idx[:-1]), X


@pytest.fixture(scope="module")
def base() -> dict:
    tab, X = synth()
    return {"tab": tab, "X": X, "out": C.walk(tab, X)}


def test_mplus_rho_zero_is_M_bitwise_for_any_g() -> None:
    M = np.array([0.0003, 0.0005, 1e-4])
    for g in (np.array([5.0, -7.0, 0.3]), np.array([np.pi, -1e9, 1e-300])):
        assert np.array_equal(C.mplus(M, g, np.zeros(3)), M)
    assert np.allclose(C.mplus(M, np.array([0.5, -0.5, 9.0]), np.array([1.0, 1.0, 1.0])), M * np.exp([0.5, -0.5, 1.0]))   # clip ±1


def test_walk_with_rho_zero_learner_gives_M_bitwise_while_g_is_nonzero(base) -> None:
    def learner(X, e):
        k, _, pred = C.gbm_learner(X, e)
        return k, 0.0, pred

    out = C.walk(base["tab"], base["X"], learner)
    assert (out["g"] != 0).any() and np.array_equal(out["F"].to_numpy(float), out["M"].to_numpy(float))
    assert not np.array_equal(base["out"]["F"].to_numpy(float), base["out"]["M"].to_numpy(float))   # а с выбранным ρ > 0 — M⁺ ≠ M


def test_pick_rho_known_answers_and_tie_goes_to_smaller() -> None:
    e = np.array([0.2, -0.1, 0.3, -0.25, 0.15])
    assert C.pick_rho(e, e) == 1.0                                    # g точный → ρ = 1
    assert C.pick_rho(e, 2 * e) == 0.5                                # g вдвое больше → ρ = 0,5
    assert C.pick_rho(e, -e) == 0.0                                   # знак перепутан → ρ = 0
    assert C.pick_rho(e, np.zeros(5)) == 0.0                          # ничья всех ρ → меньшее


def test_const_g_is_log_mean_of_ratio() -> None:
    assert abs(C.const_g(np.log([1.0, 3.0])) - np.log(2.0)) < 1e-15
    assert abs(C.const_g(np.zeros(4))) == 0.0


def test_shifted_is_circular_roll_of_full_table() -> None:
    X = pd.DataFrame({"a": np.arange(6.0)}, index=pd.bdate_range("2019-01-01", periods=6))
    s = C.shifted(X, 2)
    assert list(s["a"]) == [4.0, 5.0, 0.0, 1.0, 2.0, 3.0] and list(s.index) == list(X.index)


def test_first_refit_is_first_month_with_312_closed_pairs(base) -> None:
    tab = base["tab"]
    tp = tab["t_plus"].to_numpy("datetime64[ns]")
    firsts = [tab.index[tab.index.to_period("M") == m][0] for m in tab.index.to_period("M").unique() if m.start_time >= pd.Timestamp("2017-01-01")]
    cnt = {d: int((tp <= np.datetime64(d)).sum()) for d in firsts}
    want = next(d for d in firsts if cnt[d] >= 312)
    assert C.first_refit(tab) == want and cnt[want] >= 312
    assert all(cnt[d] < 312 for d in firsts if d < want)             # предыдущий месяц — меньше 312


def test_before_first_refit_g_zero_rho_zero_F_equals_M_and_no_refit_with_less_than_312(base) -> None:
    out, tab = base["out"], base["tab"]
    first = C.first_refit(tab)
    pre = out["d"] < first
    assert pre.any() and (out.loc[pre, "g"] == 0).all() and (out.loc[pre, "rho"] == 0).all() and out.loc[pre, "refit_date"].isna().all()
    assert np.array_equal(out.loc[pre, "F"].to_numpy(float), out.loc[pre, "M"].to_numpy(float))
    tp = tab["t_plus"].to_numpy("datetime64[ns]")
    rd = out["refit_date"].dropna().unique()
    assert len(rd) >= 3 and all((tp <= r).sum() >= 312 for r in rd)


def test_real_table_first_refit_matches_independent_count() -> None:
    tab = C.load_table()
    tp = tab["t_plus"].to_numpy("datetime64[ns]")
    mon = tab.index.to_period("M")
    firsts = [tab.index[mon == m][0] for m in mon.unique() if m.start_time >= pd.Timestamp("2017-01-01")]
    want = next(d for d in firsts if (tp <= np.datetime64(d)).sum() >= 312)
    got = C.first_refit(tab)
    assert got == want and (tp <= np.datetime64(got)).sum() >= 312 and tab["t_plus"].min() == pd.Timestamp("2017-01-03")
    assert got < pd.Timestamp("2018-06-01")                          # окно M начинается 2018-01-04: переоценки есть с первых месяцев


def test_pit_future_targets_do_not_change_past_months(base) -> None:
    tab, X, out = base["tab"], base["X"], base["out"]
    D = sorted(out["refit_date"].dropna().unique())[2]
    alt = tab.copy()
    alt.loc[tab["t_plus"] > D, "rv"] *= 50.0
    o2 = C.walk(alt, X)
    m = (out["refit_date"].notna() & (out["refit_date"] <= D)).to_numpy()
    assert m.sum() > 20
    for c in ("g", "rho", "n_trees", "F"):
        assert np.array_equal(out.loc[m, c].to_numpy(), o2.loc[m, c].to_numpy()), c
    assert not np.array_equal(out["g"].to_numpy(), o2["g"].to_numpy())  # а будущие месяцы свои цели видят — тест не пустой


def test_no_module_attribute_assignment_in_w4_comp_tests() -> None:
    for p in sorted(ROOT.glob("tests/test_v2_x1_w4_comp*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        mods = {n.asname or n.name for t in ast.walk(tree) if isinstance(t, (ast.ImportFrom, ast.Import)) for n in t.names}
        for n in ast.walk(tree):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign)) else []):
                assert not (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id in mods), (p.name, ast.unparse(n))
