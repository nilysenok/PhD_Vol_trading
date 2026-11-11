"""W4-B2, мутации ступени 1 (W4_SPEC 9, W4-B; mutations.md № 1, 99, 100, 142, 164): «закрыто + 1» (через `closed` и через `WF.closed_rows`), выбор ρ/деревьев по парам t⁺ > d₀,
блок ранней остановки в обучении (№ 99), финальная модель без последних 252 пар (№ 100), K_b с нулевым сдвигом. Всё через боевой `walk`/`job` на синтетике; подмены — только monkeypatch."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s13 import v6_1_fit as FT
from models_v2.s13 import v6_1_wf as WF
from models_v2.s13 import v6_1b_sig as SG
from models_v2.s16 import x1_w3_core as W3
from models_v2.s16 import x1_w4_comp as C

ROOT = Path(C.__file__).resolve().parents[2]
REP = ROOT / "docs/otchety/harj_x1"
REAL = [REP / "ALL_TRIALS.csv", REP / "CHECKPOINT.json", REP / "W4_STAGE1.csv", ROOT / "data/regen/harj_x1/W4/mplus_S1.parquet"]
ORIG_SELECT, ORIG_TRAIN = C.select_rho_trees, FT._train


def real_state() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in REAL}


@pytest.fixture(autouse=True)
def real_files_untouched():
    before = real_state()
    yield
    assert real_state() == before


def synth(n: int = 520, seed: int = 3) -> tuple:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-01-03", periods=n + 1)
    X = rng.normal(size=(n, 3))
    M = 0.0004 * np.exp(rng.normal(0, 0.2, n))
    e = 0.4 * np.tanh(X[:, 0]) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"t_plus": idx[1:], "rv": M * np.exp(e), "M": M}, index=idx[:-1]), X


TAB, XN = synth()


def leaks(monkeypatch, mutant=None) -> bool:
    """Меняет цели пар с t⁺ > D (D — дата третьей переоценки) в 50 раз. True, если прогнозы месяцев с переоценкой ≤ D изменились (будущее просочилось в прошлое)."""
    base = C.walk(TAB, XN) if mutant is None else (mutant(monkeypatch, TAB, XN), C.walk(TAB, XN))[1]
    D = sorted(base["refit_date"].dropna().unique())[2]
    alt = TAB.copy()
    alt.loc[TAB["t_plus"] > D, "rv"] *= 50.0
    if mutant is not None:
        mutant(monkeypatch, alt, XN)
    o2 = C.walk(alt, XN)
    m = (base["refit_date"].notna() & (base["refit_date"] <= D)).to_numpy()
    assert m.sum() > 20
    return any(not np.array_equal(base.loc[m, c].to_numpy(), o2.loc[m, c].to_numpy()) for c in ("g", "rho", "n_trees", "F"))


def m_closed_plus_one(mp, tab, X):
    mp.setattr(C, "closed", lambda tp, d0: np.flatnonzero(np.asarray(tp, "datetime64[ns]") <= np.datetime64(pd.Timestamp(d0) + pd.offsets.BDay(1))))


def m_closed_rows_plus_one(mp, tab, X):
    mp.setattr(WF, "closed_rows", lambda tp, d0: np.flatnonzero(np.asarray(tp, "datetime64[ns]") <= np.datetime64(pd.Timestamp(d0) + pd.offsets.BDay(1))))


def m_select_by_future_pairs(mp, tab, X):
    e_all = np.log(tab["rv"].to_numpy(float) / tab["M"].to_numpy(float))
    mp.setattr(C, "select_rho_trees", lambda Xc, e: ORIG_SELECT(X[-400:], e_all[-400:]))     # ρ и деревья по последним парам таблицы (t⁺ > d₀)


def test_honest_walk_has_no_leak(monkeypatch) -> None:
    assert leaks(monkeypatch) is False


@pytest.mark.parametrize("mutant", [m_closed_plus_one, m_closed_rows_plus_one, m_select_by_future_pairs], ids=lambda f: f.__name__)
def test_mutation_future_pairs_are_caught_through_battle_walk(monkeypatch, mutant) -> None:
    assert leaks(monkeypatch, mutant) is True


def train_calls(monkeypatch) -> list:
    calls = []

    def rec(X, e, n_trees):
        calls.append(len(e))
        return ORIG_TRAIN(X, e, n_trees)

    monkeypatch.setattr(FT, "_train", rec)
    return calls


def expected_calls(out: pd.DataFrame) -> list:
    tp = TAB["t_plus"].to_numpy("datetime64[ns]")
    seq = []
    for r in out["refit_date"].dropna().unique():
        n = int((tp <= r).sum())
        seq += [n - 252, n]                                            # проверка/остановка — обучение без последних 252; итог — все закрытые пары
    return seq


def test_honest_training_sets_are_closed_minus_252_then_all_closed(monkeypatch) -> None:
    calls = train_calls(monkeypatch)
    out = C.walk(TAB, XN)
    assert calls == expected_calls(out) and len(calls) >= 6


def test_mutation_99_early_stopping_block_in_training_is_caught(monkeypatch) -> None:
    monkeypatch.setattr(FT, "split_es", lambda n, hold=252: (np.arange(n), np.arange(n - hold, n)))
    calls = train_calls(monkeypatch)
    out = C.walk(TAB, XN)
    assert calls != expected_calls(out)
    assert all(calls[i] == calls[i + 1] for i in range(0, len(calls), 2))      # обучение на всех закрытых парах вместо n − 252


def test_mutation_100_final_model_without_last_252_is_caught(monkeypatch) -> None:
    monkeypatch.setattr(FT, "final_idx", lambda n, hold=252: np.arange(n - hold))
    calls = train_calls(monkeypatch)
    out = C.walk(TAB, XN)
    assert calls != expected_calls(out) and all(calls[i] == calls[i + 1] for i in range(0, len(calls), 2))


def patch_inputs(monkeypatch, n: int = 520) -> None:
    S = pd.DataFrame(XN, index=TAB.index, columns=["s1", "s2", "s3"])
    for c in C.S_COLS[3:]:
        S[c] = np.random.default_rng(5).normal(size=len(S))
    G = pd.DataFrame(np.random.default_rng(6).normal(size=(len(S), 12)), index=S.index, columns=list(SG.SIGMA_NAMES))
    monkeypatch.setattr(C, "load_table", lambda: TAB)
    monkeypatch.setattr(C, "feature_source", lambda: (S[C.S_COLS], G))


def test_kb_shifted_differs_from_mplus_and_is_worse_on_signal(monkeypatch) -> None:
    patch_inputs(monkeypatch)
    _, a = C.job(("S1", 0))
    _, b = C.job(("S1", C.SHIFT))
    assert len(a) == len(b) and (a["d"] == b["d"]).all() and not np.array_equal(a["F"].to_numpy(), b["F"].to_numpy())
    q = lambda df: float(W3.qlike(df["rv"].to_numpy(float), df["F"].to_numpy(float)).mean())  # noqa: E731
    assert q(a) < q(b)                                                   # сигнал есть: K_b (признаки со сдвигом 252) хуже M⁺


def test_mutation_kb_with_zero_shift_equals_mplus_and_is_caught(monkeypatch) -> None:
    patch_inputs(monkeypatch)
    _, a = C.job(("S1", 0))
    monkeypatch.setattr(C, "shifted", lambda X, k: X)                    # мутация: сдвиг не применён
    _, b = C.job(("S1", C.SHIFT))
    assert np.array_equal(a["F"].to_numpy(), b["F"].to_numpy())          # K_b = M⁺ побитно — тест «K_b ≠ M⁺» упал бы


def test_job_window_and_ka_control(monkeypatch) -> None:
    patch_inputs(monkeypatch)
    _, a = C.job(("S1", 0))
    _, k = C.job(("Ka", 0))
    assert a["t_plus"].min() >= C.WINDOW_START and len(a) == int((TAB["t_plus"] >= C.WINDOW_START).sum()) == len(k)
    assert (k["n_trees"] == 0).all() and np.isfinite(k["F"]).all()
