"""S-A-EVAL, таблицы `a_eval_tab` (S_A_SPEC разд. 6.1, 6.3, 7; mutations.md № 103, 125, 140, 158, 160): Δ = 100·(Q_h/Q_b − 1) от базы (знак: минус — лучше), QLIKE окна = среднее дневных потерь (не среднее годовых),
согласованные слои (X1_A только против X1), Холм N = 831 / 3367 и m = 12 против независимой реализации, NaN-p не становится 1, «подтверждено» только при p Холма < 0,05 и p SPA < 0,05 по своей строке, окна h1/h2 с лагом по их n.
Только monkeypatch / копия исходника (№ 43); реальные файлы только читаются. Синтетика: 700 дат 2021-01-04…, h1 = 2021, h2 = 2022+."""
from __future__ import annotations

import math
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import a_eval as E
from models_v2.s18 import a_eval_spa as SP
from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import a_eval_tab as T

CANDS, BASES = ("S-A0", "S-A1", "S-A2"), ("X1", "W5-G", "K2", "K3")


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        ps = [E.CKPT, E.R.TRIALS] + [E.REP / n for n in E.OUTS] + sorted(E.A_DIR.glob("*.parquet"))
        return {str(p): p.read_bytes() for p in ps if p.exists() and p.is_file()}
    before = st()
    yield
    assert st() == before


def mutant(old: str, new: str) -> types.ModuleType:
    src = Path(T.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    m = types.ModuleType("a_eval_tab_mutant")
    m.__file__ = T.__file__
    exec(compile(src.replace(old, new), T.__file__, "exec"), m.__dict__)
    return m


def synth(n: int = 700):
    rng = np.random.default_rng(9)
    tp = pd.bdate_range("2021-01-04", periods=n)
    Lv = np.exp(rng.normal(-8, 0.3, n))
    rv = Lv * np.exp(rng.normal(0, 0.45, n) + 0.1)

    def q(mult, noise, s):
        return np.exp(np.random.default_rng(s).normal(0, noise, n)) * Lv * mult
    F = {"K1": q(1.9, 0.1, 5), "K2": q(1.7, 0.1, 3), "K3": q(1.8, 0.1, 4), "X1": q(1.45, 0.1, 1), "W5-G": q(1.22, 0.1, 2), "W8-G": q(1.25, 0.1, 6), "B_mix": q(1.6, 0.1, 7), "ZB": q(1.4, 0.1, 8)}
    for i, c in enumerate(CANDS + ("S-A2p",)):
        F[c] = q(1.32 + 0.01 * i, 0.1, 10 + i)                                           # F_A хуже W5-G (плюс), лучше K2 / K3 (минус)
        F["X1_A(%s)" % c] = q(1.24 + 0.01 * i, 0.1, 20 + i)
    F["X1_A(S-A0)"] = F["X1"].copy()                                                     # ровно равен X1: разности нулевые ⇒ DM даёт NaN-p (проверка, что NaN не становится 1)
    return rv, tp, F


RV, TP, F = synth()
LS = T.losses(RV, F)
SPA0 = T.spa_blocks(LS)
SPA = {**SPA0, "K2": {**SPA0["K2"], "p_c": 0.5}}                                         # семейство K2: SPA не значим при значимом Холме строк — проверяет «и» в условии «подтверждено» (S-A2|X1: наоборот)


def key(c: str, b: str) -> str:
    return "X1_A(%s)" % c if b == "X1" else c                                            # независимая карта слоёв


def nw_p(lb, lh, lag: int) -> float:
    d = lb - lh
    n, u = len(d), d - d.mean()
    lrv = float(u @ u) / n + sum(2 * (1 - k / (lag + 1)) * float(u[k:] @ u[:-k]) / n for k in range(1, lag + 1))
    if not lrv > 0:
        return float("nan")
    return 0.5 * math.erfc(d.mean() / math.sqrt(lrv / n) / math.sqrt(2))


def holm(p, N: int) -> np.ndarray:
    p = np.asarray(p, float)
    out, run = np.full(len(p), np.nan), 0.0
    f = np.flatnonzero(np.isfinite(p))
    for r, i in enumerate(f[np.argsort(p[f], kind="stable")]):
        run = max(run, min(1.0, (N - r) * p[i]))
        out[i] = run
    return out


def ql(f) -> np.ndarray:
    r = RV / f
    return r - np.log(r) - 1.0


def check_stats(st: pd.DataFrame, spa: dict = SPA) -> None:
    m = st[st["kind"] == "main"].reset_index(drop=True)
    assert m["hypothesis"].tolist() == ["%s|%s" % (c, b) for c in CANDS for b in BASES] and (m["window"] == "M").all()
    for r in m.itertuples():                                                             # знак и база: Δ = 100 (Q_h / Q_b − 1); минус — кандидат лучше
        h, b = key(r.hypothesis.split("|")[0], r.base), r.base
        assert r.series == h and r.delta_pct == pytest.approx(100.0 * (ql(F[h]).mean() / ql(F[b]).mean() - 1.0), rel=1e-9)
        assert r.qlike_h == pytest.approx(ql(F[h]).mean(), rel=1e-12) and r.qlike_b == pytest.approx(ql(F[b]).mean(), rel=1e-12)      # № 160: среднее дневных потерь
        lag = math.ceil(4 * (len(TP) / 100) ** (2 / 9))                                  # NW-94 по n (при n = 2198 — 8, см. test_s18_a_eval_stat); здесь n = 700 → 7
        assert r.lag == lag != 5 and r.n == len(TP) and r.first == str(TP.min().date()) and r.last == str(TP.max().date())
        assert r.p_dm == pytest.approx(nw_p(ql(F[b]), ql(F[h]), lag), rel=1e-9, nan_ok=True)      # основная колонка — лаг NW-94, не 5
        assert r.p_dm_lag5 == pytest.approx(nw_p(ql(F[b]), ql(F[h]), 5), rel=1e-9, nan_ok=True)
        assert r.p_spa_c == spa[b]["p_c"]                                                # SPA — бенчмарк = база, слой согласован
    assert math.isnan(m.loc[m["hypothesis"] == "S-A0|X1", "p_dm"].iloc[0])               # предусловие: есть NaN-p
    p = m["p_dm"].to_numpy()
    for col, N in (("p_holm_m12", 12), ("p_holm_N", 831), ("p_holm_Nwide", 3367)):
        assert m[col].to_numpy() == pytest.approx(holm(p, N), rel=1e-9, nan_ok=True) and np.isnan(m[col][m["hypothesis"] == "S-A0|X1"]).all()      # NaN-p не стал 1
    assert (st["n_holm"] == 831).all() and (st["n_holm_wide"] == 3367).all() and (st["m_family"] == 12).all() and (st["B"] == 10000).all()
    for r in m.itertuples():                                                             # № 158: «подтверждено» — по своей строке, только при p Холма N и p SPA < 0,05
        conf = r.delta_pct < 0 and r.p_holm_N < 0.05 and r.p_spa_c < 0.05
        assert ("подтверждено (" in r.verdict and "не подтверждено" not in r.verdict) == conf
        assert ("численно лучше базы" in r.verdict) == (r.delta_pct < 0)                 # № 125/140: слово ↔ знак Δ
    assert (st[st["kind"] != "main"]["verdict"] == "справочно, критериев не получает").all() and st.loc[st["kind"] != "main", "p_holm_N"].isna().all()


def test_stats_known_answer_and_preconditions_are_not_vacuous() -> None:
    st = T.stats(RV, TP, F, LS, SPA)
    check_stats(st)
    m = st[st["kind"] == "main"]
    assert (m["verdict"].str.contains("не подтверждено")).any() and (m["verdict"].str.contains("подтверждено") & ~m["verdict"].str.contains("не подтверждено")).any()      # есть и то, и другое
    assert (m["delta_pct"] < 0).any() and (m["delta_pct"] > 0).any() and abs(m["p_dm"] - m["p_dm_lag5"]).max() > 1e-6
    assert st["kind"].value_counts().to_dict() == {"main": 12, "window": 24, "reference": 16, "ablation": 8}


def test_windows_references_and_ablations() -> None:
    st = T.stats(RV, TP, F, LS, SPA)
    w = st[st["kind"] == "window"]
    y = TP.year.to_numpy()
    assert set(w["window"]) == {"h1_2018-2021", "h2_2022-2026"} and w["p_holm_N"].isna().all()
    for r in w.itertuples():
        m = (y <= 2021) if r.window.startswith("h1") else (y >= 2022)
        h, b = r.series, r.base
        assert r.n == int(m.sum()) and r.lag == math.ceil(4 * (m.sum() / 100) ** (2 / 9)) and r.delta_pct == pytest.approx(100 * (ql(F[h])[m].mean() / ql(F[b])[m].mean() - 1), rel=1e-9)
    a = st[st["kind"] == "ablation"].set_index("hypothesis")
    assert a.at["S-A2|S-A1 F", "delta_pct"] == pytest.approx(100 * (ql(F["S-A2"]).mean() / ql(F["S-A1"]).mean() - 1), rel=1e-9)
    assert a.at["S-A2|S-A1 слой X1_A", "delta_pct"] == pytest.approx(100 * (ql(F["X1_A(S-A2)"]).mean() / ql(F["X1_A(S-A1)"]).mean() - 1), rel=1e-9)
    assert set(a.index) == {"%s|%s %s" % (h, b, lab) for h, b in T.ABL for lab in ("F", "слой X1_A")}
    ref = st[st["kind"] == "reference"]
    assert set(ref["base"]) == {"W8-G", "K1", "B_mix", "ZB"} and set(ref["series"]) == set(CANDS) | {"S-A2p"}


def test_table_dates_n_and_qlike_is_mean_of_daily_losses_mutation_160_caught() -> None:
    t = T.table(RV, TP, F)
    assert (t["n"] == len(TP)).all() and (t["first_tplus"] == str(TP.min().date())).all() and (t["last_tplus"] == str(TP.max().date())).all()
    assert list(t["series"][:6]) == ["K1", "K2", "K3", "X1", "W5-G", "W8-G"] and len(t) == 6 + 4 + 2 + 4
    x = t.set_index("series")
    for s in x.index:
        assert x.at[s, "qlike"] == pytest.approx(ql(F[s]).mean(), rel=1e-12)
        for b in BASES:                                                                  # знак и база: Δ к каждой из четырёх баз, от QLIKE базы
            assert x.at[s, "d_%s_pct" % b] == pytest.approx(100 * (ql(F[s]).mean() / ql(F[b]).mean() - 1), rel=1e-9)
    assert x.at["S-A2", "d_K3_pct"] < 0 < x.at["K3", "d_X1_pct"]                         # F_A лучше K3 (минус); K3 хуже X1 (плюс)
    years = [ql(F["X1"])[TP.year == yy].mean() for yy in sorted(set(TP.year))]
    assert abs(float(np.mean(years)) - ql(F["X1"]).mean()) > 1e-6                          # предусловие: «среднее годовых» ≠ среднее дневных
    mut = mutant("float(Q.ql(rv, F[k]).mean())", "float(np.mean([Q.ql(rv, F[k])[tp.year == yy].mean() for yy in sorted(set(tp.year))]))")      # мутация № 160
    tm = mut.table(RV, TP, F).set_index("series")
    assert abs(tm.at["X1", "qlike"] - ql(F["X1"]).mean()) > 1e-6


def test_holm_n_sign_base_and_confirmed_mutations_are_caught(monkeypatch) -> None:
    orig_ci, orig_v = S.boot_ci, T.verdict

    def caught() -> bool:
        try:
            check_stats(T.stats(RV, TP, F, LS, SPA))
        except AssertionError:
            return True
        return False
    assert not caught()
    monkeypatch.setattr(T, "N_HOLM", 12)                                                 # № 103: N = числу строк семейства
    assert caught()
    monkeypatch.setattr(T, "N_HOLM", 831)
    monkeypatch.setattr(T, "N_WIDE", 12)
    assert caught()
    monkeypatch.setattr(T, "N_WIDE", 3367)
    monkeypatch.setattr(S, "holm_n", lambda p, N: holm(p, len(p)))                       # Холм по числу строк вместо N
    assert caught()
    monkeypatch.setattr(S, "holm_n", lambda p, N: np.where(np.isfinite(p), holm(np.nan_to_num(p, nan=1.0), N), 1.0))        # NaN-p → 1
    assert caught()
    monkeypatch.undo()
    monkeypatch.setattr(S, "boot_ci", lambda lh, lb, idx: {**orig_ci(lh, lb, idx), "delta_pct": -orig_ci(lh, lb, idx)["delta_pct"]})       # знак Δ (№ 125)
    assert caught()
    monkeypatch.setattr(S, "boot_ci", lambda lh, lb, idx: {**orig_ci(lh, lb, idx), "delta_pct": 100 * float(np.mean(lh - lb)) / float(np.mean(lh))})      # база процентов — кандидат
    assert caught()
    monkeypatch.undo()
    monkeypatch.setattr(T, "verdict", lambda d, ph, ps: orig_v(-d, ph, ps))              # «лучше / хуже» перевёрнуто
    assert caught()
    monkeypatch.setattr(T, "verdict", lambda d, ph, ps: orig_v(d, ph, 0.0))              # «подтверждено» без условия p SPA
    assert caught()
    monkeypatch.setattr(T, "verdict", lambda d, ph, ps: orig_v(d, 0.0, ps))              # …и без условия p Холма
    assert caught()


def test_lag_5_in_main_column_and_inconsistent_layers_mutations_are_caught(monkeypatch) -> None:
    monkeypatch.setattr(S, "nw_lag", lambda n: 5)                                        # мутация: лаг NW 5 в основной колонке
    with pytest.raises(AssertionError):
        check_stats(T.stats(RV, TP, F, LS, SPA))
    monkeypatch.undo()
    monkeypatch.setattr(T, "hkey", lambda c, b: c)                                       # мутация: F_A против базы X1 (слой X1_A потерян)
    with pytest.raises(AssertionError):
        check_stats(T.stats(RV, TP, F, LS, SPA))
    monkeypatch.undo()
    idx = S.idx_for(len(TP))
    wrong = {b: SP.spa_stepm(LS["X1"], np.column_stack([LS[c] for c in CANDS]), idx) for b in BASES}      # мутация: SPA с бенчмарком X1 для F_A
    assert wrong["W5-G"]["p_c"] != SPA["W5-G"]["p_c"]
    with pytest.raises(AssertionError):
        check_stats(T.stats(RV, TP, F, LS, wrong))


def test_verdict_truth_table() -> None:
    v = T.verdict
    assert "подтверждено (" in v(-1.0, 0.04, 0.04) and "не подтверждено" not in v(-1.0, 0.04, 0.04) and v(-1.0, 0.04, 0.04).startswith("численно лучше базы")
    for d, ph, ps in ((-1.0, 0.04, 0.06), (-1.0, 0.06, 0.04), (-1.0, 0.05, 0.04), (-1.0, 0.04, 0.05), (-1.0, np.nan, 0.01), (-1.0, 0.01, np.nan)):
        assert "не подтверждено" in v(d, ph, ps) and v(d, ph, ps).startswith("численно лучше базы")        # граница 0,05 строгая; NaN не подтверждает
    for d in (0.5, 0.0, np.nan):
        assert v(d, 0.0, 0.0) == "численно не лучше базы; не подтверждено"
    assert T.N_HOLM == 831 and T.N_WIDE == 3367 and T.M_FAMILY == 12 and T.CANDS == CANDS and T.BASES == BASES
    assert T.hkey("S-A1", "X1") == "X1_A(S-A1)" and T.hkey("S-A1", "K3") == "S-A1" and T.hkey("S-A1", "W5-G") == "S-A1"
