"""S-A-EVAL, SPA / StepM (`a_eval_spa`; S_A_SPEC разд. 6.3, 8, гейт E.G2): известный ответ на синтетике — один кандидат лучше на 1 % QLIKE базы при σ_LR из данных ⇒ p_c < 0,05 и StepM отвергает ровно его;
все равны ⇒ p > 0,1; под нулём размер теста мал; кандидат хуже базы p не получает; p_l ≤ p_c ≤ p_u; NaN в потерях — отказ. Мутации: знак d (база − кандидат) перевёрнут, «бенчмарк X1 для F_A» (несогласованные слои).
Только monkeypatch / копия исходника (№ 43); реальные файлы только читаются."""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from models_v2.s18 import a_eval as E
from models_v2.s18 import a_eval_spa as SP
from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import a_eval_tab as T


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        ps = [E.CKPT, E.R.TRIALS] + [E.REP / n for n in E.OUTS] + sorted(E.A_DIR.glob("*.parquet"))
        return {str(p): p.read_bytes() for p in ps if p.exists() and p.is_file()}
    before = st()
    yield
    assert st() == before


def mutant(old: str, new: str) -> types.ModuleType:
    src = Path(SP.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    m = types.ModuleType("a_eval_spa_mutant")
    m.__file__ = SP.__file__
    exec(compile(src.replace(old, new), SP.__file__, "exec"), m.__dict__)
    return m


LVL, SIG = 0.2155, 0.02


def draw(seed: int, n: int, effs, phi: float = 0.3):
    """База > 0; кандидат k лучше базы на effs[k]·LVL; разности — AR(1), собственный шум (σ задан в данных, не подогнан под ответ)."""
    rng = np.random.default_rng(seed)
    base = LVL * rng.chisquare(1, n)
    e = rng.normal(0.0, SIG * (1 - phi), (n, len(effs)))
    for t in range(1, n):
        e[t] += phi * e[t - 1]
    return base, base[:, None] + e - np.asarray(effs)[None, :] * LVL


def test_one_candidate_better_by_1pct_rejects_and_equal_does_not() -> None:
    n = 2198
    idx = S.idx_for(n)
    base, L = draw(0, n, (0.01, 0.0, 0.0))
    sig = S.sigma_lr(base - L[:, 0], S.nw_lag(n))
    assert 0.005 < sig < 0.2 and (base - L[:, 0]).mean() == pytest.approx(0.01 * LVL, rel=0.5)      # предусловие: эффект 1 % при данном σ_LR различим
    r = SP.spa_stepm(base, L, idx)
    assert r["p_c"] < 0.05 and r["p_l"] <= r["p_c"] <= r["p_u"] and r["T"] == pytest.approx(r["t"].max())
    assert r["reject"].tolist() == [True, False, False] and r["p_step"][0] < 0.05 <= r["p_step"][1:].min()      # StepM: ровно первый
    eq = SP.spa_stepm(base, np.column_stack([base] * 3), idx)
    assert eq["p_c"] > 0.1 and eq["T"] == 0.0 and not eq["reject"].any()               # копии базы: t = 0, p = 1
    _, Lnull = draw(1, n, (0.0, 0.0, 0.0))
    nul = SP.spa_stepm(base, Lnull, idx)
    assert nul["p_c"] > 0.1 and not nul["reject"].any()                                # все равны в среднем ⇒ p > 0,1


def test_size_and_power_over_repeats_small_b() -> None:
    n, B = 1000, 500
    r0, r1 = [], []
    for i in range(40):
        idx = S.stat_idx(n, B, 20, i)
        b0, L0 = draw(100 + i, n, (0.0, 0.0, 0.0))
        b1, L1 = draw(200 + i, n, (0.03, 0.0, 0.0))
        r0.append(SP.spa_stepm(b0, L0, idx)["p_c"])
        r1.append(SP.spa_stepm(b1, L1, idx)["p_c"])
    assert np.mean(np.array(r0) < 0.05) <= 0.2                                         # размер под нулём
    assert np.mean(np.array(r1) < 0.05) >= 0.9                                         # мощность при 3 % на n = 500


def test_worse_candidate_gets_no_rejection_and_nan_is_refused() -> None:
    n = 800
    idx = S.stat_idx(n, 500, 20, 0)
    base, L = draw(5, n, (-0.05, -0.05))                                               # оба кандидата хуже базы на 5 %
    r = SP.spa_stepm(base, L, idx)
    assert r["p_c"] > 0.5 and not r["reject"].any() and (r["t"] < 0).all()
    L2 = L.copy()
    L2[3, 0] = np.nan
    with pytest.raises(ValueError):
        SP.spa_stepm(base, L2, idx)
    with pytest.raises(ValueError):
        SP.spa_stepm(np.where(np.arange(n) == 4, np.inf, base), L, idx)


def test_sign_of_difference_flipped_mutation_is_caught() -> None:
    n = 800
    idx = S.stat_idx(n, 500, 20, 0)
    good, worse = draw(6, n, (0.05,)), draw(6, n, (-0.05,))
    honest = SP.spa_stepm(good[0], good[1], idx)["p_c"] < 0.05 and SP.spa_stepm(worse[0], worse[1], idx)["p_c"] > 0.5
    assert honest
    mut = mutant("d = lb[:, None] - L", "d = L - lb[:, None]")                          # мутация: d = кандидат − база (плюс — хуже)
    ok = mut.spa_stepm(good[0], good[1], idx)["p_c"] < 0.05 and mut.spa_stepm(worse[0], worse[1], idx)["p_c"] > 0.5
    assert not ok


def make_ls(n: int = 600) -> dict:
    """Слои согласованы: X1_A(S-A*) — над базой X1 (М1), S-A* — над W5-G, K2, K3 (М2). X1_A заметно лучше X1, F_A не лучше W5-G."""
    rng = np.random.default_rng(9)
    Lv = np.exp(rng.normal(-8, 0.3, n))
    rv = Lv * np.exp(rng.normal(0, 0.45, n) + 0.1)
    def q(mult, noise, s):
        return np.exp(np.random.default_rng(s).normal(0, noise, n)) * Lv * mult
    F = {"X1": q(1.45, 0.1, 1), "W5-G": q(1.23, 0.1, 2), "K2": q(1.7, 0.1, 3), "K3": q(1.8, 0.1, 4)}      # оптимум QLIKE ≈ 1,22·L (смещение rv): X1 далеко, W5-G у оптимума
    for i, c in enumerate(T.CANDS):
        F[c] = q(1.26 + 0.01 * i, 0.1, 10 + i)
        F["X1_A(%s)" % c] = q(1.24 + 0.01 * i, 0.1, 20 + i)
    return T.losses(rv, F)


def test_spa_blocks_use_consistent_layers_and_x1_benchmark_mutation_is_caught(monkeypatch) -> None:
    ls = make_ls()
    n = len(ls["X1"])
    idx = S.idx_for(n)
    key = lambda c, b: "X1_A(%s)" % c if b == "X1" else c                              # независимая карта слоёв: X1_A — только против X1
    want = {b: SP.spa_stepm(ls[b], np.column_stack([ls[key(c, b)] for c in T.CANDS]), idx) for b in T.BASES}
    got = T.spa_blocks(ls)
    assert set(got) == set(T.BASES) and T.hkey("S-A2", "X1") == "X1_A(S-A2)" and all(T.hkey("S-A2", b) == "S-A2" for b in ("W5-G", "K2", "K3"))
    for b in T.BASES:
        assert got[b]["p_c"] == want[b]["p_c"] and np.array_equal(got[b]["t"], want[b]["t"]) and np.array_equal(got[b]["reject"], want[b]["reject"])
    assert want["X1"]["p_c"] < 0.05 <= want["W5-G"]["p_c"]                              # предусловие: слои различимы — иначе мутация невидима

    def bench_x1_for_f_a(ls_):                                                          # мутация: бенчмарк X1 для F_A (и слой X1_A против W5-G / K2 / K3)
        return {b: SP.spa_stepm(ls_["X1"], np.column_stack([ls_[c] for c in T.CANDS]), idx) for b in T.BASES}
    mut = bench_x1_for_f_a(ls)
    assert any(mut[b]["p_c"] != want[b]["p_c"] for b in ("W5-G", "K2", "K3"))
    monkeypatch.setattr(T, "hkey", lambda c, b: c)                                      # мутация: несогласованные слои — F_A против бенчмарка X1
    bad = T.spa_blocks(ls)
    assert not all(np.array_equal(bad[b]["t"], want[b]["t"]) for b in T.BASES)
