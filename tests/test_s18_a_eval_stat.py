"""S-A-EVAL, статистические процедуры `a_eval_stat` (S_A_SPEC разд. 6.3, 8; mutations.md № 103, 160; S_STAT_AUDIT разд. 4): лаг NW-94 = 8 при n = 2198, B = 10 000, блок 20, seed 0;
DM Ньюи–Уэст против независимой формулы; Холм с N программы — известный ответ, мутации «N = числу строк», «NaN-p → 1 бегущим максимумом», «без монотонности»;
MCS только на конечных положительных датах (мутации «без фильтра», «без положительности»). Только monkeypatch / копия исходника (№ 43); реальные файлы только читаются (autouse, сравнение содержимого)."""
from __future__ import annotations

import math
import types
from pathlib import Path

import numpy as np
import pytest

from models_v2.s18 import a_eval as E
from models_v2.s18 import a_eval_stat as S


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        ps = [E.CKPT, E.R.TRIALS] + [E.REP / n for n in E.OUTS] + [E.REP / "S_A_REPORT.md"] + sorted(E.A_DIR.glob("*.parquet"))
        return {str(p): p.read_bytes() for p in ps if p.exists() and p.is_file()}
    before = st()
    yield
    assert st() == before


def mutant(old: str, new: str) -> types.ModuleType:
    src = Path(S.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    m = types.ModuleType("a_eval_stat_mutant")
    m.__file__ = S.__file__
    exec(compile(src.replace(old, new), S.__file__, "exec"), m.__dict__)
    return m


def nw_p(d: np.ndarray, lag: int) -> tuple:
    """Независимая реализация: Бартлетт, d = l_base − l_model, односторонний p «модель лучше»."""
    n, u = len(d), d - d.mean()
    lrv = float(u @ u) / n + sum(2 * (1 - k / (lag + 1)) * float(u[k:] @ u[:-k]) / n for k in range(1, lag + 1))
    z = d.mean() / math.sqrt(lrv / n)
    return z, 0.5 * math.erfc(z / math.sqrt(2))


def holm(p, N: int) -> np.ndarray:
    p = np.asarray(p, float)
    out, run = np.full(len(p), np.nan), 0.0
    f = np.flatnonzero(np.isfinite(p))
    for r, i in enumerate(f[np.argsort(p[f], kind="stable")]):
        run = max(run, min(1.0, (N - r) * p[i]))
        out[i] = run
    return out


def ar_losses(n: int = 800, shift: float = 0.0, seed: int = 3) -> tuple:
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.05, n)
    for t in range(1, n):
        e[t] += 0.5 * e[t - 1]
    base = 0.2 + 0.1 * rng.chisquare(1, n)
    return base, base - shift + e


def test_constants_and_nw94_lag_known_answers() -> None:
    assert (S.B, S.MEAN_BLOCK, S.SEED, S.ALPHA, S.MCS_ALPHA, S.LAG_REF) == (10000, 20, 0, 0.05, 0.10, 5)
    assert S.nw_lag(2198) == 8 and S.nw_lag(100) == 4 and S.nw_lag(1010) == math.ceil(4 * (10.1) ** (2 / 9)) and S.nw_lag(1188) == math.ceil(4 * (11.88) ** (2 / 9))
    assert S.nw_lag(2198) != S.LAG_REF                                                  # основная колонка — 8, а не справочные 5


def test_stationary_bootstrap_block_length_seed_and_shape() -> None:
    idx = S.stat_idx(400, 2000, 20, 0)
    assert idx.shape == (2000, 400) and idx.min() >= 0 and idx.max() < 400
    jump = float(np.mean(idx[:, 1:] != (idx[:, :-1] + 1) % 400))                         # вероятность нового блока ≈ 1/20 (старт мог совпасть: поправка 1/400)
    assert 0.045 < jump < 0.055
    assert np.array_equal(idx, S.stat_idx(400, 2000, 20, 0)) and not np.array_equal(idx, S.stat_idx(400, 2000, 20, 1))
    assert S.idx_for(2198).shape == (10000, 2198)                                       # B = 10 000 на D_M


def test_dm_nw_matches_independent_formula_sign_and_lag_matters() -> None:
    lb, lm = ar_losses(shift=0.01)
    r8, r5 = S.dm_nw(lb, lm, 8), S.dm_nw(lb, lm, 5)
    z, p = nw_p(lb - lm, 8)
    assert r8["stat"] == pytest.approx(z, rel=1e-10) and r8["p_one"] == pytest.approx(p, rel=1e-10) and r8["lag"] == 8
    assert r8["mean_d"] > 0 and r8["p_one"] < 0.5                                       # d = l_base − l_model: плюс — модель лучше
    assert S.dm_nw(lm, lb, 8)["p_one"] == pytest.approx(1 - r8["p_one"], abs=1e-12)     # знак: при перестановке ролей p зеркален
    assert abs(r8["p_one"] - r5["p_one"]) > 1e-6                                        # предусловие: лаг меняет p (иначе мутация «лаг 5» не видна)
    assert math.isnan(S.dm_nw(lb, lb, 8)["p_one"]) and math.isnan(S.dm_nw([1.0], [2.0], 8)["p_one"])      # нулевая дисперсия / одна точка → NaN, не 0 и не 1


def test_boot_ci_delta_base_and_p_known_answer() -> None:
    lb, lh = ar_losses(n=300, shift=0.02)
    idx = S.stat_idx(300, 500, 20, 0)
    c = S.boot_ci(lh, lb, idx)
    assert c["delta_pct"] == pytest.approx(100 * (lh.mean() - lb.mean()) / lb.mean(), rel=1e-12) and c["lo90"] < c["delta_pct"] < c["hi90"]
    assert S.boot_ci(lb + 1.0, lb, idx)["p_boot"] == 1.0 and S.boot_ci(lb - 1.0, lb, idx)["p_boot"] == pytest.approx(1 / 501)     # (#{≥ 0} + 1) / (B + 1)


def test_holm_known_answers_n_not_row_count() -> None:
    assert S.holm_n([0.01, 0.04, 0.03], 3) == pytest.approx([0.03, 0.06, 0.06], rel=1e-12)
    p = [5e-4, 1e-4, 2e-4]
    assert S.holm_n(p, 831) == pytest.approx([0.4145, 0.0831, 0.166], rel=1e-12)       # № 103: N программы, не число строк (на 3 строках было бы 0,0015 / 0,0003 / 0,0004)
    assert S.holm_n(p, 3) != pytest.approx(S.holm_n(p, 831)) and S.holm_n([0.4], 12)[0] == 1.0
    assert S.holm_n(p, 3367)[1] == pytest.approx(0.3367, rel=1e-12)
    with pytest.raises(ValueError):
        S.holm_n([0.1, 0.2, 0.3], 2)


def test_holm_nan_p_stays_nan_and_does_not_poison_others_mutations_caught() -> None:
    out = S.holm_n([0.02, np.nan, 0.01], 10)
    assert out[0] == pytest.approx(0.18) and out[2] == pytest.approx(0.10) and np.isnan(out[1])      # NaN не становится 1 и не сдвигает ранги других
    assert S.holm_n([0.02, np.nan, 0.01], 3)[0] == pytest.approx(0.04)                  # N = 3 при двух конечных p: множитель 3 и 2
    m1 = mutant("ok = np.isfinite(p)", "ok = np.ones(len(p), bool)")                    # мутация: NaN идёт в ранжирование, min(1, nan) = 1
    assert not np.isnan(m1.holm_n([0.02, np.nan, 0.01], 10)[1])
    m2 = mutant("run = max(run, min(1.0, (N - r) * p[i]))", "run = min(1.0, (N - r) * p[i])")      # мутация: без монотонности
    assert not m2.holm_n([0.01, 0.04, 0.03], 3) == pytest.approx([0.03, 0.06, 0.06])
    assert S.holm_n([0.01, 0.04, 0.03], 3) == pytest.approx(holm([0.01, 0.04, 0.03], 3))


def mcs_data(n: int = 300):
    rng = np.random.default_rng(1)
    L = np.exp(rng.normal(-8, 0.3, n))
    rv = L * np.exp(rng.normal(0.0, 0.4, n) + 0.08)
    F = {"good": L * np.exp(rng.normal(0, 0.05, n)), "bad": 3.0 * L, "mid": L * np.exp(rng.normal(0, 0.05, n))}
    rv[5] = 0.0                                                                           # нулевой rv
    F["good"][7] = np.nan                                                                 # NaN в прогнозе
    F["bad"][9] = -1.0                                                                    # отрицательный прогноз
    return rv, F


def test_mcs_only_finite_positive_dates_known_answer() -> None:
    rv, F = mcs_data()
    m = S.finite_pos_mask(rv, F)
    assert int(m.sum()) == len(rv) - 3 and not m[[5, 7, 9]].any()
    r = S.mcs(rv, F, S.stat_idx(int(m.sum()), 1000, 20, 0))
    assert (r["n_used"] == len(rv) - 3).all() and np.isfinite(r["mean_qlike"]).all() and np.isfinite(r["p"]).all()
    want = ((rv[m] / F["good"][m]) - np.log(rv[m] / F["good"][m]) - 1).mean()
    assert r.set_index("series").at["good", "mean_qlike"] == pytest.approx(want, rel=1e-12)
    ix = r.set_index("series")
    assert bool(ix.at["good", "in_mcs"]) and bool(ix.at["mid", "in_mcs"]) and not bool(ix.at["bad", "in_mcs"])      # 3L хуже в 3 раза — вне множества


def test_mcs_without_filter_mutations_are_caught(monkeypatch) -> None:
    rv, F = mcs_data()
    n_ok = int(S.finite_pos_mask(rv, F).sum())
    idx = S.stat_idx(n_ok, 500, 20, 0)

    def honest(mod) -> bool:
        r = mod.mcs(rv, F, idx)
        return bool((r["n_used"] == len(rv) - 3).all() and np.isfinite(r["mean_qlike"]).all())
    assert honest(S)
    for old, new in (("m &= np.isfinite(f) & (f > 0)", "m &= np.isfinite(f)"), ("m = np.isfinite(rv) & (np.asarray(rv, float) > 0)", "m = np.isfinite(rv)"),
                     ("m &= np.isfinite(f) & (f > 0)", "pass")):
        mod = mutant(old, new)
        try:
            ok = honest(mod)
        except Exception:                                                                 # падение на нефильтрованных данных — тоже «поймано»
            ok = False
        assert not ok, new
    monkeypatch.setattr(S, "finite_pos_mask", lambda rv, F: np.ones(len(rv), bool))      # фильтр подменён пропуском всех дат
    with pytest.raises(Exception):
        assert honest(S)
