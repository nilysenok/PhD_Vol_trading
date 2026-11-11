"""W8-B, проход W8-G на синтетике (W8_SPEC 2–4, 9; mutations.md № 1, 164): известный ответ β_G = Δū + ½Δs² (ddof 1) по группе, границы [0; 0,5]; в днях gap = 0 F = X1 побитно; группа не включена (< 30 пар gap = 1
группы в закрытом пуле) ⇒ F = W5-G побитно; метка пары — её собственная; «закрыто + 1» через боевой `walk`; порча RV пар с t⁺ > D не меняет β_{m,G}; сдвиг 0 даёт W8-Gb = W8-G (ловится).
Мутации кода — подменой исходника в отдельном модуле (боевой файл не меняется) и monkeypatch (№ 43). Реальные файлы только читаются (autouse-хэши)."""
from __future__ import annotations

import hashlib
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w5_gap as G5
from models_v2.s16 import x1_w8_gap as G
from models_v2.s16 import x1_z_walk as ZW


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        s = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (G.REG.TRIALS, G.REG.CKPT) if p.exists()}
        s.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(G.REG.REP.glob("W8_*")) if p.is_file()})
        s["W8"] = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(G.OUT.rglob("*")) if p.is_file()} if G.OUT.exists() else {}
        return s
    before = st()
    yield
    assert st() == before


def mutant(old: str, new: str) -> types.ModuleType:
    src = Path(G.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    m = types.ModuleType("x1_w8_gap_mutant")
    m.__file__ = G.__file__
    exec(compile(src.replace(old, new), G.__file__, "exec"), m.__dict__)
    return m


def assert_known_betas(m) -> None:
    k = np.arange(40)
    u1, u0 = 0.2 + 0.01 * k, 0.005 * k                                               # ū = 0,395 / 0,0975; s² = 1,366667e-2 / 3,416667e-3 (ddof 1)
    u = np.r_[u1, u0]
    g = np.r_[np.ones(40, int), np.zeros(40, int)]
    sel = g == 1
    assert m._beta(u, g, sel) == pytest.approx(0.2975 + 0.5 * 0.01025, abs=1e-12)    # 0,302625 = Δū + ½Δs²
    assert m._beta(np.r_[u0 - 0.2, u0], g, sel) == 0.0                               # отрицательный эффект → нижняя граница 0
    assert m._beta(np.r_[u1 + 3.0, u0], g, sel) == 0.5                               # верхняя граница 0,5
    two = np.r_[u1, u1 + 9.0, u0]                                                    # группа — подмножество дней gap = 1: другие дни gap = 1 в ū₁,G не входят
    gg = np.r_[np.ones(80, int), np.zeros(40, int)]
    assert m._beta(two, gg, np.r_[np.ones(40, bool), np.zeros(80, bool)]) == pytest.approx(0.302625, abs=1e-12)


def test_known_answer_beta_on_real_code() -> None:
    assert_known_betas(G)
    assert (G.MIN_G, G.BMAX, G.SHIFT, G.GROUPS) == (30, 0.5, 252, ("T1", "T2", "T3"))        # константы спецификации


@pytest.mark.parametrize("old,new", [
    (" + 0.5 * (u1.var(ddof=1) - u0.var(ddof=1))", ""),                              # без члена ½Δs²
    ("u1.var(ddof=1) - u0.var(ddof=1)", "u1.var(ddof=0) - u0.var(ddof=0)"),            # ddof = 0
    ("), 0.0, BMAX))", "), -9.0, BMAX))"),                                           # без нижней границы 0
    ("), 0.0, BMAX))", "), 0.0, 9.0))"),                                             # без верхней границы
    ("u1, u0 = u[sel], u[g == 0]", "u1, u0 = u[g == 1], u[g == 0]"),                  # ū₁ по всем дням gap = 1, а не по группе
])
def test_beta_mutations_are_caught(old, new) -> None:
    with pytest.raises(AssertionError):
        assert_known_betas(mutant(old, new))


def synth(n: int = 2200, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    d, tp = pd.bdate_range("2018-01-03", periods=n), pd.bdate_range("2018-01-04", periods=n)
    g = (np.arange(n) % 2).astype(int)
    grp = np.array(["T1", "T2", "T3"], object)[np.arange(n) // 2 % 3]
    mu, sd = {"T1": 0.25, "T2": 0.05, "T3": 0.15}, {"T1": 0.3, "T2": 0.2, "T3": 0.25}      # u ~ N(μ_G, σ²_G) в днях gap = 1, N(0, 0,2²) в gap = 0
    u = np.where(g == 1, [mu[a] + sd[a] * rng.normal() for a in grp], 0.2 * rng.normal(size=n))
    base = np.exp(rng.normal(-8, 0.3, n))
    return dict(d=d, tp=tp, g=g, grp=grp, base=base, rv=base * np.exp(u), u=u)


S = synth()
W5 = G5.walk(S["d"], S["tp"], S["rv"], S["base"], S["g"])


def run(s=S, grp=None, **kw) -> pd.DataFrame:
    w5 = G5.walk(s["d"], s["tp"], s["rv"], s["base"], s["g"])
    return G.walk(s["d"], s["tp"], s["rv"], s["base"], s["g"], s["grp"] if grp is None else grp, w5["beta"], w5["refit_date"], **kw)


def test_walk_beta_equals_independent_formula_and_synthetic_truth() -> None:
    w = run()
    d, tp = pd.DatetimeIndex(S["d"]), pd.DatetimeIndex(S["tp"])
    refits = sorted(w["refit_date"].dropna().unique())
    assert pd.Timestamp(refits[0]) == pd.Timestamp("2018-07-02")
    for D in refits[:15]:
        pool = np.flatnonzero(tp <= D)
        u, g, gr = S["u"][pool], S["g"][pool], S["grp"][pool]
        row = w[w["refit_date"] == D].iloc[0]
        for T in G.GROUPS:
            a, b = u[(g == 1) & (gr == T)], u[g == 0]
            assert row["n_" + T] == len(a)
            if len(a) >= 30:                                                          # включена: β_{m,G} = clip(Δū + ½Δs², 0; 0,5) по закрытым парам t⁺ ≤ D
                assert row["b_" + T] == pytest.approx(float(np.clip(a.mean() - b.mean() + 0.5 * (a.var(ddof=1) - b.var(ddof=1)), 0, 0.5)), abs=1e-12)
            else:
                assert row["b_" + T] == W5.loc[W5["refit_date"] == D, "beta"].iloc[0]       # не включена → β W5-G
    last = w.iloc[-1]
    for T in G.GROUPS:                                                                # истинное β_G = Δμ + ½Δσ² (σ₀ = 0,2): T1 0,275, T2 0,05, T3 0,1563
        assert abs(last["b_" + T] - ({"T1": 0.275, "T2": 0.05, "T3": 0.15 + 0.5 * (0.0625 - 0.04)}[T])) < 0.035
    assert w["beta"].between(0, 0.5).all() and G.check_series(w, W5, len(w), "W8-G") == []


def test_f_is_x1_in_gap_zero_and_group_off_equals_w5g_bitwise_and_group_beta_acts_on_own_day() -> None:
    w = run()
    on = ((w["gap"] == 1) & w["refit_date"].notna()).to_numpy()
    assert (w.loc[~on, "F"].to_numpy() == S["base"][~on]).all()                      # gap = 0 и до включения: F = X1 побитно
    assert (w.loc[on, "F"].to_numpy() == S["base"][on] * np.exp(w.loc[on, "beta"].to_numpy())).all()
    for T in G.GROUPS:                                                                # β дня = β своей группы (метка дня d действует на день d)
        m = on & (S["grp"] == T)
        assert (w.loc[m, "beta"].to_numpy() == w.loc[m, "b_" + T].to_numpy()).all()
    off = on & np.array([w.at[i, "n_" + a] < 30 for i, a in enumerate(S["grp"])])
    assert off.any() and (w.loc[off, "F"].to_numpy() == W5.loc[off, "F"].to_numpy()).all()       # группа не включена ⇒ F = W5-G побитно
    assert (w.loc[w["refit_date"].isna(), "beta"] == 0).all()


def test_off_group_mutation_f_not_w5g_and_correction_in_gap_zero_are_caught_by_check_series() -> None:
    w = run()
    bad = w.copy()
    z = (bad["gap"] == 0).to_numpy() & bad["refit_date"].notna().to_numpy()
    bad.loc[z, "F"] = bad.loc[z, "X1"] * 1.01                                         # поправка в днях gap = 0
    assert any("F ≠ X1" in b for b in G.check_series(bad, W5, len(bad), "t"))
    m = mutant("else float(w5_beta[a])", "else 0.0").walk                             # мутация: не включённая группа получает 0, а не β W5-G
    f = m(S["d"], S["tp"], S["rv"], S["base"], S["g"], S["grp"], W5["beta"], W5["refit_date"])
    assert any("F ≠ W5-G" in b for b in G.check_series(f, W5, len(f), "t"))
    m = mutant("F = np.where((gap == 1) & ~np.isnat(refit), base * np.exp(beta), base)", "F = np.where(~np.isnat(refit), base * np.exp(beta), base)").walk
    f = m(S["d"], S["tp"], S["rv"], S["base"], S["g"], S["grp"], W5["beta"], W5["refit_date"])
    assert any("F ≠ X1" in b for b in G.check_series(f, W5, len(f), "t"))             # мутация «поправка в днях gap = 0»
    assert G.check_series(w.assign(beta=w["beta"] + 0.6), W5, len(w), "t") and G.check_series(w, W5, len(w) + 1, "t")           # β вне [0; 0,5]; не 2198 дат


def thirty_case(ones: int, **kw) -> pd.Series:
    """Первая переоценка D = 2018-07-02: в закрытом пуле ровно `ones` пар gap = 1 группы T3 (остальные gap = 1 — T1); W5-G включена."""
    tpd = pd.DatetimeIndex(S["tp"])
    pool = np.flatnonzero(tpd <= pd.Timestamp("2018-07-02"))
    ids = pool[(S["g"][pool] == 1)]
    grp = S["grp"].copy()
    grp[pool] = "T1"
    grp[ids[:ones]] = "T3"
    assert (grp[pool][S["g"][pool] == 1] == "T3").sum() == ones
    w = run(grp=grp, **kw)
    return w[w["refit_date"] == pd.Timestamp("2018-07-02")].iloc[0], W5.loc[W5["refit_date"] == pd.Timestamp("2018-07-02"), "beta"].iloc[0]


def assert_thirty(**kw) -> None:
    for ones, incl in ((30, True), (29, False)):
        r, b5 = thirty_case(ones, **kw)
        assert r["n_T3"] == ones
        assert bool(r["b_T3"] != b5) is incl, ones                                      # ровно 30 пар — группа включена (своё β), 29 — β W5-G


def test_group_included_only_at_thirty_pairs_and_mutation_is_caught() -> None:
    assert_thirty()
    with pytest.raises(AssertionError):
        assert_thirty(min_g=5)                                                       # мутация: группа включена при < 30 парах
    with pytest.raises(AssertionError):
        assert_thirty(min_g=31)                                                      # и обратная: порог выше 30
    assert G.walk.__defaults__[-1] == 30                                             # умолчание боевого кода


def leaks(s=S, grp=None) -> bool:
    """Порча RV пар с t⁺ > D (×3) не должна менять β_{m,G} и F строк d ≤ D."""
    base, tpd = run(s, grp), pd.DatetimeIndex(s["tp"])
    cols = ["beta", "F", "b_T1", "b_T2", "b_T3"]
    for D in sorted(base["refit_date"].dropna().unique())[:10]:
        rv = s["rv"].copy()
        rv[tpd > D] *= 3.0
        alt = run(dict(s, rv=rv), grp)
        m = (base["d"] <= D).to_numpy()
        if not np.array_equal(base.loc[m, cols].to_numpy(), alt.loc[m, cols].to_numpy(), equal_nan=True):
            return True
    return False


def test_pit_corrupted_future_pairs_do_not_change_beta_and_closed_plus_one_is_caught(monkeypatch) -> None:
    assert leaks() is False
    monkeypatch.setattr(ZW, "closed_date", lambda tp, d: d)                          # мутация № 164: «закрыто + 1» — пара с d = D (t⁺ = D + 1) в пуле
    assert leaks() is True


def test_pair_label_is_its_own_and_recomputed_late_label_mutation_is_caught() -> None:
    from models_v2.s16 import x1_w8_gap_z as Z
    rng = np.random.default_rng(9)
    z = rng.normal(size=400)
    lab = Z.labels(z)
    assert (lab[:150] == Z.labels(z[:150])).all()                                     # метка пары не зависит от будущих z (пороги по дням строго до d)
    z2 = z.copy()
    z2[200:] += 50.0
    assert (Z.labels(z2)[:200] == lab[:200]).all()
    late = lambda zz: Z.terc_past(zz)                                                 # noqa: E731
    whole = lambda zz: np.where(np.isfinite(zz), np.where(zz <= np.quantile(zz, 1 / 3), "T1", np.where(zz <= np.quantile(zz, 2 / 3), "T2", "T3")), "na").astype(object)    # noqa: E731
    assert (late(z2)[:200] == lab[:200]).all() and not (whole(z2)[:200] == whole(z)[:200]).all()       # мутация «поздние пороги»: порча будущего меняет метки прошлого


def test_shift_zero_makes_w8gb_equal_w8g_and_is_caught() -> None:
    from models_v2.s16 import x1_w8_gap_z as Z
    z = np.log(np.abs(np.random.default_rng(1).normal(size=len(S["d"]))) + 0.1)
    assert (Z.shift_z(z, 252) != z).any() and Z.SHIFT == 252 and (Z.shift_z(z, 0) == z).all()
    w, gb, g0 = run(grp=Z.labels(z)), run(grp=Z.labels(Z.shift_z(z))), run(grp=Z.labels(Z.shift_z(z, 0)))
    assert not w["F"].equals(gb["F"]) and w["F"].to_numpy().tobytes() == g0["F"].to_numpy().tobytes()      # мутация «сдвиг 0»: W8-Gb = W8-G побитно — её ловит первое утверждение
    assert (Z.shift_z(np.arange(10.0), 3) == np.roll(np.arange(10.0), 3)).all()
