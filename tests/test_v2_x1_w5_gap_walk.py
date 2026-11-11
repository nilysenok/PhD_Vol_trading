"""W5-B, поправка W5-G на синтетике (W5_SPEC 1, 2, 11.7; mutations.md № 1, 164): известный ответ β = Δū + ½Δs² (ddof 1); граница [0; 0,5]; в днях gap = 0 и до включения F = X1 побитно; включение не раньше 2018-07-02
и при ≥ 30 парах в каждой группе; «закрыто + 1» через боевой `walk`; порча RV пар с t⁺ > D не меняет β; сдвиг меток 252. Мутации кода — подменой исходника в отдельном модуле (не правит боевой файл), либо monkeypatch (№ 43)."""
from __future__ import annotations

import hashlib
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w5_gap as G
from models_v2.s16 import x1_z_walk as ZW

REAL = (G.GAP_PIT, G.REG.TRIALS, G.REG.CKPT)


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        s = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in REAL}
        s["W5"] = sorted(str(p) for p in G.OUT.rglob("*")) if G.OUT.exists() else []
        return s
    before = st()
    yield
    assert st() == before


def mutant(old: str, new: str) -> types.ModuleType:
    """Копия боевого `x1_w5_gap` с одной подложенной ошибкой в исходнике; сам файл не меняется."""
    src = Path(G.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    m = types.ModuleType("x1_w5_gap_mutant")
    m.__file__ = G.__file__
    exec(compile(src.replace(old, new), G.__file__, "exec"), m.__dict__)
    return m


def assert_known_betas(m) -> None:
    k = np.arange(40)
    u1, u0 = 0.2 + 0.01 * k, 0.005 * k                                               # ū = 0,395 / 0,0975; s² = 1,366667e-2 / 3,416667e-3 (ddof 1)
    g = np.r_[np.ones(40, int), np.zeros(40, int)]
    assert m.beta_gap(np.r_[u1, u0], g) == pytest.approx(0.2975 + 0.5 * 0.01025, abs=1e-12)          # 0,302625
    assert m.beta_gap(np.r_[u0 - 0.2, u0], g) == 0.0                                  # отрицательный эффект → нижняя граница 0
    assert m.beta_gap(np.r_[u1 + 3.0, u0], g) == 0.5                                  # верхняя граница 0,5
    assert m.beta_gap(np.r_[u1[:29], u0], np.r_[np.ones(29, int), np.zeros(40, int)]) is None          # < 30 пар gap = 1
    assert m.beta_gap(np.r_[u1, u0[:29]], np.r_[np.ones(40, int), np.zeros(29, int)]) is None          # < 30 пар gap = 0
    assert m.beta_gap(np.r_[u1[:30], u0[:30]], np.r_[np.ones(30, int), np.zeros(30, int)]) is not None  # ровно 30 — достаточно
    v = 0.01 * np.arange(60)
    assert m.beta_all(v) == pytest.approx(0.295 + 0.5 * 1e-4 * 60 * 61 / 12, abs=1e-12) and m.beta_all(v[:59]) is None       # ū + ½s², ≥ 60 пар
    assert m.beta_all(np.full(60, -2.0)) == -0.5 and m.beta_all(np.full(60, 2.0)) == 0.5          # контроль (а): границы ±0,5


def test_known_answer_beta_on_real_code() -> None:
    assert_known_betas(G)


@pytest.mark.parametrize("old,new", [
    (" + 0.5 * (u1.var(ddof=1) - u0.var(ddof=1))", ""),                              # без члена ½Δs²
    ("u1.var(ddof=1) - u0.var(ddof=1)", "u1.var(ddof=0) - u0.var(ddof=0)"),            # ddof = 0
    ("), 0.0, BMAX))", "), -9.0, BMAX))"),                                           # без нижней границы 0
    ("), 0.0, BMAX))", "), 0.0, 9.0))"),                                             # без верхней границы
    ("np.clip(u.mean() + 0.5 * u.var(ddof=1), -BMAX, BMAX)", "np.clip(u.mean() + 0.5 * u.var(ddof=1), 0.0, BMAX)"),      # контроль (а) без отрицательной стороны
    ("u.mean() + 0.5 * u.var(ddof=1)", "u.mean()"),                                  # контроль (а) без ½s²
])
def test_beta_mutations_are_caught(old, new) -> None:
    with pytest.raises(AssertionError):
        assert_known_betas(mutant(old, new))


def synth(n: int = 700, seed: int = 2, effect: float = 0.2) -> dict:
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2018-01-03", periods=n)
    tp = pd.bdate_range("2018-01-04", periods=n)
    g = (np.arange(n) % 2).astype(int)
    x1 = np.exp(rng.normal(-8, 0.3, n))
    return dict(d=d, tp=tp, g=g, base=x1, rv=x1 * np.exp(effect * g + rng.normal(0, 0.2, n)))


S = synth()


def run(s=S, **kw) -> pd.DataFrame:
    return G.walk(s["d"], s["tp"], s["rv"], s["base"], s["g"], **kw)


def test_walk_beta_equals_independent_formula_and_f_is_base_where_off() -> None:
    w = run()
    d, tp = pd.DatetimeIndex(S["d"]), pd.DatetimeIndex(S["tp"])
    u = np.log(S["rv"] / S["base"])
    refits = sorted(w["refit_date"].dropna().unique())
    assert pd.Timestamp(refits[0]) == pd.Timestamp("2018-07-02")                      # включение: первый торговый день ≥ 2018-07-02
    for D in refits[:12]:
        pool = np.flatnonzero(tp <= D)
        a, b = u[pool][S["g"][pool] == 1], u[pool][S["g"][pool] == 0]
        want = float(np.clip(a.mean() - b.mean() + 0.5 * (a.var(ddof=1) - b.var(ddof=1)), 0, 0.5))
        assert w.loc[w["refit_date"] == D, "beta"].eq(want).all()                     # β месяца — известный ответ по закрытым парам t⁺ ≤ D
    on = (w["gap"] == 1) & w["refit_date"].notna()
    assert (w.loc[~on, "F"].to_numpy() == S["base"][~on.to_numpy()]).all()           # в днях gap = 0 и до включения F = X1 побитно (11.7)
    assert (w.loc[on, "F"].to_numpy() == S["base"][on.to_numpy()] * np.exp(w.loc[on, "beta"].to_numpy())).all()
    assert w["beta"].between(0, 0.5).all() and abs(w["beta"].iloc[-1] - 0.2) < 0.06  # эффект 0,2 восстановлен
    assert G.check_series({"W5-G": w}, len(w)) == []
    for p in refits[:-1]:                                                             # (последний месяц ряда неполный) β постоянна внутри месяца и меняется только в первый торговый день
        m = w["refit_date"] == p
        assert m.sum() >= 15 and w.loc[m, "beta"].nunique() == 1 and d[np.flatnonzero(m)[0]] == pd.Timestamp(p)


def test_correction_in_gap_zero_days_is_caught_by_check_series() -> None:
    assert G.check_series({"W5-G": run(all_days=True)}, len(S["d"]))                 # мутация «поправка в днях gap = 0»: F ≠ X1 побитно
    assert G.check_series({"W5-G": run(beta_fn=lambda u, g: 9.0)}, len(S["d"]))      # β вне [0; 0,5]


def test_activation_and_thirty_pairs_rule_and_mutations(monkeypatch) -> None:
    tpd = pd.DatetimeIndex(S["tp"])
    pool = np.flatnonzero(tpd <= pd.Timestamp("2018-07-02"))
    n = len(pool)
    for ones, enabled in ((30, True), (29, False), (n - 30, True), (n - 29, False)):
        s = dict(S)
        g = (np.arange(len(S["d"])) % 2).astype(int)
        g[pool] = 0
        g[pool[np.linspace(0, n - 1, ones).astype(int)]] = 1
        s["g"] = g
        assert int(g[pool].sum()) == ones
        r = run(s)
        assert bool(r.loc[r["d"] == pd.Timestamp("2018-07-02"), "refit_date"].notna().iloc[0]) is enabled, ones          # граница 30 пар в каждой группе
        if not enabled:
            july = r["d"].dt.month.eq(7) & r["d"].dt.year.eq(2018)
            assert (r.loc[july, "F"].to_numpy() == s["base"][july.to_numpy()]).all() and (r.loc[july, "beta"] == 0).all()
    s29 = dict(S)
    s29["g"] = np.zeros(len(S["d"]), int)
    s29["g"][pool[:29]] = 1
    s29["g"][len(pool):] = (np.arange(len(S["d"]) - len(pool)) % 2)
    ok = run(s29)["refit_date"].min()
    monkeypatch.setattr(G, "MIN_G", 5)                                               # мутация: включение при 5 парах
    assert run(s29)["refit_date"].min() < ok                                         # раньше (v=29 пар хватает при пороге 5): мутация видна
    monkeypatch.setattr(G, "MIN_G", 30)
    monkeypatch.setattr(G, "ACTIVATION", pd.Timestamp("2018-05-01"))                 # мутация: включение раньше 2018-07-02
    assert run()["refit_date"].min() < pd.Timestamp("2018-07-02")                    # …видна; честный код: см. первый тест (refit == 2018-07-02)


def leaks(s=S) -> bool:
    """Порча RV пар с t⁺ > D (×3) не должна менять β и F строк d ≤ D."""
    base, tpd = run(s), pd.DatetimeIndex(s["tp"])
    for D in sorted(base["refit_date"].dropna().unique())[:10]:
        rv = s["rv"].copy()
        rv[(tpd > D)] *= 3.0
        alt = run(dict(s, rv=rv))
        m = (base["d"] <= D).to_numpy()
        if not np.array_equal(base.loc[m, ["beta", "F"]].to_numpy(), alt.loc[m, ["beta", "F"]].to_numpy()):
            return True
    return False


def test_pit_corrupted_future_pairs_do_not_change_beta_and_closed_plus_one_is_caught(monkeypatch) -> None:
    assert leaks() is False
    monkeypatch.setattr(ZW, "closed_date", lambda tp, d: d)                          # мутация № 164: «закрыто + 1» — пара с d = D (t⁺ = D + 1) в пуле
    assert leaks() is True


def test_label_of_own_day_is_used_and_next_day_label_mutation_is_caught() -> None:
    w = run()
    on = (w["refit_date"].notna() & (w["beta"] > 0)).to_numpy()
    assert (w.loc[on & (S["g"] == 1), "F"].to_numpy() != S["base"][on & (S["g"] == 1)]).all()        # метка дня d действует на день d
    mut = run(dict(S, g=np.roll(S["g"], -1)))                                         # мутация № 1: метка дня d + 1 (сдвиг на строку)
    assert (mut["F"] != w["F"]).any() and (mut.loc[on & (S["g"] == 1), "F"].to_numpy() == S["base"][on & (S["g"] == 1)]).any()


def test_shift_labels_known_answer_and_control_b_differs_from_w5g_and_zero_shift_mutation_is_caught() -> None:
    a = np.arange(10)
    assert G.shift_labels(a, 3).tolist() == [(i - 3) % 10 for i in range(10)] and G.SHIFT == 252       # gap^b_i = gap_{(i − 252) mod n}
    s3 = dict(S, g=(np.arange(len(S["d"])) % 3 == 0).astype(int))
    w, b, b0 = run(s3), run(dict(s3, g=G.shift_labels(s3["g"]))), run(dict(s3, g=G.shift_labels(s3["g"], 0)))
    assert not w["F"].equals(b["F"])                                                  # контроль (б) со сдвигом 252 отличается от W5-G
    assert w["F"].to_numpy().tobytes() == b0["F"].to_numpy().tobytes()               # мутация «сдвиг 0»: W5-Gb = W5-G побитно — именно это ловят assert выше и тест на реальных рядах
