"""S-C-EVAL, `cm_eval_tab` (S_C_SPEC разд. 4.1, 4.2, 4.4; mutations.md № 103, 125, 140, 158, 160): QLIKE = среднее дневных потерь (не среднее годовых), знак Δ (минус — лучше, база — второй ряд), согласованные слои X1_C,
Холм N = 836 / N_wide = 3377 и m = 8 / 3 против независимой реализации, NaN-p не превращается в 1, «подтверждено» только по своей строке (Δ < 0, p Холма N < 0,05, p SPA < 0,05), лаг NW по n окна.
Синтетика: 700 торговых дней с 2021-01-04; кандидаты S-CK / S-CKA близки к истине (σ 0,05), W5-G тоже (кандидат против него не значим), остальные — шумнее. Реальные файлы не читаются и не пишутся."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import cm_eval_tab as T

SD = {"K1": 0.4, "K2": 0.35, "K3": 0.35, "X1": 0.3, "W5-G": 0.05, "W8-G": 0.3, "B_mix": 0.3, "ZB": 0.3, "S-A2": 0.3, "S-CD": 0.3, "S-CK": 0.05, "S-CKA": 0.05, "S-C7": 0.3, "S-CKL": 0.3}


def syn(n: int = 700, seed: int = 3):
    rng = np.random.default_rng(seed)
    tp = pd.bdate_range("2021-01-04", periods=n)
    ft = np.exp(rng.normal(-1.0, 0.4, n))
    rv = ft * rng.chisquare(3, n) / 3
    F = {k: ft * np.exp(s * rng.normal(size=n)) for k, s in SD.items()}
    F.update({"X1_C(%s)" % v: F[v] * np.exp(0.01 * rng.normal(size=n)) for v in T.VARS})
    F["X1_C(S-CD)"], F["X1_C(S-C7)"], F["X1_C(S-CKL)"] = (ft * np.exp(0.3 * rng.normal(size=n)) for _ in range(3))
    return rv, tp, F


@pytest.fixture(scope="module")
def world():
    rv, tp, F = syn()
    ls = T.losses(rv, F)
    spa, sspa = T.spa_blocks(ls), T.struct_spa(ls)
    return dict(rv=rv, tp=tp, F=F, ls=ls, spa=spa, sspa=sspa, st=T.stats(rv, tp, F, ls, spa, sspa))


def holm_indep(p, N):
    """Независимая реализация: step-down, множитель N − r, монотонность."""
    order, out, run = np.argsort(p, kind="stable"), np.empty(len(p)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (N - r) * p[i]))
        out[i] = run
    return out


def test_hkey_layers():
    assert T.hkey("S-CK", "X1") == "X1_C(S-CK)" and T.hkey("S-CK", "W5-G") == "S-CK" and T.hkey("S-CKA", "K2") == "S-CKA"
    assert T.CANDS == ("S-CK", "S-CKA") and "S-CD" not in T.CANDS and T.N_HOLM == 836 and T.N_WIDE == 3377 and (T.M_FAMILY, T.M_STRUCT) == (8, 3)


def test_table_qlike_is_daily_mean_not_mean_of_years(world):
    t = T.table(world["rv"], world["tp"], world["F"]).set_index("series")
    q = world["ls"]["S-CK"]
    y = world["tp"].year.to_numpy()
    mean_of_years = float(np.mean([q[y == k].mean() for k in np.unique(y)]))
    assert t.at["S-CK", "qlike"] == pytest.approx(float(q.mean()), abs=1e-15)
    assert abs(mean_of_years - float(q.mean())) > 1e-8          # годы разного размера: среднее годовых — иное число (№ 160)
    assert (t["n"] == 700).all() and (t["first_tplus"] == "2021-01-04").all()


def test_delta_sign_and_base_known_answer(world):
    t = T.table(world["rv"], world["tp"], world["F"]).set_index("series")
    for b in T.BASES:
        want = 100.0 * (t.at["S-CK", "qlike"] - t.at[b, "qlike"]) / t.at[b, "qlike"]
        assert t.at["S-CK", "d_%s_pct" % b] == pytest.approx(want, abs=1e-12)
    assert t.at["S-CK", "d_K2_pct"] < 0 < t.at["K2", "d_X1_pct"] + 100 and t.at["K1", "d_K2_pct"] > 0   # кандидат близок к истине: лучше K2 (минус); K1 хуже (плюс)
    assert abs(t.at["K2", "d_K2_pct"]) < 1e-12


def test_stats_shape_and_candidates(world):
    st = world["st"]
    m = st[st["kind"] == "main"]
    assert len(m) == 8 and set(m["hypothesis"]) == {"%s|%s" % (c, b) for c in ("S-CK", "S-CKA") for b in T.BASES}
    assert not m["hypothesis"].str.startswith("S-CD").any()                  # S-CD — не кандидат
    assert (st["kind"] == "window").sum() == 16 and (st["kind"] == "struct").sum() == 3 and (st["kind"] == "cd_base").sum() == 4
    assert set(st[st["kind"] == "struct"]["hypothesis"]) == {"К1", "К2", "К3"}
    x1 = m[m["hypothesis"] == "S-CK|X1"].iloc[0]
    assert x1["series"] == "X1_C(S-CK)" and x1["base"] == "X1"               # М1 — слой X1_C против X1


def test_holm_program_n_against_independent(world):
    st = world["st"]
    for kind, mm in (("main", 8), ("struct", 3)):
        part = st[st["kind"] == kind]
        p = part["p_dm"].to_numpy()
        for col, N in (("p_holm_m", mm), ("p_holm_N", 836), ("p_holm_Nwide", 3377)):
            assert np.allclose(part[col].to_numpy(), holm_indep(p, N), rtol=1e-12, atol=0), (kind, col)
        small = part[(part["p_dm"] > 0) & (part["p_holm_m"] < 1)]
        assert (small["p_holm_N"] > small["p_holm_m"]).all()
    assert (st["n_holm"] == 836).all() and (st["n_holm_wide"] == 3377).all()


def test_mutation_holm_n_equals_number_of_rows_is_caught(world, monkeypatch):
    monkeypatch.setattr(T, "N_HOLM", 8)                                      # № 103: N = число строк семейства
    mut = T.stats(world["rv"], world["tp"], world["F"], world["ls"], world["spa"], world["sspa"])
    m = mut[mut["kind"] == "main"]
    assert not np.allclose(m["p_holm_N"].to_numpy(), holm_indep(m["p_dm"].to_numpy(), 836), rtol=1e-12, atol=0)


def test_nan_p_not_turned_into_one():
    t = pd.DataFrame(dict(kind=["main"] * 4, p_dm=[1e-4, np.nan, 1e-3, 0.2]))
    for k in ("p_holm_m", "p_holm_N", "p_holm_Nwide", "m_family"):
        t[k] = np.nan
    T._holm(t, np.ones(4, bool), 8)
    assert np.isnan(t.loc[1, "p_holm_N"]) and np.isfinite(t.loc[[0, 2, 3], "p_holm_N"]).all()
    assert t.loc[0, "p_holm_N"] == pytest.approx(836 * 1e-4) and t.loc[2, "p_holm_N"] == pytest.approx(835 * 1e-3) and t.loc[3, "p_holm_N"] == 1.0    # N − r с r по конечным p; NaN не участвует


def test_verdict_own_row_rules():
    assert T.verdict(-1.0, 0.01, 0.01).startswith("численно лучше базы; подтверждено")
    assert "не подтверждено" in T.verdict(-1.0, 0.01, 0.20) and "не подтверждено" in T.verdict(-1.0, 0.20, 0.01)
    assert T.verdict(1.0, 0.001, 0.001).startswith("численно не лучше базы; не подтверждено")
    assert "не подтверждено" in T.verdict(-1.0, float("nan"), 0.01)


def test_stats_verdicts_follow_own_row_and_foreign_row_mutation_is_caught(world):
    st = world["st"]
    fam = st[st["kind"].isin(["main", "struct"])].reset_index(drop=True)
    own = [T.verdict(d, ph, ps) for d, ph, ps in zip(fam["delta_pct"], fam["p_holm_N"], fam["p_spa_c"])]
    assert list(fam["verdict"]) == own
    assert any("подтверждено" in v and "не подтверждено" not in v for v in own) and any("не подтверждено" in v for v in own)
    foreign = [T.verdict(d, ph, ps) for d, ph, ps in zip(fam["delta_pct"], fam["p_holm_N"], np.roll(fam["p_spa_c"].to_numpy(), 1))]
    assert foreign != own                                                    # p SPA чужой строки изменил бы вывод
    ref = st[st["kind"].isin(["reference", "cd_base", "ablation", "window"])]
    assert (ref["verdict"] == "справочно, критериев не получает").all()


def test_lag_nw8_for_real_n_and_by_window_n(world):
    assert S.nw_lag(2198) == 8 and S.nw_lag(5) != S.nw_lag(2198)
    st = world["st"]
    assert all(r.lag == S.nw_lag(r.n) for r in st.itertuples())
    h = st[(st["kind"] == "window") & (st["hypothesis"] == "S-CK|K2")]
    assert sorted(h["window"]) == ["h1_2018-2021", "h2_2022-2026"] and int(h["n"].sum()) == 700


def test_mcs_filters_nonpositive_forecasts_of_the_stage_set(world):
    from models_v2.s18 import cm_eval as E
    F = {k: np.array(world["F"][k]) for k in E.MCS_SET}
    F["K3"][5] = -1.0                                                        # неположительный прогноз: дата исключается из MCS у всех рядов
    r = S.mcs(world["rv"], F)
    assert list(r["series"]) == list(E.MCS_SET) and (r["n_used"] == 699).all() and np.isfinite(r["mean_qlike"]).all()
    unfiltered = T.losses(world["rv"], F)["K3"]                               # мутация «без фильтра»: потеря на плохой дате — NaN / inf, MCS получил бы мусор
    assert not np.isfinite(unfiltered).all()


def test_k1_pair_is_ck_against_cd_same_loss_level(world):
    s = world["st"]
    k1 = s[(s["kind"] == "struct") & (s["hypothesis"] == "К1")].iloc[0]
    assert (k1["series"], k1["base"]) == ("S-CK", "S-CD")
    assert k1["delta_pct"] == pytest.approx(100.0 * (world["ls"]["S-CK"].mean() / world["ls"]["S-CD"].mean() - 1.0), abs=1e-10) and k1["delta_pct"] < 0
