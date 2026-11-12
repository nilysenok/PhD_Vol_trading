"""S-RCD-EVAL (Ж-259, RE.G1), тестировщик: `rd_eval_chk.problems` и таблицы `rd_eval_*` на синтетическом мире (2198 дат D_M, B = 200); всё пишется только в tmp.
Мутации спецификации: обрезка дат, база не S-CDF / не S-RCK, H-RCD1 против W5-G, Холм N = строки (№ 103), знак Δ / «лучше–хуже» (№ 125), подменённое число, «эквивалентно» при p ≥ 0,05, кандидат по минимуму QLIKE,
RCD-DEC «превзошёл» без подтверждения / без набора Б, QLIKE окна как среднее годовых (№ 160), У5-S без исключения лучшего периода, тождества id1–id3, правило нулевого распределения со знаком.
Мутант портит таблицу t (или подменяет настоящую функцию через monkeypatch); настоящий проверяющий обязан вернуть непустой список."""
import numpy as np
import pandas as pd
import pytest

import test_s18_rd_eval_fx as FX
from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import rd_eval_chk as K
from models_v2.s18 import rd_eval_cfg as C
from models_v2.s18 import rd_eval_dc as DC
from models_v2.s18 import rd_eval_gd as GD
from models_v2.s18 import rd_eval_null as RN
from models_v2.s18 import rd_eval_tab as T

R, CDF, RCK = FX.R, FX.CDF, FX.RCK
MID = {R: 0.10, CDF: 0.04, RCK: 0.12}                      # S-RCD лучше S-CDF, но хуже S-RCK
STRONG = {R: 0.5, CDF: 0.1, RCK: 0.1}                      # S-RCD сильно лучше обеих баз: RCD-DEC = 1
REVERSED = {R: 0.02, CDF: 0.06}                            # S-RCD лучше W5-G, но хуже S-CDF: H-RCD1 не выполнена
FX_ORDER = None


def build(gain, noise=0.03):
    with pytest.MonkeyPatch.context() as mp:
        FX.patch_boot(mp)
        inp = FX.inputs(gain, noise=noise)
        t = FX.synth_t(inp)
    S.idx_for.cache_clear()
    return inp, t


@pytest.fixture(scope="module")
def mid():
    return build(MID)


@pytest.fixture(scope="module")
def strong():
    return build(STRONG)


@pytest.fixture(scope="module")
def rev():
    return build(REVERSED)


@pytest.fixture(scope="module")
def conc():
    y = FX.calendar().year.to_numpy()
    return build({R: np.where(y == 2022, 0.9, -0.0015)}, noise=0.001)         # весь выигрыш в 2022; в остальные годы чуть хуже базы (в пределах порогов У1-S…У3-S)


@pytest.fixture
def boot(monkeypatch):
    FX.patch_boot(monkeypatch)
    yield
    monkeypatch.undo()
    S.idx_for.cache_clear()


def all_p(t, inp):
    return K.p_const() + K.p_qlike(t, inp) + K.p_stats(t, inp) + K.p_dec(t, inp) + K.p_ident(t, inp) + K.p_slices(t, inp)


def dec(t):
    return t["dec"].set_index("id")


def test_good_worlds_have_no_problems_and_decisions_are_derived(boot, mid, strong, rev):
    for inp, t in (mid, strong, rev):
        assert all_p(t, inp) == []
        assert t["identity"] <= 1e-12 and (inp["N"], inp["N_wide"], C.M_FAMILY) == (858, 3421, 6)
    assert str(dec(strong[1]).loc["RCD-DEC", "value"]) == "1" and bool(dec(strong[1]).loc["подтверждение над S-RCK", "passed"])
    assert str(dec(mid[1]).loc["RCD-DEC", "value"]) in "23" and bool(dec(mid[1]).loc["H-RCD1", "passed"]) and not bool(dec(mid[1]).loc["S-RCD против S-RCK", "passed"])
    assert str(dec(rev[1]).loc["RCD-DEC", "value"]) == "4" and not bool(dec(rev[1]).loc["H-RCD1", "passed"])
    assert list(t["dec"]["id"][:2]) == ["H-RCD1", "S-RCD против S-RCK"]


def test_mut_trimmed_dates_in_main_delta_caught(boot, mid):
    inp, t = mid
    t2 = FX.copy_t(t)
    t2["table"]["n"] = C.N_D - 44
    assert K.p_qlike(t2, inp)
    t3 = FX.copy_t(t)
    t3["stats"].loc[t3["stats"]["kind"] == "main", "n"] = C.N_D - 44
    assert any("обрезка" in b for b in K.p_slices(t3, inp))
    t4 = FX.copy_t(t)
    t4["table"]["first_tplus"] = "2018-03-01"
    assert any("границы дат" in b for b in K.p_qlike(t4, inp))


def test_mut_base_not_cdf_and_not_rck_caught(boot, mid):
    inp, t = mid
    for b in (CDF, RCK):
        t2 = FX.copy_t(t)
        t2["table"]["d_%s_pct" % b] = t2["table"]["d_W5-G_pct"]
        assert any("Δ" in b_ for b_ in K.p_qlike(t2, inp)), b
        t3 = FX.copy_t(t)
        s = t3["stats"]
        s.loc[s["hypothesis"] == "%s|%s" % (R, b), "delta_pct"] = s.loc[s["hypothesis"] == "%s|W5-G" % R, "delta_pct"].iloc[0]
        assert any("знак или база" in b_ for b_ in K.p_stats(t3, inp)), b
    # набор Б (база S-RCK): подмена базы процентов в первой таблице на S-CDF
    t4 = FX.copy_t(t)
    t4["table"]["d_S-RCK_pct"] = t4["table"]["d_S-CDF_pct"]
    assert any("S-RCK" in b for b in K.p_qlike(t4, inp))
    # H-RCD1: значение в процентах QLIKE W5-G вместо S-CDF
    t5 = FX.copy_t(t)
    q = t5["table"].set_index("series")["qlike"]
    t5["dec"].loc[t5["dec"]["id"] == "H-RCD1", "value"] = 100.0 * (q[R] - q["W5-G"]) / q["W5-G"]
    assert any("Δ не в % QLIKE S-CDF" in b for b in K.p_dec(t5, inp))


def test_mut_hrcd1_against_w5g_instead_of_cdf_caught(boot, rev):
    inp, t = rev                                                  # S-RCD лучше W5-G, хуже S-CDF: ответ H-RCD1 — «нет»
    q = t["table"].set_index("series")["qlike"]
    assert q[R] < q["W5-G"] and q[R] > q[CDF]
    t2 = FX.copy_t(t)
    d = t2["dec"]
    d.loc[d["id"] == "H-RCD1", "passed"] = bool(q[R] < q["W5-G"])
    d.loc[d["id"] == "H-RCD1", "value"] = 100.0 * (q[R] - q["W5-G"]) / q["W5-G"]
    bad = K.p_dec(t2, inp)
    assert any("H-RCD1" in b for b in bad) and len(bad) >= 2


def test_mut_holm_n_equals_row_count_caught_n103(boot, mid):
    inp, t = mid
    m = t["stats"]["kind"] == "main"
    p = t["stats"].loc[m, "p_dm"].to_numpy()
    t2 = FX.copy_t(t)
    t2["stats"].loc[m, "p_holm_N"] = S.holm_n(p, len(p))
    assert any("p_holm_N" in b for b in K.p_stats(t2, inp))
    t3 = FX.copy_t(t)
    t3["stats"]["n_holm"] = 6
    assert any("N Холма" in b for b in K.p_stats(t3, inp))
    t4 = FX.copy_t(t)
    t4["stats"].loc[m, "p_holm_Nwide"] = S.holm_n(p, 858)
    assert any("p_holm_Nwide" in b for b in K.p_stats(t4, inp))
    t5 = FX.copy_t(t)
    dd = t5["stats"]["kind"] == "decomp"
    t5["stats"].loc[dd, "p_holm_N"] = S.holm_n(t5["stats"].loc[dd, "p_dm"].to_numpy(), 6)
    assert any("Холм разложения" in b for b in K.p_stats(t5, inp))
    m1 = FX.copy_t(t)
    assert inp["N"] == 858 and (m1["stats"]["n_holm"] == 858).all()
    assert len(p) == 6 and S.holm_n(p, 858)[0] >= S.holm_n(p, 6)[0]                 # N = 858 строже, чем m = 6 (направление поправки)


def test_mut_sign_and_better_worse_words_caught_n125(boot, mid):
    inp, t = mid
    t2 = FX.copy_t(t)
    t2["stats"]["delta_pct"] = -t2["stats"]["delta_pct"]
    assert any("знак или база" in b for b in K.p_stats(t2, inp))
    t3 = FX.copy_t(t)
    m = t3["stats"]["kind"] == "main"
    t3["stats"].loc[m, "p_holm_N"] = 0.5
    t3["stats"].loc[m, "verdict"] = "лучше X; подтверждено"
    assert any("подтверждено" in b for b in K.p_stats(t3, inp))
    t4 = FX.copy_t(t)
    mm = t4["stats"]["kind"] == "main"
    t4["stats"].loc[mm, "verdict"] = "лучше S-RCK; подтверждено (p Холма N < 0,05 и p SPA < 0,05)"
    t4["stats"].loc[mm, "delta_pct"] = 0.1
    assert any("№ 125" in b for b in K.p_stats(t4, inp))
    t5 = FX.copy_t(t)
    mm = t5["stats"]["kind"] == "main"
    t5["stats"].loc[mm, "delta_pct"] = 0.1
    t5["stats"].loc[mm, "verdict"] = "численно лучше S-CDF; не подтверждено (p Холма N ≥ 0,05 или p SPA ≥ 0,05)"
    assert any("Δ ≥ 0 названо «лучше»" in b for b in K.p_stats(t5, inp))


def test_verdict_known_answers_minus_is_better_and_mutants(monkeypatch):
    v = T.verdict
    assert v(-0.5, 0.01, 0.01, "B").startswith("лучше B; подтверждено")
    assert v(-0.5, 0.01, 0.2, "B") == "численно лучше B; не подтверждено (p Холма N ≥ 0,05 или p SPA ≥ 0,05)"      # SPA ≥ 0,05 — не подтверждено
    assert v(-0.5, 0.2, 0.01, "B").startswith("численно лучше")
    assert v(0.5, 0.001, 0.001, "B") == "численно не лучше B; не подтверждено"                                      # знак: плюс — хуже (№ 125)
    assert v(-0.5, 0.05, 0.01, "B").startswith("численно")                                                          # граница: p = 0,05 не проходит


def test_mut_spa_ignored_in_verdict_is_caught(boot, mid, monkeypatch):
    inp, t = mid
    ls = T.losses(inp["rv"], inp["F"])
    spa = T.spa_blocks(ls)
    for k in spa:
        spa[k] = dict(spa[k], p_c=0.9)                              # SPA ни в одном блоке не отвергает
    monkeypatch.setattr(T, "verdict", lambda d, ph, ps, b: ("лучше %s; подтверждено (p Холма N < 0,05 и p SPA < 0,05)" % b) if d < 0 and ph < 0.05 else "численно не лучше %s; не подтверждено" % b)
    t2 = FX.copy_t(t)
    t2["stats"] = T.stats(inp["rv"], inp["tp"], ls, spa, 858, 3421, DC.act_mask(inp))
    m = t2["stats"]["kind"] == "main"
    t2["stats"].loc[m, "p_holm_N"] = 0.001
    t2["stats"].loc[m, "verdict"] = "лучше X; подтверждено (p Холма N < 0,05 и p SPA < 0,05)"
    assert any("подтверждено" in b for b in K.p_stats(t2, inp))


def test_mut_substituted_number_caught(boot, mid):
    inp, t = mid
    for col, f in (("qlike", lambda v: v * 1.0001), ("d_S-CDF_pct", lambda v: v + 1e-6), ("d_S-RCK_pct", lambda v: v + 1e-6)):
        t2 = FX.copy_t(t)
        i = t2["table"].index[t2["table"]["series"] == R][0]
        t2["table"].loc[i, col] = f(t2["table"].loc[i, col])
        assert K.p_qlike(t2, inp), col
    t3 = FX.copy_t(t)
    j = t3["stats"].index[(t3["stats"]["hypothesis"] == "S-RCD|K2") & (t3["stats"]["kind"] == "main")][0]
    t3["stats"].loc[j, "lo90"] += 1e-6
    assert any("интервал" in b for b in K.p_stats(t3, inp))
    t4 = FX.copy_t(t)
    t4["stats"].loc[j, "delta_pct"] += 1e-6
    assert any("знак или база" in b for b in K.p_stats(t4, inp))


def test_mut_equivalent_word_absent_and_confirmed_at_p_ge_005_caught(boot, mid):
    inp, t = mid
    assert not t["stats"]["verdict"].str.contains("эквивал").any()
    assert not t["dec"].astype(str).apply(lambda c: c.str.contains("эквивал")).any().any()
    t2 = FX.copy_t(t)
    k = t2["stats"]["kind"] == "main"
    t2["stats"].loc[k, "p_holm_N"] = 0.2
    t2["stats"].loc[k, "verdict"] = "лучше S-CDF; подтверждено (p Холма N < 0,05 и p SPA < 0,05)"
    assert K.p_stats(t2, inp)


def test_mut_rcddec_overtook_without_confirmation_or_without_set_b_caught(boot, strong, mid):
    inp, t = strong
    t2 = FX.copy_t(t)                                               # код 1 остался, подтверждение над S-RCK пропало
    s = t2["stats"]
    s.loc[(s["kind"] == "main") & (s["hypothesis"] == "S-RCD|S-RCK"), "p_holm_N"] = 0.5
    bad = K.p_dec(t2, inp)
    assert any("«превзошёл» без подтверждения или без набора Б" in b for b in bad)
    t3 = FX.copy_t(t)                                               # набор Б не выполнен, код 1 остался
    g = t3["dec"]
    g.loc[g["id"] == "У к S-RCK", "passed"] = False
    assert any("набор к S-RCK" in b for b in K.p_dec(t3, inp))
    t4 = FX.copy_t(t)
    t4["dec"].loc[t4["dec"]["id"] == "подтверждение над S-RCK", "passed"] = False
    assert any("подтверждение над S-RCK" in b for b in K.p_dec(t4, inp))
    inp2, tm = mid                                                  # S-RCD хуже S-RCK: код 1 обязан не пройти
    t5 = FX.copy_t(tm)
    t5["dec"].loc[t5["dec"]["id"] == "RCD-DEC", "value"] = "1"
    assert any("RCD-DEC" in b for b in K.p_dec(t5, inp2))


def test_decide_known_answers_each_component_is_needed():
    base = {"H-RCD1": True, "У к W5-G": True, "S-RCD против S-RCK": True, "У к S-RCK": True, "подтверждение над S-RCK": True}

    def mk(**o):
        p = dict(base, **o)
        d = pd.DataFrame([dict(id=k, passed=v, value=0.0) for k, v in p.items()])
        g = pd.DataFrame(dict(set=["к W5-G"], id=["У1-S"], passed=[True], series=[R], value=[0.0]))
        return GD.decide(d, g)[0]
    assert mk() == "1"
    for k in ("S-RCD против S-RCK", "У к S-RCK", "подтверждение над S-RCK"):
        assert mk(**{k: False}) == "2"                              # без любого из условий «превзошёл» невозможен
    assert mk(**{"У к W5-G": False}) == "3" and mk(**{"H-RCD1": False, "У к W5-G": False}) == "4"
    assert mk(**{"H-RCD1": False}) == "4"                           # набор А без H-RCD1 не даёт «2»


def test_mut_candidate_by_min_qlike_caught(boot, mid, monkeypatch):
    inp, t = mid
    q = t["table"].set_index("series")["qlike"]

    def min_qlike_decide(d, g):                                     # «кандидат» = ряд с минимумом QLIKE, а не S-RCD по правилам разд. 7
        best = q[["S-RCD", "S-RCK", "S-CDF", "S-CD", "S-CK", "S-CKF"]].idxmin()
        return ("1", "победитель по QLIKE") if best == R else ("4", "не победитель по QLIKE")
    real = GD.decide(t["dec"].iloc[:-1], t["guard"])[0]
    monkeypatch.setattr(GD, "decide", min_qlike_decide)
    d2 = GD.dec_rows(t["table"], t["guard"], t["stats"])
    monkeypatch.undo()
    got = str(d2.set_index("id").loc["RCD-DEC", "value"])
    t2 = FX.copy_t(t)
    t2["dec"] = d2
    if got != real:
        assert any("RCD-DEC" in b for b in K.p_dec(t2, inp))
    # S-RCK лучше S-RCD по QLIKE (мир mid): мутант «S-RCD — лучший» обязан расходиться с независимым выводом
    assert q["S-RCK"] < q[R]
    t3 = FX.copy_t(t)
    t3["dec"].loc[t3["dec"]["id"] == "S-RCD против S-RCK", "passed"] = True
    t3["dec"].loc[t3["dec"]["id"] == "RCD-DEC", "value"] = "1"
    assert any("RCD-DEC" in b for b in K.p_dec(t3, inp))


# Был xfail(strict): p_dec не сверял строку «S-RCD против S-RCK» (найдено тестировщиком 12.11); исправлено автором, метку снял дирижёр.
def test_gap_p_dec_does_not_compare_row_rcd_vs_rck(boot, mid):
    inp, t = mid
    t2 = FX.copy_t(t)
    d = t2["dec"]
    d.loc[d["id"] == "S-RCD против S-RCK", "passed"] = True        # строка противоречит QLIKE (S-RCD хуже S-RCK), код решения прежний
    assert any("S-RCD против S-RCK" in b for b in K.p_dec(t2, inp)), "p_dec не сверяет строку «S-RCD против S-RCK» с QLIKE"


def test_mut_contributions_do_not_sum_to_delta_caught(boot, mid):
    inp, t = mid
    t2 = FX.copy_t(t)
    y = t2["years"]
    i = y.index[(y["series"] == R) & (y["base"] == CDF) & (y["period"] == "2020")][0]
    y.loc[i, "contrib_pct"] += 1e-9
    assert any("Σ вкладов" in b for b in K.p_dec(t2, inp))


def test_mut_qlike_mean_of_annual_means_caught_n160(boot, mid):
    inp, t = mid
    y = inp["tp"].year.to_numpy()
    ls = T.losses(inp["rv"], inp["F"])
    t2 = FX.copy_t(t)
    t2["table"]["qlike"] = [float(np.mean([ls[s][y == g].mean() for g in np.unique(y)])) for s in t2["table"]["series"]]
    assert any("QLIKE" in b for b in K.p_qlike(t2, inp))
    assert abs(t["table"].set_index("series").loc[R, "qlike"] - float(ls[R].mean())) < 1e-12           # настоящая таблица: среднее дневных потерь


def test_mut_u5s_without_exclusion_of_best_period_caught(boot, conc, monkeypatch):
    inp, t = conc
    w = t["guard"]
    u5 = w[(w["id"] == "У5-S") & (w["set"] == "к W5-G")].iloc[0]
    assert not bool(u5["passed"])                                   # без лучшего периода (2022) Δ > 0
    others = w[(w["set"] == "к W5-G") & (w["id"].isin(["У1-S", "У2-S", "У3-S", "У6-R"]))]
    assert others["passed"].map(bool).all(), others[["id", "value", "passed"]]
    assert all_p(t, inp) == []
    # мутант: «Δ без лучшего периода» заменён общей Δ (период не исключён) во всех строках loyo
    t2 = FX.copy_t(t)
    c, y = t2["conc"], t2["years"]
    tot = float(y[(y["series"] == R) & (y["base"] == "W5-G")]["delta_total_pct"].iloc[0])
    assert tot < 0
    c.loc[(c["series"] == R) & (c["base"] == "W5-G") & (c["item"] == "loyo"), "value"] = tot
    g2 = GD.guard(y, c)
    t2["guard"] = g2
    t2["dec"] = GD.dec_rows(t2["table"], g2, t2["stats"])
    assert bool(g2[(g2["id"] == "У5-S") & (g2["set"] == "к W5-G")].iloc[0]["passed"])                  # мутант пропустил У5-S
    assert any("набор к W5-G" in b for b in K.p_dec(t2, inp))


def test_ident_known_answers_and_mutations_caught(boot, mid, monkeypatch):
    inp, t = mid
    ls = T.losses(inp["rv"], inp["F"])
    allm = np.ones(len(inp["rv"]), bool)
    e = DC.e_win(ls, allm)
    ids = DC._id(e)
    assert max(abs(ids[k]) for k in ("id1_gap", "id2_gap", "id3_gap")) <= 1e-12
    assert abs(ids["id1_R_on_CD"] - (e["S-RCD"] - e["S-CDF"])) <= 1e-12 and abs(ids["id3_interaction"] - (ids["id2_R_on_CK_at_F"] - ids["id1_R_on_CD"])) <= 1e-12
    # 1) невязка id1 нарушена
    t2 = FX.copy_t(t)
    dm = t2["decomp"]
    dm.loc[(dm["section"] == "ident_D_M") & (dm["item"] == "id1_gap"), "value"] = 1e-6
    assert any("id1_gap" in b for b in K.p_ident(t2, inp))
    # 2) id2 и id3 — в активном окне
    for item in ("id2_gap", "id3_gap"):
        t3 = FX.copy_t(t)
        dm = t3["decomp"]
        dm.loc[(dm["section"] == "ident_active") & (dm["item"] == item), "value"] = -1e-6
        assert any(item in b for b in K.p_ident(t3, inp)), item
    # 3) слагаемое «R на C-D» неверно
    t4 = FX.copy_t(t)
    dm = t4["decomp"]
    dm.loc[(dm["section"] == "ident_D_M") & (dm["item"] == "id1_R_on_CD"), "value"] += 1e-6
    assert any("R на C-D" in b for b in K.p_ident(t4, inp))
    # 4) выброшенная дата в e(S-CD) (настоящая функция заменена мутантом): невязка тождеств остаётся нулём (телескоп), ловит независимый e(·)
    real = DC.e_win

    def dropped(ls_, m):
        out = real(ls_, m)
        b = float(ls_["W5-G"][m].sum())
        m2 = m.copy()
        m2[np.flatnonzero(m)[0]] = False
        out["S-CD"] = 100.0 * (float(ls_["S-CD"][m2].sum()) - b) / b
        return out
    monkeypatch.setattr(DC, "e_win", dropped)
    t5 = FX.copy_t(t)
    t5["decomp"] = DC.decomp(inp, ls, inp["null"])
    monkeypatch.undo()
    assert any("e(S-CD)" in b for b in K.p_ident(t5, inp))
    # 5) неверная база e(·): S-CDF вместо W5-G
    def wrong_base(ls_, m):
        b = float(ls_["S-CDF"][m].sum())
        return {k: 100.0 * (float(ls_[k][m].sum()) - b) / b for k in DC.SER}
    monkeypatch.setattr(DC, "e_win", wrong_base)
    t6 = FX.copy_t(t)
    t6["decomp"] = DC.decomp(inp, ls, inp["null"])
    monkeypatch.undo()
    bad = K.p_ident(t6, inp)
    assert any("e(" in b for b in bad)
    # 6) тождество нарушено внутри `_id`: слагаемое «уровень без R» с обратным знаком
    real_id = DC._id

    def bad_id(e_):
        r = real_id(e_)
        r["id1_gap"] = r["id1_gap"] + 2 * r["id1_level_without_R"]
        return r
    monkeypatch.setattr(DC, "_id", bad_id)
    t7 = FX.copy_t(t)
    t7["decomp"] = DC.decomp(inp, ls, inp["null"])
    monkeypatch.undo()
    assert any("id1_gap" in b for b in K.p_ident(t7, inp))


def test_ident_active_window_is_1549_dates_and_dropped_date_changes_n(boot, mid):
    inp, t = mid
    dm = t["decomp"]
    n = dm[(dm["section"] == "share_active") & (dm["item"] == "n_dates")]["value"].iloc[0]
    assert int(n) == 1549 and int(dm[(dm["section"] == "share_D_M") & (dm["item"] == "n_dates")]["value"].iloc[0]) == C.N_D
    a = t["stats"][t["stats"]["kind"] == "active"]
    assert (a["n"] == 1549).all()
    t2 = FX.copy_t(t)
    t2["stats"].loc[t2["stats"]["kind"] == "active", "n"] = 1548
    assert any("активное окно" in b for b in K.p_stats(t2, inp))


def test_null_rule_uses_abs_delta_not_signed(boot, mid, monkeypatch):
    inp, t = mid
    ls = T.losses(inp["rv"], inp["F"])
    d0 = abs(DC._dp(ls[R], ls[CDF], np.ones(len(inp["rv"]), bool)))
    assert d0 > 0 and DC._dp(ls[R], ls[CDF], np.ones(len(inp["rv"]), bool)) < 0                # Δ(S-RCD к S-CDF) < 0: знак отличим от модуля
    inp = dict(inp, null=FX.null_seeds(1.5 * d0))                                               # |Δ_null| в [≈0,17·1,5·|Δ|; 1,5·|Δ|]: реальная |Δ| внутри
    dm = DC.decomp(inp, ls, inp["null"])
    assert RN.rule_abs(-0.3, 0.1, 0.5) == "не отличим от шума процедуры" and RN.rule_abs(0.3, 0.1, 0.5) == "не отличим от шума процедуры"
    assert RN.rule_abs(-0.7, 0.1, 0.5) == "вне 5–95 % распределения модуля Δ_null" and RN.rule_abs(0.05, 0.1, 0.5).startswith("вне")

    def expected(d_):
        lo, hi = RN.abs_bounds(inp["null"][0])
        r = dm[dm["section"] == "noise_real"].set_index("item")["value"]
        return {k: ("не отличим от шума процедуры" if lo <= abs(v) <= hi else "вне 5–95 % распределения модуля Δ_null") for k, v in r.items()}
    exp = expected(dm)
    got = dm[dm["section"] == "noise_rule"].set_index("item")["key"].to_dict()
    assert got == exp and len(got) == 4
    r = dm[dm["section"] == "noise_real"].set_index("item")["value"]
    assert (r < 0).any()                                           # знак учитывается: мутант «знак вместо модуля» отличим на отрицательной Δ
    monkeypatch.setattr(RN, "rule_abs", lambda d_, lo, hi: "не отличим от шума процедуры" if lo <= d_ <= hi else "вне 5–95 % распределения модуля Δ_null")
    dm2 = DC.decomp(inp, ls, inp["null"])
    monkeypatch.undo()
    got2 = dm2[dm2["section"] == "noise_rule"].set_index("item")["key"].to_dict()
    assert got["RCD_vs_CDF|D_M"] == "не отличим от шума процедуры" and got2["RCD_vs_CDF|D_M"].startswith("вне")
    assert got2 != exp, "мутант (Δ со знаком вместо |Δ|) не отличим от настоящего правила"


def test_mut_slice_without_day_and_constants_changed_caught(boot, mid, monkeypatch):
    inp, t = mid
    t2 = FX.copy_t(t)
    c = t2["conc"]
    i = c.index[(c["series"] == R) & (c["base"] == CDF) & (c["item"] == "no_day")][0]
    c.loc[i, "value"] += 1e-6
    assert any("2022-02-24" in b for b in K.p_slices(t2, inp))
    for name, val in (("KTRIM", 21), ("N_HOLM", 6), ("N_WIDE", 6), ("M_FAMILY", 5)):
        monkeypatch.setattr(C, name, val)
        assert K.p_const(), name
        monkeypatch.undo()
    monkeypatch.setitem(C.THR, "У2-S", 0.6)
    assert K.p_const()
    monkeypatch.undo()
    monkeypatch.setattr(C, "SHARE_HALF", 0.4)
    assert K.p_const()
    monkeypatch.undo()
    monkeypatch.setattr(S, "SEED", 1)
    assert K.p_const()
    monkeypatch.undo()
    assert K.p_const() == []
