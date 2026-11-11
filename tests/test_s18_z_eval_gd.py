"""S-Z-EVAL (Ж-258, ZE.G1/ZE.G3), тестировщик: `z_eval_gd` — вклады лет (Σ = Δ, 10⁻¹²), срезы, условия У1-S…У6-R (два набора), H-Z1 / H-Z2, Z-DEC.
Известные ответы на синтетике с контролируемыми потерями: база ≡ 1, у кандидата потери заданы по периодам. Мутант = подмена настоящей функции monkeypatch; оракул обязан падать. Выходы не пишутся."""
import numpy as np
import pandas as pd
import pytest

import test_s18_z_eval_fx as FX
from models_v2.s18 import a_eval_yr as AY
from models_v2.s18 import z_eval_cfg as C
from models_v2.s18 import z_eval_gd as GD
from test_s18_z_eval_tab import caught

Z, ZL = FX.Z, FX.ZL
TP = FX.calendar()
Y = TP.year.to_numpy()
KEYS = [Z, "S-RCK", "W5-G", ZL, "X1", "X1_R(S-RCK)", "S-RCKZs", "S-RZ"]


def ls_of(lz=1.0, lzl=None) -> dict:
    ls = {k: np.ones(len(TP)) for k in KEYS}
    ls[Z] = np.broadcast_to(np.asarray(lz, float), (len(TP),)).copy()
    ls[ZL] = ls[Z].copy() if lzl is None else np.broadcast_to(np.asarray(lzl, float), (len(TP),)).copy()
    return ls


def guard_of(ls):
    return GD.guard(GD.years(TP, ls), GD.conc(TP, ls))


def pas(g, name, i) -> bool:
    return bool(g[(g["set"] == name) & (g["id"] == i)]["passed"].iloc[0])


ALL5 = ("У1-S", "У2-S", "У3-S", "У5-S", "У6-R")


# ───────── вклады лет, тождество Σ = Δ ─────────
def oracle_years(yrs, ls):
    for (h, b, kd), g in yrs[yrs["base"] != "-"].groupby(["series", "base", "kind"]):
        assert abs(g["contrib_pct"].sum() - g["delta_total_pct"].iloc[0]) <= 1e-12
        assert abs(g["delta_total_pct"].iloc[0] - 100 * (ls[h].mean() - ls[b].mean()) / ls[b].mean()) <= 1e-9     # Δ — на всех датах, не среднее по периодам
    assert GD.identity_gap(yrs) <= 1e-12


def test_contributions_sum_to_delta_known_answer_and_periods():
    rv, tp, F = FX.world(gain={"S-RCKZ": 0.05})
    ls = {k: FX.ql(rv, v) for k, v in F.items()}
    yrs = GD.years(tp, ls)
    oracle_years(yrs, ls)
    w = yrs[(yrs["series"] == Z) & (yrs["base"] == "S-RCK")].set_index("period")
    assert list(w.index) == [str(y) for y in range(2018, 2026)] + ["2026_до_27.02", "2026_F", "h1_2018-2021", "h2_2022-2026"]
    n = len(tp)
    for p in ("2019", "2026_F"):
        m = {"2019": tp.year == 2019, "2026_F": tp > pd.Timestamp("2026-02-27")}[p]
        want = 100 * (ls[Z][m] - ls["S-RCK"][m]).sum() / (n * ls["S-RCK"].mean())
        assert abs(w.at[p, "contrib_pct"] - want) < 1e-12 and w.at[p, "n"] == int(m.sum())
    assert w.at["h1_2018-2021", "n"] == 1010 or w.at["h1_2018-2021", "n"] == int((tp.year <= 2021).sum())


def test_mut_period_dropped_breaks_identity_caught(monkeypatch):
    rv, tp, F = FX.world(gain={"S-RCKZ": 0.05})
    ls = {k: FX.ql(rv, v) for k, v in F.items()}
    oracle_years(GD.years(tp, ls), ls)
    orig = AY.T5.masks
    monkeypatch.setattr(AY.T5, "masks", lambda t: {k: v for k, v in orig(t).items() if k != "2020"})        # год выброшен
    assert caught(oracle_years, GD.years(tp, ls), ls)


def test_mut_contribution_shifted_by_1e9_caught():
    rv, tp, F = FX.world()
    ls = {k: FX.ql(rv, v) for k, v in F.items()}
    y = GD.years(tp, ls)
    i = y[(y["series"] == Z) & (y["base"] == "S-RCK") & (y["period"] == "2019")].index[0]
    y.loc[i, "contrib_pct"] += 1e-9
    assert caught(oracle_years, y, ls)


# ───────── H-Z1, H-Z2 и «кандидат по минимуму QLIKE» ─────────
def qt(z, rck, zs, rz=9.0) -> pd.DataFrame:
    return pd.DataFrame({"series": [Z, "S-RCK", "S-RCKZs", "S-RZ", "W5-G"], "qlike": [z, rck, zs, rz, 5.0]})


def oracle_hz(fn):
    assert fn(qt(1.0, 2.0, 3.0)) == {"H-Z1": True, "H-Z2": True}
    assert fn(qt(2.0, 3.0, 1.0)) == {"H-Z1": True, "H-Z2": False}                  # H-Z2 — против S-RCKZs
    assert fn(qt(2.0, 1.0, 3.0)) == {"H-Z1": False, "H-Z2": True}
    assert fn(qt(2.0, 2.0, 2.0)) == {"H-Z1": False, "H-Z2": False}                  # строго
    assert fn(qt(2.0, 3.0, 3.0, rz=0.1)) == {"H-Z1": True, "H-Z2": True}            # минимум QLIKE у S-RZ кандидата не меняет
    assert fn(qt(4.0, 3.0, 5.0, rz=0.1)) == {"H-Z1": False, "H-Z2": True}
    assert fn(qt(4.0, 5.0, 3.0, rz=0.1)) == {"H-Z1": True, "H-Z2": False}


def test_hz_known_answers():
    oracle_hz(GD.hz)


@pytest.mark.parametrize("name,mut", [
    ("H-Z2 против W5-G вместо S-RCKZs", lambda t: {"H-Z1": GD.hz(t)["H-Z1"], "H-Z2": bool(t.set_index("series")["qlike"][Z] < t.set_index("series")["qlike"]["W5-G"])}),
    ("H-Z1 против W5-G вместо S-RCK", lambda t: {"H-Z1": bool(t.set_index("series")["qlike"][Z] < t.set_index("series")["qlike"]["W5-G"]), "H-Z2": GD.hz(t)["H-Z2"]}),
    ("нестрогое ≤", lambda t: {k: bool(a <= b) for k, (a, b) in zip(("H-Z1", "H-Z2"), ((t.set_index("series")["qlike"][Z], t.set_index("series")["qlike"]["S-RCK"]), (t.set_index("series")["qlike"][Z], t.set_index("series")["qlike"]["S-RCKZs"])))}),
    ("кандидат — ряд с минимумом QLIKE", lambda t: {k: bool(t.set_index("series")["qlike"][t.iloc[:4].set_index("series")["qlike"].idxmin()] < t.set_index("series")["qlike"][b]) for k, b in (("H-Z1", "S-RCK"), ("H-Z2", "S-RCKZs"))}),
])
def test_mut_hz_caught(name, mut):
    assert caught(oracle_hz, mut), name


def test_mut_candidate_series_replaced_by_srz_caught(monkeypatch):
    monkeypatch.setattr(GD, "Z", "S-RZ")
    assert caught(oracle_hz, GD.hz)


# ───────── защитные условия (два набора), пороги, У5-S, У6-R ─────────
def test_guard_all_pass_when_candidate_better_everywhere_and_constants_are_spec():
    g = guard_of(ls_of(0.98))
    assert list(g["id"]).count("набор") == 2 and len(g) == 12 and set(g["set"]) == {"к W5-G", "к S-RCK"}
    for name in ("к W5-G", "к S-RCK"):
        assert all(pas(g, name, i) for i in ALL5 + ("набор",))
    assert dict(C.THR) == {"У1-S": 0.415, "У2-S": 0.50, "У3-S": 1.00} and C.KTRIM == 22


def worst(v1=None, v2=None, v3=None):
    """Потери кандидата: h1 = v1 везде; h2 = v2; 2018 = v3 (остальной h1 = 0,99 при заданном v3)."""
    lz = np.full(len(TP), 0.99)
    if v1 is not None:
        lz[Y <= 2021] = v1
    if v2 is not None:
        lz[Y >= 2022] = v2
    if v3 is not None:
        lz[Y == 2018] = v3
    return lz


@pytest.mark.parametrize("i,good,bad", [("У1-S", worst(v1=1.0040), worst(v1=1.0045)), ("У2-S", worst(v2=1.0049), worst(v2=1.0051)), ("У3-S", worst(v3=1.0099), worst(v3=1.0101))])
def test_thresholds_042_050_100_known_answers_and_mutated_thresholds_caught(i, good, bad, monkeypatch):
    def oracle(thr_mut=None):
        if thr_mut is not None:
            monkeypatch.setitem(C.THR, i, thr_mut)
        try:
            gg, gb = guard_of(ls_of(good)), guard_of(ls_of(bad))
            for name in ("к W5-G", "к S-RCK"):
                assert pas(gg, name, i) and not pas(gb, name, i) and not pas(gb, name, "набор")
        finally:
            monkeypatch.setitem(C.THR, i, {"У1-S": 0.415, "У2-S": 0.50, "У3-S": 1.00}[i])
    oracle()
    assert caught(oracle, {"У1-S": 0.50, "У2-S": 0.60, "У3-S": 1.20}[i])                # порог ослаблен — должен ловиться


def scenario_single_gain():
    """Кандидат выигрывает только в 2020 (вклад −), в остальных периодах чуть хуже: Δ < 0, но без лучшего периода Δ > 0."""
    lz = np.full(len(TP), 1.0002)
    lz[Y == 2020] = 0.97
    return lz


def test_u5s_known_answer_single_period_gain_fails_and_spread_gain_passes():
    g = guard_of(ls_of(scenario_single_gain()))
    row = g[(g["set"] == "к S-RCK") & (g["id"] == "У5-S")].iloc[0]
    assert "2020" in row["condition"] and row["value"] > 0 and not row["passed"]
    assert pas(g, "к S-RCK", "У1-S") and pas(g, "к S-RCK", "У2-S")                      # прочие условия — не повод падения
    lz = np.full(len(TP), 0.995)                                                          # выигрыш во всех периодах
    assert pas(guard_of(ls_of(lz)), "к S-RCK", "У5-S")
    lz2 = lz.copy()
    lz2[Y == 2020] = 0.90                                                                 # лучший период 2020, но без него всё равно < 0
    assert pas(guard_of(ls_of(lz2)), "к W5-G", "У5-S")


def test_mut_u5s_without_best_period_variants_caught(monkeypatch):
    def oracle():
        assert not pas(guard_of(ls_of(scenario_single_gain())), "к S-RCK", "У5-S")
    oracle()
    orig = GD._cv

    def total_instead(cnc, h, b, item, key=None):                                         # «Δ целиком» вместо Δ без лучшего периода
        return orig(cnc, h, b, "summary", "delta_total_pct") if item == "loyo" else orig(cnc, h, b, item, key)

    def worst_instead(cnc, h, b, item, key=None):                                         # без худшего периода вместо лучшего
        if item != "loyo":
            return orig(cnc, h, b, item, key)
        m = (cnc["series"] == h) & (cnc["base"] == b) & (cnc["item"] == "loyo")
        return float(cnc[m]["value"].min())
    for mut in (total_instead, worst_instead):
        monkeypatch.setattr(GD, "_cv", mut)
        assert caught(oracle), mut.__name__
        monkeypatch.setattr(GD, "_cv", orig)


def scenario_ten_days_gain():
    lz = np.full(len(TP), 1.001)
    lz[np.linspace(50, len(TP) - 50, 10).astype(int)] = 0.5
    return lz


def test_u6r_trim_known_answer_and_n_slice():
    ls = ls_of(scenario_ten_days_gain())
    g, cnc = guard_of(ls), GD.conc(TP, ls)
    assert pas(g, "к S-RCK", "У1-S") and pas(g, "к S-RCK", "У2-S") and pas(g, "к S-RCK", "У5-S")      # выигрыш есть и не из одного периода
    assert not pas(g, "к S-RCK", "У6-R") and not pas(g, "к W5-G", "У6-R") and not pas(g, "к S-RCK", "набор")
    t = cnc[(cnc["series"] == Z) & (cnc["base"] == "S-RCK") & (cnc["item"] == "trim")]["value"].iloc[0]
    assert abs(t - 100 * (0.001 * (len(TP) - 44)) / (len(TP) - 44)) < 1e-9 and t > 0                    # знаменатель — оставшиеся даты
    n = cnc[(cnc["series"] == Z) & (cnc["base"] == "S-RCK") & (cnc["item"] == "n_slice") & (cnc["key"] == "trim")]["value"].iloc[0]
    assert n == len(TP) - 44
    ok = guard_of(ls_of(0.98))
    assert pas(ok, "к S-RCK", "У6-R")
    lay_bad = guard_of(ls_of(0.98, scenario_ten_days_gain()))                                           # F хорош, слой X1_R — нет
    assert pas(lay_bad, "к S-RCK", "У1-S") and not pas(lay_bad, "к S-RCK", "У6-R")


@pytest.mark.parametrize("k", [0, 1, 5])         # K = 10 здесь не ловится: срезает ровно все 10 дат выигрыша — граница чувствительности сценария
def test_mut_trim_k_not_22_caught(k, monkeypatch):
    def oracle():
        assert not pas(guard_of(ls_of(scenario_ten_days_gain())), "к S-RCK", "У6-R")
    oracle()
    monkeypatch.setattr(C, "KTRIM", k)
    assert caught(oracle)


# ───────── Z-DEC ─────────
def st_row(delta=-0.3, ph=0.01, ps=0.01) -> pd.DataFrame:
    return pd.DataFrame([dict(kind="main", hypothesis="S-RCKZ|S-RCK", delta_pct=delta, p_holm_N=ph, p_spa_c=ps, n_holm=855)])


def dec_of(q=(1.0, 2.0, 3.0), g=None, st=None):
    g = guard_of(ls_of(0.98)) if g is None else g
    return GD.dec_rows(qt(*q), g, st_row() if st is None else st)


def code(d) -> str:
    return str(d.set_index("id").loc["Z-DEC", "value"])


def oracle_dec():
    d = dec_of()
    assert code(d) == "1" and d.set_index("id").loc["Z-DEC", "text"].startswith("S-RCKZ превзошёл S-RCK") and bool(d.iloc[-1]["passed"])
    assert code(dec_of(st=st_row(ph=0.06))) == "2" and code(dec_of(st=st_row(ps=0.06))) == "2" and code(dec_of(st=st_row(ph=0.05))) == "2"       # «превзошёл» — только с подтверждением
    t2 = dec_of(st=st_row(ph=0.06)).set_index("id").loc["Z-DEC", "text"]
    assert "превосходство не установлено" in t2 and "S-RCK остаётся кандидатом" in t2 and "превзошёл" not in t2
    assert code(dec_of(q=(2.0, 3.0, 1.0))) == "3"                                                                                  # H-Z2 не выполнено
    assert code(dec_of(q=(3.0, 2.0, 4.0), st=st_row(delta=0.3, ph=0.001, ps=0.001))) == "3"                                       # H-Z1 не выполнено
    bad = guard_of(ls_of(worst(v1=1.0045)))
    d3 = dec_of(g=bad)
    assert code(d3) == "3" and "У1-S|к S-RCK" in d3.set_index("id").loc["Z-DEC", "text"] and "S-RCK остаётся кандидатом" in d3.set_index("id").loc["Z-DEC", "text"]
    only_w5 = guard_of(ls_of(0.98))
    only_w5.loc[(only_w5["set"] == "к S-RCK") & (only_w5["id"] == "набор"), "passed"] = False                                     # набор «к S-RCK» не пройден
    assert code(dec_of(g=only_w5)) == "3"
    only_w5.loc[(only_w5["set"] == "к S-RCK") & (only_w5["id"] == "набор"), "passed"] = True
    only_w5.loc[(only_w5["set"] == "к W5-G") & (only_w5["id"] == "набор"), "passed"] = False                                      # набор «к W5-G» не пройден
    assert code(dec_of(g=only_w5)) == "3"


def test_zdec_known_answers():
    oracle_dec()


def _mutated_decide(skip):
    def decide(dec, g):
        p = dec.set_index("id")["passed"].map(bool)
        ok = bool(p["H-Z1"] and (skip == "hz2" or p["H-Z2"]) and (skip == "set_w5" or p["У к W5-G"]) and (skip == "set_rck" or p["У к S-RCK"]))
        if ok and (skip == "confirm" or p["подтверждение H-Z1"]):
            return "1", "S-RCKZ превзошёл S-RCK (мутант)"
        return ("2", "S-RCKZ численно лучше S-RCK; превосходство не установлено. S-RCK остаётся кандидатом.") if ok else ("3", "S-RCKZ не прошёл: мутант. S-RCK остаётся кандидатом.")
    return decide


@pytest.mark.parametrize("skip", ["confirm", "hz2", "set_w5", "set_rck"])
def test_mut_zdec_skips_condition_caught(skip, monkeypatch):
    monkeypatch.setattr(GD, "decide", _mutated_decide(skip))
    assert caught(oracle_dec), skip
    monkeypatch.undo()
    oracle_dec()
