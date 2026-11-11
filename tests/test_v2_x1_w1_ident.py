"""W1.4: тождества вкладов — Σ_y c_y = полному Δ и Σ_групп = полному Δ до 1e-12; выброшенная дата ломает тождество (мутация). Синтетика + реальные таблицы."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w1 as W, x1_w1_join as J, x1_w1_st as S, x1_w1_stat as SS


def synth_al() -> dict:
    d = pd.bdate_range("2018-01-03", "2026-11-09")
    tp = pd.DatetimeIndex(list(d[1:]) + [pd.Timestamp("2026-11-10")])
    rng = np.random.default_rng(3)
    rv = np.exp(rng.normal(0, 0.3, len(d)))
    return {k: pd.DataFrame(dict(d=d, t_plus=tp, rv=rv, F=rv * np.exp(rng.normal(0.05 * i, 0.2, len(d))))) for i, k in enumerate(("K1", "K2", "K3", "X1", "ZB"))}


def test_years_identity_known_answer_and_period_split() -> None:
    al = synth_al()
    y, c = SS.years_conc(al)
    tp = pd.DatetimeIndex(al["K1"]["t_plus"])
    assert y[(y.series == "X1") & (y.base == "K2")]["n"].sum() == len(tp)                       # периоды разбивают окно без пересечений и дыр
    assert set(y["period"]) == set(SS.PERIODS)
    for (s, b), g in y[y.base != "-"].groupby(["series", "base"]):
        assert g["contrib_pct"].sum() == pytest.approx(g["delta_total_pct"].iloc[0], abs=1e-12)
    rv = al["K1"]["rv"].to_numpy(float)
    lx, lk = SS.Q.ql(rv, al["X1"]["F"].to_numpy(float)), SS.Q.ql(rv, al["K2"]["F"].to_numpy(float))
    assert y[(y.series == "X1") & (y.base == "K2")]["delta_total_pct"].iloc[0] == pytest.approx(100 * (lx.mean() - lk.mean()) / lk.mean(), abs=1e-12)
    loyo = c[(c.series == "X1") & (c.base == "K2") & (c.item == "loyo") & (c.key == "2018")]["value"].iloc[0]
    m = tp.year != 2018
    assert loyo == pytest.approx(100 * (lx[m] - lk[m]).sum() / lk[m].sum(), abs=1e-12)          # Δ без года пересчитывается по остатку


def test_mutation_dropped_date_breaks_years_identity() -> None:
    al = synth_al()
    rv, tp = al["K1"]["rv"].to_numpy(float), pd.DatetimeIndex(al["K1"]["t_plus"])
    lx, lk = SS.Q.ql(rv, al["X1"]["F"].to_numpy(float)), SS.Q.ql(rv, al["K2"]["F"].to_numpy(float))
    mk = SS.period_masks(tp)
    tot = 100 * (lx.mean() - lk.mean()) / lk.mean()
    assert sum(SS.contrib(lx, lk, mk).values()) == pytest.approx(tot, abs=1e-12)
    bad = {k: v.copy() for k, v in mk.items()}
    bad["2020"][int(np.flatnonzero(bad["2020"])[5])] = False                                    # одна дата выброшена из периода
    assert abs(sum(SS.contrib(lx, lk, bad).values()) - tot) > 1e-6


def test_groups_identity_every_state_and_mutation_dropped_date() -> None:
    rng = np.random.default_rng(1)
    n = 500
    dif, yr = rng.normal(0, 0.01, n), np.repeat(np.arange(2018, 2023), 100)
    lab = np.array(rng.choice(["T1 низ", "T2 середина", "T3 верх", S.NONE], n), dtype=object)
    qk = 0.2
    rows = S.group_rows("X1|K2", "S1", lab, dif, qk, yr)
    tot = 100 * dif.sum() / (n * qk)
    assert sum(r["contrib_pct"] for r in rows) == pytest.approx(tot, abs=1e-12) and sum(r["n"] for r in rows) == n
    assert all(r["delta_total_pct"] == pytest.approx(tot, abs=1e-12) for r in rows)
    lab2, dif2 = lab[1:], dif[1:]                                                               # мутант: выброшена одна дата из групп, итог считается по полному ряду
    bad = S.group_rows("X1|K2", "S1", lab2, dif2, qk, yr[1:])
    assert abs(sum(r["contrib_pct"] for r in bad) - tot) > 1e-6


@pytest.fixture(scope="module")
def real():
    fr, jn, sha = J.load_joined()
    wins = W.windows(fr)
    y, c = SS.years_conc(wins["M"])
    s, b = S.states_bias(fr, wins["M"])
    return y, s, b


def test_real_tables_identities_1e12(real) -> None:
    y, s, b = real
    for _, g in y[y.base != "-"].groupby(["series", "base"]):
        assert g["contrib_pct"].sum() == pytest.approx(g["delta_total_pct"].iloc[0], abs=1e-12)
    g = s[~s["group"].str.startswith("ORACLE")]
    assert len(g) > 0
    for (pair, st), x in g.groupby(["pair", "state"]):
        assert x["contrib_pct"].sum() == pytest.approx(x["delta_total_pct"].iloc[0], abs=1e-12), (pair, st)
        assert abs(x["share"].sum() - 1.0) < 1e-12
    n = pd.DataFrame(g.groupby(["pair", "state"])["n"].sum())
    assert n["n"].nunique() == 1                                                                 # во всех состояниях группы покрывают одни и те же даты окна M
    o = s[s["group"].str.startswith("ORACLE")]
    assert (o["contrib_pct"] <= 1e-12 + s[(s.pair == "X1|K2")].groupby("state")["delta_total_pct"].first().reindex(o["state"]).to_numpy()).all()  # потолок не хуже самой разности X1 − K2
