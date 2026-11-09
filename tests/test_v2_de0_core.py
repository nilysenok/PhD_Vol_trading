"""DE0, G2: тождества (i), (iii), (iv) и знак Δ — известные ответы на синтетике; мутации «выброшенная дата» и «знак Δ перевёрнут» ловятся."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s15 import de0_core as C, de0_tables as T


def _syn(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    year = np.repeat(np.arange(2018, 2026), n // 8)
    rv = np.exp(rng.normal(0, 0.5, len(year)))
    return rv, year, rng


def test_qlike_known_answer_and_nonneg():
    assert C.qlike_day([2.0, 3.0], [2.0, 3.0]).tolist() == [0.0, 0.0]
    assert abs(C.qlike_day([2.0], [1.0])[0] - (2.0 - np.log(2.0) - 1.0)) < 1e-15
    rv, _, rng = _syn()
    assert (C.qlike_day(rv, rv * np.exp(rng.normal(0, 0.3, len(rv)))) >= 0).all()


def test_identity_i_sum_of_year_contributions_equals_delta():
    rv, year, rng = _syn()
    dl = C.qlike_day(rv, rv * np.exp(rng.normal(0, .4, len(rv)))) - C.qlike_day(rv, rv * np.exp(rng.normal(0, .3, len(rv))))
    t = C.contrib_years(dl, year)
    assert C.identity_years(t) < 1e-12 and abs(t[t["year"] == "итого"]["contrib"].iloc[0] - dl.mean()) < 1e-12


def test_mutation_dropped_date_breaks_identity_i():
    rv, year, rng = _syn()
    dl = rng.normal(0, 1, len(rv))
    t = C.contrib_years(dl, year)
    bad = t.copy()
    row = bad.index[bad["year"] == "2020"][0]
    bad.loc[row, "contrib"] -= dl[np.flatnonzero(year == 2020)[3]] / len(dl)         # одна дата выброшена из вклада года
    assert C.identity_years(t) < 1e-12 and C.identity_years(bad) > 1e-6


def test_identity_iii_under_plus_over_and_dropped_date():
    rv, year, rng = _syn()
    f = rv * np.exp(rng.normal(0, .4, len(rv)))
    r = C.under_over(rv, f, year, "2018_2025")
    assert C.identity_under_over(r) < 1e-12 and r["under"] > 0 and r["over"] > 0
    m = C.wmask(year, "2018_2025")
    u = C.u_of(rv, f)[m]
    l = C.qlike_day(rv, f)[m]
    k = int(np.flatnonzero(u <= 0)[0])
    bad = dict(r, over=float((l[u <= 0].sum() - l[k]) / m.sum()))                  # выброшена одна дата переоценки
    assert C.identity_under_over(bad) > 1e-6


def test_identity_iv_tails_and_dropped_date():
    _, _, rng = _syn()
    dl = rng.normal(0, 1, 300)
    for q in (0.01, 0.05):
        r = C.tail_share(dl, q)
        assert r["identity"] < 1e-12 and C.tail_identity(r) < 1e-12 and r["k"] == int(np.ceil(q * 300))
        sel = np.argsort(-np.abs(dl))[: r["k"]]
        assert set(np.flatnonzero(np.abs(dl) >= np.abs(dl[sel]).min())) >= set(sel)
        bad = dict(r, delta_ex_top=float((np.delete(dl, sel).sum() - np.delete(dl, sel)[0]) / (300 - r["k"])))
        assert C.tail_identity(bad) > 1e-6


def test_sign_of_delta_known_answer_and_flipped_sign_is_caught():
    """Гибрид хуже K2 (прогноз вдвое мимо, K2 точен) → Δ > 0, adv < 0; перевёрнутый знак даёт обратное."""
    rv, year, _ = _syn()
    f = {"K1": rv * 1.3, "K2": rv.copy(), "A2": rv * 2.0, "NTP2": rv * 1.1, "cfg8": rv * 1.05, "cfg4": rv * 1.5}
    gb = T.gap_base(rv, f, year)
    a2 = gb[(gb["hybrid"] == "A2") & (gb["window"] == "2018_2025")].iloc[0]
    assert a2["delta_hybrid_vs_K2"] > 0 and a2["q_K2"] == 0.0
    c = C.cause_row(rv, f, year, "A2")
    assert c["adv_2018_2021"] < 0
    flipped = (a2["q_K2"] - a2["q_hybrid"])                                          # мутация: K2 − гибрид вместо гибрид − K2
    assert flipped < 0 < a2["delta_hybrid_vs_K2"]


def test_decomposition_parts_sum_to_change_in_advantage():
    rv, year, rng = _syn(800)
    f = {"K1": rv * np.exp(rng.normal(.1, .5, len(rv))), "K2": rv * np.exp(rng.normal(0, .3, len(rv))), "h": rv * np.exp(rng.normal(0, .4, len(rv)))}
    c = C.cause_row(rv, f, year, "h")
    assert c["identity"] < 1e-12 and abs(c["share_base"] + c["share_gu"] + c["share_go"] - 1.0) < 1e-9


def test_acf_and_runs_known_answer():
    x = np.tile([1.0, -1.0], 50)
    a = C.acf(x)
    assert a[1] < -0.95 and a[2] > 0.95
    assert C.run_lengths(np.array([1, 1, 0, 1, 0, 0, 1, 1, 1])).tolist() == [2, 1, 3]
    assert C.run_lengths(np.array([-1, -1])).tolist() == []
    s = C.run_summary(np.array([1, 1, 0, 1, 0, 0, 1, 1, 1]))
    assert s["n_runs"] == 3 and s["max_len"] == 3 and s["n_len_1"] == 1 and s["n_len_5"] == 0


def test_real_delta_signs_and_values_equal_v7_fin_all():
    """Знак и значение Δ на сохранённых прогнозах = разность табличных QLIKE FIN_ALL V7 (2022–2025: cfg8 хуже K2)."""
    from models_v2.s15 import de0_data as D, de0_io as IO
    d = D.load()
    year = np.asarray(d["tp"].year)
    gb = T.gap_base(d["rv"], d["f"], year)
    fin = pd.read_csv(IO.V7 / "FIN_ALL.csv", comment="#").set_index(["series", "window"])["qlike"]
    for h, s in (("cfg8", "cfg8"), ("A2", "V6:A2"), ("NTP2", "V4:NTP2"), ("cfg4", "cfg4")):
        for w in ("2018_2021", "2022_2025"):
            got = gb[(gb["hybrid"] == h) & (gb["window"] == w)].iloc[0]["delta_hybrid_vs_K2"]
            assert abs(got - (fin[(s, w)] - fin[("K2", w)])) < 1e-12
    assert gb[(gb["hybrid"] == "cfg8") & (gb["window"] == "2022_2025")].iloc[0]["delta_hybrid_vs_K2"] > 0
    assert gb[(gb["hybrid"] == "cfg8") & (gb["window"] == "2018_2021")].iloc[0]["delta_hybrid_vs_K2"] < 0
