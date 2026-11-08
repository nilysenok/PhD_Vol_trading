"""V5-FIN (G23, G25): сводные таблицы. Известный ответ на синтетике (совместные ошибки, межбумажная корреляция), сверка с таблицами листов 0/1a/2 ≤ 1e-12 с подложенной ошибкой, стоимость, реестр и CHECKPOINT[FIN]. Реальные таблицы — только чтение."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s12 import v5_0_io as IO, v5_fin_check as CK, v5_fin_cost as CO, v5_fin_data as D, v5_fin_joint as JT

CP = D.checkpoint()
TAB = {n: IO.read_csv(n + ".csv") for n in D.FIN_TABLES}
R = {"t": TAB, "d": {"cp": CP, "best_gs": CP["FIN"]["best_gs"], "best_sp": CP["FIN"]["best_sp"], "keys": CP["FIN"]["cache_keys"]}}


def synth(over: float) -> dict:
    """2 бумаги × 1800 дат (2018–2024, все окна непусты), RV = 1; шесть прогнозов = RV·over·(1 + 0,01·k): при over > 1 все переоценивают."""
    dates = pd.bdate_range("2018-01-01", periods=1800)
    rows = [dict(ticker=t, group="M16", date=d, t_plus=d, rv_tplus=1.0, **{s: over * (1 + 0.01 * k) for k, s in enumerate(["k1", "k2", "k3", "K8", "GS4", "SP1"])}) for t in ("A", "B") for d in dates]
    return {"cc": pd.DataFrame(rows), "best_gs": "GS4", "best_sp": "SP1"}


def test_joint_known_answer_all_overestimate_then_bracketed():
    j = JT.joint(synth(2.0))
    v = lambda sec, w="2022_2025": float(j[(j["section"] == sec) & (j["window"] == w)]["value"].iloc[0])
    assert v("all_same_sign") == 1.0 and v("rv_below_min") == 1.0 and v("rv_above_max") == 0.0 and v("rv_outside_range") == 1.0
    assert (j[j["section"] == "same_sign"]["value"] == 1.0).all()
    j = JT.joint(synth(0.5))                                              # все недооценивают: RV выше max
    assert float(j[(j["section"] == "rv_above_max") & (j["window"] == "2022_2025")]["value"].iloc[0]) == 1.0
    assert float(j[(j["section"] == "rv_below_min") & (j["window"] == "2022_2025")]["value"].iloc[0]) == 0.0


def test_joint_swapped_sign_mutation_caught_by_known_answer():
    d = synth(2.0)
    d["cc"]["K8"] = 1.0 / d["cc"]["K8"]                                  # мутация: прогноз K8 «по другую сторону» RV
    j = JT.joint(d)
    assert float(j[(j["section"] == "rv_outside_range") & (j["window"] == "2022_2025")]["value"].iloc[0]) == 0.0
    ss = j[(j["section"] == "same_sign") & (j["a"] == "k1") & (j["b"] == "K8") & (j["window"] == "2022_2025")]["value"].iloc[0]
    assert ss == 0.0


def test_cross_ticker_corr_known_answer():
    dates = pd.bdate_range("2023-01-02", periods=120)
    a = pd.Series(np.sin(np.arange(120) * 0.7) + np.cos(np.arange(120) * 0.31))
    m = pd.DataFrame({"date": np.tile(dates, 3), "ticker": np.repeat(["A", "B", "C"], 120)})
    e = pd.Series(np.concatenate([a, a, -a]))
    assert abs(JT.cross_ticker_corr(m, e) - (1 - 1 - 1) / 3) < 1e-12


def test_cmp_catches_shift_missing_key_and_nan():
    ref = pd.DataFrame({"k": ["a", "b"], "x": [1.0, 2.0]})
    assert CK.cmp("t", ref.copy(), ref, ["k"], ["x"]) == []
    assert CK.cmp("t", ref.assign(x=[1.0, 2.0 + 1e-9]), ref, ["k"], ["x"]) != []
    assert CK.cmp("t", ref.iloc[:1], ref, ["k"], ["x"]) != []
    assert CK.cmp("t", ref.assign(x=[1.0, np.nan]), ref, ["k"], ["x"]) != []


def test_fin_tables_equal_sheets_of_leaves_0_1a_2():
    assert CK.cross(R) == []


def test_mutation_substituted_qlike_in_fin_all_is_caught():
    t = dict(TAB)
    a = TAB["FIN_ALL"].copy()
    i = a.index[(a["section"] == "metrics") & (a["series"] == "GS4") & (a["window"] == "all")][0]
    a.loc[i, "qlike"] += 1e-9
    t["FIN_ALL"] = a
    assert CK.cross({**R, "t": t}) != []


def test_mutation_wrong_n_improved_tickers_is_caught():
    t = dict(TAB)
    a = TAB["FIN_TICKERS"].copy()
    i = a.index[(a["ticker"] == "COUNT") & (a["series"] == "GS4")][0]
    a.loc[i, "n_improve_vs_K8"] += 1
    t["FIN_TICKERS"] = a
    assert CK.cross({**R, "t": t}) != []


def test_tables_have_status_header_and_17_tickers_and_nine_configs():
    for n in D.FIN_TABLES:
        head = (IO.REP / (n + ".csv")).read_text(encoding="utf-8").splitlines()[0]
        assert head.startswith("# DEVELOPMENT / HISTORICAL WALK-FORWARD") and "входы " in head and "код v5.fin " in head, n
    tk = TAB["FIN_TICKERS"]
    assert tk[tk["ticker"] != "COUNT"]["ticker"].nunique() == 17 and set(D.GS + D.SP) <= set(tk["series"])
    assert set(TAB["FIN_ALL"][TAB["FIN_ALL"]["section"] == "go"]["series"]) >= set(D.GS + D.SP)
    assert sorted(TAB["FIN_ALL"]["window"].dropna().unique()) == sorted(["2017", "2018_2021", "2022", "2023_2025", "2026", "2022_2025", "2017_2021", "all"])


def test_sber_has_reference_on_same_2271_dates():
    s = TAB["FIN_SBER"]
    a = s[s["window"] == "all"]
    assert (a["n"] == 2271).all() and abs(float(a["ref_a1_roll_126"].iloc[0]) - 0.2081) < 1e-12 and (a["d_to_ref"] > 0).all()


def test_cost_has_four_stages_rc0_and_parameter_counts():
    c = TAB["FIN_COST"]
    tm, pr = c[c["section"] == "time"], c[c["section"] == "params"].set_index("stage")["n_params"]
    assert list(tm["stage"].astype(str)) == ["0", "GS", "SP", "DX1"] and (tm["rc"] == 0).all() and (tm["seconds"] > 0).all()
    assert {k: int(v) for k, v in pr.items()} == {"GS1": 2, "GS2": 2, "GS3": 3, "GS4": 3, "GS5": 3, "SP1": 272, "SP2": 9, "SP3": 12, "SP4": 18}
    mem = CO.table(False, c)
    assert CK.cost_ok(c, mem) and not CK.cost_ok(c.assign(rc=1), mem)


def test_registry_and_checkpoint_ok_and_mutations_caught(monkeypatch):
    assert CK.registry(R) == []
    bad = json.loads(json.dumps(CP))
    bad["FIN"]["used_total"] = 10
    monkeypatch.setattr(D, "checkpoint", lambda: bad)
    assert CK.registry(R) != []
    bad2 = json.loads(json.dumps(CP))
    bad2["N_wide"] = 3105                                                # прежние ключи тронуты
    monkeypatch.setattr(D, "checkpoint", lambda: bad2)
    assert CK.registry(R) != []
    assert CP["FIN"]["used_total"] == 9 and CP["FIN"]["budget"] == 24 and CP["N_declared_before_stats"] == 704 and CP["N_wide"] == 3104
