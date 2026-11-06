"""Ж-214, лист 4: проводка стадий `ml` и `mkt` на синтетике (весь поток: реестр, кэш, строки, V13, абляции, Cb, таблицы). Реальные реестр, CHECKPOINT и ряды НЕ трогаются: пути и запись реестра
перенаправлены во временный каталог. Синтетика мала (мин. пар и закрытых сокращены), поэтому проверяется проводка и инварианты, не качество."""
import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s9 import hv2_eval as E, hv2_ledger as L, hv2_mktstage as MKS, hv2_mlout as O, hv2_mlstage as MLS, hv2_msfeat as MS, hv2_paths as P, hv2_synth_ml as SY

MP, MC = 100, 15


def fake_ctx(dw):
    el = dw.el
    a0 = MS.base_a0(dw, el)[el]
    df = pd.DataFrame({"d": dw.p.dates[el], "target_date": dw.p.dates[el + 1], "rv": dw.p.rv[el + 1], "HAR": a0, "XJ": a0 * 1.1, "LF": a0 * 1.2})
    return {"w": None, "df": df, "refs": {"A0": a0, "XJ": a0 * 1.1, "LF": a0 * 1.2}, "same_grid": True, "max_rel_rv": 0.0}


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    cp, allp = tmp_path / "CP.json", tmp_path / "ALL.csv"
    cp.write_text(json.dumps({"controls": [], "decisions": {"V6_rest": {"B_star_V9": "H2"}, "V7": {"passed_V13": ["T"]}}}))
    monkeypatch.setattr(P, "OUT", tmp_path / "out")
    monkeypatch.setattr(P, "REP", tmp_path / "rep")
    monkeypatch.setattr(L, "CHECK", cp)
    o_reg, o_load, o_wc = L.register, L.load, L.write_check
    monkeypatch.setattr(L, "register", lambda d, c: o_reg(d, c, dst=allp, check=cp))
    monkeypatch.setattr(L, "load", lambda dst=None: o_load(allp))
    monkeypatch.setattr(L, "write_check", lambda df, path=None, **kw: o_wc(df, path=cp, **kw))
    o_bases = MS.bases
    monkeypatch.setattr(MS, "bases", lambda dw, upto=None, need=("A0", "H2", "T3"), min_pairs=MP: o_bases(dw, upto, need, MP))
    o_run = MLS.run_cfg
    monkeypatch.setattr(MLS, "run_cfg", lambda dw, B, sp, procs=1, **kw: o_run(dw, B, sp, 1, min_pairs=MP, min_closed=MC, **kw))
    o_tr = MLS.trunc
    monkeypatch.setattr(MLS, "trunc", lambda dw, sp, procs, tail=20: o_tr(dw, sp, 1, tail))
    monkeypatch.setattr(E, "TRUNC_AT", (50,))
    monkeypatch.setattr(O, "base_check", lambda ctx, dw, B: {"A0": 0.0, "H2": 0.0, "T3": 0.0})
    def refs(ctx, dw, B):
        ctx["refs"]["Bstar_H2"], ctx["refs"]["T3"] = B["H2"][dw.el], B["T3"]["s2"][dw.el]
    monkeypatch.setattr(O, "add_refs", refs)
    forced = lambda rows, controls, spent, limit, _o=E.v13: {**_o(rows, controls, spent, limit), "continue": True}
    monkeypatch.setattr(E, "v13", forced)
    dw = SY.dworld(n=520, tail=80, signal=1.0)
    monkeypatch.setattr(E, "context", lambda: fake_ctx(dw))
    monkeypatch.setattr(MS, "build_world", lambda w, m10, ix: dw)
    return tmp_path, dw, allp, cp


@pytest.mark.skipif(not (P.M10.exists() and P.IMOEX.exists()), reason="нет данных")
def test_stage_ml_full_flow_registers_budget_tables_and_separate_libraries(sandbox):
    tmp, dw, allp, cp = sandbox
    assert MLS.stage_ml(procs=1) == 0
    names = ["N1x_A0", "N1l_A0", "N1x_B", "N1l_B", "N2x", "N2l", "AblQ_x", "AblS_x", "AblT_x", "AblM_x"]
    reg = pd.read_csv(allp)
    new = reg[reg["program"] == "Ж-214"]
    assert len(new) == 11 and set(new["direction"]) == {"structural_ml"} and len(reg) == 541 + 11
    cfgs = [json.loads(c) for c in new["config"]]
    assert [c["name"] for c in cfgs[:10]] == names and cfgs[10]["name"].startswith("Cb_") and all(c[L.DIFF].startswith("Ж-212:residual_qlike:") for c in cfgs)
    assert all(c["lib"] == MLS.spec(c["name"])["lib"] and c["lib"] in ("XGB", "LGBM") for c in cfgs)
    assert {c["lib"] for c in cfgs[:2]} == {"XGB", "LGBM"} and cfgs[0]["lib"] != cfgs[1]["lib"]       # XGB и LGBM — отдельные конфигурации, не объединены
    cpj = json.loads(cp.read_text())
    assert cpj["used"]["structural_ml"] == 11 and cpj["N_final"] == 541 + 11 and "V9" in cpj["decisions"] and cpj["decisions"]["V9"]["unused_trials"] == 5
    x, g = (pd.read_csv(tmp / "rep" / d / ("STRUCT_%s.csv" % k), comment="#") for d, k in (("xgboost_structural", "XGB"), ("lightgbm_structural", "LGBM")))
    assert set(x[x["config"].notna() & x["lib"].notna()]["lib"]) == {"XGB"} and set(g[g["lib"].notna()]["lib"]) == {"LGBM"}
    chk = pd.read_csv(tmp / "rep" / "xgboost_structural" / "ML_CHECKS.csv", comment="#")
    lam = chk[chk["check"].str.startswith("lam_")]
    assert len(lam) == 11 and lam["lam0_bitwise"].all() and lam["lam1_equals_full"].all() and lam["positive"].all()
    d = chk[chk["check"] == "deriv_real"]
    assert (d["max_err_grad"] < 1e-6).all() and (d["min_hess"] > 0).all()
    assert (pd.read_csv(tmp / "rep" / "xgboost_structural" / "ABLATION_GROUPS_XGB.csv", comment="#")["parent"] != "").all()


@pytest.mark.skipif(not (P.M10.exists() and P.IMOEX.exists()), reason="нет данных")
def test_stage_mkt_full_flow_min_set_then_rest_and_budget(sandbox):
    tmp, dw, allp, cp = sandbox
    assert MKS.stage_mkt() == 0
    reg = pd.read_csv(allp)
    new = reg[reg["program"] == "Ж-214"]
    assert set(new["direction"]) == {"market"} and 6 <= len(new) <= 10
    t = pd.read_csv(tmp / "rep" / "market_commonality" / "MARKET_COMMONALITY.csv", comment="#")
    mb = t[t["config"] == "E1"].iloc[0]
    assert abs(mb["d_qlike_Mb"]) < 1e-15                                                  # E1 против самой себя (crit_wf на синтетике 2018–2019 не определён)
    assert (t[t["config"].isin(["E1", "E2", "E3", "E4", "E5", "E6", "E9"])]["trunc_max_diff"] == 0).all()
    cpj = json.loads(cp.read_text())
    assert cpj["used"]["market"] == len(new) and cpj["decisions"]["V10"]["ran"][:4] == ["E1", "E2", "E3", "E4"]
    assert np.isfinite(pd.read_csv(tmp / "rep" / "market_commonality" / "MKT_BLOCKS_INFORMATIVE.csv", comment="#")["d_qlike_Mb"]).all()
