"""Ж-214, лист 1: стадии `ctrl` (контроли + A0), `m10` (таблица M10-величин + тождества + расписание), `diagv` (этап V), `ledger` (реестр 541 + CHECKPOINT).
Запуск: python3 -W ignore -m models_v2.s9.run_hv2 --stage ctrl|m10|diagv|ledger  (по одной, run_in_background). Лист 2 (после «коммит есть»): стадии `q` (V5), `h` (V6) — hv2_qstage / hv2_hstage."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s8 import hs_ctrl
from models_v2.s9 import hv2_a0 as A0, hv2_diagv as DV, hv2_info as INFO, hv2_ledger as L, hv2_m10 as M, hv2_paths as P, hv2_sched as S, hv2_seas as Z, hv2_sync as Y


def sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stage_ctrl() -> int:
    ctl = hs_ctrl.check()
    P.write_csv(ctl, "CONTROL_VALUES.csv", "сверка контролей (hs_ctrl.check), SHA-256 файлов", "harj_qlike")
    bars = pd.read_parquet(P.SBER)
    d, p, el = A0.panel_grid(bars)
    s = A0.harj_series(p, el)
    comp = pd.read_parquet(P.COMP)
    cmp_ = A0.compare(s, comp)
    P.OUT.mkdir(parents=True, exist_ok=True)
    s.to_parquet(P.OUT / "A0_BETA.parquet")
    facts = {**A0.spec_facts(p, el, s), **cmp_, "components_sha256": sha(P.COMP), "a0_beta_sha256": sha(P.OUT / "A0_BETA.parquet")}
    (P.REP / "harj_qlike").mkdir(parents=True, exist_ok=True)
    (P.REP / "harj_qlike" / "A0_FACTS.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1, default=float))
    print(ctl[["model", "qlike", "declared", "match_4dp", "n"]].to_string(index=False))
    print(json.dumps(facts, ensure_ascii=False, indent=1, default=float))
    return 0 if ctl["match_4dp"].all() and cmp_["same_grid"] and cmp_["max_rel_har"] <= 1e-12 else 1


def stage_m10() -> int:
    sb, ix = pd.read_parquet(P.SBER), pd.read_parquet(P.IMOEX)
    t = M.build(sb)
    r = M.bar_returns(sb)
    t = t.join(Y.build(sb, ix, t.index)).join(Z.build(r, t.index))
    _, p, el = A0.panel_grid(sb)
    t["grid_d"] = t.index.isin(pd.DatetimeIndex(p.dates[el]))
    t["rs_neg_share"] = t["rs_neg"] / t["rv"]
    P.OUT.mkdir(parents=True, exist_ok=True)
    t.to_parquet(P.M10)
    idn = M.identities(t)
    idn["cov_split"] = Y.identity(t.dropna(subset=["cov"]))
    idn["tol"] = P.TOL_ID
    idn["all_ok"] = all(idn[k] <= P.TOL_ID for k in ("parts_sum", "sign_sum", "c_plus_j", "bars_vs_canon", "cov_split"))
    idn["bv_gt_rv_days"] = int(t["bv_gt_rv"].sum())
    idn["days"], idn["days_grid"], idn["days_sched_not_ok"] = len(t), int(t["grid_d"].sum()), int((~t["sched_ok"]).sum())
    idn["days_no_sync"] = int(t["cov"].isna().sum())
    idn["sha256"] = sha(P.M10)
    P.write_csv(pd.DataFrame([idn]), "M10_IDENTITIES.csv", "тождества разложений, относительные отклонения; допуск %g" % P.TOL_ID, "intraday_decomposition")
    sched = S.day_schedule(D.main_session(sb))
    P.write_csv(S.regime_periods(sched), "SCHEDULE_BY_YEAR.csv", "расписание основной сессии из фактических баров (regime = последний бар, минуты)", "intraday_decomposition")
    print(json.dumps(idn, ensure_ascii=False, indent=1, default=float))
    return 0 if idn["all_ok"] else 1


def stage_diagv() -> int:
    sb = pd.read_parquet(P.SBER)
    t = pd.read_parquet(P.M10)
    r = M.bar_returns(sb)
    M2, R = Z.slot_matrix(r, t.index), Z.slot_matrix(r, t.index, power=1)
    pt = DV.profile_table(M2, t.index)
    out = {"PROFILE_BY_SLOT.csv": pt, "PROFILE_SUMMARY.csv": DV.profile_summary(pt), "SIGNATURE_NOISE.csv": DV.signature(R, t.index),
           "DATA_QUALITY.csv": DV.quality(M2, R, t.index, sb), "EDGE_AUCTION_GAP.csv": DV.edge_and_gap(sb, t.index, t["rv"]),
           "RV_SA_VS_RV.csv": DV.sa_compare(t)}
    for name, df in out.items():
        P.write_csv(df, name, "этап V, диагностика измерения (не прогноз)", "seasonality")
        print("==", name); print(df.drop(columns=[c for c in ("slot",) if c in df and name != "PROFILE_BY_SLOT.csv"]).head(14 if name != "PROFILE_BY_SLOT.csv" else 0).to_string(index=False))
    top = t.assign(maxbar=np.nanmax(np.where(np.isfinite(M2), M2, -1), axis=1) / np.nansum(M2, axis=1)).nlargest(10, "rv")[["rv", "maxbar", "n_bars"]]
    P.write_csv(top.reset_index(), "TOP10_RV_DAYS.csv", "10 дней с наибольшей RV и доля максимального бара", "seasonality")
    print(top)
    return 0


def stage_info() -> int:
    tb = INFO.table(pd.read_parquet(P.M10), pd.read_parquet(P.COMP))
    P.write_csv(tb, "M10_VS_HAR_ERROR.csv", "ОПИСАТЕЛЬНО, после GATES_V2: связь M10-величин d с ln-ошибкой A0; не trials, на сетки не влияет")
    print(tb.round(4).to_string(index=False))
    return 0


def stage_ledger() -> int:
    L.init()
    ctl = [{"name": "A0", "note": "HAR-J(б), воспроизведена побитно (harj_qlike/A0_SPEC.md); не trial"},
           {"name": "XJ/LF/A1_roll_126", "note": "замороженные контроли 0,2113 / 0,2117 / 0,2081; не trials"},
           {"name": "Ж-212 D0_full/D0_drop_a", "note": "QLIKE-HAR с логарифмической связью — ссылки на Ж-212 (ключи в GATES_V2 V5), не trials Ж-214"}]
    st = L.write_check(L.load(), controls=ctl)
    print(st["used_total"], st["N_final"], L.n_final())
    return 0 if st["N_final"] == 541 and st["used_total"] == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["ctrl", "m10", "diagv", "ledger", "info", "q", "h", "h2"])
    st = ap.parse_args().stage
    if st in ("q", "h", "h2"):   # лист 2 (после «коммит есть»): V5 / V6 / остаток V6
        from models_v2.s9 import hv2_hstage, hv2_hstage2, hv2_qstage
        raise SystemExit({"q": hv2_qstage.stage_q, "h": hv2_hstage.stage_h, "h2": hv2_hstage2.stage_h2}[st]())
    raise SystemExit({"ctrl": stage_ctrl, "m10": stage_m10, "diagv": stage_diagv, "ledger": stage_ledger, "info": stage_info}[st]())
