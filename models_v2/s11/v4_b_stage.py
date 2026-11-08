"""V4-B: стадия `B` (этап операторов объединения, G9–G14). Порядок: порог «go» и N_wide в CHECKPOINT.json -> прогнозы B1–B4, B6 -> правило B5 (по OOF B2) -> B5 при правиле -> статистика -> LOTO -> реестр.
Базовые модели не переобучаются; гиперпараметры (κ, приор, частота) фиксированы."""
from __future__ import annotations

import hashlib
import json
import time

import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_b_cache as CA, v4_b_data as D, v4_b_decl as DECL, v4_b_loto as LO, v4_b_report as RP, v4_b_rule as RU, v4_b_run as R, v4_b_stats as ST, v4_io as IO, v4_trials as TR

BASE = ["B1", "B2", "B3", "B4", "B6"]
REF_NAME = "A1_roll_126"


def forecasts(cfg: str, P: pd.DataFrame, key: str) -> tuple:
    r = CA.cached("B_" + cfg, key, lambda: dict(zip(("", "_weights"), R.forecast(cfg, P))))
    return r[""], r["_weights"]


def sber_ref() -> float:
    b = IO.read_csv("SBER_BENCH.csv")
    return float(b[b["series"].str.startswith(REF_NAME)]["qlike"].iloc[0])


def trials_rows(cfgs: list, key: str) -> pd.DataFrame:
    rows = [{"key": "Ж-222:%s:%s" % (c, hashlib.sha256((c + "|" + key).encode()).hexdigest()[:12]), "program": "Ж-222", "source": "harj_hybrid_v4",
             "direction": "operators_B", "config": json.dumps({"cfg": c, "kappa": 250, "prior": 250, "cache_key": key[:16]}, ensure_ascii=False), "kind": "trial"} for c in cfgs]
    return pd.DataFrame(rows, columns=["key", "program", "source", "direction", "config", "kind"])


def stage_b() -> dict:
    t0 = time.time()
    P = D.panel()
    cp = DECL.declare(D.eval_frame(P))            # порог и N_wide — до первого файла результатов
    ks = CA.keys()
    note = "входы sha256=%s…, код sha256=%s…" % (ks["inputs"][:12], ks["code"][:12])
    fc, wt = {}, {}
    for cfg in BASE:
        fc[cfg], wt[cfg] = forecasts(cfg, P, ks["key"])
    st = RU.rule_stats(fc["B2"])
    IO.write_csv(pd.DataFrame([{**st, "passes": RU.rule_passes(st), "b5_computed": RU.rule_passes(st)}]), "B5_RULE.csv", "правило B5 по OOF B2, M16 × цели 2017–2021; " + note)
    cfgs = RU.cfgs_to_run(st)
    if "B5" in cfgs:
        fc["B5"], wt["B5"] = forecasts("B5", P, ks["key"])
    cfgs = [c for c in ["B1", "B2", "B3", "B4", "B5", "B6"] if c in cfgs]
    k = D.eval_frame(P, {c: fc[c] for c in cfgs})
    cols = DECL.CTRL + cfgs
    c = S.common(k, cols)
    crit = S.crit_panel(k, cols)
    pt = ST.per_ticker(c, cols)
    out = {"P": P, "cp": cp, "st": st, "cfgs": cfgs, "fc": fc, "wt": wt, "c": c, "crit": crit, "ks": ks, "note": note, "t0": t0}
    out["tests"], out["mcs"] = ST.tests(c, cfgs, cp["N_declared_before_stats"], cp["N_wide"]), ST.mcs(c, cols)
    out["comp"], out["pt"], out["sber"] = ST.comparison(c, cols, crit, pt), pt, ST.sber(c, cols, sber_ref())
    out["best"] = min(cfgs, key=lambda x: float(crit.set_index("series").loc[x, "crit_panel"]))
    out["loto"], out["gche"] = loto_all(out, ks["key"])
    write_all(out)
    if cp.get("B_cache_key") != ks["key"]:       # время — только полного счёта (повтор из кэша его не затирает)
        DECL.update(B_runtime_sec=round(time.time() - t0, 1), B_processes=1)
    DECL.update(B_configs_spent=len(cfgs), B_best=out["best"], B_cache_key=ks["key"])
    return out


def loto_all(o: dict, key: str) -> tuple:
    """LOTO для лучшей B по crit_panel и для B2; GCHE — перенос панельных весов."""
    lt, gc = [], []
    for cfg in dict.fromkeys([o["best"], "B2"]):
        def build(cfg=cfg):
            lo, zero = LO.loto_forecasts(cfg, o["P"])
            return {"": lo, "_zero": pd.DataFrame([{"ticker": t, "ok": v} for t, v in zero.items()])}
        r = CA.cached("B_loto_" + cfg, key, build)
        zero = dict(zip(r["_zero"]["ticker"], r["_zero"]["ok"]))
        lt.append(LO.loto_table(cfg, r[""], o["fc"][cfg], zero))
        gc.append(LO.transfer_gche(cfg, o["wt"][cfg]))
    na = pd.DataFrame([{"config": "все", "held_out": t, "window": "—", "series": "нет M (нет K2/K3), не применимо"} for t in IO.P.E11 if t != "GCHE"])
    return pd.concat(lt, ignore_index=True), pd.concat(gc + [na], ignore_index=True)


def write_all(o: dict) -> None:
    n, note = o["cp"], o["note"]
    IO.write_csv(o["comp"], "B_COMPARISON.csv", "crit_panel = hp_stats_sel.crit_panel (M16 × 2017–2021, Ж-217); метрики — среднее по M16 на общих датах; порог go = %.6f; %s" % (n["go_threshold_B"], note))
    IO.write_csv(o["pt"], "B_PER_TICKER.csv", "17 бумаг (SBER — отдельная диагностика, в выбор не входит); общие даты; " + note)
    IO.write_csv(o["tests"], "B_TESTS.csv", "DM НУ лаг 5 и блочный бутстрап дат; Δ = QLIKE(B) − QLIKE(база); Холм: N=%d (p_holm), N_wide=%d (p_holm_wide), основное семейство — M16 × {2022_2025, all}; %s" % (n["N_declared_before_stats"], n["N_wide"], note))
    IO.write_csv(o["mcs"], "B_MCS.csv", "MCS 90 %, T_max, блочный бутстрап дат (блок 20, 5000, seed 42), панельный QLIKE M16; " + note)
    IO.write_csv(pd.concat([o["wt"][c] for c in o["cfgs"]], ignore_index=True), "B_WEIGHTS.csv", "параметры оператора по месяцам × бумагам (w — вес на H; B6: wa, wb, wc на H, K2, K3), PANEL — панельное значение; converged — сходимость; " + note)
    IO.write_csv(o["sber"], "B_SBER.csv", "SBER отдельно, в выбор не входит; ориентир 0,2081 — потолок; " + o["note"])
    IO.write_csv(pd.concat([o["loto"], o["gche"]], ignore_index=True), "B_LOTO.csv", "LOTO (16 бумаг, δ = 0 у исключённой) и перенос на GCHE; прочие E11 — нет M, не применимо; " + note)
    (IO.REP / "B_REPORT.md").write_text(RP.build(o))
    (IO.REP / "ALL_TRIALS.csv").write_text(TR.HEADER + trials_rows(o["cfgs"], o["ks"]["key"]).to_csv(header=False, index=False, lineterminator="\n"))
