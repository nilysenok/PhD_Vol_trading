"""V4-RQC: стадия `RQC` (G35–G37). Порядок: объявление (CHECKPOINT, до первого файла результатов) -> прогнозы K8, RQC1–RQC3 (кэш, ключ = хэш входов и кода, Д5) -> общие таблицы (контроли, K8, оценённые веса B–CLD, RQC) ->
LOTO лучшей RQC -> годы, SBER, a -> выходы -> реестр. Базовые модели и прежние гибриды не переобучаются (только чтение); выбор — только `crit_panel` (M16 × цели 2017–2021).
Отчёт и проверки — вне путей прогноза (`v4_rqc_report`, `v4_rqc_check`)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import (v4_b_data as D, v4_b_decl as DECL, v4_b_loto as LO, v4_b_stage as BS, v4_b_stats as ST, v4_c_cache as CC, v4_cb_years as Y, v4_e_go as GO, v4_io as IO, v4_rqc_cache as RC,
                           v4_rqc_run as R, v4_rqc_stats as RS, v4_trials as TR, v4_z_build as ZB)

CFGS = R.CFGS
REFS = ["C1", "C4", "C8", "NTP1", "CLD1"]
STAGE_W = {"B": ["B1", "B2", "B3", "B4", "B6"], "C": ["C%d" % i for i in range(1, 11)], "E": ["E%d" % i for i in range(1, 7)], "CS": ["CS1", "CS2"], "NTP": ["NTP%d" % i for i in range(1, 5)], "CLD": ["CLD%d" % i for i in range(1, 5)]}
ALLW = [c for v in STAGE_W.values() for c in v]
RQC_RULE = ("лист K8-RQC (Ж-225 п. 6): K8 = sqrt(H·M), H = K1, M = sqrt(K2·K3) (контроль, 1 конфигурация); F = a·K8, a = mean(RV/K8) по закрытым парам t⁺ ≤ d, обрезка [0,6; 1,6], до 60 закрытых пар a = 1: "
            "RQC1 — все закрытые пары 17 бумаг (расширяющееся окно); RQC2 — пары панели года Y−1 (2017: a = 1); RQC3 — ln a бумаги = иерархическая (κ = 250) EWMA (полураспад 60) winsorized (p5/p95 прошлого окна бумаги) ln(RV/K8); "
            "заморозка на 5 дней с дня-скачка (доля J > p90 или |Δ ln RV| > p95 по прошлому окну бумаги), шаг |Δ ln a| ≤ 0,02; панель — 17 бумаг, в LOTO — 16; «go» — три условия (Ж-225 п. 10); выбор — crit_panel")


def declare() -> dict:
    """Один раз: время объявления листа до первого файла результатов; повтор не сдвигает."""
    cp = json.loads(DECL.CP.read_text())
    if "RQC_declared_at" not in cp:
        DECL.update(RQC_declared_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), RQC_rule=RQC_RULE, RQC_configs=CFGS, RQC_prior_sha=RC.prior_sha())
        cp = json.loads(DECL.CP.read_text())
    return cp


def forecasts(P: pd.DataFrame, Zl: pd.DataFrame, key: str) -> dict:
    pre = {}
    def build(cfg):
        if cfg == "RQC3" and "v" not in pre:
            pre["v"] = R.make_pre(P, Zl)
        return {"": R.forecast(cfg, P, Zl, pre=pre.get("v"))[0]}
    return {cfg: RC.cached(cfg, key, lambda cfg=cfg: build(cfg))[""] for cfg in CFGS}


def zero_ok(best: str, P, Zl, t: str, out: pd.DataFrame, pre) -> bool:
    """Исключённая бумага не входит в панельные величины: подмена её RV не меняет прогнозы остальных бумаг (побитно)."""
    Q = P.copy()
    Q.loc[Q["ticker"] == t, "rv"] *= 5.0
    o2, m = R.forecast(best, Q, Zl, exclude=t, pre=pre)
    f = lambda x: x[x["ticker"] != t].sort_values(["date", "ticker"])["F"].to_numpy()
    return bool(np.array_equal(f(out), f(o2)) and t not in m["panel"] and len(m["panel"]) == len(IO.T17) - 1)


def loto(best: str, P, Zl, fc: dict, key: str) -> pd.DataFrame:
    """LOTO: панельные величины — по 16 бумагам без исключённой; RQC3 — бумажная часть по собственным закрытым парам исключённой (как у любой бумаги)."""
    def build():
        pre = R.make_pre(P, Zl) if best == "RQC3" else None
        parts, zero = [], []
        for t in IO.M16:
            out = R.forecast(best, P, Zl, exclude=t, pre=pre)[0]
            parts.append(out[out["ticker"] == t][["ticker", "date", "t_plus", "rv", "h", "F"]])
            zero.append({"ticker": t, "ok": zero_ok(best, P, Zl, t, out, pre)})
        return {"": pd.concat(parts, ignore_index=True), "_zero": pd.DataFrame(zero)}
    r = RC.cached("RQC_loto_" + best, key, build)
    return LO.loto_table(best, r[""], fc[best], dict(zip(r["_zero"]["ticker"], r["_zero"]["ok"])))


def trials_rows(key: str) -> list:
    spec = lambda c: json.dumps({"cfg": c, "base": "K8=sqrt(H*sqrt(K2*K3))", "a_clip": [0.6, 1.6], "kappa": 250 if c == "RQC3" else None, "cache_key": key[:16]}, ensure_ascii=False)
    d = {"K8": "equal_weight_control_K8"}
    return ["Ж-222:%s:%s,Ж-222,harj_hybrid_v4,%s,\"%s\",trial" % (c, hashlib.sha256((c + "|" + key).encode()).hexdigest()[:12], d.get(c, "robust_qlike_calibration_RQC"), spec(c).replace('"', '""')) for c in CFGS]


def write_trials(key: str) -> None:
    TR.replace_rows(["Ж-222:%s:" % c for c in CFGS], trials_rows(key))       # только K8, RQC1–RQC3; строки других стадий сохраняются


def stage_rqc() -> dict:
    t0 = time.time()
    cp = declare()
    P, kc = D.panel(), RC.keys()
    Zl = ZB.z_by_ticker(CC.keys("Z")["key"])
    ex = {c: pd.read_parquet(RC.OUT / (("B_" + c if c.startswith("B") else c) + ".parquet")) for c in ALLW}
    fc = forecasts(P, Zl, kc["key"])
    k = D.eval_frame(P, {**ex, **fc})
    cols_main, cols_all = list(DECL.CTRL) + REFS + CFGS, list(DECL.CTRL) + ALLW + CFGS
    c, crit = S.common(k, cols_all), S.crit_panel(k, cols_all)
    pt, cs = ST.per_ticker(c, cols_all), crit.set_index("series")["crit_panel"]
    best = min(CFGS[1:], key=lambda x: float(cs[x]))
    o = {"P": P, "Zl": Zl, "cp": cp, "k": k, "c": c, "crit": crit, "pt": pt, "cols_main": cols_main, "cols_all": cols_all, "fc": fc, "kc": kc, "best": best, "t0": t0, "ex": ex,
         "note": "входы sha256=%s…, код sha256=%s…" % (kc["inputs"][:12], kc["code"][:12])}
    o["comp"] = ST.comparison(c, cols_all, crit, pt)
    o["tests"] = RS.tests(c, CFGS, cp["N_declared_before_stats"], cp["N_wide"])
    o["sber"], o["years"] = ST.sber(c, cols_main, BS.sber_ref()), Y.years_table(k, cols_main)
    o["go"] = GO.rules(k, ["k1", "K8"] + CFGS[1:], cs, cp["go_threshold_B"])
    o["a"] = RS.a_table(fc, {e: pd.read_parquet(RC.OUT / (e + ".parquet")) for e in STAGE_W["E"][:5]})
    o["loto"] = loto(best, P, Zl, fc, kc["key"])
    write_all(o)
    if cp.get("RQC_cache_key") != kc["key"]:
        DECL.update(RQC_runtime_sec=round(time.time() - t0, 1), RQC_processes=1)
    DECL.update(RQC_configs_spent=len(CFGS), used_after_RQC=int(cp["used_after_CLD"]) + len(CFGS), RQC_best=best, RQC_cache_key=kc["key"], RQC_stop_go="go" if o["go"]["cfg"][best]["go"] else "stop", RQC_go_rules=o["go"])
    return o


def write_all(o: dict) -> None:
    n, note, thr = o["cp"], o["note"], o["cp"]["go_threshold_B"]
    grp = {**{x: "контроль" for x in DECL.CTRL}, "K8": "контроль K8", **{x: "RQC" for x in CFGS[1:]}, **{x: "веса " + s for s, v in STAGE_W.items() for x in v}}
    cmp_ = o["comp"].assign(group=lambda d: d["series"].map(grp), in_main=lambda d: d["series"].isin(o["cols_main"]))
    IO.write_csv(cmp_, "RQC_COMPARISON.csv", "лист K8-RQC (Ж-225 п. 6): crit_panel = hp_stats_sel.crit_panel (M16 × 2017–2021); qlike за окно 2017_2021 — простое среднее по M16; ряды: контроли, K8, RQC1–3, ориентиры C1, C4, C8, NTP1, CLD1 (in_main) "
                 "и все оценённые веса B–CLD (group «веса …»); все ряды на одинаковых датах; порог go = %.6f; %s" % (thr, note))
    IO.write_csv(o["pt"][o["pt"]["series"].isin(o["cols_main"])], "RQC_PER_TICKER.csv", "17 бумаг (SBER — отдельная диагностика, в выбор не входит); общие даты; " + note)
    IO.write_csv(o["tests"], "RQC_TESTS.csv", "DM НУ лаг 5 и блочный бутстрап дат; Δ = QLIKE(cfg) − QLIKE(база); Холм: N=%d (p_holm), N_wide=%d (p_holm_wide), семейство M16 × {2022_2025, all} × (K8, RQC1–3) × компараторы; %s" % (n["N_declared_before_stats"], n["N_wide"], note))
    IO.write_csv(o["years"], "RQC_YEARS.csv", "crit_panel по годам цели: crit_part = pen_Y/4 суммируется в crit_panel; ann_qlike — средний QLIKE по M16 за год цели (2017 и 2018 — отдельные строки); 2022–2026 описательно; " + note)
    IO.write_csv(o["sber"], "RQC_SBER.csv", "SBER отдельно, в выбор не входит; ориентир 0,2081 — потолок; " + note)
    IO.write_csv(o["loto"], "RQC_LOTO.csv", "LOTO лучшей RQC: панельные величины по 16 бумагам, исключённая бумага не входит в панель (RQC3 — бумажная часть по её закрытым парам); " + note)
    IO.write_csv(o["a"], "RQC_A.csv", "ln a (kind=month: среднее по дням месяца; PANEL — среднее по бумагам; kind=window: окна шоков, описательно). E1–E5: ln-множитель b к F0 (H или B2), не к K8; " + note)
    write_trials(o["kc"]["key"])
