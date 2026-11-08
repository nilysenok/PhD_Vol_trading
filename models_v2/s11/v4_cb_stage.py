"""V4-C, лист C-b: стадия `Cb` (G19–G21). Порядок: объявление (CHECKPOINT, до первого файла результатов) -> стадия C-a (C1–C4 из кэша, без изменений) -> прогнозы C5–C10 (кэш) -> общие таблицы C1–C10 ->
LOTO лучшей из C1–C10 (если не C1) -> C_YEARS -> выходы -> реестр. Базовые модели не переобучаются; гиперпараметры фиксированы; выбор — только `crit_panel` (M16 × цели 2017–2021)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time

import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import (v4_b_data as D, v4_b_decl as DECL, v4_b_loto as LO, v4_b_stage as BS, v4_b_stats as ST, v4_c_cache as CC, v4_c_stage as CA, v4_c_stats as CS, v4_cb_report as RP,
                           v4_cb_run as R, v4_cb_years as Y, v4_io as IO, v4_trials as TR, v4_z_feat as ZF)

CFGS = R.CFGS
ALLC = CA.BASE + CFGS
CB_RULE = ("лист C-b: C5 GAM (B-сплайн ℓ, 5 ст. св.); C6 монотонный (γ ≥ 0 при EWMA преимущества M); C7 QLIKE-бустинг (200 деревьев, глубина 2, шаг 0,05, лист ≥ 500, набор Z C4, раз в год); "
           "C8 два режима (π = σ(a + b·z_mkt_level)); C9 три эксперта (softmax); C10 наклоны β_i = β + η_i; κ = λ = 250·s²; месячная переоценка по парам t⁺ < d0 (C7 — годовая); выбор — crit_panel")


def declare() -> dict:
    """Один раз: время объявления листа C-b до первого файла результатов; повтор не сдвигает."""
    cp = json.loads(DECL.CP.read_text())
    if "C_b_declared_at" not in cp:
        DECL.update(C_b_declared_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), C_b_rule=CB_RULE, C_b_configs=CFGS)
        cp = json.loads(DECL.CP.read_text())
    return cp


def keys() -> dict:
    code = CC.c_code() + sorted(CC.CODE_DIR.glob("v4_cb_*.py"))
    return CC.keys("C", extra_code=code)


def forecasts(Zl: pd.DataFrame, P: pd.DataFrame, key: str) -> tuple:
    fc, wt = {}, {}
    for cfg in CFGS:
        r = CC.cached("C", cfg, key, lambda cfg=cfg: dict(zip(("", "_weights"), R.forecast(cfg, P, Zl))))
        fc[cfg], wt[cfg] = r[""], r["_weights"]
    return fc, wt


def loto(best: str, P: pd.DataFrame, Zl: pd.DataFrame, fc: dict, key: str) -> pd.DataFrame:
    """LOTO: панель на 16, поперечные Z по 16, отклонения исключённой бумаги = 0 (`dev` в таблице весов)."""
    def build():
        parts, zero = [], []
        for t in IO.M16:
            out, w = R.forecast(best, P, ZF.recross(Zl, [x for x in IO.T17 if x != t]), exclude=t)
            parts.append(out[out["ticker"] == t][["ticker", "date", "t_plus", "rv", "h", "F"]])
            wt = w[w["ticker"] == t]
            zero.append({"ticker": t, "ok": bool((wt["dev"] == 0).all() and (wt["n_pairs"] == 0).all() and wt["excluded"].all())})
        return {"": pd.concat(parts, ignore_index=True), "_zero": pd.DataFrame(zero)}
    r = CC.cached("C", "C_loto_" + best, key, build)
    return LO.loto_table(best, r[""], fc[best], dict(zip(r["_zero"]["ticker"], r["_zero"]["ok"])))


def trials_rows(key: str) -> list:
    return ["Ж-222:%s:%s,Ж-222,harj_hybrid_v4,conditional_mixture_Cb,%s,trial" % (c, hashlib.sha256((c + "|" + key).encode()).hexdigest()[:12],
                                                                              '"' + json.dumps({"cfg": c, "kind": R.SPECS[c][0], "feats": R.SPECS[c][1], "freq": R.SPECS[c][2], "kappa": 250, "lam": 250, "cache_key": key[:16]},
                                                                                               ensure_ascii=False).replace('"', '""') + '"') for c in CFGS]


def write_trials(key: str) -> None:
    TR.replace_rows(["Ж-222:%s:" % c for c in CFGS], trials_rows(key))


def stage_cb() -> dict:
    t0 = time.time()
    cp = declare()
    o = CA.stage_ca()                                                   # C1–C4, Z, B6 — из кэша и без изменений
    P, Zl, kb = o["P"], o["Zl"], keys()
    fcb, wtb = forecasts(Zl, P, kb["key"])
    fc = {**o["fc"], **fcb}
    diag = {"C4_diag_" + a: fc["C4"][["ticker", "date", "F_abl_" + a]].rename(columns={"F_abl_" + a: "F"}) for a in CA.ABL}
    b6 = pd.read_parquet(CC.OUT / "B_B6.parquet")
    k = D.eval_frame(P, {"B6": b6, **{c: fc[c] for c in ALLC}, **diag})
    cols_main = DECL.CTRL + ["B6"] + ALLC
    cols = cols_main + CA.DIAG
    c, crit = S.common(k, cols), S.crit_panel(k, cols)
    pt = ST.per_ticker(c, cols)
    cs = crit.set_index("series")["crit_panel"]
    best = min(ALLC, key=lambda x: float(cs[x]))
    lo = o["loto"] if best == "C1" else pd.concat([o["loto"], loto(best, P, Zl, fc, kb["key"])], ignore_index=True)
    r = {**o, "ca": o, "k": k, "c": c, "crit": crit, "pt": pt, "cols": cols, "cols_main": cols_main, "fc": fc, "wt_b": wtb, "kb": kb, "cp": cp, "best": best, "loto": lo, "t0": t0,
         "comp": ST.comparison(c, cols, crit, pt), "tests": CS.tests(c, ALLC, cp["N_declared_before_stats"], cp["N_wide"]), "mcs": ST.mcs(c, cols_main), "sber": ST.sber(c, cols_main, BS.sber_ref()),
         "years": Y.years_table(k, cols_main), "best_ca": o["best"], "note": "входы sha256=%s…, код sha256=%s…" % (kb["inputs"][:12], kb["code"][:12])}
    write_all(r)
    if cp.get("C_b_cache_key") != kb["key"]:
        DECL.update(C_b_runtime_sec=round(time.time() - t0, 1), C_b_processes=1)
    DECL.update(C_b_configs_spent=len(CFGS), used_after_C_b=int(cp["used_after_C_a"]) + len(CFGS), C_best_overall=best, C_b_cache_key=kb["key"], C_stop_go="go" if cs[best] <= cp["go_threshold_B"] else "stop")
    return r


def write_all(r: dict) -> None:
    n, note, thr = r["cp"], r["note"], r["cp"]["go_threshold_B"]
    IO.write_csv(r["comp"], "C_COMPARISON.csv", "лист C-a + C-b: crit_panel = hp_stats_sel.crit_panel (M16 × 2017–2021, Ж-217); метрики — среднее по M16 на общих датах (формулы hp_hyb_out); порог go = %.6f; "
                 "C4_diag_* — диагностика обнулением без переоценки, не конфигурации; %s" % (thr, note))
    IO.write_csv(r["pt"], "C_PER_TICKER.csv", "17 бумаг (SBER — отдельная диагностика, в выбор не входит); общие даты C1–C10; " + note)
    IO.write_csv(r["tests"], "C_TESTS.csv", "DM НУ лаг 5 и блочный бутстрап дат; Δ = QLIKE(C) − QLIKE(база); Холм: N=%d (p_holm), N_wide=%d (p_holm_wide), семейство M16 × {2022_2025, all} × C1–C10; %s" % (n["N_declared_before_stats"], n["N_wide"], note))
    IO.write_csv(r["mcs"], "C_MCS.csv", "MCS 90 %, T_max, блочный бутстрап дат (блок 20, 5000, seed 42), панельный QLIKE M16, ряды K1–K3, D2, B6, C1–C10; " + note)
    wt = pd.concat([r["ca"]["wt"][c] for c in CA.BASE] + [r["wt_b"][c] for c in CFGS], ignore_index=True)
    IO.write_csv(wt, "C_WEIGHTS.csv", "mean_w — средний вес на M за период; PANEL: коэффициенты (b_<признак> на стандартизованных z), alpha, converged (C7 — не применимо, True), nit, grad; dev — наибольшее отклонение параметров бумаги от панельных; " + note)
    IO.write_csv(r["sber"], "C_SBER.csv", "SBER отдельно, в выбор не входит; ориентир 0,2081 — потолок; " + note)
    IO.write_csv(r["loto"], "C_LOTO.csv", "LOTO: C1 (лучшая C-a) и лучшая из C1–C10 (если не C1): панель на 16 бумагах, поперечные Z по 16, параметры исключённой = панельные; " + note)
    IO.write_csv(r["years"], "C_YEARS.csv", "crit_panel по годам цели: crit_part = pen_Y/4 (2018–2021) суммируется в crit_panel; ann_qlike — средний QLIKE по M16 за год цели; 2022–2026 описательно; " + note)
    (IO.REP / "C_REPORT.md").write_text(RP.build(r))
    write_trials(r["kb"]["key"])
