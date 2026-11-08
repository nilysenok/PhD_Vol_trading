"""V4-C: стадия `Ca` (лист C-a, G15–G18). Порядок: объявление (CHECKPOINT) -> Z по 17 бумагам -> прогнозы C1–C4 (кэш) -> статистика -> LOTO лучшей -> выходы -> реестр.
Базовые модели не переобучаются; гиперпараметры (κ, λ, полураспад, частота) фиксированы; выбор — только `crit_panel` (M16 × цели 2017–2021)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import (v4_b_data as D, v4_b_decl as DECL, v4_b_loto as LO, v4_b_stage as BS, v4_b_stats as ST, v4_c_cache as CC, v4_c_report as RP, v4_c_run as CR, v4_c_stats as CS, v4_io as IO,
                           v4_trials as TR, v4_z_build as ZB, v4_z_feat as ZF)

BASE = ["C1", "C2", "C3", "C4"]
ABL = {"no_har_surprise": ["sur_mkt"], "no_intraday_shape": ["lasth", "herf"]}
DIAG = ["C4_diag_" + a for a in ABL]
C_RULE = ("лист C-a: C1 = z_div; C2 = + z_perf; C3 = + z_mkt; C4 = + z_new (HAR-surprise, Intraday Shape); F = H·exp(w(Z)·ln(M/H)), w = σ(α + δ_i + βᵀz); κ = λ = 250; переоценка раз в месяц по парам t⁺ < d0; "
          "выбор — crit_panel; stop/go этапа C — после C-b")


def declare() -> dict:
    """Один раз: время объявления листа до первого файла результатов (Z, C*); повтор не сдвигает."""
    cp = json.loads(DECL.CP.read_text())
    if "C_declared_at" not in cp:
        DECL.update(C_declared_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), C_rule=C_RULE, C_a_configs=BASE)
        cp = json.loads(DECL.CP.read_text())
    return cp


def archive_b_metrics() -> None:
    """Прежние B_COMPARISON/B_PER_TICKER (rmse_ln, mae_ln, r2_lvl) уже в archive_pre_metrics/ (копия до пересчёта); здесь — SHA-256 прежних и текущих файлов."""
    rows = []
    for name in ("B_COMPARISON.csv", "B_PER_TICKER.csv"):
        old = IO.REP / "archive_pre_metrics" / name
        rows += [{"file": "archive_pre_metrics/" + name, "role": "до пересчёта метрик (rmse_ln, mae_ln, r2_lvl)", "sha256": IO.sha256(old),
                  "reason": "RMSE/MAE/R² приведены к единому определению проекта (hp_hyb_out), Д7; прогнозы B не менялись"},
                 {"file": name, "role": "после пересчёта (rmse, mae, r2_rv, r2_ln)", "sha256": IO.sha256(IO.REP / name), "reason": "то же"}]
    IO.write_csv(pd.DataFrame(rows), "archive_pre_metrics/HASHES.csv", "SHA-256 прежних и пересчитанных таблиц B (лист C-a, 08.11)")


def forecasts(Zl: pd.DataFrame, P: pd.DataFrame, key: str) -> tuple:
    fc, wt = {}, {}
    for cfg in BASE:
        abl = ABL if cfg == "C4" else None
        r = CC.cached("C", cfg, key, lambda cfg=cfg, abl=abl: dict(zip(("", "_weights"), CR.forecast(cfg, P, Zl, ablate=abl))))
        fc[cfg], wt[cfg] = r[""], r["_weights"]
    return fc, wt


def loto(best: str, P: pd.DataFrame, Zl: pd.DataFrame, fc: dict, key: str) -> pd.DataFrame:
    def build():
        parts, zero = [], []
        for t in IO.M16:
            Z16 = ZF.recross(Zl, [x for x in IO.T17 if x != t])          # поперечные признаки по 16 (исключённая — относительно 16)
            out, w = CR.forecast(best, P, Z16, exclude=t)
            parts.append(out[out["ticker"] == t][["ticker", "date", "t_plus", "rv", "h", "F"]])
            wt = w[(w["ticker"] == t)]
            zero.append({"ticker": t, "ok": bool((wt["delta"] == 0).all() and (wt["n_pairs"] == 0).all() and wt["excluded"].all())})
        return {"": pd.concat(parts, ignore_index=True), "_zero": pd.DataFrame(zero)}
    r = CC.cached("C", "C_loto_" + best, key, build)
    return LO.loto_table(best, r[""], fc[best], dict(zip(r["_zero"]["ticker"], r["_zero"]["ok"])))


def trials_rows(key: str) -> list:
    rows = ["Ж-222:%s:%s,Ж-222,harj_hybrid_v4,conditional_mixture_Ca,%s,trial" % (c, hashlib.sha256((c + "|" + key).encode()).hexdigest()[:12],
                                                                              '"' + json.dumps({"cfg": c, "feats": ZF.cfg_feats(c), "kappa": 250, "lam": 250, "cache_key": key[:16]}, ensure_ascii=False).replace('"', '""') + '"') for c in BASE]
    return rows


def write_trials(key: str) -> None:
    f = IO.REP / "ALL_TRIALS.csv"
    keep = [x for x in f.read_text().splitlines()[1:] if not x.startswith("Ж-222:C")]
    f.write_text(TR.HEADER + "".join(x + "\n" for x in keep + trials_rows(key)))


def stage_ca() -> dict:
    t0 = time.time()
    cp = declare()
    P = D.panel()
    kz, kc = CC.keys("Z"), CC.keys("C")
    Zl = ZB.z_by_ticker(kz["key"])
    fc, wt = forecasts(Zl, P, kc["key"])
    b6 = pd.read_parquet(CC.OUT / "B_B6.parquet")
    diag = {"C4_diag_" + a: fc["C4"][["ticker", "date", "F_abl_" + a]].rename(columns={"F_abl_" + a: "F"}) for a in ABL}
    k = D.eval_frame(P, {"B6": b6, **{c: fc[c] for c in BASE}, **diag})
    cols_main = DECL.CTRL + ["B6"] + BASE
    cols = cols_main + DIAG
    c, crit = S.common(k, cols), S.crit_panel(k, cols)
    pt = ST.per_ticker(c, cols)
    o = {"P": P, "cp": cp, "fc": fc, "wt": wt, "Zl": Zl, "c": c, "crit": crit, "pt": pt, "cols": cols, "cols_main": cols_main, "kz": kz, "kc": kc, "t0": t0,
         "note": "входы sha256=%s…, код sha256=%s…" % (kc["inputs"][:12], kc["code"][:12])}
    o["best"] = min(BASE, key=lambda x: float(crit.set_index("series").loc[x, "crit_panel"]))
    o["comp"], o["tests"], o["mcs"] = ST.comparison(c, cols, crit, pt), CS.tests(c, BASE, cp["N_declared_before_stats"], cp["N_wide"]), ST.mcs(c, cols_main)
    o["sber"] = ST.sber(c, cols_main, BS.sber_ref())
    o["loto"] = loto(o["best"], P, Zl, fc, kc["key"])
    o["zrep"] = ZB.report(Zl, P, ZB.real_inputs(), kz)
    write_all(o)
    if cp.get("C_cache_key") != kc["key"]:
        DECL.update(C_runtime_sec=round(time.time() - t0, 1), C_processes=1)
    DECL.update(C_configs_spent=len(BASE), used_after_C_a=int(cp.get("B_configs_spent", 5)) + len(BASE), C_best=o["best"], C_cache_key=kc["key"], Z_cache_key=kz["key"])
    return o


def write_all(o: dict) -> None:
    n, note = o["cp"], o["note"]
    thr = n["go_threshold_B"]
    IO.write_csv(o["zrep"], "Z_REPORT.csv", "покрытие, источник и проверка сдвига (Д1: корреляция признака с ln RV цели при сдвигах −2…+1, цели ≤ 2021) каждого признака Z; " + note)
    IO.write_csv(o["comp"], "C_COMPARISON.csv", "crit_panel = hp_stats_sel.crit_panel (M16 × 2017–2021, Ж-217); метрики — среднее по M16 на общих датах (RMSE/MAE/R² — формулы hp_hyb_out); порог go = %.6f; "
                 "C4_diag_* — диагностика обнулением коэффициентов без переоценки, не конфигурации; %s" % (thr, note))
    IO.write_csv(o["pt"], "C_PER_TICKER.csv", "17 бумаг (SBER — отдельная диагностика, в выбор не входит); общие даты; " + note)
    IO.write_csv(o["tests"], "C_TESTS.csv", "DM НУ лаг 5 и блочный бутстрап дат; Δ = QLIKE(C) − QLIKE(база); Холм: N=%d (p_holm), N_wide=%d (p_holm_wide), семейство M16 × {2022_2025, all}; %s" % (n["N_declared_before_stats"], n["N_wide"], note))
    IO.write_csv(o["mcs"], "C_MCS.csv", "MCS 90 %, T_max, блочный бутстрап дат (блок 20, 5000, seed 42), панельный QLIKE M16; " + note)
    IO.write_csv(pd.concat([o["wt"][c] for c in BASE], ignore_index=True), "C_WEIGHTS.csv", "mean_w — средний вес на M за месяц; PANEL: коэффициенты β (b_<признак>, на стандартизованных z), alpha, converged, nit, grad; ticker: alpha_i = α + δ_i; " + note)
    IO.write_csv(o["sber"], "C_SBER.csv", "SBER отдельно, в выбор не входит; ориентир 0,2081 — потолок; " + note)
    IO.write_csv(o["loto"], "C_LOTO.csv", "LOTO лучшей C-a: панель на 16 бумагах, поперечные Z по 16, δ = 0 у исключённой; " + note)
    (IO.REP / "C_REPORT.md").write_text(RP.build(o))
    write_trials(o["kc"]["key"])
    archive_b_metrics()
