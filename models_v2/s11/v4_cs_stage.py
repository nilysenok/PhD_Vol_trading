"""V4-CS: стадия `CS` (G27–G28). Порядок: объявление (CHECKPOINT, до первого файла результатов) -> Z (из кэша листа C-a) -> прогнозы CS1, CS2 (кэш, ключ = хэш входов и кода, Д5) -> общие таблицы (контроли, C1, C4, C8, CS1, CS2) ->
LOTO лучшей CS -> годы -> выходы -> реестр. Базовые модели не переобучаются (K1–K3, D2, C1, C4, C8 — только чтение); гиперпараметры фиксированы; выбор — только `crit_panel` (M16 × цели 2017–2021).
Диагностика холодного старта и отчёт (`CS_DIAG.csv`, `CS_REPORT.md`) — вне путей прогноза: `v4_dx_cold`, `v4_dx_cs_report`, вызываются из `run_v4`."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time

import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import (v4_b_data as D, v4_b_decl as DECL, v4_b_loto as LO, v4_b_stage as BS, v4_b_stats as ST, v4_c_cache as CC, v4_cs_run as R, v4_cs_stats as CST, v4_cb_years as Y, v4_e_go as GO, v4_io as IO,
                           v4_trials as TR, v4_z_build as ZB, v4_z_feat as ZF)

CFGS = R.CFGS
REFS = ["C1", "C4", "C8"]
CS_RULE = ("лист DX-CS, Cold Start (Ж-225 п. 3): база — C8 (два режима, med_lrv), всё прочее как в C8 (κ = λ = 250·s², месячная переоценка по парам t⁺ < d0, до 60 закрытых пар — приор w = 0,5); "
           "CS1: λ_prior(n) = 250·250/(250 + n), n — закрытые пары панели; CS2: κ_i(n_i) = 250·250/(250 + n_i), n_i — закрытые пары бумаги i; «go» — три условия (Ж-225 п. 10); выбор — crit_panel")


def declare() -> dict:
    """Один раз: время объявления листа до первого файла результатов; повтор не сдвигает."""
    cp = json.loads(DECL.CP.read_text())
    if "CS_declared_at" not in cp:
        DECL.update(CS_declared_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), CS_rule=CS_RULE, CS_configs=CFGS)
        cp = json.loads(DECL.CP.read_text())
    return cp


def keys() -> dict:
    code = CC.c_code() + sorted(CC.CODE_DIR.glob("v4_cb_*.py")) + sorted(CC.CODE_DIR.glob("v4_cs_*.py"))
    return CC.keys("C", extra_code=code)


def forecasts(Zl: pd.DataFrame, P: pd.DataFrame, key: str) -> tuple:
    fc, wt = {}, {}
    for cfg in CFGS:
        r = CC.cached("CS", cfg, key, lambda cfg=cfg: dict(zip(("", "_weights"), R.forecast(cfg, P, Zl))))
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
    r = CC.cached("CS", "CS_loto_" + best, key, build)
    return LO.loto_table(best, r[""], fc[best], dict(zip(r["_zero"]["ticker"], r["_zero"]["ok"])))


def trials_rows(key: str) -> list:
    spec = lambda c: json.dumps({"cfg": c, "base": "C8", "lam": "250*250/(250+n_panel)" if c == "CS1" else 250, "kappa": 250 if c == "CS1" else "250*250/(250+n_i)", "cache_key": key[:16]}, ensure_ascii=False)
    return ["Ж-222:%s:%s,Ж-222,harj_hybrid_v4,cold_start_CS,\"%s\",trial" % (c, hashlib.sha256((c + "|" + key).encode()).hexdigest()[:12], spec(c).replace('"', '""')) for c in CFGS]


def write_trials(key: str) -> None:
    TR.replace_rows(["Ж-222:%s:" % c for c in CFGS], trials_rows(key))        # только CS1, CS2; строки других стадий сохраняются


def stage_cs() -> dict:
    t0 = time.time()
    cp = declare()
    P, kc = D.panel(), keys()
    kz = CC.keys("Z")
    Zl = ZB.z_by_ticker(kz["key"])
    fc, wt = forecasts(Zl, P, kc["key"])
    ex = {c: pd.read_parquet(CC.OUT / (c + ".parquet")) for c in REFS}
    k = D.eval_frame(P, {**ex, **fc})
    cols = DECL.CTRL + REFS + CFGS
    c, crit = S.common(k, cols), S.crit_panel(k, cols)
    pt = ST.per_ticker(c, cols)
    cs = crit.set_index("series")["crit_panel"]
    best = min(CFGS, key=lambda x: float(cs[x]))
    o = {"P": P, "Zl": Zl, "cp": cp, "k": k, "c": c, "crit": crit, "pt": pt, "cols": cols, "fc": fc, "wt": wt, "kc": kc, "best": best, "t0": t0, "refs": REFS,
         "note": "входы sha256=%s…, код sha256=%s…" % (kc["inputs"][:12], kc["code"][:12])}
    o["comp"], o["tests"], o["mcs"] = ST.comparison(c, cols, crit, pt), CST.tests(c, CFGS, cp["N_declared_before_stats"], cp["N_wide"]), ST.mcs(c, cols)
    o["sber"], o["years"] = ST.sber(c, cols, BS.sber_ref()), Y.years_table(k, cols)
    o["go"] = GO.rules(k, ["k1"] + REFS + CFGS, cs, cp["go_threshold_B"])
    o["loto"] = loto(best, P, Zl, fc, kc["key"])
    write_all(o)
    if cp.get("CS_cache_key") != kc["key"]:
        DECL.update(CS_runtime_sec=round(time.time() - t0, 1), CS_processes=1)
    DECL.update(CS_configs_spent=len(CFGS), used_after_CS=int(cp["used_after_E"]) + len(CFGS), CS_best=best, CS_cache_key=kc["key"], CS_stop_go="go" if o["go"]["cfg"][best]["go"] else "stop", CS_go_rules=o["go"])
    return o


def write_all(o: dict) -> None:
    n, note, thr = o["cp"], o["note"], o["cp"]["go_threshold_B"]
    IO.write_csv(o["comp"], "CS_COMPARISON.csv", "Cold Start: crit_panel = hp_stats_sel.crit_panel (M16 × 2017–2021, Ж-217); метрики — среднее по M16 на общих датах (hp_hyb_out); порог go = %.6f; C1, C4, C8 — прежние прогнозы; %s" % (thr, note))
    IO.write_csv(o["pt"], "CS_PER_TICKER.csv", "17 бумаг (SBER — отдельная диагностика, в выбор не входит); общие даты; " + note)
    IO.write_csv(o["tests"], "CS_TESTS.csv", "DM НУ лаг 5 и блочный бутстрап дат; Δ = QLIKE(CS) − QLIKE(база); Холм: N=%d (p_holm), N_wide=%d (p_holm_wide), семейство M16 × {2022_2025, all} × CS1–CS2; %s" % (n["N_declared_before_stats"], n["N_wide"], note))
    IO.write_csv(o["mcs"], "CS_MCS.csv", "MCS 90 %, T_max, блочный бутстрап дат (блок 20, 5000, seed 42), панельный QLIKE M16, ряды K1–K3, D2, C1, C4, C8, CS1, CS2; " + note)
    IO.write_csv(o["years"], "CS_YEARS.csv", "crit_panel по годам цели: crit_part = pen_Y/4 (2018–2021) суммируется в crit_panel; ann_qlike — средний QLIKE по M16 за год цели (2017 и 2018 — отдельные строки); 2022–2026 описательно; " + note)
    IO.write_csv(o["sber"], "CS_SBER.csv", "SBER отдельно, в выбор не входит; ориентир 0,2081 — потолок; " + note)
    IO.write_csv(o["loto"], "CS_LOTO.csv", "LOTO лучшей CS: панель на 16 бумагах, поперечные Z по 16, параметры исключённой = панельные; " + note)
    IO.write_csv(pd.concat([o["wt"][c] for c in CFGS], ignore_index=True), "CS_WEIGHTS.csv", "mean_w — средний вес на M за месяц; PANEL: параметры режимов, lam, kappa_mean (сила штрафа в парах-эквивалентах); ticker: kappa_i; " + note)
    write_trials(o["kc"]["key"])
