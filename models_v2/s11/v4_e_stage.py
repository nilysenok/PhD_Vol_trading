"""V4-E: стадия `E` (G22–G25). Порядок: объявление (CHECKPOINT, до первого файла результатов) -> прогнозы E1–E6 (кэш, ключ = хэш входов и кода) -> общие таблицы -> LOTO лучшей E -> годы, шоки, SBER -> выходы -> реестр.
Базовые модели не переобучаются (K1, K2, K3, D2, B2, C8 — только чтение); гиперпараметры фиксированы объявлением; выбор — только `crit_panel` (M16 × цели 2017–2021)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time

import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import (v4_b_data as D, v4_b_decl as DECL, v4_b_loto as LO, v4_b_stage as BS, v4_b_stats as ST, v4_cb_years as Y, v4_e_cache as EC, v4_e_go as GO, v4_e_report as RP, v4_e_run as R,
                           v4_e_stats as ES, v4_io as IO, v4_trials as TR)

CFGS = R.CFGS
E_RULE = ("этап E: F = F0·exp(b_t); E1 F0 = H; E2–E5 F0 = G = B2; E6 = H·exp(b + w·ℓ) + дневная EWMA; b — по парам t⁺ ≤ d (обновление ежедневно), до 60 закрытых пар бумаги b = 0; b ∈ [−0,5; 0,5]; "
          "κ = 250; E1/E2 hl 60, E3 hl 20, E4 hl 20 × hl 120, E5 Калман (q/r раз в год), E6 месячная (b, w) hl 250 + EWMA hl 20; панель — 17 бумаг, в LOTO — 16; выбор — crit_panel")


def declare() -> dict:
    """Один раз: время объявления этапа E до первого файла результатов; повтор не сдвигает."""
    cp = json.loads(DECL.CP.read_text())
    if "E_declared_at" not in cp:
        DECL.update(E_declared_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), E_rule=E_RULE, E_configs=CFGS)
        cp = json.loads(DECL.CP.read_text())
    return cp


def forecasts(P: pd.DataFrame, G: pd.DataFrame, key: str) -> tuple:
    fc, meta = {}, {}
    for cfg in CFGS:
        def build(cfg=cfg):
            out, m = R.forecast(cfg, P, G)
            return {"": out, "_meta": pd.DataFrame({"json": [json.dumps(m, default=str, ensure_ascii=False)]})}
        r = EC.cached(cfg, key, build)
        fc[cfg], meta[cfg] = r[""], json.loads(r["_meta"]["json"].iloc[0])
    return fc, meta


def loto(best: str, P: pd.DataFrame, G: pd.DataFrame, fc: dict, key: str) -> pd.DataFrame:
    """LOTO: панельные величины (ĉ_panel, ρ_panel, месячная панель E6) — по 16 бумагам без исключённой; бумажная часть — по собственным закрытым парам исключённой (как у любой бумаги)."""
    def build():
        parts, zero = [], []
        for t in IO.M16:
            out, m = R.forecast(best, P, G, exclude=t)
            parts.append(out[out["ticker"] == t][["ticker", "date", "t_plus", "rv", "h", "F"]])
            zero.append({"ticker": t, "ok": bool(t not in m["panel"] and len(m["panel"]) == len(IO.T17) - 1)})
        return {"": pd.concat(parts, ignore_index=True), "_zero": pd.DataFrame(zero)}
    r = EC.cached("E_loto_" + best, key, build)
    return LO.loto_table(best, r[""], fc[best], dict(zip(r["_zero"]["ticker"], r["_zero"]["ok"])))


def trials_rows(key: str) -> list:
    spec = lambda c: json.dumps({"cfg": c, "F0": R.SPEC[c][0], "mode": R.SPEC[c][1], "hl_own": R.SPEC[c][2], "hl_panel": R.SPEC[c][3], "kappa": 250, "cache_key": key[:16]}, ensure_ascii=False)
    return ["Ж-222:%s:%s,Ж-222,harj_hybrid_v4,bias_calibration_E,\"%s\",trial" % (c, hashlib.sha256((c + "|" + key).encode()).hexdigest()[:12], spec(c).replace('"', '""')) for c in CFGS]


def write_trials(key: str) -> None:
    TR.replace_rows(["Ж-222:%s:" % c for c in CFGS], trials_rows(key))       # только E1–E6; строки других стадий сохраняются


def stage_e() -> dict:
    t0 = time.time()
    cp = declare()
    P = D.panel()
    G, ke = pd.read_parquet(EC.OUT / "B_B2.parquet"), EC.keys()
    fc, meta = forecasts(P, G, ke["key"])
    ex = {"B2": G, "C8": pd.read_parquet(EC.OUT / "C8.parquet"), "B6": pd.read_parquet(EC.OUT / "B_B6.parquet")}
    k = D.eval_frame(P, {**ex, **fc})
    cols_main = list(DECL.CTRL) + ["B2", "C8"] + CFGS
    c, crit = S.common(k, cols_main), S.crit_panel(k, cols_main)
    pt = ST.per_ticker(c, cols_main)
    cs = crit.set_index("series")["crit_panel"]
    best = min(CFGS, key=lambda x: float(cs[x]))
    o = {"P": P, "cp": cp, "k": k, "c": c, "crit": crit, "pt": pt, "cols_main": cols_main, "fc": fc, "meta": meta, "best": best, "t0": t0, "ke": ke,
         "note": "входы sha256=%s…, код sha256=%s…" % (ke["inputs"][:12], ke["code"][:12])}
    o["comp"], o["tests"], o["mcs"] = ST.comparison(c, cols_main, crit, pt), ES.tests(c, CFGS, cp["N_declared_before_stats"], cp["N_wide"]), ST.mcs(c, cols_main)
    o["sber"], o["years"] = ST.sber(c, cols_main, BS.sber_ref()), Y.years_table(k, cols_main)
    o["go"] = GO.rules(k, CFGS, cs, cp["go_threshold_B"])
    o["bias"], o["shocks"], o["loto"] = ES.bias_table(fc), ES.shocks(fc), loto(best, P, G, fc, ke["key"])
    write_all(o)
    if cp.get("E_cache_key") != ke["key"]:
        DECL.update(E_runtime_sec=round(time.time() - t0, 1), E_processes=1)
    DECL.update(E_configs_spent=len(CFGS), used_after_E=int(cp["used_after_C_b"]) + len(CFGS), E_best=best, E_cache_key=ke["key"], E_stop_go="go" if o["go"]["cfg"][best]["go"] else "stop", E_go_rules=o["go"])
    return o


def write_all(o: dict) -> None:
    n, note, thr = o["cp"], o["note"], o["cp"]["go_threshold_B"]
    IO.write_csv(o["comp"], "E_COMPARISON.csv", "этап E: crit_panel = hp_stats_sel.crit_panel (M16 × 2017–2021, Ж-217); метрики — среднее по M16 на общих датах (hp_hyb_out); порог go = %.6f; B2 и C8 — прежние прогнозы; %s" % (thr, note))
    IO.write_csv(o["pt"], "E_PER_TICKER.csv", "17 бумаг (SBER — отдельная диагностика, в выбор не входит); общие даты; " + note)
    IO.write_csv(o["tests"], "E_TESTS.csv", "DM НУ лаг 5 и блочный бутстрап дат; Δ = QLIKE(E) − QLIKE(база); Холм: N=%d (p_holm), N_wide=%d (p_holm_wide), семейство M16 × {2022_2025, all} × E1–E6; %s" % (n["N_declared_before_stats"], n["N_wide"], note))
    IO.write_csv(o["mcs"], "E_MCS.csv", "MCS 90 %, T_max, блочный бутстрап дат (блок 20, 5000, seed 42), панельный QLIKE M16, ряды K1–K3, D2, B2, C8, E1–E6; " + note)
    IO.write_csv(o["years"], "E_YEARS.csv", "crit_panel по годам цели: crit_part = pen_Y/4 (2018–2021) суммируется в crit_panel; ann_qlike — средний QLIKE по M16 за год цели; 2022–2026 описательно; " + note)
    IO.write_csv(o["sber"], "E_SBER.csv", "SBER отдельно, в выбор не входит; ориентир 0,2081 — потолок; " + note)
    IO.write_csv(o["loto"], "E_LOTO.csv", "LOTO лучшей E: панельные величины по 16 бумагам без исключённой, бумажная часть — по её собственным закрытым парам; " + note)
    IO.write_csv(pd.concat([o["bias"], o["shocks"].assign(ticker="PANEL-окно")], ignore_index=True), "E_BIAS.csv", "b по месяцам × бумагам (среднее по дням) и среднее по панели; строки PANEL-окно — шоки (описательно); " + note)
    (IO.REP / "E_REPORT.md").write_text(RP.build(o))
    write_trials(o["ke"]["key"])
