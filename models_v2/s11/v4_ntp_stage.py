"""V4-NTP: стадия `NTP` (G29–G31). Порядок: объявление (CHECKPOINT, до первого файла результатов) -> Z (кэш листа C-a) -> прогнозы NTP1–NTP4 (кэш, ключ = хэш входов и кода, Д5) -> общие таблицы (контроли, C1, C4, C8, NTP1–NTP4) ->
LOTO лучшей NTP -> годы -> выходы -> реестр. Базовые модели не переобучаются (K1–K3, D2, C1, C4, C8 — только чтение), гиперпараметры фиксированы; выбор — только `crit_panel` (M16 × цели 2017–2021).
Состояния, отчёт и проверки (`NTP_STATES.csv`, `NTP_REPORT.md`) — вне путей прогноза: `v4_ntpd_*` (тест изоляции), вызываются из `run_v4`."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import (v4_b_data as D, v4_b_decl as DECL, v4_b_loto as LO, v4_b_stage as BS, v4_b_stats as ST, v4_c_cache as CC, v4_cb_years as Y, v4_cs_stats as CST, v4_e_go as GO, v4_io as IO,
                           v4_ntp_run as R, v4_trials as TR, v4_z_build as ZB, v4_z_feat as ZF)

CFGS = R.CFGS
REFS = ["C1", "C4", "C8"]
NTP_RULE = ("лист NTP (Ж-225 п. 4): F = M^(1−π)·C8^π, M = √(K2·K3), C8 — сохранённый прогноз; всё по закрытым парам (t⁺ ≤ d); D′ = QLIKE(C8) − QLIKE(M); до 60 закрытых пар бумаги π = 0,5; "
            "NTP1 π = σ(−m/s), EWMA 60, κ = 250; NTP2 fixed share α = 0,02, η = 1/медиана QLIKE панели; NTP3 гейт σ(a_i + b·m/s + c·|ℓ| + e·z_mkt), κ = λ = 250·s²; NTP4 — NTP1 с шагом ≤ 0,02; «go» — три условия (Ж-225 п. 10); выбор — crit_panel")


def declare() -> dict:
    """Один раз: время объявления листа до первого файла результатов; повтор не сдвигает."""
    cp = json.loads(DECL.CP.read_text())
    if "NTP_declared_at" not in cp:
        DECL.update(NTP_declared_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), NTP_rule=NTP_RULE, NTP_configs=CFGS)
        cp = json.loads(DECL.CP.read_text())
    return cp


def keys() -> dict:
    code = sorted(CC.CODE_DIR.glob("v4_ntp_*.py")) + CC.c_code()
    return CC.keys("NTP", extra_in=CC.z_inputs() + [CC.OUT / "C8.parquet"], extra_code=code)


def forecasts(Zl: pd.DataFrame, P: pd.DataFrame, G: pd.DataFrame, key: str) -> tuple:
    fc, wt = {}, {}
    for cfg in CFGS:
        r = CC.cached("NTP", cfg, key, lambda cfg=cfg: dict(zip(("", "_weights"), R.forecast(cfg, P, G, Zl))))
        fc[cfg], wt[cfg] = r[""], r["_weights"]
    return fc, wt


def zero_ok(best: str, P, G, Z16, t: str, w: pd.DataFrame) -> bool:
    """Исключённая бумага не входит в панельную величину: NTP3 — δ = 0 и метка excluded в каждом месяце; NTP1/2/4 — подмена RV исключённой не меняет панельную величину дня."""
    if best == "NTP3":
        wt = w[w["ticker"] == t]
        return bool((wt["delta"] == 0).all() and wt["excluded"].all())
    Q = P.copy()
    Q.loc[Q["ticker"] == t, "rv"] *= 5.0
    a, b = (R.forecast(best, X, G, Z16, exclude=t)[0]["pan"].to_numpy() for X in (P, Q))
    return bool(np.array_equal(a, b, equal_nan=True))


def loto(best: str, P, G, Zl, fc: dict, key: str) -> pd.DataFrame:
    """LOTO: панельная величина (NTP3 — и оценка гейта) — по 16 бумагам, поперечные Z по 16, исключённая получает панельные параметры (δ = 0)."""
    def build():
        parts, zero = [], []
        for t in IO.M16:
            Z16 = ZF.recross(Zl, [x for x in IO.T17 if x != t])
            out, w = R.forecast(best, P, G, Z16, exclude=t)
            parts.append(out[out["ticker"] == t][["ticker", "date", "t_plus", "rv", "h", "F"]])
            zero.append({"ticker": t, "ok": zero_ok(best, P, G, Z16, t, w)})
        return {"": pd.concat(parts, ignore_index=True), "_zero": pd.DataFrame(zero)}
    r = CC.cached("NTP", "NTP_loto_" + best, key, build)
    return LO.loto_table(best, r[""], fc[best], dict(zip(r["_zero"]["ticker"], r["_zero"]["ok"])))


def trials_rows(key: str) -> list:
    spec = lambda c: json.dumps({"cfg": c, "base": "M=sqrt(K2*K3) vs C8", "hl": 60, "kappa": 250, "alpha": 0.02 if c == "NTP2" else None, "step": 0.02 if c == "NTP4" else None, "cache_key": key[:16]}, ensure_ascii=False)
    return ["Ж-222:%s:%s,Ж-222,harj_hybrid_v4,negative_transfer_NTP,\"%s\",trial" % (c, hashlib.sha256((c + "|" + key).encode()).hexdigest()[:12], spec(c).replace('"', '""')) for c in CFGS]


def write_trials(key: str) -> None:
    TR.replace_rows(["Ж-222:%s:" % c for c in CFGS], trials_rows(key))        # только NTP1–NTP4; строки других стадий сохраняются


def stage_ntp() -> dict:
    t0 = time.time()
    cp = declare()
    P, kc = D.panel(), keys()
    Zl = ZB.z_by_ticker(CC.keys("Z")["key"])
    ex = {c: pd.read_parquet(CC.OUT / (c + ".parquet")) for c in REFS}
    fc, wt = forecasts(Zl, P, ex["C8"], kc["key"])
    k = D.eval_frame(P, {**ex, **fc})
    cols = DECL.CTRL + REFS + CFGS
    c, crit = S.common(k, cols), S.crit_panel(k, cols)
    pt, cs = ST.per_ticker(c, cols), crit.set_index("series")["crit_panel"]
    best = min(CFGS, key=lambda x: float(cs[x]))
    o = {"P": P, "G": ex["C8"], "Zl": Zl, "cp": cp, "k": k, "c": c, "crit": crit, "pt": pt, "cols": cols, "fc": fc, "wt": wt, "kc": kc, "best": best, "t0": t0, "refs": REFS,
         "note": "входы sha256=%s…, код sha256=%s…" % (kc["inputs"][:12], kc["code"][:12])}
    o["comp"], o["tests"] = ST.comparison(c, cols, crit, pt), CST.tests(c, CFGS, cp["N_declared_before_stats"], cp["N_wide"])
    o["sber"], o["years"] = ST.sber(c, cols, BS.sber_ref()), Y.years_table(k, cols)
    o["go"] = GO.rules(k, ["k1"] + REFS + CFGS, cs, cp["go_threshold_B"])
    o["loto"] = loto(best, P, ex["C8"], Zl, fc, kc["key"])
    write_all(o)
    if cp.get("NTP_cache_key") != kc["key"]:
        DECL.update(NTP_runtime_sec=round(time.time() - t0, 1), NTP_processes=1)
    DECL.update(NTP_configs_spent=len(CFGS), used_after_NTP=int(cp["used_after_CS"]) + len(CFGS), NTP_best=best, NTP_cache_key=kc["key"], NTP_stop_go="go" if o["go"]["cfg"][best]["go"] else "stop", NTP_go_rules=o["go"])
    return o


def write_all(o: dict) -> None:
    n, note, thr = o["cp"], o["note"], o["cp"]["go_threshold_B"]
    IO.write_csv(o["comp"], "NTP_COMPARISON.csv", "NTP (Ж-225): crit_panel = hp_stats_sel.crit_panel (M16 × 2017–2021, Ж-217); метрики — среднее по M16 на общих датах (hp_hyb_out); порог go = %.6f; C1, C4, C8 — прежние прогнозы; %s" % (thr, note))
    IO.write_csv(o["pt"], "NTP_PER_TICKER.csv", "17 бумаг (SBER — отдельная диагностика, в выбор не входит); общие даты; " + note)
    IO.write_csv(o["tests"], "NTP_TESTS.csv", "DM НУ лаг 5 и блочный бутстрап дат; Δ = QLIKE(NTP) − QLIKE(база); Холм: N=%d (p_holm), N_wide=%d (p_holm_wide), семейство M16 × {2022_2025, all} × NTP1–NTP4; %s" % (n["N_declared_before_stats"], n["N_wide"], note))
    IO.write_csv(o["years"], "NTP_YEARS.csv", "crit_panel по годам цели: crit_part = pen_Y/4 (2018–2021) суммируется в crit_panel; ann_qlike — средний QLIKE по M16 за год цели (2017 и 2018 — отдельные строки); 2022–2026 описательно; " + note)
    IO.write_csv(o["sber"], "NTP_SBER.csv", "SBER отдельно, в выбор не входит; ориентир 0,2081 — потолок; " + note)
    IO.write_csv(o["loto"], "NTP_LOTO.csv", "LOTO лучшей NTP: панельные величины (NTP3 — и оценка) по 16 бумагам, поперечные Z по 16, параметры исключённой = панельные; " + note)
    IO.write_csv(pd.concat([o["wt"][c] for c in CFGS], ignore_index=True), "NTP_WEIGHTS.csv", "mean_pi — средний π (доверие гибриду C8) за месяц; PANEL — среднее по панели; NTP3: alpha, b_*, delta, converged; " + note)
    write_trials(o["kc"]["key"])
