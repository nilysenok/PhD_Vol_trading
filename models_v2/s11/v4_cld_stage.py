"""V4-CLD: стадия `CLD` (G32–G34). Порядок: объявление (CHECKPOINT, до первого файла результатов) -> Z (кэш листа C-a) -> BRK -> прогнозы CLD1–CLD4 (кэш, ключ = хэш входов и кода, Д5) -> общие таблицы (контроли, C1, C4, C8, NTP1, CLD1–CLD4) ->
LOTO лучшей CLD -> годы -> выходы -> реестр. Базовые модели не переобучаются (K1–K3, D2, C1, C4, C8, NTP1 — только чтение); выбор — только `crit_panel` (M16 × цели 2017–2021).
Предсказуемость D, отчёт и проверки — вне путей прогноза: `v4_cldd_*` (тест изоляции), вызываются из `run_v4`."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time

import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import (v4_b_data as D, v4_b_decl as DECL, v4_b_loto as LO, v4_b_stage as BS, v4_b_stats as ST, v4_c_cache as CC, v4_cb_years as Y, v4_cld_brk as BK, v4_cld_run as R,
                           v4_cld_stats as CST, v4_e_go as GO, v4_io as IO, v4_trials as TR, v4_z_build as ZB, v4_z_feat as ZF)

CFGS = R.CFGS
REFS = ["C1", "C4", "C8", "NTP1", "NTP3"]          # NTP3 — только в таблицах сравнения (прямое обучение), в DM-компараторах — по заданию без него
CLD_RULE = ("лист CLD (Ж-225 п. 5): F = H·exp(w·ℓ), ℓ = ln(M/H), M = √(K2·K3), H = K1; D = QLIKE(H) − QLIKE(M); D̂ = E[D|z_d] — регрессия раз в месяц по закрытым парам 17 бумаг (t⁺ < d0), расширяющееся окно; до 60 пар w = 0,5; "
            "CLD1 гребень λ = 250, w = σ(D̂/ŝ); CLD2 та же регрессия, w = clip(0,5 + D̂/(2c)), c = p90 |D|; CLD3 Хьюбер δ = 1,345·ŝ; CLD4 аддитивная (кубический B-сплайн, 4 ст. св., гребень 250); "
            "z = ℓ, |ℓ|, lerr_h, lerr_m, ln RV, медиана ln RV, Δ ln RV, Херфиндаль, доля последнего часа, HAR-surprise рынка, время после излома (CUSUM h = 5, k = 0,5); «go» — три условия (Ж-225 п. 10); выбор — crit_panel")


def declare() -> dict:
    """Один раз: время объявления листа до первого файла результатов; повтор не сдвигает."""
    cp = json.loads(DECL.CP.read_text())
    if "CLD_declared_at" not in cp:
        DECL.update(CLD_declared_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), CLD_rule=CLD_RULE, CLD_configs=CFGS)
        cp = json.loads(DECL.CP.read_text())
    return cp


def keys() -> dict:
    code = sorted(CC.CODE_DIR.glob("v4_cld_*.py")) + CC.c_code()
    return CC.keys("CLD", extra_in=CC.z_inputs(), extra_code=code)


def forecasts(Zl: pd.DataFrame, P: pd.DataFrame, key: str) -> tuple:
    fc, wt, cf = {}, {}, {}
    for cfg in CFGS:
        r = CC.cached("CLD", cfg, key, lambda cfg=cfg: dict(zip(("", "_weights", "_coef"), R.forecast(cfg, P, Zl))))
        fc[cfg], wt[cfg], cf[cfg] = r[""], r["_weights"], r["_coef"]
    return fc, wt, cf


def brk_table(Zl: pd.DataFrame, key: str) -> pd.DataFrame:
    return CC.cached("CLD", "BRK", key, lambda: {"": BK.build_table(Zl)})[""]


def zero_ok(best: str, P, Z16, t: str, out: pd.DataFrame) -> bool:
    """Исключённая бумага не входит в оценку: подмена её RV (и D) не меняет прогнозы остальных бумаг при тех же поперечных z по 16."""
    Q = P.copy()
    Q.loc[Q["ticker"] == t, "rv"] *= 5.0
    o2 = R.forecast(best, Q, Z16, exclude=t)[0]
    a, b = (x[x["ticker"] != t].sort_values(["date", "ticker"])[["F", "w"]].to_numpy() for x in (out, o2))
    return bool((a == b).all())


def loto(best: str, P, Zl, fc: dict, key: str) -> pd.DataFrame:
    """LOTO: поперечные z (медианы, brk) — по 16 бумагам, исключённая бумага не входит в оценку; её прогноз — по панельной модели."""
    def build():
        parts, zero = [], []
        for t in IO.M16:
            Z16 = ZF.recross(Zl, [x for x in IO.T17 if x != t])
            out = R.forecast(best, P, Z16, exclude=t)[0]
            parts.append(out[out["ticker"] == t][["ticker", "date", "t_plus", "rv", "h", "F"]])
            zero.append({"ticker": t, "ok": zero_ok(best, P, Z16, t, out)})
        return {"": pd.concat(parts, ignore_index=True), "_zero": pd.DataFrame(zero)}
    r = CC.cached("CLD", "CLD_loto_" + best, key, build)
    return LO.loto_table(best, r[""], fc[best], dict(zip(r["_zero"]["ticker"], r["_zero"]["ok"])))


def trials_rows(key: str) -> list:
    spec = lambda c: json.dumps({"cfg": c, "base": "H=K1, M=sqrt(K2*K3)", "lam": 250, "huber": 1.345 if c == "CLD3" else None, "spline_df": 4 if c == "CLD4" else None, "cache_key": key[:16]}, ensure_ascii=False)
    return ["Ж-222:%s:%s,Ж-222,harj_hybrid_v4,conditional_loss_differential_CLD,\"%s\",trial" % (c, hashlib.sha256((c + "|" + key).encode()).hexdigest()[:12], spec(c).replace('"', '""')) for c in CFGS]


def write_trials(key: str) -> None:
    TR.replace_rows(["Ж-222:%s:" % c for c in CFGS], trials_rows(key))        # только CLD1–CLD4; строки других стадий сохраняются


def stage_cld() -> dict:
    t0 = time.time()
    cp = declare()
    P, kc = D.panel(), keys()
    Zl = ZB.z_by_ticker(CC.keys("Z")["key"])
    ex = {c: pd.read_parquet(CC.OUT / (c + ".parquet")) for c in REFS}
    brk = brk_table(Zl, kc["key"])
    fc, wt, cf = forecasts(Zl, P, kc["key"])
    k = D.eval_frame(P, {**ex, **fc})
    cols = DECL.CTRL + REFS + CFGS
    c, crit = S.common(k, cols), S.crit_panel(k, cols)
    pt, cs = ST.per_ticker(c, cols), crit.set_index("series")["crit_panel"]
    best = min(CFGS, key=lambda x: float(cs[x]))
    o = {"P": P, "Zl": Zl, "brk": brk, "cp": cp, "k": k, "c": c, "crit": crit, "pt": pt, "cols": cols, "fc": fc, "wt": wt, "cf": cf, "kc": kc, "best": best, "t0": t0, "refs": REFS,
         "note": "входы sha256=%s…, код sha256=%s…" % (kc["inputs"][:12], kc["code"][:12])}
    o["comp"], o["tests"] = ST.comparison(c, cols, crit, pt), CST.tests(c, CFGS, cp["N_declared_before_stats"], cp["N_wide"])
    o["sber"], o["years"] = ST.sber(c, cols, BS.sber_ref()), Y.years_table(k, cols)
    o["go"] = GO.rules(k, ["k1"] + REFS + CFGS, cs, cp["go_threshold_B"])
    o["loto"] = loto(best, P, Zl, fc, kc["key"])
    write_all(o)
    if cp.get("CLD_cache_key") != kc["key"]:
        DECL.update(CLD_runtime_sec=round(time.time() - t0, 1), CLD_processes=1)
    DECL.update(CLD_configs_spent=len(CFGS), used_after_CLD=int(cp["used_after_NTP"]) + len(CFGS), CLD_best=best, CLD_cache_key=kc["key"], CLD_stop_go="go" if o["go"]["cfg"][best]["go"] else "stop", CLD_go_rules=o["go"])
    return o


def write_all(o: dict) -> None:
    n, note, thr = o["cp"], o["note"], o["cp"]["go_threshold_B"]
    IO.write_csv(o["comp"], "CLD_COMPARISON.csv", "CLD (Ж-225): crit_panel = hp_stats_sel.crit_panel (M16 × 2017–2021, Ж-217); метрики — среднее по M16 на общих датах (hp_hyb_out); порог go = %.6f; C1, C4, C8, NTP1 — прежние прогнозы; %s" % (thr, note))
    IO.write_csv(o["pt"], "CLD_PER_TICKER.csv", "17 бумаг (SBER — отдельная диагностика, в выбор не входит); общие даты; " + note)
    IO.write_csv(o["tests"], "CLD_TESTS.csv", "DM НУ лаг 5 и блочный бутстрап дат; Δ = QLIKE(CLD) − QLIKE(база); Холм: N=%d (p_holm), N_wide=%d (p_holm_wide), семейство M16 × {2022_2025, all} × CLD1–CLD4; %s" % (n["N_declared_before_stats"], n["N_wide"], note))
    IO.write_csv(o["years"], "CLD_YEARS.csv", "crit_panel по годам цели: crit_part = pen_Y/4 (2018–2021) суммируется в crit_panel; ann_qlike — средний QLIKE по M16 за год цели (2017 и 2018 — отдельные строки); 2022–2026 описательно; " + note)
    IO.write_csv(o["sber"], "CLD_SBER.csv", "SBER отдельно, в выбор не входит; ориентир 0,2081 — потолок; " + note)
    IO.write_csv(o["loto"], "CLD_LOTO.csv", "LOTO лучшей CLD: оценка по 16 бумагам, поперечные z и brk по 16, исключённая получает панельную модель; " + note)
    IO.write_csv(pd.concat([o["wt"][c] for c in CFGS], ignore_index=True), "CLD_WEIGHTS.csv", "mean_w — средний вес на M (w = 0,5 — приор) за месяц; PANEL — среднее по панели; " + note)
    IO.write_csv(pd.concat([o["cf"][c] for c in CFGS], ignore_index=True), "CLD_COEF.csv", "coef: CLD1–CLD3 — коэффициент при стандартизованном признаке; CLD4 — с. о. вклада признака (сплайн); s_hat, c90 — масштабы отображения; месячная оценка; " + note)
    write_trials(o["kc"]["key"])
