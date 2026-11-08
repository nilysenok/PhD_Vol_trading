"""V4-H: стадия `H` (G38–G41). Порядок записи: объявление (CHECKPOINT, до первого файла результатов) -> признаки ZH_<T> (кэш) -> прогнозы H1–H8 (кэш, ключ = хэш входов и кода, Д5) -> общие таблицы (контроли, K8, C8, CLD1, H1–H8) ->
LOTO лучшей H -> годы, SBER, DM, «go» -> выходы -> реестр (`v4_trials.replace_rows`). `stage_h(write=False)` — режим проверки: только чтение (кэш и объявление должны уже быть; ни файла, ни ключа не пишется).
Базовые модели, прежние гибриды, Z и BRK не переобучаются и не меняются (только чтение); выбор — только `crit_panel` (M16 × цели 2017–2021). Отчёт, значения источников и проверки — вне путей прогноза (`v4_hd_*`)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import (v4_b_data as D, v4_b_decl as DECL, v4_b_loto as LO, v4_b_stage as BS, v4_b_stats as ST, v4_c_cache as CC, v4_cb_years as Y, v4_e_go as GO, v4_h_cache as HC, v4_h_feat as HF, v4_h_run as R,
                           v4_io as IO, v4_rqc_stats as RS, v4_trials as TR, v4_z_build as ZB, v4_z_feat as ZF)

CFGS = R.CFGS
REFS = ["K8", "C8", "CLD1"]
H_RULE = ("этап H (Ж-225, Ж-222 п. 9): F = K8·exp(clip(θᵀz̃, ±0,5)), K8 = √(H·√(K2·K3)), H = K1; z̃ — признак, ортогонализованный к (ln H, ln M, ℓ, |ℓ|) гребневой регрессией (λ = 250) на закрытых парах окна оценки; θ — минимум среднего QLIKE "
          "на закрытых парах 17 бумаг (t⁺ < d0, расширяющееся окно с 2017-01, раз в месяц), гребень λ = 250·s², без свободного члена; до 60 пар g = 0; H1 рыночный HAR-surprise, H2 секторный, H3 внутридневная форма, H4 вероятность скачка, "
          "H5 согласованность прогнозов, H6 нестабильность режима (CUSUM, с. о. Δ ln медианной RV), H7 всё вместе, H8 = H7 с иерархией θ_i = θ + η_i (κ = 250·s²); «go» — три условия (Ж-225 п. 10); выбор — crit_panel")


def declare() -> dict:
    """Один раз: время объявления этапа до первого файла результатов; повтор не сдвигает."""
    cp = json.loads(DECL.CP.read_text())
    if "H_declared_at" not in cp:
        DECL.update(H_declared_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), H_rule=H_RULE, H_configs=CFGS, H_prior_sha=HC.prior_sha())
        cp = json.loads(DECL.CP.read_text())
    return cp


def features(Zl: pd.DataFrame, key: str, write: bool) -> pd.DataFrame:
    memo = {}

    def mk(t):
        def build():
            if "z" not in memo:
                memo["z"] = HF.build(Zl, HF.firsth_table(IO.T17), IO.T17)
            return {"": memo["z"][memo["z"]["ticker"] == t].reset_index(drop=True)}
        return build
    return pd.concat([HC.cached("ZH_" + t, key, mk(t), write)[""] for t in IO.T17], ignore_index=True)


def forecasts(P: pd.DataFrame, Zh: pd.DataFrame, key: str, write: bool) -> tuple:
    fc, cf = {}, {}
    for cfg in CFGS:
        r = HC.cached(cfg, key, lambda cfg=cfg: dict(zip(("", "_coef"), R.forecast(cfg, P, Zh))), write)
        fc[cfg], cf[cfg] = r[""], r["_coef"]
    return fc, cf


def zero_ok(best: str, P, Zh16, t: str, out: pd.DataFrame) -> bool:
    """Исключённая бумага не входит в оценку: подмена её RV не меняет прогнозы остальных бумаг (побитно) при тех же поперечных величинах по 16."""
    Q = P.copy()
    Q.loc[Q["ticker"] == t, "rv"] *= 5.0
    o2 = R.forecast(best, Q, Zh16, exclude=t)[0]
    f = lambda x: x[x["ticker"] != t].sort_values(["date", "ticker"])["F"].to_numpy()
    return bool(np.array_equal(f(out), f(o2)))


def loto(best: str, P, Zl, fc: dict, key: str, write: bool) -> pd.DataFrame:
    """LOTO: поперечные величины (медианы, секторные соседи, brk) — по 16 бумагам без исключённой; исключённая бумага в оценку не входит, получает панельные θ (η = 0)."""
    def build():
        fh, parts, zero = HF.firsth_table(IO.T17), [], []
        for t in IO.M16:
            tk = [x for x in IO.T17 if x != t]
            Zh16 = HF.build(ZF.recross(Zl, tk), fh, tk)
            out = R.forecast(best, P, Zh16, exclude=t)[0]
            parts.append(out[out["ticker"] == t][["ticker", "date", "t_plus", "rv", "h", "F"]])
            zero.append({"ticker": t, "ok": zero_ok(best, P, Zh16, t, out)})
        return {"": pd.concat(parts, ignore_index=True), "_zero": pd.DataFrame(zero)}
    r = HC.cached("H_loto_" + best, key, build, write)
    return LO.loto_table(best, r[""], fc[best], dict(zip(r["_zero"]["ticker"], r["_zero"]["ok"])))


def trials_rows(key: str) -> list:
    spec = lambda c: json.dumps({"cfg": c, "base": "K8=sqrt(H*sqrt(K2*K3))", "kappa": 250, "clip": 0.5, "hier": c == "H8", "cache_key": key[:16]}, ensure_ascii=False)
    return ["Ж-222:%s:%s,Ж-222,harj_hybrid_v4,orthogonal_new_information_H,\"%s\",trial" % (c, hashlib.sha256((c + "|" + key).encode()).hexdigest()[:12], spec(c).replace('"', '""')) for c in CFGS]


def write_trials(key: str) -> None:
    TR.replace_rows(["Ж-222:%s:" % c for c in CFGS], trials_rows(key))       # только H1–H8; строки других стадий сохраняются


def stage_h(write: bool = True) -> dict:
    t0 = time.time()
    cp = declare() if write else json.loads(DECL.CP.read_text())
    if "H_declared_at" not in cp:
        raise RuntimeError("этап H не объявлен: сначала `run_v4 --stage H`")
    P, kc = D.panel(), HC.keys()
    Zl = ZB.z_by_ticker(CC.keys("Z")["key"])
    Zh = features(Zl, kc["key"], write)
    ex = {c: pd.read_parquet(HC.OUT / (c + ".parquet")) for c in REFS}
    fc, cf = forecasts(P, Zh, kc["key"], write)
    k = D.eval_frame(P, {**ex, **fc})
    cols = list(DECL.CTRL) + REFS + CFGS
    c, crit = S.common(k, cols), S.crit_panel(k, cols)
    pt, cs = ST.per_ticker(c, cols), crit.set_index("series")["crit_panel"]
    best = min(CFGS, key=lambda x: float(cs[x]))
    o = {"P": P, "Zl": Zl, "Zh": Zh, "cp": cp, "k": k, "c": c, "crit": crit, "pt": pt, "cols": cols, "fc": fc, "cf": cf, "kc": kc, "best": best, "t0": t0,
         "note": "входы sha256=%s…, код sha256=%s…" % (kc["inputs"][:12], kc["code"][:12])}
    o["comp"], o["tests"] = ST.comparison(c, cols, crit, pt), RS.tests(c, CFGS, cp["N_declared_before_stats"], cp["N_wide"])
    o["sber"], o["years"] = ST.sber(c, cols, BS.sber_ref()), Y.years_table(k, cols)
    o["go"] = GO.rules(k, ["k1"] + REFS + CFGS, cs, cp["go_threshold_B"])
    o["loto"] = loto(best, P, Zl, fc, kc["key"], write)
    if write:
        from models_v2.s11 import v4_hd_write as W
        W.write_all(o)
        if cp.get("H_cache_key") != kc["key"]:
            DECL.update(H_runtime_sec=round(time.time() - t0, 1), H_processes=1)
        DECL.update(H_configs_spent=len(CFGS), used_after_H=int(cp["used_after_RQC"]) + len(CFGS), H_best=best, H_cache_key=kc["key"], H_stop_go="go" if o["go"]["cfg"][best]["go"] else "stop", H_go_rules=o["go"])
    return o
