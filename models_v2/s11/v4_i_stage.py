"""V4-I: стадия `I` (G42–G44). Порядок записи: объявление (CHECKPOINT, до первого файла результатов) -> признаки компонент (в памяти, из daily_<T>) -> компонентный прогноз I_comp (кэш, ключ = хэш входов и кода, Д5) -> прогнозы I1–I3 (кэш) ->
общие таблицы (контроли, K8, C8, CLD1, H2, I1–I3) -> LOTO лучшей I -> годы, SBER, DM, «go», точность компонент -> выходы -> реестр (`v4_trials.replace_rows`). `stage_i(write=False)` — режим проверки: только чтение (кэш и объявление уже на диске).
Базовые модели, прежние гибриды, Z и BRK не переобучаются и не меняются; выбор — только `crit_panel` (M16 × цели 2017–2021). Таблицы, отчёт и проверки — вне путей прогноза (`v4_i_out`, `v4_i_check`)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import (v4_b_data as D, v4_b_decl as DECL, v4_b_loto as LO, v4_b_stage as BS, v4_b_stats as ST, v4_cb_years as Y, v4_e_go as GO, v4_i_cache as IC_, v4_i_comp as COMP, v4_i_run as R, v4_io as IO,
                           v4_rqc_stats as RS, v4_trials as TR)

CFGS = R.CFGS
REFS = ["K8", "C8", "CLD1", "H2"]
I_RULE = ("этап I (Ж-222 п. 10): RV = C + J (J = max(RV − BV, 0), C = RV − J, как в HAR-J K1); Ĉ — HAR на ln C (день, неделя, месяц), гребень λ = 250, поправка exp(σ̂²/2); Ĵ = p̂·m̂, p̂ — логит события J > 0 на "
          "(EWMA₂₀ частоты, J/RV, ln RV), m̂ — EWMA₆₀ величины J при J > 0, стянутая к панельной (κ = 250); свободные члены бумаг — κ = 250; оценка раз в месяц по закрытым парам 17 бумаг с 2017-01, до 60 пар S = K8; "
          "I1 = Ĉ + Ĵ, I2 = √(K8·S), I3 = K8·exp(θ(ln S − ln K8)), θ ∈ [0, 1] — QLIKE, гребень λ = 250·s²; «go» — три условия (Ж-225 п. 10); выбор — crit_panel")


def declare() -> dict:
    """Один раз: время объявления этапа до первого файла результатов; повтор не сдвигает."""
    cp = json.loads(DECL.CP.read_text())
    if "I_declared_at" not in cp:
        DECL.update(I_declared_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), I_rule=I_RULE, I_configs=CFGS, I_prior_sha=IC_.prior_sha())
        cp = json.loads(DECL.CP.read_text())
    return cp


def components(P: pd.DataFrame, Fe: pd.DataFrame, key: str, write: bool) -> tuple:
    r = IC_.cached("I_comp", key, lambda: dict(zip(("", "_coef"), R.components(P, Fe))), write)
    return r[""], r["_coef"]


def forecasts(P, cmp: pd.DataFrame, co: pd.DataFrame, key: str, write: bool) -> dict:
    th = co[["month", "theta"]] if "theta" in co else None
    return {c: IC_.cached(c, key, lambda c=c: {"": R.combine(c, cmp, th)}, write)[""] for c in CFGS}


def thetas(cmp, P, key, write) -> pd.DataFrame:
    return IC_.cached("I_theta", key, lambda: {"": R.thetas(cmp, P)}, write)[""]


def zero_ok(best: str, P, Fe, t: str, out: pd.DataFrame) -> bool:
    """Исключённая бумага не входит в оценку: подмена её RV, J, C (и признаков) не меняет прогнозы остальных бумаг (побитно)."""
    Q, G = P.copy(), Fe.copy()
    Q.loc[Q["ticker"] == t, "rv"] *= 5.0
    m = G["ticker"] == t
    G.loc[m, ["c_t", "j_t"]] = G.loc[m, ["c_t", "j_t"]].to_numpy() * 5.0
    G.loc[m, ["lc", "lcw", "lcm", "lrv", "m_ew"]] += 0.7
    o2 = R.forecast(best, Q, G, exclude=t)[0]
    f = lambda x: x[x["ticker"] != t].sort_values(["date", "ticker"])["F"].to_numpy()
    return bool(np.array_equal(f(out), f(o2)))


def loto(best: str, P, Fe, fc: dict, key: str, write: bool) -> pd.DataFrame:
    """LOTO: оценка по 16 бумагам; исключённая бумага в оценку не входит, получает панельные параметры (собственные признаки — из её же дневного ряда)."""
    def build():
        parts, zero = [], []
        for t in IO.M16:
            out = R.forecast(best, P, Fe, exclude=t)[0]
            parts.append(out[out["ticker"] == t][["ticker", "date", "t_plus", "rv", "h", "F"]])
            zero.append({"ticker": t, "ok": zero_ok(best, P, Fe, t, out)})
        return {"": pd.concat(parts, ignore_index=True), "_zero": pd.DataFrame(zero)}
    r = IC_.cached("I_loto_" + best, key, build, write)
    return LO.loto_table(best, r[""], fc[best], dict(zip(r["_zero"]["ticker"], r["_zero"]["ok"])))


def trials_rows(key: str) -> list:
    spec = lambda c: json.dumps({"cfg": c, "components": "C=RV-J, J=max(RV-BV,0)", "kappa": 250, "lam": 250, "base": "K8=sqrt(H*sqrt(K2*K3))", "cache_key": key[:16]}, ensure_ascii=False)
    return ["Ж-222:%s:%s,Ж-222,harj_hybrid_v4,consistent_components_I,\"%s\",trial" % (c, hashlib.sha256((c + "|" + key).encode()).hexdigest()[:12], spec(c).replace('"', '""')) for c in CFGS]


def write_trials(key: str) -> None:
    TR.replace_rows(["Ж-222:%s:" % c for c in CFGS], trials_rows(key))       # только I1–I3; строки других стадий сохраняются


def stage_i(write: bool = True) -> dict:
    t0 = time.time()
    cp = declare() if write else json.loads(DECL.CP.read_text())
    if "I_declared_at" not in cp:
        raise RuntimeError("этап I не объявлен: сначала `run_v4 --stage I`")
    P, kc = D.panel(), IC_.keys()
    dl = COMP.load_dailies()
    Fe = COMP.build(P, dl)
    cmp, co = components(P, Fe, kc["key"], write)
    co = co.merge(thetas(cmp, P, kc["key"], write)[["month", "n_train", "theta"]], on="month")
    fc = forecasts(P, cmp, co, kc["key"], write)
    ex = {c: pd.read_parquet(IC_.OUT / (c + ".parquet")) for c in REFS}
    k = D.eval_frame(P, {**ex, **fc})
    cols = list(DECL.CTRL) + REFS + CFGS
    c, crit = S.common(k, cols), S.crit_panel(k, cols)
    pt, cs = ST.per_ticker(c, cols), crit.set_index("series")["crit_panel"]
    best = min(CFGS, key=lambda x: float(cs[x]))
    o = {"P": P, "dl": dl, "Fe": Fe, "cp": cp, "k": k, "c": c, "crit": crit, "pt": pt, "cols": cols, "fc": fc, "cmp": cmp, "co": co, "kc": kc, "best": best, "t0": t0, "cfgs": CFGS,
         "note": "входы sha256=%s…, код sha256=%s…" % (kc["inputs"][:12], kc["code"][:12])}
    o["comp"], o["tests"] = ST.comparison(c, cols, crit, pt), RS.tests(c, CFGS, cp["N_declared_before_stats"], cp["N_wide"])
    o["sber"], o["years"] = ST.sber(c, cols, BS.sber_ref()), Y.years_table(k, cols)
    o["go"] = GO.rules(k, ["k1"] + REFS + CFGS, cs, cp["go_threshold_B"])
    o["loto"] = loto(best, P, Fe, fc, kc["key"], write)
    if write:
        from models_v2.s11 import v4_i_write as W
        W.write_all(o)
        if cp.get("I_cache_key") != kc["key"]:
            DECL.update(I_runtime_sec=round(time.time() - t0, 1), I_processes=1)
        DECL.update(I_configs_spent=len(CFGS), used_after_I=int(cp["used_after_H"]) + len(CFGS), I_best=best, I_cache_key=kc["key"], I_stop_go="go" if o["go"]["cfg"][best]["go"] else "stop", I_go_rules=o["go"])
    return o
