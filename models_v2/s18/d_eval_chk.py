"""Проверки S-D-EVAL (E.G1): `problems(t, inp, rep, ckpt, tx)` — только чтение. Независимый пересчёт: QLIKE формулой numpy, Δ, бутстрап (свежий `stat_idx`, без кэша), вердикты, тождество вкладов лет, срезы, предсказания, ключ `S.D.eval` до таблиц."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import d_eval_tab as T

N_D, MARK0 = 2198, "# диагностика вклада элементов R, исследовательский результат, не OOS"
SPEC = dict(thr_pm=-0.0313, thr_pf=0.0157, ktrim=22, B=10000, block=20, seed=0, lag=8)


def q_indep(rv, F) -> float:
    r = np.asarray(rv, float) / np.asarray(F, float)
    return float(np.mean(r - np.log(r) - 1.0))


def code_of(delta: float, lo: float, hi: float) -> str:
    """Разд. 5 независимо от `d_eval_tab.verdict`."""
    if delta > 0 and lo > 0:
        return "needed"
    if delta > 0:
        return "worse_noise"
    if delta < 0 and hi < 0:
        return "harms"
    return "not_found"


def p_const() -> list:
    ok = T.THR_PM == SPEC["thr_pm"] and T.THR_PF == SPEC["thr_pf"] and T.KTRIM == SPEC["ktrim"] and (S.B, S.MEAN_BLOCK, S.SEED) == (SPEC["B"], SPEC["block"], SPEC["seed"]) and S.nw_lag(N_D) == SPEC["lag"]
    return [] if ok else ["константы спецификации изменены (пороги 0 / −0,0313 / 0,0157, K = 22, B = 10 000, блок 20, seed 0, NW-8)"]


def p_qlike(t: dict, inp: dict) -> list:
    rv, F, bad = inp["rv"], inp["F"], []
    bad += ["n ≠ 2198 или границы дат"] * (not (len(rv) == N_D and (t["table"]["n"] == N_D).all() and (t["table"]["first_tplus"] == "2018-01-04").all() and (t["table"]["last_tplus"] == "2026-11-10").all()))
    for _, r in t["table"].iterrows():
        bad += ["QLIKE %s ≠ независимому расчёту (обрезки нет)" % r["series"]] * (abs(r["qlike"] / q_indep(rv, F[r["series"]]) - 1.0) > 1e-12)
    return bad


def p_delta(t: dict, inp: dict) -> list:
    rv, F, bad = inp["rv"], inp["F"], []
    idx = S.stat_idx(N_D, SPEC["B"], SPEC["block"], SPEC["seed"])
    for _, r in t["delta"].iterrows():
        qh, qb = q_indep(rv, F[r["series"]]), q_indep(rv, F[r["base"]])
        want = 100.0 * (qh - qb) / qb
        bad += ["Δ %s к %s: знак или база процентов (%.6f ≠ %.6f)" % (r["series"], r["base"], r["delta_pct"], want)] * (abs(r["delta_pct"] - want) > 1e-9)
        ci = S.boot_ci(T.ql(rv, F[r["series"]]), T.ql(rv, F[r["base"]]), idx)
        bad += ["интервал %s к %s ≠ независимому бутстрапу" % (r["series"], r["base"])] * (abs(ci["lo90"] - r["lo90"]) > 1e-9 or abs(ci["hi90"] - r["hi90"]) > 1e-9)
        bad += ["параметры бутстрапа/лага не B = 10 000, блок 20, seed 0, NW-8"] * (not (r["boot_B"] == 10000 and r["boot_block"] == 20 and r["boot_seed"] == 0 and r["nw_lag"] == 8))
    return bad


def p_verdict(t: dict) -> list:
    el, bad = t["elements"], []
    for i, (name, x, b) in enumerate(T.ELEMS):
        d = t["delta"][(t["delta"]["series"] == x) & (t["delta"]["base"] == b)].iloc[0]
        want = code_of(d["delta_pct"], d["lo90"], d["hi90"])
        bad += ["вердикт %s: %s ≠ выведенному %s" % (name, el["code"].iloc[i], want)] * (el["code"].iloc[i] != want)
        bad += ["текст вердикта %s не содержит Δ таблицы" % name] * ("%+.4f" % d["delta_pct"] not in el["verdict"].iloc[i])
    codes = {"iii": el["code"].iloc[0], "i": el["code"].iloc[1]}
    bad += ["сводный вывод ≠ выведенному из таблицы"] * (el["verdict"].iloc[-1] != T.summary(codes))
    return bad


def p_years_slices(t: dict, inp: dict) -> list:
    bad, y, tp = [], t["years"], inp["tp"]
    for (h, b), g in y.groupby(["series", "base"]):
        bad += ["Σ вкладов лет ≠ Δ: %s к %s" % (h, b)] * bool(abs(g["contrib_pct"].sum() - g["delta_total_pct"].iloc[0]) > 1e-12)
        d = t["delta"][(t["delta"]["series"] == h) & (t["delta"]["base"] == b)]
        bad += ["Δ итого лет ≠ Δ основной таблицы: %s к %s" % (h, b)] * bool(len(d) == 1 and abs(g["delta_total_pct"].iloc[0] - d["delta_pct"].iloc[0]) > 1e-9)
    s = t["slices"]
    ctl = s[(s["series"] == "S-RCKm") & (s["base"] == "S-RCK")].set_index("slice")
    n2022 = int((tp.year == 2022).sum())
    bad += ["срезы: n оставшихся ≠ 2197 / %d" % (N_D - n2022)] * (not (ctl.loc[T.SLICES[1], "n"] == N_D - 1 and ctl.loc[T.SLICES[2], "n"] == N_D - n2022 and N_D - n2022 == 1991))
    bad += ["основной срез ≠ Δ таблицы"] * any(abs(r["delta_pct"] - t["delta"][(t["delta"]["series"] == r["series"]) & (t["delta"]["base"] == r["base"])]["delta_pct"].iloc[0]) > 1e-9 for _, r in s[s["kind"] == "основной"].iterrows() if len(t["delta"][(t["delta"]["series"] == r["series"]) & (t["delta"]["base"] == r["base"])]) == 1)
    return bad + ["обрезка попала в основную Δ"] * (not (s[s["kind"] == "основной"]["n_removed"] == 0).all())


def p_pred(t: dict) -> list:
    m, p, bad = t["mech"], t["pred"].set_index("id"), []
    g = lambda s, i: float(m[(m["series"] == s) & (m["item"] == i)]["value"].iloc[0])
    sh = lambda s: g(s, "level_only|F_vs_W5G_pct") / g(s, "full|F_vs_W5G_pct") if g(s, "full|F_vs_W5G_pct") < 0 else float("nan")
    want = {"P-m": g("S-RCKm", "mean_ln_C_cfg8") < -0.0313, "P-f": abs(g("S-RCKf", "mean_ln_C_cfg8")) < 0.0157, "P-f2": bool(sh("S-RCKf") < sh("S-RCK"))}
    for k, v in want.items():
        bad += ["предсказание %s: %s ≠ выведенному" % (k, p.loc[k, "fulfilled"])] * (p.loc[k, "fulfilled"] != ("выполнено" if v else "нет"))
    for s in T.SERIES[:3]:
        bad += ["разложение при фиксированных θ %s: пересчитанный F ≠ сохранённому (%.2e)" % (s, g(s, "full|max_rel_F_vs_saved"))] * (g(s, "full|max_rel_F_vs_saved") > 1e-12)
    return bad


def p_key(t: dict, rep, ckpt, tx: dict) -> list:
    cp = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}
    k, reg, bad = cp.get("S.D.eval"), (cp.get("S.D.registered") or {}), []
    if not k:
        return ["ключа S.D.eval нет"]
    pt = lambda s: dt.datetime.strptime(s, "%d.%m.%Y %H:%M:%S")
    after_reg = bool(reg) and pt(reg["registered_at"]) < pt(k["written_at"])
    bad += ["S.D.eval не позже S.D.registered"] * (not after_reg)
    first = min((dt.datetime.fromtimestamp((rep / n).stat().st_mtime) for n in tx if (rep / n).exists()), default=None)
    bad += ["ключ S.D.eval позже самой ранней таблицы"] * (first is not None and pt(k["written_at"]) > first.replace(microsecond=0) + dt.timedelta(seconds=1))
    import hashlib
    bad += ["хэш таблицы %s ≠ ключу" % n for n, v in tx.items() if k["outputs"].get(n) != hashlib.sha256(v.encode("utf-8")).hexdigest()]
    return bad + ["первая строка таблицы не пометка"] * (not all(v.startswith(MARK0) for v in tx.values()))


def problems(t: dict, inp: dict, rep, ckpt, tx: dict) -> list:
    return p_const() + p_qlike(t, inp) + p_delta(t, inp) + p_verdict(t) + p_years_slices(t, inp) + p_pred(t) + p_key(t, rep, ckpt, tx)
