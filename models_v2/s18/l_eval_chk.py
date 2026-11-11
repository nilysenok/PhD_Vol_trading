"""Проверки S-L-EVAL (E.G1): `problems(t, inp, rep, ckpt, tx)` — только чтение. Независимый пересчёт: QLIKE формулой numpy, Δ, бутстрап (свежий `stat_idx`), Холм (своя реализация, N из ключей регистрации), условия 5.2 (в т. ч. условие 3 из `S_D_ELEMENTS.csv`),
вердикт 5.4, P-L1…P-L4, тождество вкладов лет, срезы, порядок ключа `S.L.eval` и таблиц."""
from __future__ import annotations

import datetime as dt
import hashlib
import json

import numpy as np
import pandas as pd

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import l_eval_tab as LT

N_D = 2198
SPEC = dict(ktrim=22, B=10000, block=20, seed=0, lag=8, pl1=0.90, pl2=(-0.063, -0.015), thr=(0.415, 0.50, 1.00))


def q_indep(rv, F) -> float:
    r = np.asarray(rv, float) / np.asarray(F, float)
    return float(np.mean(r - np.log(r) - 1.0))


def holm_indep(p, N: int) -> np.ndarray:
    o, out, run = np.argsort(p, kind="stable"), np.zeros(len(p)), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (N - r) * p[i]))
        out[i] = run
    return out


def p_const() -> list:
    from models_v2.s18 import a_eval_yr as AY
    ok = LT.KTRIM == SPEC["ktrim"] and (S.B, S.MEAN_BLOCK, S.SEED) == (SPEC["B"], SPEC["block"], SPEC["seed"]) and S.nw_lag(N_D) == SPEC["lag"] and LT.PL1 == SPEC["pl1"] and LT.PL2 == SPEC["pl2"] and tuple(AY.THR.values()) == SPEC["thr"]
    return [] if ok else ["константы спецификации изменены (K = 22, B = 10 000, блок 20, seed 0, NW-8, 0,90, [−0,063; −0,015], 0,415 / 0,50 / 1,00)"]


def p_qlike_delta(t: dict, inp: dict) -> list:
    rv, F, bad = inp["rv"], inp["F"], []
    bad += ["n ≠ 2198 или границы дат"] * (not (len(rv) == N_D and (t["table"]["n"] == N_D).all() and (t["table"]["first_tplus"] == "2018-01-04").all() and (t["table"]["last_tplus"] == "2026-11-10").all()))
    for _, r in t["table"].iterrows():
        bad += ["QLIKE %s ≠ независимому расчёту" % r["series"]] * (abs(r["qlike"] / q_indep(rv, F[r["series"]]) - 1.0) > 1e-12)
    idx = S.stat_idx(N_D, SPEC["B"], SPEC["block"], SPEC["seed"])
    for _, r in t["delta"].iterrows():
        qh, qb = q_indep(rv, F[r["series"]]), q_indep(rv, F[r["base"]])
        want = 100.0 * (qh - qb) / qb
        bad += ["Δ %s к %s: знак или база (%.6f ≠ %.6f)" % (r["series"], r["base"], r["delta_pct"], want)] * (abs(r["delta_pct"] - want) > 1e-9)
        ci = S.boot_ci(LT.ql(rv, F[r["series"]]), LT.ql(rv, F[r["base"]]), idx)
        bad += ["интервал %s к %s ≠ независимому бутстрапу" % (r["series"], r["base"])] * (abs(ci["lo90"] - r["lo90"]) > 1e-9 or abs(ci["hi90"] - r["hi90"]) > 1e-9)
    return bad


def p_holm(t: dict, inp: dict) -> list:
    d = t["delta"]
    f = d[d["m_family"].notna()]
    bad = ["N Холма ≠ N программы %d / %d" % (inp["N"], inp["N_wide"])] * (not (inp["N"] == 852 and inp["N_wide"] == 3409 and (d["N"] == inp["N"]).all() and (d["N_wide"] == inp["N_wide"]).all()))
    bad += ["семейство Холма ≠ 4 гипотезам S-RL1"] * (len(f) != 4 or bool(set(f["series"]) - {"S-RL1", "X1_R(S-RL1)"}))
    for col, N in (("p_holm_N", inp["N"]), ("p_holm_Nwide", inp["N_wide"]), ("p_holm_m", 4)):
        bad += ["%s ≠ независимому Холму" % col] * (not np.allclose(f[col], holm_indep(f["p_dm_nw8"].to_numpy(), N), rtol=1e-12, atol=0))
    for _, r in d.iterrows():
        conf = bool(r["delta_pct"] < 0 and r["p_holm_N"] < 0.05 and r["p_spa_c"] < 0.05) if pd.notna(r["m_family"]) else None
        bad += ["вердикт %s|%s: «подтверждено» ≠ правилу (№ 158)" % (r["series"], r["base"])] * (conf is not None and r["verdict"].startswith("численно лучше базы; подтверждено") != conf)
    return bad


def p_verdicts(t: dict, inp: dict) -> list:
    v, q, bad = t["verdict"].set_index("id"), t["table"].set_index("series")["qlike"], []
    from models_v2.s18 import l_eval as E
    el = E.elements()
    c3 = float(el[(el["series"] == "S-RCKf") & (el["base"] == "S-RCK")]["delta_pct"].iloc[0]) > 0
    want = q["S-RL1"] < q["W5-G"] and q["S-RL1"] < q["S-RL1b2"] and c3
    ref = q["S-RL1"] < q["W5-G"] and q["S-RL1"] < q["S-RL1b"] and c3
    bad += ["критерий взаимодействия (основной, по S-RL1b2) ≠ независимому: %s" % v.loc["interaction", "value"]] * ((v.loc["interaction", "value"] == "да") != want)
    bad += ["справочный критерий (по S-RL1b) ≠ независимому"] * ((v.loc["interaction_ref", "value"] == "да") != ref)
    r = t["delta"][(t["delta"]["series"] == "S-RL1") & (t["delta"]["base"] == "S-RCK")].iloc[0]
    code = "needed" if r["delta_pct"] > 0 and r["lo90"] > 0 else "worse_noise" if r["delta_pct"] > 0 else "harms" if r["hi90"] < 0 else "not_found"
    bad += ["вердикт 5.4 %s ≠ %s" % (v.loc["5.4", "value"], code)] * (v.loc["5.4", "value"] != code)
    m, y = t["months"], t["years"]
    pos = m[(m["series"] == "S-RL1") & (m["lam"] > 0)]
    c4 = [float(y[(y["series"] == "S-RL1") & (y["base"] == "W5-G") & (y["period"] == p)]["contrib_pct"].iloc[0]) for p in ("2021", "2025", "2026")]
    pl = {"P-L1": (pos["a"] < 0).mean() >= 0.9, "P-L2": -0.063 <= pos["a"].median() <= -0.015, "P-L3": bool(np.array_equal(inp["F"]["S-RL1a"], inp["F"]["W5-G"])), "P-L4": min(c4) >= 0}
    bad += ["предсказание %s ≠ выведенному" % k for k, ok in pl.items() if (v.loc[k, "value"] == "выполнено") != bool(ok)]
    g = t["guard"]
    bad += ["строка «этап S» ≠ выведенной из условий"] * (bool(g.iloc[-1]["passed"]) != bool(g.iloc[:-1]["passed"].all()))
    return bad


def p_years_slices(t: dict, inp: dict) -> list:
    bad, tp = [], inp["tp"]
    for (h, b), g in t["years"].groupby(["series", "base"]):
        bad += ["Σ вкладов лет ≠ Δ: %s к %s" % (h, b)] * bool(abs(g["contrib_pct"].sum() - g["delta_total_pct"].iloc[0]) > 1e-12)
    s, n22 = t["slices"], int((tp.year == 2022).sum())
    c = s[(s["series"] == "S-RL1") & (s["base"] == "W5-G")].set_index("slice")
    bad += ["срезы: n ≠ 2197 / 1991"] * (not (c.iloc[1]["n"] == N_D - 1 and c.iloc[2]["n"] == N_D - n22 == 1991))
    return bad + ["обрезка в основной Δ"] * (not (s[s["kind"] == "основной"]["n_removed"] == 0).all())


def p_key(t: dict, rep, ckpt, tx: dict) -> list:
    cp = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}
    k, bad = cp.get("S.L.eval"), []
    if not k:
        return ["ключа S.L.eval нет"]
    pt = lambda s: dt.datetime.strptime(s, "%d.%m.%Y %H:%M:%S")
    bad += ["S.L.eval не позже регистраций S.L.registered / S.L2.registered"] * (not all(key in cp and pt(cp[key]["registered_at"]) < pt(k["written_at"]) for key in ("S.L.registered", "S.L2.registered")))
    first = min((dt.datetime.fromtimestamp((rep / n).stat().st_mtime) for n in tx if (rep / n).exists()), default=None)
    bad += ["ключ S.L.eval позже самой ранней таблицы"] * (first is not None and pt(k["written_at"]) > first.replace(microsecond=0) + dt.timedelta(seconds=1))
    bad += ["хэш таблицы %s ≠ ключу" % n for n, v in tx.items() if k["outputs"].get(n) != hashlib.sha256(v.encode("utf-8")).hexdigest()]
    return bad + ["первая строка таблицы не пометка"] * (not all(v.startswith("# исследовательский результат, не OOS") for v in tx.values()))


def problems(t: dict, inp: dict, rep, ckpt, tx: dict) -> list:
    return p_const() + p_qlike_delta(t, inp) + p_holm(t, inp) + p_verdicts(t, inp) + p_years_slices(t, inp) + p_key(t, rep, ckpt, tx)
