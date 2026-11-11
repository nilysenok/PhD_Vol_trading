"""Проверки S-Z-EVAL (ZE.G1): `problems(t, inp, rep, ckpt, tx)` — только чтение. Независимый пересчёт: QLIKE формулой numpy, Δ, бутстрап (свежий `stat_idx`), Холм (своя реализация, N из ключей регистрации и константы 855 / 3415),
H-Z1, H-Z2, условия У1-S…У3-S, У5-S, У6-R (два набора), подтверждение и решение Z-DEC, тождество вкладов лет, срез без 2022-02-24, нулевая диагностика, порядок ключа `S.Z.eval` и таблиц."""
from __future__ import annotations

import datetime as dt
import hashlib
import json

import numpy as np
import pandas as pd

from models_v2.s11 import v4_ntp_state as NS
from models_v2.s16 import x1_w1_stat as SS
from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import z_eval_cfg as C
from models_v2.s18 import z_eval_gd as GD
from models_v2.s18 import z_eval_null as ZN
from models_v2.s18 import z_eval_tab as T

SPEC = dict(ktrim=22, B=10000, block=20, seed=0, lag=8, n=855, nw=3415, m=6, thr=(0.415, 0.50, 1.00))


def q_indep(rv, F) -> float:
    """QLIKE окна — `v4_ntp_state.qlike` (ZE.G1), среднее дневных потерь на всех датах, без обрезки; расчёт оценки идёт через `v6_0_q.ql` — сверка двух реализаций."""
    return float(np.mean(NS.qlike(rv, F)))


def holm_indep(p, N: int) -> np.ndarray:
    o, out, run = np.argsort(p, kind="stable"), np.zeros(len(p)), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (N - r) * p[i]))
        out[i] = run
    return out


def trim_indep(lh, lb, k: int = 22) -> float:
    d = np.asarray(lh, float) - np.asarray(lb, float)
    drop = np.concatenate([np.argsort(d, kind="stable")[:k], np.argsort(d, kind="stable")[len(d) - k:]])
    keep = np.setdiff1d(np.arange(len(d)), drop)
    return 100.0 * float(d[keep].sum()) / float(np.asarray(lb, float)[keep].sum())


def p_const() -> list:
    ok = (C.KTRIM, C.N_HOLM, C.N_WIDE, C.M_FAMILY) == (SPEC["ktrim"], SPEC["n"], SPEC["nw"], SPEC["m"]) and (S.B, S.MEAN_BLOCK, S.SEED) == (SPEC["B"], SPEC["block"], SPEC["seed"]) and S.nw_lag(C.N_D) == SPEC["lag"]
    ok = ok and tuple(C.THR.values()) == SPEC["thr"]
    return [] if ok else ["константы спецификации изменены (K = 22, N = 855 / 3415, m = 6, B = 10 000, блок 20, seed 0, NW-8, 0,415 / 0,50 / 1,00)"]


def p_qlike(t: dict, inp: dict) -> list:
    rv, F, tb, bad = inp["rv"], inp["F"], t["table"], []
    bad += ["n ≠ 2198 или границы дат"] * (not (len(rv) == C.N_D and (tb["n"] == C.N_D).all() and (tb["first_tplus"] == C.FIRST).all() and (tb["last_tplus"] == C.LAST).all()))
    bad += ["в первой таблице нет рядов или порядок ≠ заданию"] * (list(tb["series"][:len(C.SHOWN)]) != list(C.SHOWN))
    for _, r in tb.iterrows():
        bad += ["QLIKE %s ≠ независимому расчёту" % r["series"]] * (abs(r["qlike"] / q_indep(rv, F[r["series"]]) - 1.0) > 1e-12)
        for b in C.BASES:
            want = 100.0 * (q_indep(rv, F[r["series"]]) - q_indep(rv, F[b])) / q_indep(rv, F[b])
            bad += ["Δ %s к %s: знак или база" % (r["series"], b)] * (abs(r["d_%s_pct" % b] - want) > 1e-9)
    return bad


def p_stats(t: dict, inp: dict) -> list:
    st, rv, F, bad = t["stats"], inp["rv"], inp["F"], []
    idx = S.stat_idx(C.N_D, SPEC["B"], SPEC["block"], SPEC["seed"])
    m = st[st["kind"] == "main"]
    bad += ["семейство Холма ≠ 6 гипотезам S-RCKZ"] * (len(m) != 6 or set(m["series"]) - {"S-RCKZ", "X1_R(S-RCKZ)"} != set())
    bad += ["N Холма ≠ 855 / 3415 или N ≠ ключам регистрации"] * (not ((st["n_holm"] == SPEC["n"]).all() and (st["n_holm_wide"] == SPEC["nw"]).all() and (inp["N"], inp["N_wide"]) == (SPEC["n"], SPEC["nw"])))
    for col, N in (("p_holm_N", SPEC["n"]), ("p_holm_Nwide", SPEC["nw"]), ("p_holm_m", 6)):
        bad += ["%s ≠ независимому Холму" % col] * (not np.allclose(m[col], holm_indep(m["p_dm"].to_numpy(), N), rtol=1e-12, atol=0))
    for _, r in st[st["window"] == "M"].iterrows():
        qh, qb = q_indep(rv, F[r["series"]]), q_indep(rv, F[r["base"]])
        bad += ["Δ %s к %s: знак или база" % (r["series"], r["base"])] * (abs(r["delta_pct"] - 100.0 * (qh - qb) / qb) > 1e-9)
        ci = S.boot_ci(S.Q.ql(rv, F[r["series"]]), S.Q.ql(rv, F[r["base"]]), idx)
        bad += ["интервал %s к %s ≠ независимому бутстрапу" % (r["series"], r["base"])] * (abs(ci["lo90"] - r["lo90"]) > 1e-9 or abs(ci["hi90"] - r["hi90"]) > 1e-9)
    for _, r in m.iterrows():
        conf = bool(r["delta_pct"] < 0 and r["p_holm_N"] < 0.05 and r["p_spa_c"] < 0.05)
        bad += ["вердикт %s: «подтверждено» ≠ правилу (№ 158)" % r["hypothesis"]] * (r["verdict"].startswith("лучше") != conf)
        bad += ["вердикт %s: Δ ≥ 0 названо «лучше» (№ 125)" % r["hypothesis"]] * (r["delta_pct"] >= 0 and "численно не лучше" not in r["verdict"])
    return bad


def p_dec(t: dict, inp: dict) -> list:
    rv, F, bad = inp["rv"], inp["F"], []
    q = {k: q_indep(rv, F[k]) for k in ("S-RCKZ", "S-RCK", "S-RCKZs")}
    d, y, tp = t["dec"].set_index("id"), t["years"], inp["tp"]
    hz1, hz2 = q["S-RCKZ"] < q["S-RCK"], q["S-RCKZ"] < q["S-RCKZs"]
    bad += ["H-Z1 ≠ независимому"] * (bool(d.loc["H-Z1", "passed"]) != hz1)
    bad += ["H-Z2 ≠ независимому (база должна быть S-RCKZs)"] * (bool(d.loc["H-Z2", "passed"]) != hz2)
    yy, ok_sets = tp.year.to_numpy(), []
    for name, b, lay in GD.SETS:
        lz, lb = S.Q.ql(rv, F["S-RCKZ"]), S.Q.ql(rv, F[b])
        dl = lambda m: 100.0 * float(lz[m].mean() - lb[m].mean()) / float(lb[m].mean())
        ok = dl(yy <= 2021) <= 0.415 and dl(yy >= 2022) <= 0.50 and dl(yy == 2018) <= 1.00
        mk = {p: m for p, m in SS.period_masks(tp).items()}
        c = SS.contrib(lz, lb, mk)
        best = min(c, key=c.get)
        loyo = 100.0 * float((lz[~mk[best]] - lb[~mk[best]]).sum()) / float(lb[~mk[best]].sum())
        ok = ok and loyo <= 0 and trim_indep(lz, lb) < 0 and trim_indep(S.Q.ql(rv, F["X1_R(S-RCKZ)"]), S.Q.ql(rv, F[lay])) < 0
        ok_sets.append(ok)
        bad += ["набор %s: итог ≠ независимому" % name] * (bool(d.loc["У " + name, "passed"]) != ok)
    r = t["stats"][(t["stats"]["kind"] == "main") & (t["stats"]["hypothesis"] == "S-RCKZ|S-RCK")].iloc[0]
    conf = bool(hz1 and r["p_holm_N"] < 0.05 and r["p_spa_c"] < 0.05)
    want = "1" if hz1 and hz2 and all(ok_sets) and conf else "2" if hz1 and hz2 and all(ok_sets) else "3"
    bad += ["Z-DEC %s ≠ независимому %s" % (d.loc["Z-DEC", "value"], want)] * (str(d.loc["Z-DEC", "value"]) != want)
    bad += ["«превзошёл» без подтверждения"] * (str(d.loc["Z-DEC", "value"]) == "1" and not conf)
    bad += ["Σ вкладов %s ≠ Δ: %s к %s" % (kd, h, b) for (h, b, kd), g in y[y["base"] != "-"].groupby(["series", "base", "kind"]) if abs(g["contrib_pct"].sum() - g["delta_total_pct"].iloc[0]) > 1e-12]
    return bad


def p_slices(t: dict, inp: dict) -> list:
    rv, F, tp, c, bad = inp["rv"], inp["F"], inp["tp"], t["conc"], []
    lz, lb = S.Q.ql(rv, F["S-RCKZ"]), S.Q.ql(rv, F["S-RCK"])
    k = ~(tp == pd.Timestamp("2022-02-24"))
    got = float(c[(c["series"] == "S-RCKZ") & (c["base"] == "S-RCK") & (c["item"] == "no_day")]["value"].iloc[0])
    bad += ["срез без 2022-02-24 ≠ независимому"] * (abs(got - 100.0 * float((lz - lb)[k].sum()) / float(lb[k].sum())) > 1e-9)
    bad += ["обрезка в основной Δ"] * (not (t["stats"][t["stats"]["kind"] == "main"]["n"] == C.N_D).all())
    return bad


def p_key(t: dict, rep, ckpt, tx: dict) -> list:
    cp = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}
    k, bad = cp.get(C.KEY), []
    if not k:
        return ["ключа S.Z.eval нет"]
    pt = lambda s: dt.datetime.strptime(s, "%d.%m.%Y %H:%M:%S")
    bad += ["S.Z.eval не позже регистрации S.Z.registered"] * (not ("S.Z.registered" in cp and pt(cp["S.Z.registered"]["registered_at"]) < pt(k["written_at"])))
    first = min((dt.datetime.fromtimestamp((rep / n).stat().st_mtime) for n in tx if (rep / n).exists()), default=None)
    bad += ["ключ S.Z.eval позже самой ранней таблицы"] * (first is not None and pt(k["written_at"]) > first.replace(microsecond=0) + dt.timedelta(seconds=1))
    bad += ["хэш таблицы %s ≠ ключу" % n for n, v in tx.items() if k["outputs"].get(n) != hashlib.sha256(v.encode("utf-8")).hexdigest()]
    bad += ["sha S_Z_NULL.csv ≠ ключу"] * (k["inputs"].get("docs/otchety/harj_struct/S_Z_NULL.csv") != hashlib.sha256(ZN.OUT.read_bytes()).hexdigest())
    return bad + ["первая строка таблицы не пометка"] * (not all(v.startswith("# исследовательский результат на раскрытой истории, не OOS; механизм выбран после просмотра истории") for v in tx.values()))


def problems(t: dict, inp: dict, rep, ckpt, tx: dict) -> list:
    return p_const() + p_qlike(t, inp) + p_stats(t, inp) + p_dec(t, inp) + p_slices(t, inp) + ZN.check() + p_key(t, rep, ckpt, tx)
