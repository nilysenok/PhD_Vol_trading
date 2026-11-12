"""Проверки S-RCD-EVAL (RE.G1): `problems(t, inp, rep, ckpt, tx)` — только чтение. Независимый пересчёт: QLIKE (`v4_ntp_state.qlike`), Δ, бутстрап (свежий `stat_idx`), Холм (своя реализация, N из ключей и константы 858 / 3421),
H-RCD1, наборы А и Б (У1-S…У3-S, У5-S, У6-R), подтверждения, s и RCD-DEC, тождества разложения (разд. 6.2) и вклады лет, срез без 2022-02-24, окно D_M без обрезки, сохранённые S-CD, S-CK, S-RCK, S-RCKf против таблиц прежних этапов,
нулевая диагностика, порядок ключа `S.RCD.eval` и таблиц."""
from __future__ import annotations

import datetime as dt
import hashlib
import json

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w1_stat as SS
from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import rd_eval_cfg as C
from models_v2.s18 import rd_eval_dc as DC
from models_v2.s18 import rd_eval_gd as GD
from models_v2.s18 import rd_eval_null as RN
from models_v2.s18.z_eval_chk import holm_indep, q_indep, trim_indep

SPEC = dict(ktrim=22, B=10000, block=20, seed=0, lag=8, n=858, nw=3421, m=6, thr=(0.415, 0.50, 1.00), win=1549)
R, SAVED = "S-RCD", {"S-CD": ("S_C_TABLE.csv", "series"), "S-CK": ("S_C_TABLE.csv", "series"), "S-RCK": ("S_D_TABLE.csv", "series"), "S-RCKf": ("S_D_TABLE.csv", "series")}


def p_const() -> list:
    ok = (C.KTRIM, C.N_HOLM, C.N_WIDE, C.M_FAMILY) == (SPEC["ktrim"], SPEC["n"], SPEC["nw"], SPEC["m"]) and (S.B, S.MEAN_BLOCK, S.SEED) == (SPEC["B"], SPEC["block"], SPEC["seed"]) and S.nw_lag(C.N_D) == SPEC["lag"]
    return [] if ok and tuple(C.THR.values()) == SPEC["thr"] and C.SHARE_HALF == 0.5 else ["константы спецификации изменены (K = 22, N = 858 / 3421, m = 6, B = 10 000, блок 20, seed 0, NW-8, 0,415 / 0,50 / 1,00, 0,5)"]


def p_qlike(t: dict, inp: dict) -> list:
    rv, F, tb, bad = inp["rv"], inp["F"], t["table"], []
    bad += ["n ≠ 2198 или границы дат"] * (not (len(rv) == C.N_D and (tb["n"] == C.N_D).all() and (tb["first_tplus"] == C.FIRST).all() and (tb["last_tplus"] == C.LAST).all()))
    bad += ["в первой таблице нет рядов или порядок ≠ заданию"] * (list(tb["series"][:len(C.SHOWN)]) != list(C.SHOWN))
    for _, r in tb.iterrows():
        qs = q_indep(rv, F[r["series"]])
        bad += ["QLIKE %s ≠ независимому расчёту" % r["series"]] * (abs(r["qlike"] / qs - 1.0) > 1e-12)
        for b in C.BASES:
            bad += ["Δ %s к %s: знак или база" % (r["series"], b)] * (abs(r["d_%s_pct" % b] - 100.0 * (qs - q_indep(rv, F[b])) / q_indep(rv, F[b])) > 1e-9)
    return bad


def p_saved(t: dict, rep) -> list:
    """Сохранённые S-CD, S-CK, S-RCK, S-RCKf — QLIKE из таблиц прежних этапов (12 значащих знаков в CSV)."""
    bad, q = [], t["table"].set_index("series")["qlike"]
    for k, (f, col) in SAVED.items():
        d = pd.read_csv(rep / f, comment="#")
        v = float(d[d[col] == k]["qlike"].iloc[0]) if k in set(d[col]) else float("nan")
        bad += ["сохранённый %s: QLIKE не сходится с %s" % (k, f)] * (not abs(q[k] / v - 1.0) <= 1e-10) if k in q.index else []
    bad += ["S-RCKf нет среди сверяемых"] * ("S-RCKf" not in set(pd.read_csv(rep / "S_D_TABLE.csv", comment="#")["series"]))
    return bad


def p_stats(t: dict, inp: dict) -> list:
    st, rv, F, bad = t["stats"], inp["rv"], inp["F"], []
    idx = S.stat_idx(C.N_D, SPEC["B"], SPEC["block"], SPEC["seed"])
    m = st[st["kind"] == "main"]
    bad += ["семейство Холма ≠ 6 гипотезам S-RCD"] * (len(m) != 6 or set(m["series"]) - {R, "X1_R(S-RCD)"} != set())
    bad += ["N Холма ≠ 858 / 3421 или N ≠ ключам регистрации"] * (not ((st["n_holm"] == SPEC["n"]).all() and (st["n_holm_wide"] == SPEC["nw"]).all() and (inp["N"], inp["N_wide"]) == (SPEC["n"], SPEC["nw"])))
    for col, N in (("p_holm_N", SPEC["n"]), ("p_holm_Nwide", SPEC["nw"]), ("p_holm_m", 6)):
        bad += ["%s ≠ независимому Холму" % col] * (not np.allclose(m[col], holm_indep(m["p_dm"].to_numpy(), N), rtol=1e-12, atol=0))
    dd = st[st["kind"] == "decomp"]
    bad += ["Холм разложения ≠ независимому по своим 4 p"] * (len(dd) != 4 or not all(np.allclose(dd[c], holm_indep(dd["p_dm"].to_numpy(), n), rtol=1e-12) for c, n in (("p_holm_N", SPEC["n"]), ("p_holm_m", 4))))
    for _, r in st[st["window"] == "M"].iterrows():
        qh, qb = q_indep(rv, F[r["series"]]), q_indep(rv, F[r["base"]])
        bad += ["Δ %s к %s: знак или база" % (r["series"], r["base"])] * (abs(r["delta_pct"] - 100.0 * (qh - qb) / qb) > 1e-9)
        ci = S.boot_ci(S.Q.ql(rv, F[r["series"]]), S.Q.ql(rv, F[r["base"]]), idx)
        bad += ["интервал %s к %s ≠ независимому бутстрапу" % (r["series"], r["base"])] * (abs(ci["lo90"] - r["lo90"]) > 1e-9 or abs(ci["hi90"] - r["hi90"]) > 1e-9)
    for _, r in m.iterrows():
        conf = bool(r["delta_pct"] < 0 and r["p_holm_N"] < 0.05 and r["p_spa_c"] < 0.05)
        bad += ["вердикт %s: «подтверждено» ≠ правилу (№ 158)" % r["hypothesis"]] * (r["verdict"].startswith("лучше") != conf)
        bad += ["вердикт %s: Δ ≥ 0 названо «лучше» (№ 125)" % r["hypothesis"]] * (r["delta_pct"] >= 0 and "численно не лучше" not in r["verdict"])
    a = st[st["kind"] == "active"]
    bad += ["активное окно: n ≠ 1549 или даты"] * (not ((a["n"] == SPEC["win"]).all() and (a["first"] == "2020-08-04").all() and (a["last"] == C.LAST).all()))
    return bad


def _set_ok(rv, F, tp, b, lay) -> bool:
    yy, lz, lb = tp.year.to_numpy(), S.Q.ql(rv, F[R]), S.Q.ql(rv, F[b])
    dl = lambda m: 100.0 * float(lz[m].mean() - lb[m].mean()) / float(lb[m].mean())
    mk = SS.period_masks(tp)
    c = SS.contrib(lz, lb, mk)
    best = min(c, key=c.get)
    loyo = 100.0 * float((lz[~mk[best]] - lb[~mk[best]]).sum()) / float(lb[~mk[best]].sum())
    return bool(dl(yy <= 2021) <= 0.415 and dl(yy >= 2022) <= 0.50 and dl(yy == 2018) <= 1.00 and loyo <= 0 and trim_indep(lz, lb) < 0 and trim_indep(S.Q.ql(rv, F["X1_R(S-RCD)"]), S.Q.ql(rv, F[lay])) < 0)


def p_dec(t: dict, inp: dict) -> list:
    rv, F, tp, bad = inp["rv"], inp["F"], inp["tp"], []
    q = {k: q_indep(rv, F[k]) for k in (R, "S-CDF", "S-RCK", "W5-G", "S-CD", "S-CK", "S-CKF")}
    d, y, st = t["dec"].set_index("id"), t["years"], t["stats"]
    h1, lt = q[R] < q["S-CDF"], q[R] < q["S-RCK"]
    bad += ["H-RCD1 ≠ независимому (база должна быть S-CDF)"] * (bool(d.loc["H-RCD1", "passed"]) != h1)
    bad += ["H-RCD1: Δ не в % QLIKE S-CDF"] * (abs(d.loc["H-RCD1", "value"] - 100.0 * (q[R] - q["S-CDF"]) / q["S-CDF"]) > 1e-9)
    oks = [_set_ok(rv, F, tp, b, lay) for _, b, lay in GD.SETS]
    for (name, *_), ok in zip(GD.SETS, oks):
        bad += ["набор %s: итог ≠ независимому" % name] * (bool(d.loc["У " + name, "passed"]) != ok)
    g = lambda b: st[(st["kind"] == "main") & (st["hypothesis"] == "%s|%s" % (R, b))].iloc[0]
    c1 = bool(h1 and g("S-CDF")["p_holm_N"] < 0.05 and g("S-CDF")["p_spa_c"] < 0.05)
    ck = bool(lt and g("S-RCK")["p_holm_N"] < 0.05 and g("S-RCK")["p_spa_c"] < 0.05)
    want = "1" if h1 and lt and all(oks) and ck else "2" if h1 and oks[0] else "3" if h1 else "4"
    bad += ["RCD-DEC %s ≠ независимому %s" % (d.loc["RCD-DEC", "value"], want)] * (str(d.loc["RCD-DEC", "value"]) != want)
    bad += ["подтверждение H-RCD1 ≠ независимому"] * (bool(d.loc["подтверждение H-RCD1", "passed"]) != c1)
    bad += ["подтверждение над S-RCK ≠ независимому"] * (bool(d.loc["подтверждение над S-RCK", "passed"]) != ck)
    e = lambda k: 100.0 * (q[k] - q["W5-G"]) / q["W5-G"]
    s = e(R) / e("S-RCK") if e("S-RCK") < 0 else float("nan")
    bad += ["доля s ≠ независимой"] * (not np.isclose(float(d.loc["доля s", "value"]), s, rtol=1e-9, equal_nan=True))
    bad += ["строка «S-RCD против S-RCK»: passed ≠ QLIKE(S-RCD) < QLIKE(S-RCK)"] * (bool(d.loc["S-RCD против S-RCK", "passed"]) != lt)
    bad += ["строка «S-RCD против S-RCK»: Δ не в % QLIKE(S-RCK)"] * (abs(float(d.loc["S-RCD против S-RCK", "value"]) - 100.0 * (q[R] - q["S-RCK"]) / q["S-RCK"]) > 1e-9)
    bad += ["строка «доля s»: passed ≠ (s ≥ 0,5)"] * (bool(d.loc["доля s", "passed"]) != bool(s >= 0.5))
    bad += ["«превзошёл» без подтверждения или без набора Б"] * (str(d.loc["RCD-DEC", "value"]) == "1" and not (ck and oks[1]))
    bad += ["Σ вкладов %s ≠ Δ: %s к %s" % (kd, h, b) for (h, b, kd), gr in y[y["base"] != "-"].groupby(["series", "base", "kind"]) if abs(gr["contrib_pct"].sum() - gr["delta_total_pct"].iloc[0]) > 1e-12]
    return bad


def p_ident(t: dict, inp: dict) -> list:
    rv, F, bad = inp["rv"], inp["F"], []
    dm = {(s, i): v for s, i, v in zip(t["decomp"]["section"], t["decomp"]["item"], t["decomp"]["value"])}   # ключи (section, item) без повторов — в разделах e_*, ident_*
    for w, m in (("D_M", np.ones(len(rv), bool)), ("active", DC.act_mask(inp))):
        e = {k: 100.0 * (float(S.Q.ql(rv, F[k])[m].sum()) - float(S.Q.ql(rv, F["W5-G"])[m].sum())) / float(S.Q.ql(rv, F["W5-G"])[m].sum()) for k in DC.SER}
        bad += ["e(%s) %s ≠ независимому" % (k, w) for k in e if abs(dm[("e_" + w, k)] - e[k]) > 1e-9]
        bad += ["тождество %s %s > 1e-12" % (k, w) for k in ("id1_gap", "id2_gap", "id3_gap") if abs(dm[("ident_" + w, k)]) > 1e-12]
        want = e["S-RCK"] - e["S-RCD"] + e["S-RCD"] - e["S-CDF"] + e["S-CDF"] - e["S-CD"] + e["S-CD"]
        bad += ["тождество 1 не сходится с e(S-RCK) (%s)" % w] * (abs(want - e["S-RCK"]) > 1e-9)
        bad += ["слагаемое «R на C-D» не e(S-RCD) − e(S-CDF) (%s)" % w] * (abs(dm[("ident_" + w, "id1_R_on_CD")] - (e["S-RCD"] - e["S-CDF"])) > 1e-9)
    return bad


def p_slices(t: dict, inp: dict) -> list:
    rv, F, tp, c, bad = inp["rv"], inp["F"], inp["tp"], t["conc"], []
    lz, lb = S.Q.ql(rv, F[R]), S.Q.ql(rv, F["S-CDF"])
    k = ~(tp == pd.Timestamp("2022-02-24"))
    got = float(c[(c["series"] == R) & (c["base"] == "S-CDF") & (c["item"] == "no_day")]["value"].iloc[0])
    bad += ["срез без 2022-02-24 ≠ независимому"] * (abs(got - 100.0 * float((lz - lb)[k].sum()) / float(lb[k].sum())) > 1e-9)
    return bad + ["обрезка в основной Δ"] * (not (t["stats"][t["stats"]["kind"] == "main"]["n"] == C.N_D).all())


def p_key(t: dict, rep, ckpt, tx: dict) -> list:
    cp = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}
    k, bad = cp.get(C.KEY), []
    if not k:
        return ["ключа S.RCD.eval нет"]
    pt = lambda s: dt.datetime.strptime(s, "%d.%m.%Y %H:%M:%S")
    bad += ["S.RCD.eval не позже регистрации S.RCD.registered"] * (not ("S.RCD.registered" in cp and pt(cp["S.RCD.registered"]["registered_at"]) < pt(k["written_at"])))
    first = min((dt.datetime.fromtimestamp((rep / n).stat().st_mtime) for n in tx if (rep / n).exists()), default=None)
    bad += ["ключ S.RCD.eval позже самой ранней таблицы"] * (first is not None and pt(k["written_at"]) > first.replace(microsecond=0) + dt.timedelta(seconds=1))
    bad += ["хэш таблицы %s ≠ ключу" % n for n, v in tx.items() if k["outputs"].get(n) != hashlib.sha256(v.encode("utf-8")).hexdigest()]
    bad += ["sha S_RCD_NULL.csv ≠ ключу"] * (k["inputs"].get("docs/otchety/harj_struct/S_RCD_NULL.csv") != hashlib.sha256(RN.OUT.read_bytes()).hexdigest())
    return bad + ["первая строка таблицы не пометка"] * (not all(v.startswith("# исследовательский результат на раскрытой истории, не OOS; гипотеза выбрана после просмотра результатов S-R, S-D, S-L, S-Z") for v in tx.values()))


def problems(t: dict, inp: dict, rep, ckpt, tx: dict) -> list:
    return p_const() + p_qlike(t, inp) + p_saved(t, rep) + p_stats(t, inp) + p_dec(t, inp) + p_ident(t, inp) + p_slices(t, inp) + RN.check() + p_key(t, rep, ckpt, tx)
