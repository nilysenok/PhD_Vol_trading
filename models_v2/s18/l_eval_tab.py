"""S-L-EVAL (Ж-256 п. 5; `S_L_SPEC.md`, разд. 5): таблицы оценки S-RL1 и контролей S-RL1a, S-RL1b — чистые функции по массивам; ничего не пишут.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD; исследовательский результат, не OOS. Основная оценка — обычная QLIKE (среднее дневных потерь, без обрезки) на всех 2198 датах D_M; Δ = 100·(QLIKE(ряд) − QLIKE(база))/QLIKE(база), минус — лучше.
Константы — из спецификации (не из результата): `N_HOLM` берётся из ключа `S.L.registered` (851 / 3407), `M_FAMILY` = 4, `PL1` = 0,90, `PL2` = (−0,063; −0,015), `KTRIM` = 22, пороги У — `a_eval_yr.THR`.
Публичное API: `table`, `deltas_ext(ls, tp, N, N_wide)` (Δ, 90 % интервал, DM NW-8, Холм m / N / N_wide и SPA для семейства S-RL1 против X1, W5-G, K2, K3), `verdict_ext`, `cond_rows`, `stage_rows`, `rck_verdict`, `mech_rows`, `months`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18 import a_eval_spa as SP
from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import a_eval_tab as AT
from models_v2.s18 import a_eval_yr as AY
from models_v2.s18 import d_eval_tab as T
from models_v2.s18 import r_eval_tab as RT
from models_v2.s18 import r_eval_yr as RY

KTRIM, M_FAMILY, PL1, PL2 = T.KTRIM, 4, 0.90, (-0.063, -0.015)
SERIES = ("S-RL1", "S-RL1a", "S-RL1b", "S-RL1b2", "S-RCK", "W5-G", "X1", "K2", "K3")    # (б): S-RL1b2 — основной контроль (Ж-257), S-RL1b — циклический, по спецификации, справочно
BASES = ("S-RCK", "W5-G", "X1", "K2", "K3", "S-RL1a", "S-RL1b", "S-RL1b2")
FAM = (("X1_R(S-RL1)", "X1"), ("S-RL1", "W5-G"), ("S-RL1", "K2"), ("S-RL1", "K3"))
PAIRS = FAM + (("S-RL1", "S-RCK"), ("S-RL1", "S-RL1a"), ("S-RL1", "S-RL1b2"), ("S-RL1", "S-RL1b"), ("X1_R(S-RL1b2)", "X1"), ("S-RL1b2", "W5-G"), ("X1_R(S-RL1b)", "X1"), ("S-RL1b", "W5-G"), ("S-RL1a", "W5-G"), ("S-RCK", "W5-G"), ("X1_R(S-RCK)", "X1"))
COMPS = [("X1_R(S-RL1)", "X1"), ("S-RL1", "W5-G"), ("S-RL1", "S-RCK"), ("S-RL1", "S-RL1b2")]
PAIRS_Y = [("S-RL1", "W5-G"), ("S-RL1", "S-RL1b2"), ("S-RL1", "S-RL1b"), ("S-RL1b2", "W5-G"), ("S-RL1b", "W5-G"), ("S-RL1a", "W5-G"), ("S-RL1", "S-RCK")]
PAIRS_S = PAIRS_Y[:6]
TXT54 = {"needed": "остальные 5 параметров C-K нужны: без них хуже на %+.4f %%", "worse_noise": "без 5 параметров хуже на %+.4f %%; от шума не отделено",
         "not_found": "вклад 5 параметров не обнаружен (Δ = %+.4f %%)", "harms": "5 параметров вредят: одного достаточно (Δ = %+.4f %%)"}
losses, ql, years, slices, trim_keep = T.losses, T.ql, T.years, T.slices, T.trim_keep


def table(rv, tp, F: dict) -> pd.DataFrame:
    tp, q = pd.DatetimeIndex(tp), {k: float(ql(rv, F[k]).mean()) for k in F}
    return pd.DataFrame([dict(series=k, n=len(tp), first_tplus=str(tp.min().date()), last_tplus=str(tp.max().date()), qlike=q[k], **{"d_vs_%s_pct" % b: 100.0 * (q[k] - q[b]) / q[b] for b in BASES}) for k in SERIES])


def deltas_ext(ls: dict, tp, N: int, N_wide: int) -> pd.DataFrame:
    dl = T.deltas(ls, tp, PAIRS)
    fam = np.array([(h, b) in FAM for h, b in zip(dl["series"], dl["base"])])
    dl["m_family"], dl["N"], dl["N_wide"] = np.where(fam, M_FAMILY, np.nan), N, N_wide
    for k, nn in (("p_holm_m", M_FAMILY), ("p_holm_N", N), ("p_holm_Nwide", N_wide)):
        dl[k] = np.nan
        dl.loc[fam, k] = S.holm_n(dl.loc[fam, "p_dm_nw8"].to_numpy(), nn)
    idx = S.idx_for(len(tp))
    dl["p_spa_c"] = np.nan
    for i in np.flatnonzero(fam):
        h, b = dl["series"][i], dl["base"][i]
        dl.loc[i, "p_spa_c"] = SP.spa_stepm(ls[b], ls[h][:, None], idx)["p_c"]
    dl["verdict"] = [AT.verdict(d, ph, ps) if f else "справочно, семейство не входит" for d, ph, ps, f in zip(dl["delta_pct"], dl["p_holm_N"], dl["p_spa_c"], fam)]
    return dl


def _d(dl: pd.DataFrame, h: str, b: str) -> pd.Series:
    return dl[(dl["series"] == h) & (dl["base"] == b)].iloc[0]


def cond_rows(tb: pd.DataFrame, dl: pd.DataFrame, elements: pd.DataFrame) -> pd.DataFrame:
    """Разд. 5.2: три условия; условие 3 — из `S_D_ELEMENTS.csv` (S-RCKf не лучше S-RCK), не вручную."""
    q = tb.set_index("series")["qlike"]
    el = elements[(elements["series"] == "S-RCKf") & (elements["base"] == "S-RCK")].iloc[0]
    c = [("1", "QLIKE(S-RL1) < QLIKE(W5-G)", q["S-RL1"], q["W5-G"], q["S-RL1"] < q["W5-G"]), ("2", "QLIKE(S-RL1) < QLIKE(S-RL1b2) — основной контроль (б), Ж-257", q["S-RL1"], q["S-RL1b2"], q["S-RL1"] < q["S-RL1b2"]),
         ("3", "S-RCKf не лучше S-RCK (S_D_ELEMENTS.csv: Δ > 0, код %s)" % el["code"], float(el["delta_pct"]), 0.0, bool(el["delta_pct"] > 0)),
         ("2ref", "справочно: QLIKE(S-RL1) < QLIKE(S-RL1b) — циклический контроль по спецификации", q["S-RL1"], q["S-RL1b"], q["S-RL1"] < q["S-RL1b"])]
    rows = [dict(id="C" + i, condition=t, value=v, threshold=th, passed=bool(ok)) for i, t, v, th, ok in c]
    ok = all(r["passed"] for r in rows if r["id"] != "C2ref")
    k = [r["id"][1:] for r in rows if not r["passed"] and r["id"] != "C2ref"]
    ok_ref = all(r["passed"] for r in rows if r["id"] != "C2")
    k_ref = [r["id"][1:] for r in rows if not r["passed"] and r["id"] != "C2"]
    w = _d(dl, "S-RL1", "W5-G")
    stat = bool(w["delta_pct"] < 0 and w["p_holm_N"] < S.ALPHA and w["p_spa_c"] < S.ALPHA)
    rows += [dict(id="interaction", condition="взаимодействие подтверждено численно (условия 1, 2 по S-RL1b2, 3) — основная версия, Ж-257", value="да" if ok else "не подтверждено (условие %s)" % ", ".join(k), threshold="—", passed=ok),
             dict(id="interaction_ref", condition="справочно: то же по циклическому S-RL1b (спецификация)", value="да" if ok_ref else "не подтверждено (условие %s)" % ", ".join(k_ref), threshold="—", passed=ok_ref),
             dict(id="statistical", condition="статистически: Δ < 0, p Холма (N) < 0,05 и p SPA < 0,05 (S-RL1 против W5-G)", value="p Холма %.4g; p SPA %.4g" % (w["p_holm_N"], w["p_spa_c"]), threshold=S.ALPHA, passed=stat)]
    return pd.DataFrame(rows)


def stage_rows(tp, ls: dict) -> pd.DataFrame:
    yrs, cnc, rows = AY.years(tp, ls, COMPS), AY.conc(tp, ls, COMPS), []
    ex = [dict(series=h, base=b, item=i, key="K=%d" % KTRIM, value=f(ls[h], ls[b])) for h, b in COMPS for i, f in (("trim", RT.trim_delta), ("share_top1pct", RT.share_top))]
    RY._guard_c(rows, "S-RL1", yrs, pd.concat([cnc, pd.DataFrame(ex)], ignore_index=True))
    g = pd.DataFrame(rows)
    bad = g[~g["passed"]]
    res = "проходит пороги этапа S" if bad.empty else "не прошёл (%s)" % "; ".join("%s%s: %s" % (r.id, "" if r.pair in ("—", r.id) else "|" + r.pair, AY._fmt(r.value)) for r in bad.itertuples())
    return pd.concat([g, pd.DataFrame([dict(candidate="S-RL1", pair="—", id="этап S", condition="М1, М2, У1-S…У3-S, У5-S, У6-R (S_R_SPEC.md разд. 4.1–4.3)", value=res, threshold="—", formula="выведено кодом из строк выше; статус кандидата не меняет", passed=bad.empty)])], ignore_index=True)


def rck_verdict(dl: pd.DataFrame) -> tuple:
    """Разд. 5.4: S-RL1 против S-RCK, база процентов S-RCK."""
    r = _d(dl, "S-RL1", "S-RCK")
    code = T.verdict(float(r["delta_pct"]), float(r["lo90"]), float(r["hi90"]))
    return code, TXT54[code] % r["delta_pct"]


def months(frames: dict) -> pd.DataFrame:
    out = []
    for k, f in frames.items():
        m = f[f["active"]].groupby("refit_R", sort=True).first().reset_index()
        t = m[["refit_R", "n_train", "c", "m", "lam", "a"] + ["w_" + z for z in ("1e-1", "1e-2", "1e-3")]].copy()
        t.insert(0, "series", k)
        t.insert(5, "kappa", t["c"] / t["m"])
        out.append(t)
    return pd.concat(out, ignore_index=True)
