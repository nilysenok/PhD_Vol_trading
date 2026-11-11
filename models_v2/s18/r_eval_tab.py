"""S-R-EVAL (Ж-254; `S_R_SPEC.md`, разд. 4.1, 4.4, 4.5): таблицы сравнения и статистики — чистые функции по массивам; ничего не пишут. Обобщение `cm_eval_tab`: потеря, DM NW-8, бутстрап, Холм, SPA — те же функции `a_eval_*`.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS. Константы N, m — этого файла (из спецификации), не из данных. QLIKE окна = среднее дневных потерь на общих датах D_M (не среднее годовых, № 160); обрезки в оценке нет.
Публичное API:
- `CANDS` = (S-RA2, S-RCK) в порядке рассмотрения; `ABL` — абляции кандидата (Rb, Rc); `OLD` — прежний ряд той же архитектуры (S-A2, S-CK); `RVARS` — 6 вариантов R; `BASES` = (X1, W5-G, K2, K3); `REFS` = (B_mix, ZB);
  `MAIN` = (K1, K2, K3, X1, W5-G, W8-G, S-A2, S-CK) + RVARS; `N_HOLM` = 846, `N_WIDE` = 3397, `M_FAMILY` = 8, `M_PROC` = 2; `KTRIM` = 22 (= ⌈0,01·2198⌉).
- `hkey(cand, base)`: X1_R(cand) — слой доверия над C_R — для базы X1 (М1); F — для прочих. `losses`, `table(rv, tp, F)`: QLIKE D_M всех рядов и слоёв X1_R, Δ к X1, W5-G, K2, K3, даты и n.
- `trim_delta(lh, lb, k)`: Δ_trim — Δ в % QLIKE базы после удаления k дат с наименьшим и k с наибольшим d_t = l_h − l_b; база процентов — QLIKE базы на оставшихся датах. `share_top(lh, lb, k)`: доля выигрыша от k лучших дат (диагностика).
- `spa_blocks(ls)`: SPA / StepM по {S-RA2, S-RCK} для каждой базы; `proc_spa(ls)`: SPA с одним кандидатом F_RA2 против F_A2 (П1), F_RCK против F_CK (П2).
- `stats(rv, tp, F, ls, spa, pspa)`: kind main (8 гипотез: Δ, 90 % интервал, p_boot, DM NW-8 и NW-5, Холм m = 8 / N / N_wide, p SPA, p StepM, verdict), window (h1, h2), proc (П1, П2; Холм m = 2 / N / N_wide), ablation (R − Rb, R − Rc: F и слой; Rb − прежний; R − прежний), reference (против W8-G, K1, B_mix, ZB).
- `verdict(delta, p_holm, p_spa)`: «подтверждено» только при Δ < 0, p Холма (N программы) < 0,05 и p SPA < 0,05 (№ 158); минус — лучше (№ 125 / 140)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from models_v2.s18 import a_eval_spa as SP
from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import a_eval_tab as T

CANDS, RVARS = ("S-RA2", "S-RCK"), ("S-RA2", "S-RCK", "S-RA2b", "S-RCKb", "S-RA2c", "S-RCKc")
ABL, OLD = {"S-RA2": ("S-RA2b", "S-RA2c"), "S-RCK": ("S-RCKb", "S-RCKc")}, {"S-RA2": "S-A2", "S-RCK": "S-CK"}
BASES, REFS = ("X1", "W5-G", "K2", "K3"), ("B_mix", "ZB")
MAIN = ("K1", "K2", "K3", "X1", "W5-G", "W8-G", "S-A2", "S-CK") + RVARS
N_HOLM, N_WIDE, M_FAMILY, M_PROC, KTRIM = 846, 3397, 8, 2, 22
PCOLS = ("p_spa_l", "p_spa_c", "p_spa_u", "p_stepm", "spa_t", "p_holm_m", "p_holm_N", "p_holm_Nwide")
verdict, losses = T.verdict, T.losses
assert KTRIM == math.ceil(0.01 * 2198)


def hkey(cand: str, base: str) -> str:
    return "X1_R(%s)" % cand if base == "X1" else cand


OLDL = {"S-A2": "X1_A(S-A2)", "S-CK": "X1_V(S-CK)"}


def lkey(s: str) -> str:
    """Слой доверия ряда: X1_R(...) у R-вариантов, X1_A / X1_V у прежних S-A2 / S-CK."""
    return OLDL[s] if s in OLDL else hkey(s, "X1")


def table(rv, tp, F) -> pd.DataFrame:
    tp, rows = pd.DatetimeIndex(tp), []
    q = {k: float(T.Q.ql(rv, F[k]).mean()) for k in F}
    for k in list(MAIN) + list(REFS) + ["X1_R(%s)" % v for v in RVARS] + list(OLDL.values()):
        role = "main" if k in MAIN else "layer X1" if k.startswith("X1_") else "reference"
        rows.append(dict(role=role, series=k, n=len(tp), first_tplus=str(tp.min().date()), last_tplus=str(tp.max().date()), qlike=q[k], **{"d_%s_pct" % b: 100.0 * (q[k] - q[b]) / q[b] for b in BASES}))
    return pd.DataFrame(rows)


def trim_delta(lh, lb, k: int = KTRIM) -> float:
    d = np.asarray(lh, float) - np.asarray(lb, float)
    keep = np.ones(len(d), bool)
    o = np.argsort(d, kind="stable")
    keep[o[:k]] = False
    keep[o[len(d) - k:]] = False
    return 100.0 * float(d[keep].sum()) / float(np.asarray(lb, float)[keep].sum())


def share_top(lh, lb, k: int = KTRIM) -> float:
    d = np.asarray(lh, float) - np.asarray(lb, float)
    g = -d.sum()
    return float(-np.sort(d)[:k].sum() / g) if g > 0 else float("nan")


def spa_blocks(ls: dict) -> dict:
    idx = S.idx_for(len(next(iter(ls.values()))))
    return {b: SP.spa_stepm(ls[b], np.column_stack([ls[hkey(c, b)] for c in CANDS]), idx) for b in BASES}


def proc_spa(ls: dict) -> dict:
    idx = S.idx_for(len(next(iter(ls.values()))))
    return {"П%d" % (i + 1): SP.spa_stepm(ls[OLD[c]], ls[c][:, None], idx) for i, c in enumerate(CANDS)}


def _holm(t: pd.DataFrame, m: np.ndarray, mm: int) -> None:
    p = t.loc[m, "p_dm"].to_numpy()
    t.loc[m, "p_holm_m"], t.loc[m, "p_holm_N"], t.loc[m, "p_holm_Nwide"] = S.holm_n(p, mm), S.holm_n(p, N_HOLM), S.holm_n(p, N_WIDE)
    t.loc[m, "m_family"] = mm


def _main_rows(tp, ls, spa) -> list:
    y, allm, rows = tp.year.to_numpy(), np.ones(len(tp), bool), []
    for c in CANDS:
        for b in BASES:
            h, hid, k = hkey(c, b), "%s|%s" % (c, b), CANDS.index(c)
            r = T._row("main", "M", hid, h, b, ls, tp, allm)
            r.update(p_spa_l=spa[b]["p_l"], p_spa_c=spa[b]["p_c"], p_spa_u=spa[b]["p_u"], p_stepm=float(spa[b]["p_step"][k]), spa_t=float(spa[b]["t"][k]))
            rows.append(r)
            rows += [T._row("window", w, hid, h, b, ls, tp, m) for w, m in (("h1_2018-2021", y <= 2021), ("h2_2022-2026", y >= 2022))]
    return rows


def _proc_rows(tp, ls, pspa) -> list:
    allm, rows = np.ones(len(tp), bool), []
    for i, c in enumerate(CANDS):
        r = T._row("proc", "M", "П%d" % (i + 1), c, OLD[c], ls, tp, allm)
        s = pspa["П%d" % (i + 1)]
        r.update(p_spa_l=s["p_l"], p_spa_c=s["p_c"], p_spa_u=s["p_u"], p_stepm=float(s["p_step"][0]), spa_t=float(s["t"][0]))
        rows.append(r)
    return rows


def _other_rows(tp, ls) -> list:
    allm, rows = np.ones(len(tp), bool), []
    for c in CANDS:
        rb, rc = ABL[c]
        for lab, (h, b) in {"R − Rb": (c, rb), "R − Rc": (c, rc), "Rb − прежний": (rb, OLD[c]), "R − прежний": (c, OLD[c])}.items():
            rows += [T._row("ablation", "M", "%s %s %s" % (c, lab, kind), hk, bk, ls, tp, allm) for kind, hk, bk in (("F", h, b), ("слой X1_R", lkey(h), lkey(b)))]
        rows += [T._row("reference", "M", "%s|%s" % (c, b), c, b, ls, tp, allm) for b in ("W8-G", "K1", "B_mix", "ZB")]
    return rows


def stats(rv, tp, F, ls, spa, pspa) -> pd.DataFrame:
    tp = pd.DatetimeIndex(tp)
    t = pd.DataFrame(_main_rows(tp, ls, spa) + _proc_rows(tp, ls, pspa) + _other_rows(tp, ls))
    for k in PCOLS:
        if k not in t:
            t[k] = np.nan
    t["m_family"] = np.nan
    for kind, mm in (("main", M_FAMILY), ("proc", M_PROC)):
        _holm(t, (t["kind"] == kind).to_numpy(), mm)
    t["n_holm"], t["n_holm_wide"], t["B"] = N_HOLM, N_WIDE, S.B
    t["verdict"] = [verdict(d, ph, ps) if k == "main" else "см. S_R_PROC.csv (П ∧ p Холма N < 0,05 ∧ p SPA < 0,05)" if k == "proc" else "справочно, критериев не получает"
                    for d, ph, ps, k in zip(t["delta_pct"], t["p_holm_N"], t["p_spa_c"], t["kind"])]
    return t


def spa_table(spa: dict, pspa: dict) -> pd.DataFrame:
    rows = []
    for b in BASES:
        r = spa[b]
        rows.append(dict(kind="family", base=b, benchmark=b, candidates="|".join(hkey(c, b) for c in CANDS), T=r["T"], p_l=r["p_l"], p_c=r["p_c"], p_u=r["p_u"], B=S.B, block=S.MEAN_BLOCK, note="основной p SPA этапа — p_c; бенчмарк согласован со слоем кандидатов"))
        rows += [dict(kind="candidate", base=b, benchmark=b, candidates=hkey(c, b), t_stat=float(r["t"][k]), dbar=float(r["dbar"][k]), omega=float(r["omega"][k]), p_stepm=float(r["p_step"][k]), stepm_reject=bool(r["reject"][k]))
                 for k, c in enumerate(CANDS)]
    for i, c in enumerate(CANDS):
        r = pspa["П%d" % (i + 1)]
        rows.append(dict(kind="proc", base=OLD[c], benchmark=OLD[c], candidates=c, T=r["T"], p_l=r["p_l"], p_c=r["p_c"], p_u=r["p_u"], t_stat=float(r["t"][0]), dbar=float(r["dbar"][0]), omega=float(r["omega"][0]), B=S.B, block=S.MEAN_BLOCK,
                         note="П%d: SPA с одним кандидатом %s против %s; основной p — p_c" % (i + 1, c, OLD[c])))
    return pd.DataFrame(rows)
