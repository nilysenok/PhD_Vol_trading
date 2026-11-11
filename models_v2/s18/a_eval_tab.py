"""S-A-EVAL (Ж-254; `S_A_SPEC.md`, разд. 6.1, 6.3, 8): таблицы сравнения и статистики — чистые функции по массивам; ничего не пишут. Константы N, m, пороги — этого файла (из спецификации), не из данных.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS. Потеря — `v6_0_q.ql`; QLIKE окна = среднее дневных потерь на общих датах D_M (не среднее показателей периодов, № 160).
Публичное API:
- `CANDS` = (S-A0, S-A1, S-A2); `VARS` = CANDS + (S-A2p,); `BASES` = (X1, W5-G, K2, K3); `MAIN` = (K1, K2, K3, X1, W5-G, W8-G) + VARS; `REFS` = (B_mix, ZB); `N_HOLM` = 831, `N_WIDE` = 3367, `M_FAMILY` = 12.
- `hkey(cand, base)`: ряд, сравниваемый с базой: X1_A(cand) — слой доверия над C_A — для базы X1 (М1); F_A (cand) — для прочих (М2 — W5-G, K2, K3). Несогласованные слои запрещены.
- `losses(rv, F)`; `table(rv, tp, F)`: QLIKE D_M всех рядов (и слоёв X1_A), Δ к X1, W5-G, K2, K3, даты и n.
- `spa_blocks(ls) -> {база: spa_stepm}`: SPA / StepM по кандидатам {A0, A1, A2} для каждой базы, бенчмарк = база, ряды согласованы (`hkey`).
- `stats(rv, tp, F, ls, spa) -> DataFrame`: kind main (12 гипотез, окно M: Δ, 90 % интервал, p_boot, DM NW-8 и NW-5, Холм m = 12, N = 831, N_wide = 3367 по p DM NW-8, p SPA, p StepM, verdict), window (h1, h2 тех же 12; лаг по их n; без Холма), reference (F кандидатов против W8-G, K1, B_mix, ZB), ablation (A1−A0, A2−A1, A2−A2p, A2−A0; F и X1_A).
- `verdict(delta, p_holm, p_spa)`: «подтверждено» только при Δ < 0 и p Холма (N программы) < 0,05 и p SPA < 0,05 (№ 158); знак: минус — лучше (№ 125/140).
- `spa_table(spa) -> DataFrame`: семейство (T, p_l, p_c, p_u) и кандидаты (t, d̄, ω̂, p StepM, отвергнут)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q
from models_v2.s18 import a_eval_spa as SP
from models_v2.s18 import a_eval_stat as S

CANDS, VARS, BASES = ("S-A0", "S-A1", "S-A2"), ("S-A0", "S-A1", "S-A2", "S-A2p"), ("X1", "W5-G", "K2", "K3")
MAIN, REFS = ("K1", "K2", "K3", "X1", "W5-G", "W8-G") + VARS, ("B_mix", "ZB")
N_HOLM, N_WIDE, M_FAMILY = 831, 3367, 12
ABL = (("S-A1", "S-A0"), ("S-A2", "S-A1"), ("S-A2", "S-A2p"), ("S-A2", "S-A0"))


def hkey(cand: str, base: str) -> str:
    return "X1_A(%s)" % cand if base == "X1" else cand


def losses(rv, F) -> dict:
    return {k: Q.ql(rv, np.asarray(v, float)) for k, v in F.items()}


def table(rv, tp, F) -> pd.DataFrame:
    tp, rows = pd.DatetimeIndex(tp), []
    q = {k: float(Q.ql(rv, F[k]).mean()) for k in F}
    names = list(MAIN) + list(REFS) + ["X1_A(%s)" % c for c in VARS]
    for k in names:
        role = "main" if k in MAIN else "layer X1_A" if k.startswith("X1_A") else "reference"
        rows.append(dict(role=role, series=k, n=len(tp), first_tplus=str(tp.min().date()), last_tplus=str(tp.max().date()), qlike=q[k],
                         **{"d_%s_pct" % b: 100.0 * (q[k] - q[b]) / q[b] for b in BASES}))
    return pd.DataFrame(rows)


def spa_blocks(ls: dict) -> dict:
    idx = S.idx_for(len(next(iter(ls.values()))))
    return {b: SP.spa_stepm(ls[b], np.column_stack([ls[hkey(c, b)] for c in CANDS]), idx) for b in BASES}


def _row(kind, win, hid, h, b, ls, tp, m) -> dict:
    lh, lb, n = ls[h][m], ls[b][m], int(m.sum())
    lag, t = S.nw_lag(n), tp[m]
    d8, d5, ci = S.dm_nw(lb, lh, lag), S.dm_nw(lb, lh, S.LAG_REF), S.boot_ci(lh, lb, S.idx_for(n))
    return dict(kind=kind, window=win, hypothesis=hid, series=h, base=b, n=n, first=str(t.min().date()), last=str(t.max().date()), lag=lag, qlike_h=float(lh.mean()), qlike_b=float(lb.mean()),
                **ci, dm_stat=d8["stat"], p_dm=d8["p_one"], p_dm_lag5=d5["p_one"])


def verdict(delta: float, p_holm: float, p_spa: float) -> str:
    if not delta < 0:
        return "численно не лучше базы; не подтверждено"
    ok = bool(p_holm < S.ALPHA and p_spa < S.ALPHA)
    return "численно лучше базы; " + ("подтверждено (p Холма N < 0,05 и p SPA < 0,05)" if ok else "не подтверждено (p Холма N ≥ 0,05 или p SPA ≥ 0,05)")


def stats(rv, tp, F, ls, spa) -> pd.DataFrame:
    tp, y = pd.DatetimeIndex(tp), pd.DatetimeIndex(tp).year.to_numpy()
    allm, rows = np.ones(len(tp), bool), []
    for c in CANDS:
        for j, b in enumerate(BASES):
            h, hid = hkey(c, b), "%s|%s" % (c, b)
            r = _row("main", "M", hid, h, b, ls, tp, allm)
            r.update(p_spa_l=spa[b]["p_l"], p_spa_c=spa[b]["p_c"], p_spa_u=spa[b]["p_u"], p_stepm=float(spa[b]["p_step"][CANDS.index(c)]), spa_t=float(spa[b]["t"][CANDS.index(c)]))
            rows.append(r)
            rows += [_row("window", w, hid, h, b, ls, tp, m) for w, m in (("h1_2018-2021", y <= 2021), ("h2_2022-2026", y >= 2022))]
    for c in CANDS + ("S-A2p",):
        rows += [_row("reference", "M", "%s|%s" % (c, b), c, b, ls, tp, allm) for b in ("W8-G", "K1", "B_mix", "ZB")]
    for h, b in ABL:
        rows += [_row("ablation", "M", "%s|%s %s" % (h, b, lab), hkey(h, "X1") if lab == "слой X1_A" else h, hkey(b, "X1") if lab == "слой X1_A" else b, ls, tp, allm) for lab in ("F", "слой X1_A")]
    t = pd.DataFrame(rows)
    for k in ("p_holm_m12", "p_holm_N", "p_holm_Nwide"):
        t[k] = np.nan
    m = (t["kind"] == "main").to_numpy()
    p = t.loc[m, "p_dm"].to_numpy()
    t.loc[m, "p_holm_m12"], t.loc[m, "p_holm_N"], t.loc[m, "p_holm_Nwide"] = S.holm_n(p, M_FAMILY), S.holm_n(p, N_HOLM), S.holm_n(p, N_WIDE)
    t["n_holm"], t["n_holm_wide"], t["m_family"], t["B"] = N_HOLM, N_WIDE, M_FAMILY, S.B
    t["verdict"] = [verdict(d, ph, ps) if k == "main" else "справочно, критериев не получает" for d, ph, ps, k in zip(t["delta_pct"], t["p_holm_N"], t["p_spa_c"], t["kind"])]
    return t


def spa_table(spa: dict) -> pd.DataFrame:
    rows = []
    for b in BASES:
        r = spa[b]
        rows.append(dict(kind="family", base=b, benchmark=b, candidates="|".join(hkey(c, b) for c in CANDS), T=r["T"], p_l=r["p_l"], p_c=r["p_c"], p_u=r["p_u"], B=S.B, block=S.MEAN_BLOCK,
                         note="основной p SPA этапа — p_c (центровка Hansen consistent); бенчмарк согласован со слоем кандидатов"))
        rows += [dict(kind="candidate", base=b, benchmark=b, candidates=hkey(c, b), t_stat=float(r["t"][k]), dbar=float(r["dbar"][k]), omega=float(r["omega"][k]), p_stepm=float(r["p_step"][k]),
                      stepm_reject=bool(r["reject"][k])) for k, c in enumerate(CANDS)]
    return pd.DataFrame(rows)
