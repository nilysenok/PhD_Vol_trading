"""S-RCD-EVAL (Ж-259; `S_RCD_SPEC.md`, разд. 6.1, 7.1, 7.4, 7.5): таблицы сравнения и статистики — чистые функции по массивам; ничего не пишут. Обвязка — `a_eval_stat`, `a_eval_spa`, `a_eval_tab._row` без правок.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Исследовательский результат на раскрытой истории, не OOS. Константы N, m — `rd_eval_cfg`. QLIKE окна = среднее дневных потерь на общих датах D_M, без обрезки (не среднее годовых, № 160).
Публичное API:
- `FAM` — семейство Холма m = 6: S-RCD против S-CDF, S-RCK, W5-G, X1 (слой X1_R), K2, K3; `DEC4` — четыре контраста разложения (сырые p, Холм N по своим 4 p); `REF` — справочные контрасты (разд. 6.1); `SPA_BLOCKS` — (а), (б), (в), (г).
- `losses`, `table(rv, tp, F, ls)`: QLIKE D_M рядов и слоёв, Δ к S-CDF, S-RCK, W5-G, X1, K2, K3 (+ 90 % интервалы), даты, n.
- `spa_blocks(ls)`, `spa_table(spa)`: SPA / StepM (Hansen 2005 / Romano–Wolf) по блокам (а)–(г).
- `stats(rv, tp, ls, spa, N, N_wide, act)`: kind main (6 гипотез: Δ, интервал, p_boot, DM NW-8 и NW-5, Холм m = 6 / N / N_wide, p SPA, p StepM, verdict), window (h1, h2), decomp (4 контраста), reference, active (общее активное окно d ≥ 2020-08-03).
- `verdict(delta, p_holm, p_spa, base)`: формулировки разд. 7.5; «подтверждено» только при Δ < 0, p Холма (N программы) < 0,05 и p SPA < 0,05; минус — лучше. `power_table`, `mcs`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18 import a_eval_spa as SP
from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import a_eval_tab as AT
from models_v2.s18 import rd_eval_cfg as C

R, RL, CDF, CKF = "S-RCD", "X1_R(S-RCD)", "S-CDF", "S-CKF"
FAM = ((R, CDF), (R, "S-RCK"), (R, "W5-G"), (RL, "X1"), (R, "K2"), (R, "K3"))
DEC4 = ((CDF, "S-CD"), (CKF, "S-CK"), ("S-RCK", CKF), (CKF, CDF))
REF = (("S-CK", "S-CD"), ("S-RCK", "S-RCKf"), ("S-RCK", R), (RL, "X1_V(S-CDF)"), (RL, "X1_R(S-RCK)"), ("S-RCK", "W5-G"))
SPA_BLOCKS = {"a": (CDF, (R,)), "b": ("S-RCK", (R,)), "c": ("W5-G", (R, CDF, CKF)), "d": ("X1", (RL, "X1_V(S-CDF)", "X1_V(S-CKF)"))}
SPA_OF = {(R, CDF): "a", (R, "S-RCK"): "b", (R, "W5-G"): "c", (RL, "X1"): "d"}
PCOLS = ("p_spa_c", "p_spa_block", "p_stepm", "spa_t", "p_holm_m", "p_holm_N", "p_holm_Nwide")
losses = AT.losses


def verdict(delta: float, p_holm: float, p_spa: float, base: str) -> str:
    if not delta < 0:
        return "численно не лучше %s; не подтверждено" % base
    if p_holm < S.ALPHA and p_spa < S.ALPHA:
        return "лучше %s; подтверждено (p Холма N < 0,05 и p SPA < 0,05)" % base
    return "численно лучше %s; не подтверждено (p Холма N ≥ 0,05 или p SPA ≥ 0,05)" % base


def table(rv, tp, F: dict, ls: dict) -> pd.DataFrame:
    tp, rows, idx = pd.DatetimeIndex(tp), [], S.idx_for(len(tp))
    q = {k: float(ls[k].mean()) for k in F}
    for k in list(C.SHOWN) + ["X1_R(S-RCD)", "X1_V(S-CDF)", "X1_V(S-CKF)", "X1_R(S-RCK)"]:
        r = dict(role="layer X1" if k.startswith("X1_") else "main", series=k, n=len(tp), first_tplus=str(tp.min().date()), last_tplus=str(tp.max().date()), qlike=q[k])
        for b in C.BASES:
            ci = S.boot_ci(ls[k], ls[b], idx) if k != b else dict(delta_pct=0.0, lo90=np.nan, hi90=np.nan)
            r["d_%s_pct" % b], r["lo90_%s" % b], r["hi90_%s" % b] = 100.0 * (q[k] - q[b]) / q[b], ci["lo90"], ci["hi90"]
        rows.append(r)
    return pd.DataFrame(rows)


def spa_blocks(ls: dict) -> dict:
    idx = S.idx_for(len(next(iter(ls.values()))))
    return {k: SP.spa_stepm(ls[b], np.column_stack([ls[c] for c in cs]), idx) for k, (b, cs) in SPA_BLOCKS.items()}


def spa_table(spa: dict) -> pd.DataFrame:
    rows = []
    for k, (b, cs) in SPA_BLOCKS.items():
        r = spa[k]
        rows.append(dict(kind="family", block=k, benchmark=b, candidates="|".join(cs), T=r["T"], p_l=r["p_l"], p_c=r["p_c"], p_u=r["p_u"], B=S.B, mean_block=S.MEAN_BLOCK, note="основной p SPA — p_c (Hansen consistent); бенчмарк = база"))
        rows += [dict(kind="candidate", block=k, benchmark=b, candidates=c, t_stat=float(r["t"][i]), dbar=float(r["dbar"][i]), omega=float(r["omega"][i]), p_stepm=float(r["p_step"][i]), stepm_reject=bool(r["reject"][i])) for i, c in enumerate(cs)]
    return pd.DataFrame(rows)


def _fam_rows(tp, ls, spa) -> list:
    y, allm, rows = tp.year.to_numpy(), np.ones(len(tp), bool), []
    idx = S.idx_for(len(tp))
    for h, b in FAM:
        r = AT._row("main", "M", "%s|%s" % (h, b), h, b, ls, tp, allm)
        k = SPA_OF.get((h, b))
        s = spa[k] if k is not None else SP.spa_stepm(ls[b], ls[h][:, None], idx)
        r.update(p_spa_c=s["p_c"], p_spa_block=k or "одиночный", p_stepm=float(s["p_step"][0]), spa_t=float(s["t"][0]))
        rows.append(r)
        rows += [AT._row("window", w, "%s|%s" % (h, b), h, b, ls, tp, m) for w, m in (("h1_2018-2021", y <= 2021), ("h2_2022-2026", y >= 2022))]
    return rows


def _dec_rows(tp, ls, N, N_wide) -> list:
    allm = np.ones(len(tp), bool)
    rows = [AT._row("decomp", "M", "%s|%s" % hb, hb[0], hb[1], ls, tp, allm) for hb in DEC4]
    p = np.array([r["p_dm"] for r in rows])
    for r, a, b, c in zip(rows, S.holm_n(p, len(DEC4)), S.holm_n(p, N), S.holm_n(p, N_wide)):
        r.update(p_holm_m=a, p_holm_N=b, p_holm_Nwide=c, m_family=len(DEC4))
    return rows


def stats(rv, tp, ls, spa, N: int, N_wide: int, act) -> pd.DataFrame:
    tp, allm = pd.DatetimeIndex(tp), np.ones(len(tp), bool)
    pairs = FAM[:3] + DEC4 + REF[:3]
    t = pd.DataFrame(_fam_rows(tp, ls, spa) + _dec_rows(tp, ls, N, N_wide) + [AT._row("reference", "M", "%s|%s" % hb, hb[0], hb[1], ls, tp, allm) for hb in REF]
                     + [AT._row("active", "d>=" + C.ACTIVE_FROM, "%s|%s" % hb, hb[0], hb[1], ls, tp, act) for hb in pairs])
    for k in PCOLS:
        t[k] = t[k] if k in t else np.nan
    m = (t["kind"] == "main").to_numpy()
    p = t.loc[m, "p_dm"].to_numpy()
    t.loc[m, "p_holm_m"], t.loc[m, "p_holm_N"], t.loc[m, "p_holm_Nwide"] = S.holm_n(p, C.M_FAMILY), S.holm_n(p, N), S.holm_n(p, N_wide)
    t["m_family"] = t["m_family"] if "m_family" in t else np.nan
    t.loc[m, "m_family"] = C.M_FAMILY
    t["n_holm"], t["n_holm_wide"], t["B"] = N, N_wide, S.B
    note = {"reference": "справочно, критериев не получает; p сырой", "window": "справочно (половина периода)", "decomp": "справочно (контраст разложения); Холм по 4 p", "active": "справочно (общее активное окно)"}
    t["verdict"] = [verdict(d, ph, ps, b) if k == "main" else note[k] for d, ph, ps, b, k in zip(t["delta_pct"], t["p_holm_N"], t["p_spa_c"], t["base"], t["kind"])]
    return t


def power_table(tp, ls: dict) -> pd.DataFrame:
    y, rows = pd.DatetimeIndex(tp).year.to_numpy(), []
    for h, b in FAM[:4]:
        for w, m in (("D_M", np.ones(len(y), bool)), ("без 2022", y != 2022)):
            lag = S.nw_lag(int(m.sum()))
            rows.append(dict(series=h, base=b, window=w, n=int(m.sum()), lag=lag, **S.power(ls[h][m] - ls[b][m], float(ls[b][m].mean()), lag)))
    return pd.DataFrame(rows)


def mcs(rv, F: dict) -> pd.DataFrame:
    return S.mcs(rv, {k: F[k] for k in C.MCS_SET})
