"""S-C-EVAL (Ж-254; `S_C_SPEC.md`, разд. 4.1, 4.2, 4.4): таблицы сравнения и статистики — чистые функции по массивам; ничего не пишут. Обобщение `a_eval_tab` параметрами: потеря, DM NW-8, бутстрап, Холм, SPA — те же функции `a_eval_*`.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS. Константы N, m — этого файла (из спецификации), не из данных. QLIKE окна = среднее дневных потерь на общих датах D_M (не среднее годовых, № 160).
Публичное API:
- `CANDS` = (S-CK, S-CKA) — кандидаты в порядке рассмотрения; `SPA_SET` = CANDS + (S-CD,); `VARS` = (S-CD, S-CK, S-CKA, S-C7, S-CKL); `BASES` = (X1, W5-G, K2, K3); `MAIN` = (K1, K2, K3, X1, W5-G, W8-G, S-A2) + VARS; `REFS` = (B_mix, ZB);
  `STRUCT` = ((К1, S-CK, S-CD), (К2, S-CK, S-C7), (К3, S-CK, S-CKL)); `N_HOLM` = 836, `N_WIDE` = 3377, `M_FAMILY` = 8, `M_STRUCT` = 3.
- `hkey(cand, base)`: X1_C(cand) — слой доверия над C_V — для базы X1 (М1); F (cand) — для прочих (М2, K2, K3). Несогласованные слои запрещены.
- `losses(rv, F)`, `table(rv, tp, F)`: QLIKE D_M всех рядов и слоёв X1_C, Δ к X1, W5-G, K2, K3, даты и n.
- `spa_blocks(ls)`: SPA / StepM по {S-CK, S-CKA, S-CD} для каждой базы; `struct_spa(ls)`: SPA с одним кандидатом F_CK против F_CD / F_C7 / F_CKL.
- `stats(rv, tp, F, ls, spa, sspa)`: kind main (8 гипотез, окно M; Δ, 90 % интервал, p_boot, DM NW-8 и NW-5, Холм m = 8 / N / N_wide, p SPA, p StepM, verdict), window (h1, h2 тех же 8), struct (К1–К3; Холм m = 3 / N / N_wide), cd_base (S-CD против баз,
  Холм N по 4 p, справочно), reference (кандидаты и контроли против W8-G, K1, B_mix, ZB), ablation (S-CKA − S-CK: F и слой X1_C).
- `verdict(delta, p_holm, p_spa)`: «подтверждено» только при Δ < 0, p Холма (N программы) < 0,05 и p SPA < 0,05 (№ 158); знак: минус — лучше (№ 125 / 140). `spa_table(spa, sspa)`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18 import a_eval_spa as SP
from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import a_eval_tab as T

CANDS, SPA_SET, VARS = ("S-CK", "S-CKA"), ("S-CK", "S-CKA", "S-CD"), ("S-CD", "S-CK", "S-CKA", "S-C7", "S-CKL")
BASES, REFS = ("X1", "W5-G", "K2", "K3"), ("B_mix", "ZB")
MAIN = ("K1", "K2", "K3", "X1", "W5-G", "W8-G", "S-A2") + VARS
STRUCT = (("К1", "S-CK", "S-CD"), ("К2", "S-CK", "S-C7"), ("К3", "S-CK", "S-CKL"))
N_HOLM, N_WIDE, M_FAMILY, M_STRUCT = 836, 3377, 8, 3
PCOLS = ("p_spa_l", "p_spa_c", "p_spa_u", "p_stepm", "spa_t", "p_holm_m", "p_holm_N", "p_holm_Nwide")
verdict, losses = T.verdict, T.losses


def hkey(cand: str, base: str) -> str:
    return "X1_C(%s)" % cand if base == "X1" else cand


def table(rv, tp, F) -> pd.DataFrame:
    tp, rows = pd.DatetimeIndex(tp), []
    q = {k: float(T.Q.ql(rv, F[k]).mean()) for k in F}
    for k in list(MAIN) + list(REFS) + ["X1_C(%s)" % c for c in VARS]:
        role = "main" if k in MAIN else "layer X1_C" if k.startswith("X1_C") else "reference"
        rows.append(dict(role=role, series=k, n=len(tp), first_tplus=str(tp.min().date()), last_tplus=str(tp.max().date()), qlike=q[k], **{"d_%s_pct" % b: 100.0 * (q[k] - q[b]) / q[b] for b in BASES}))
    return pd.DataFrame(rows)


def spa_blocks(ls: dict) -> dict:
    idx = S.idx_for(len(next(iter(ls.values()))))
    return {b: SP.spa_stepm(ls[b], np.column_stack([ls[hkey(c, b)] for c in SPA_SET]), idx) for b in BASES}


def struct_spa(ls: dict) -> dict:
    idx = S.idx_for(len(next(iter(ls.values()))))
    return {i: SP.spa_stepm(ls[b], ls[c][:, None], idx) for i, c, b in STRUCT}


def _holm(t: pd.DataFrame, m: np.ndarray, mm: int) -> None:
    p = t.loc[m, "p_dm"].to_numpy()
    t.loc[m, "p_holm_m"], t.loc[m, "p_holm_N"], t.loc[m, "p_holm_Nwide"] = S.holm_n(p, mm), S.holm_n(p, N_HOLM), S.holm_n(p, N_WIDE)
    t.loc[m, "m_family"] = mm


def _main_rows(tp, ls, spa) -> list:
    y, allm, rows = tp.year.to_numpy(), np.ones(len(tp), bool), []
    for c in CANDS:
        for b in BASES:
            h, hid, k = hkey(c, b), "%s|%s" % (c, b), SPA_SET.index(c)
            r = T._row("main", "M", hid, h, b, ls, tp, allm)
            r.update(p_spa_l=spa[b]["p_l"], p_spa_c=spa[b]["p_c"], p_spa_u=spa[b]["p_u"], p_stepm=float(spa[b]["p_step"][k]), spa_t=float(spa[b]["t"][k]))
            rows.append(r)
            rows += [T._row("window", w, hid, h, b, ls, tp, m) for w, m in (("h1_2018-2021", y <= 2021), ("h2_2022-2026", y >= 2022))]
    return rows


def _other_rows(tp, ls, sspa) -> tuple:
    allm, st, rf = np.ones(len(tp), bool), [], []
    for i, c, b in STRUCT:
        r = T._row("struct", "M", i, c, b, ls, tp, allm)
        r.update(p_spa_l=sspa[i]["p_l"], p_spa_c=sspa[i]["p_c"], p_spa_u=sspa[i]["p_u"], p_stepm=float(sspa[i]["p_step"][0]), spa_t=float(sspa[i]["t"][0]))
        st.append(r)
    cd = [T._row("cd_base", "M", "S-CD|%s" % b, hkey("S-CD", b), b, ls, tp, allm) for b in BASES]
    for v in VARS + ("S-A2",):
        rf += [T._row("reference", "M", "%s|%s" % (v, b), v, b, ls, tp, allm) for b in ("W8-G", "K1", "B_mix", "ZB")]
    ab = [T._row("ablation", "M", "S-CKA|S-CK %s" % lab, hkey("S-CKA", "X1") if lab == "слой X1_C" else "S-CKA", hkey("S-CK", "X1") if lab == "слой X1_C" else "S-CK", ls, tp, allm) for lab in ("F", "слой X1_C")]
    return st, cd, rf, ab


def stats(rv, tp, F, ls, spa, sspa) -> pd.DataFrame:
    tp = pd.DatetimeIndex(tp)
    st, cd, rf, ab = _other_rows(tp, ls, sspa)
    t = pd.DataFrame(_main_rows(tp, ls, spa) + st + cd + rf + ab)
    for k in PCOLS:
        if k not in t:
            t[k] = np.nan
    t["m_family"] = np.nan
    for kind, mm in (("main", M_FAMILY), ("struct", M_STRUCT)):
        _holm(t, (t["kind"] == kind).to_numpy(), mm)
    _holm(t, (t["kind"] == "cd_base").to_numpy(), 4)
    t["n_holm"], t["n_holm_wide"], t["B"] = N_HOLM, N_WIDE, S.B
    fam = t["kind"].isin(["main", "struct"])
    t["verdict"] = [verdict(d, ph, ps) if f else "справочно, критериев не получает" for d, ph, ps, f in zip(t["delta_pct"], t["p_holm_N"], t["p_spa_c"], fam)]
    return t


def spa_table(spa: dict, sspa: dict) -> pd.DataFrame:
    rows = []
    for b in BASES:
        r = spa[b]
        rows.append(dict(kind="family", base=b, benchmark=b, candidates="|".join(hkey(c, b) for c in SPA_SET), T=r["T"], p_l=r["p_l"], p_c=r["p_c"], p_u=r["p_u"], B=S.B, block=S.MEAN_BLOCK,
                         note="основной p SPA этапа — p_c (центровка Hansen consistent); бенчмарк согласован со слоем кандидатов"))
        rows += [dict(kind="candidate", base=b, benchmark=b, candidates=hkey(c, b), t_stat=float(r["t"][k]), dbar=float(r["dbar"][k]), omega=float(r["omega"][k]), p_stepm=float(r["p_step"][k]),
                      stepm_reject=bool(r["reject"][k])) for k, c in enumerate(SPA_SET)]
    for i, c, b in STRUCT:
        r = sspa[i]
        rows.append(dict(kind="struct", base=b, benchmark=b, candidates=c, T=r["T"], p_l=r["p_l"], p_c=r["p_c"], p_u=r["p_u"], t_stat=float(r["t"][0]), dbar=float(r["dbar"][0]), omega=float(r["omega"][0]), B=S.B, block=S.MEAN_BLOCK,
                         note="%s: SPA с одним кандидатом F_CK против бенчмарка %s; основной p — p_c" % (i, b)))
    return pd.DataFrame(rows)
