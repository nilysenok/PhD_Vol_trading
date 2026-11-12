"""S-RCD-EVAL (Ж-259; `S_RCD_SPEC.md`, разд. 6.2, 6.3): аддитивное разложение вклада (квадрат {C-D, C-K} × {без R, R} на уровне F), доля s, разложение «замороженный π / реакция слоя», распределение ln(C_RCD/·), сравнение с нулевым распределением.
Читает массивы потерь и сохранённые `RD/*`, `R/rck`; ничего не обучает и не пишет. Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Исследовательский результат на раскрытой истории, не OOS.
Публичное API:
- `e_win(ls, mask)`: e(X) = 100·(Σ l_X − Σ l_W5-G)/Σ l_W5-G на датах mask (единицы e — % QLIKE W5-G); `IDS` — тождества 1–3 как функции e;
- `decomp(inp, ls, null) -> DataFrame(section, item, key, value)`: окна D_M и active (d ≥ 2020-08-03): e рядов; слагаемые тождеств 1 и 2 и их невязки (≤ 1e-12); взаимодействие I; доля s и её текст; разложение Δ на F^fr и реакцию слоя; квантили ln(C_RCD/cfg8), ln(C_RCD/C_RCK); сравнение реальных Δ с нулём процедуры (правило по |Δ|)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import rd_eval_cfg as C
from models_v2.s18 import rd_eval_null as RN

QS = (0.01, 0.05, 0.5, 0.95, 0.99)
LN_EPS = 1e-3
SER = ("S-RCD", "S-CDF", "S-CKF", "S-CD", "S-CK", "S-RCK")


def e_win(ls: dict, m) -> dict:
    b = float(ls["W5-G"][m].sum())
    return {k: 100.0 * (float(ls[k][m].sum()) - b) / b for k in SER}


def _id(e: dict) -> dict:
    """Тождества 1–3 (все слагаемые — в единицах e)."""
    f, r_cd, lev, cd = e["S-RCK"] - e["S-RCD"], e["S-RCD"] - e["S-CDF"], e["S-CDF"] - e["S-CD"], e["S-CD"]
    r_ck, lev_ck = e["S-RCK"] - e["S-CKF"], e["S-CKF"] - e["S-CK"]
    return {"id1_form_at_R": f, "id1_R_on_CD": r_cd, "id1_level_without_R": lev, "id1_S_CD": cd, "id1_gap": f + r_cd + lev + cd - e["S-RCK"],
            "id2_R_on_CK_at_F": r_ck, "id2_level_without_R_CK": lev_ck, "id2_total_vs_S_CK": e["S-RCK"] - e["S-CK"], "id2_gap": r_ck + lev_ck - (e["S-RCK"] - e["S-CK"]),
            "id3_interaction": r_ck - r_cd, "id3_gap": (r_ck - r_cd) - (f - (e["S-CKF"] - e["S-CDF"]))}


def _share(e: dict) -> tuple:
    if not e["S-RCK"] < 0:
        return float("nan"), "доли нет: e(S-RCK) ≥ 0"
    s = e["S-RCD"] / e["S-RCK"]
    return s, ("выигрыш S-RCK над W5-G в основном воспроизводится прямой коррекцией под R; разложение C-K для него не обязательно" if s >= C.SHARE_HALF
               else "при R разложение C-K несёт бо́льшую часть выигрыша над W5-G")


def _dp(lh, lb, m) -> float:
    return 100.0 * float(lh[m].sum() - lb[m].sum()) / float(lb[m].sum())


def _frozen(inp: dict, ls: dict, m) -> list:
    fr = {k: inp["frames"][k]["F_fr"].to_numpy(float) for k in ("S-RCD", "S-CDF")}
    ql = {k: S.Q.ql(inp["rv"], fr[k]) for k in fr}
    out = []
    for tag, b, lb in (("vs_W5G", "W5-G", ls["W5-G"]), ("vs_CDF", "S-CDF", ls["S-CDF"])):
        base_fr = ls["W5-G"] if b == "W5-G" else ql["S-CDF"]       # при θ = 0 F^fr = W5-G (Н1); у S-CDF — своя замороженная
        tot, den = _dp(ls["S-RCD"], lb, m), float(lb[m].sum())
        frozen = 100.0 * (float(ql["S-RCD"][m].sum()) - float(base_fr[m].sum())) / den
        out += [("decomp_" + tag, "total", "", tot), ("decomp_" + tag, "frozen_pi", "", frozen), ("decomp_" + tag, "layer_reaction", "", tot - frozen)]
    return out


def act_mask(inp: dict) -> np.ndarray:
    """Общее активное окно: d ≥ 2020-08-03 (n = 1549)."""
    return np.asarray(inp["d"] >= pd.Timestamp(C.ACTIVE_FROM))


def _lnratio(inp: dict, act) -> list:
    a, rk = inp["frames"]["S-RCD"], inp["frames"]["S-RCK"]
    out = [("lnratio_n", "n_active_dates", "", int(act.sum()))]
    for tag, num, den in (("lnratio_cfg8", a["C"], a["cfg8"]), ("lnratio_rck", a["C"], rk["C"])):
        v = np.log(num.to_numpy(float) / den.to_numpy(float))[act]
        out += [(tag, "q", "q%02d" % round(100 * q), float(np.quantile(v, q))) for q in QS] + [(tag, "share_abs_gt_1e-3", "", float((np.abs(v) > LN_EPS).mean()))]
    return out


def _noise(ls: dict, act, null: tuple) -> list:
    s, m = null
    q = m.set_index(["item", "stat"])["value"]
    lo, hi = RN.abs_bounds(s)
    out = [("noise", "q05", "", float(q[("qlike_rel_pct", "q05")])), ("noise", "q95", "", float(q[("qlike_rel_pct", "q95")])), ("noise", "mean", "", float(q[("qlike_rel_pct", "mean")])),
           ("noise", "std", "", float(q[("qlike_rel_pct", "std")])), ("noise", "share_nonzero_mean", "", float(q[("share_theta_nonzero", "mean")])), ("noise", "abs_q05", "", lo), ("noise", "abs_q95", "", hi)]
    for tag, b in (("RCD_vs_CDF", "S-CDF"), ("RCD_vs_RCK", "S-RCK")):
        for w, mk in (("D_M", np.ones(len(act), bool)), ("active", act)):
            d = _dp(ls["S-RCD"], ls[b], mk)
            out += [("noise_real", "%s|%s" % (tag, w), "", d), ("noise_rule", "%s|%s" % (tag, w), RN.rule_abs(d, lo, hi), d)]
    return out


def decomp(inp: dict, ls: dict, null: tuple) -> pd.DataFrame:
    act = act_mask(inp)
    allm, rows = np.ones(len(act), bool), []
    for w, m in (("D_M", allm), ("active", act)):
        e = e_win(ls, m)
        s, txt = _share(e)
        rows += [("e_" + w, k, "", v) for k, v in e.items()] + [("ident_" + w, k, "", v) for k, v in _id(e).items()] + [("share_" + w, "s", txt, s), ("share_" + w, "n_dates", "", int(m.sum()))]
    rows += _frozen(inp, ls, allm) + _lnratio(inp, act) +_noise(ls, act, null)
    return pd.DataFrame(rows, columns=["section", "item", "key", "value"])
