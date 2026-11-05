"""D: выбор лучшего ансамбля (по внутреннему критерию Q_wf, не по описанию), честный выбор по разработке (Q_wf_dev) и правило «три лучших устойчивых» (GATES_D §4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.d import ens

EVAL_SELECTED = ("c3_f3calm_g4", "c3_f3calm_rv", "c3_f3calm_har")   # калм-модель b2_top2 выбрана на оценочном периоде — в тройку не входят


def q_wf_dev(q: np.ndarray, td: pd.Series, last: int = 2021) -> float:
    """Как evalrows.q_wf, но score_Y только для Y в 2018…last (цели 2017…last−1): выбор без оценочного периода 2022+."""
    y = pd.Series(td).dt.year.to_numpy()
    ann = {k: float(q[y == k].mean()) for k in range(2017, 2027) if (y == k).any()}
    return float(np.mean([np.mean([ann[k] for k in range(2017, Y) if k in ann]) for Y in range(2018, last + 1)]))


def best_ensemble(df: pd.DataFrame, col: str = "q_int") -> str:
    """df: row_id, <col>. Кандидаты — только ансамбли ens.TRIALS без компонентов; ничья |Δ| <= 1e-12 — меньше членов, затем id."""
    c = df[df["row_id"].isin([r for r in ens.TRIALS if r in ens.SPEC])].copy()
    if c.empty:
        raise ValueError("нет кандидатов-ансамблей")
    c["n_members"] = [len(ens.SPEC[r][0]) for r in c["row_id"]]
    lo = c[col].min()
    t = c[c[col] <= lo + 1e-12].sort_values(["n_members", "row_id"])
    return str(t["row_id"].iloc[0])


def top3(df: pd.DataFrame) -> pd.DataFrame:
    """df: row_id, q_int, q_int_b2, d_harj_no_240222, perspective, seed_ok, plateau_ok, mcs_in. Допуск: все четыре флага и выбор не на оценочном периоде.
    Порядок: относительный выигрыш внутреннего критерия (убыв.), ничья — меньший Δ к HAR-J(б) без 24.02. Меньше трёх допущенных — меньше трёх."""
    d = df.copy()
    ok = d["perspective"].astype(bool) & d["seed_ok"].astype(bool) & d["plateau_ok"].astype(bool) & d["mcs_in"].astype(bool)
    d = d[ok & ~d["row_id"].isin(EVAL_SELECTED)].copy()
    d["rel_gain"] = (d["q_int_b2"] - d["q_int"]) / d["q_int_b2"]
    return d.sort_values(["rel_gain", "d_harj_no_240222"], ascending=[False, True]).head(3)
