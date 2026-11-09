"""V6 лист FIN: FIN_ABLATION — вклад признаков против вклада обучателя (разложение разности QLIKE GBM-конфигурации и A0) и диагностика DX-A (читается с диска, задним числом).
Разложение по окнам 2017–2021, 2022–2025, вся история; Δ — разность средних QLIKE в % уровня K1 окна, 90 % интервал блоковым бутстрапом дат (блок 20, 2000, seed 0), DM — односторонний «x лучше y»:
  learner      A0g − A0   (тот же набор S, обучатель GBM вместо LIN);
  features_gbm X − A0g    (при обучателе GBM: новые признаки сверх S);
  total_gbm    X − A0     (сумма двух предыдущих);
  features_lin X − A0     (при обучателе LIN: новые признаки сверх S)."""
from __future__ import annotations

import pandas as pd

from models_v2.s13 import v6_0_io as IO, v6_0_q as Q, v6_1_eval as EV

WINS = ("2017_2021", "2022_2025", "all")
GBM_X = ["A2", "C2", "C4", "E2"]
LIN_X = ["A1", "A3", "C1", "C3", "E1", "E3", "CB1", "F1", "F2"]


def pairs() -> list:
    """(часть, x, y): порядок строк таблицы."""
    out = [("learner", "A0g", "A0")]
    for x in GBM_X:
        out += [("features_gbm", x, "A0g"), ("total_gbm", x, "A0")]
    return out + [("features_lin", x, "A0") for x in LIN_X]


def row(d: dict, part: str, x: str, y: str, win: str) -> dict:
    w, fs = d["w"], d["fs"]
    rv, tp = w["rv"].to_numpy(), w["t_plus"].to_numpy()
    m = Q.wmask(tp, win)
    qx, qy = Q.ql(rv[m], fs[x][m]), Q.ql(rv[m], fs[y][m])
    k1 = float(Q.ql(rv[m], fs["K1"][m]).mean())
    dm = EV.dm_one(rv[m], fs[y][m], fs[x][m])
    return dict(part=part, x=x, y=y, window=win, n=int(m.sum()), delta_qlike=float(qx.mean() - qy.mean()), **EV.delta_ci(qx, qy, k1), dm_p_one=dm["p_one"])


def ablation(d: dict) -> pd.DataFrame:
    rows = [row(d, p, x, y, win) for p, x, y in pairs() for win in WINS]
    dx = IO.read_csv("DXA.csv").assign(part="DX-A")
    return pd.concat([pd.DataFrame(rows), dx], ignore_index=True)
