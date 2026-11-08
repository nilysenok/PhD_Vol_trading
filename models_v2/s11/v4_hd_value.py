"""V4-H, G40: ценность источников сверх K8 (H_VALUE.csv). Δ QLIKE к K8 по окнам (среднее по M16, как в таблице сравнения), DM к K8 (из H_TESTS), доля строк (бумага, дата) M16, где RV вне [min(H, M), max(H, M)],
и снижение потерь на них; доля общего выигрыша, пришедшая с этих строк. Все ряды — на общих датах (`S.common`). Диагностика задним числом, вне путей прогноза."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_b_stats as ST

SOURCE = {"H1": "рыночный HAR-surprise", "H2": "секторный сюрприз", "H3": "внутридневная форма", "H4": "вероятность скачка", "H5": "согласованность прогнозов", "H6": "нестабильность режима (CUSUM)",
          "H7": "все источники", "H8": "все источники, иерархия по бумагам"}


def out_of_range(c: pd.DataFrame) -> np.ndarray:
    """RV вне диапазона [min(H, M), max(H, M)], H = K1, M = √(K2·K3) — по строкам."""
    M = np.sqrt(c["k2"].to_numpy(float) * c["k3"].to_numpy(float))
    h, rv = c["k1"].to_numpy(float), c["rv_tplus"].to_numpy(float)
    return (rv < np.minimum(h, M)) | (rv > np.maximum(h, M))


def build(c: pd.DataFrame, comp: pd.DataFrame, tests: pd.DataFrame, cfgs: list) -> pd.DataFrame:
    q = comp.set_index(["series", "window"])["qlike"]
    rows = []
    for w in ST.WINS:
        m = (c["group"] == "M16").to_numpy() & ST.win_mask(c, w)
        s = c[m]
        out = out_of_range(s)
        base = ST.qrow(s, "K8")
        for cfg in cfgs:
            d = base - ST.qrow(s, cfg)                         # выигрыш cfg к K8 по строкам (плюс = лучше)
            t = tests[(tests["config"] == cfg) & (tests["comparator"] == "K8") & (tests["window"] == w)]
            rows.append({"config": cfg, "source": SOURCE[cfg], "window": w, "d_qlike_k8": float(q[(cfg, w)] - q[("K8", w)]), "dm_stat": float(t["dm_stat"].iloc[0]) if len(t) else np.nan,
                         "dm_p": float(t["dm_p"].iloc[0]) if len(t) else np.nan, "p_holm": float(t["p_holm"].iloc[0]) if len(t) else np.nan, "share_rows_out_of_range": float(out.mean()),
                         "n_rows": int(len(s)), "gain_per_row_out": float(d[out].mean()) if out.any() else np.nan, "gain_per_row_in": float(d[~out].mean()) if (~out).any() else np.nan,
                         "loss_k8_out": float(base[out].mean()) if out.any() else np.nan, "share_of_gain_out": float(d[out].sum() / d.sum()) if d.sum() != 0 else np.nan})
    return pd.DataFrame(rows)
