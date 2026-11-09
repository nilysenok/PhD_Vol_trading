"""V6-2, G26: главная проверка направления C — точность в дни после неустойчиво измеренной RV. Группа дня d: u_d выше порога верхнего квинтиля (порог и нормировка u — по закрытым парам на дату переоценки месяца, `v6_2_wf`) против остальных дней.
Для каждого окна цели (2017–2021, 2022–2025, вся история) и группы: Δ QLIKE конфигурации к A0 и к K3 в % среднего QLIKE K1 ЭТОЙ группы и окна, 90 % интервал блокового бутстрапа (блок 20, 2000, seed 0) по датам группы в порядке времени.
Дни до первой переоценки (g = 0, группы нет) в проверку не входят. Чистая функция от прогнозов."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q, v6_1_eval as EV

GROUPS = (("высокая u (верхний квинтиль)", True), ("остальные дни", False))
WINDOWS = ("2017_2021", "2022_2025", "all")


def unstable_table(w: pd.DataFrame, high: np.ndarray, valid: np.ndarray, fs: dict, cids: list) -> pd.DataFrame:
    """w — общие даты (rv, t_plus, K1, K3); high/valid — булевы массивы на этих датах; fs — прогнозы ({A0, C*})."""
    rv, tp, rows = w["rv"].to_numpy(), w["t_plus"].to_numpy(), []
    high, valid = np.asarray(high, bool), np.asarray(valid, bool)
    for win in WINDOWS:
        mw = Q.wmask(tp, win) & valid
        for gname, flag in GROUPS:
            mg = mw & (high if flag else ~high)
            k1 = float(Q.ql(rv[mg], w["K1"].to_numpy()[mg]).mean())
            for c in cids:
                for base, fb in (("A0", fs["A0"]), ("K3", w["K3"].to_numpy())):
                    qx, qy = Q.ql(rv[mg], fs[c][mg]), Q.ql(rv[mg], fb[mg])
                    rows.append(dict(window=win, group=gname, n_days=int(mg.sum()), share_of_window=float(mg.sum() / mw.sum()), config=c, base=base, qlike_config=float(qx.mean()), qlike_base=float(qy.mean()),
                                     qlike_K1=k1, delta_qlike=float(qx.mean() - qy.mean()), **EV.delta_ci(qx, qy, k1)))
    return pd.DataFrame(rows)
