"""V6-1b, G22: уровни 1–5 этапа 21 блока для конфигураций (да / нет и основание, числа из таблиц). Чистая функция от прогнозов и готовых таблиц.
1 — техническая корректность (F > 0, g = 0 → F = H побитно, цель = цели листа 0, данные только SBER); 2 — QLIKE ниже замороженной HAR-J (K1) в 2017–2021 и на всей истории;
3 — превосходит K2, K3 и лучший PIT-сопоставимый прежний гибрид (V4:NTP2, лист 0 G3) на 2017–2021; 4 — устойчивость: Δ к K1 < 0 без 2017, без 2017–2018, без 1 % крайних дат, не менее чем в 3 годах из 5 и во всех соседних точках (λ / число деревьев);
5 — p Холма (N = 722) к K1 на 2017–2021 < 0,05, иначе «статистически неподтверждённое историческое улучшение» (если уровень 2 выполнен)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q

DATA_ONLY_SBER = "данные только SBER (аудит путей G11: два файла SBER)"


def level_rows(cid: str, a: pd.DataFrame, w: pd.DataFrame, t: dict) -> list:
    """a — прогноз конфигурации (rv, H, g, F на общих датах); w — контроли; t — таблицы конфигурации: tests, stress, sens."""
    rv, tp, f = w["rv"].to_numpy(), w["t_plus"].to_numpy(), a["F"].to_numpy()
    m = Q.wmask(tp, "2017_2021")
    z = (a["g"] == 0).to_numpy()
    l1 = bool(np.isfinite(f).all() and (f > 0).all() and (a.loc[z, "F"] == a.loc[z, "H"]).all() and float(np.abs(a["rv"].to_numpy() - rv).max()) == 0.0)
    d = t["tests"][(t["tests"]["section"] == "delta_dm") & (t["tests"]["base"] == "K1")].set_index("window")
    l2 = bool(d.loc["2017_2021", "delta_pct"] < 0 and d.loc["all", "delta_pct"] < 0)
    q = {k: float(Q.ql(rv[m], w[k].to_numpy()[m]).mean()) for k in ("K1", "K2", "K3", "V4:NTP2")}
    qf = float(Q.ql(rv[m], f[m]).mean())
    l3 = bool(qf < min(q["K2"], q["K3"], q["V4:NTP2"]))
    yf, y1 = Q.annual(Q.ql(rv[m], f[m]), tp[m]), Q.annual(Q.ql(rv[m], w["K1"].to_numpy()[m]), tp[m])
    yb = int(sum(yf[k] < y1[k] for k in yf))
    st = t["stress"].set_index("scenario")["d_pct_K1"]
    sc = {k: float(v) for k, v in st.items() if k.startswith("без")}
    nb = t["sens"]["d_pct_K1_2017_2021"].to_numpy(float)
    l4 = bool(all(v < 0 for v in sc.values()) and yb >= 3 and (nb < 0).all())
    p = float(d.loc["2017_2021", "p_holm_N"])
    return [
        (cid, 1, l1, "F > 0 на %d датах; g = 0 → F = H побитно (%d дат); цель = цели листа 0; %s" % (len(f), int(z.sum()), DATA_ONLY_SBER)),
        (cid, 2, l2, "Δ к K1 2017–2021 %+.2f %%, вся история %+.2f %% (в %% уровня K1)" % (d.loc["2017_2021", "delta_pct"], d.loc["all", "delta_pct"])),
        (cid, 3, l3, "QLIKE 2017–2021 %.5f; K2 %.5f, K3 %.5f, V4:NTP2 %.5f" % (qf, q["K2"], q["K3"], q["V4:NTP2"])),
        (cid, 4, l4, "Δ к K1 в окнах %s; лет лучше K1 %d из 5; соседние точки Δ 2017–2021 %s" % ({k: round(v, 2) for k, v in sc.items()}, yb, [round(float(v), 2) for v in nb])),
        (cid, 5, bool(p < 0.05), "p Холма (N = 722) к K1 2017–2021: %.3g%s" % (p, "" if p < 0.05 else ("; статистически неподтверждённое историческое улучшение" if l2 else "; улучшения к K1 нет")))]


def levels_table(rows: list) -> pd.DataFrame:
    return pd.DataFrame([dict(config=c, level=lv, passed="да" if ok else "нет", basis=b) for c, lv, ok, b in rows])
