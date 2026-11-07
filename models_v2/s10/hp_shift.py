"""P0, G3 (Д1): проверка сдвига каждого входного ряда каждой бумаги. Ряд x_d (известен к 18:50 d, строка собственного дня бумаги) сопоставляется с целевой ln RV бумаги:
corr(x_d, ln RV_{d+k}), k = −2…+1 (k — число СОБСТВЕННЫХ строк бумаги); максимум |corr| обязан быть на k = 0 (последний известный день). Оконные ряды (rv_w, rv_m) — по приращениям Δx_d.
Прогноз K1 хранится на дате ИНФОРМАЦИИ d (как все входные ряды; цель — RV_{d+1}): ожидаемый максимум тоже k = 0 (σ̂_d построен из RV_d, корреляция с RV_d выше, чем с шумной целью RV_{d+1}).
Выравнивание по дате цели дало бы максимум k = −1 — это тот же ряд, а не ошибка (проверено: первый прогон, 29 строк). Сдвиг ряда на +1 строку (утечка) даёт максимум на k = +1 — тест."""
from __future__ import annotations

import numpy as np
import pandas as pd

KS = (-2, -1, 0, 1)
LEVEL = ["rv", "rq", "rv_rel", "rs_pos", "rs_neg", "rv_open", "rv_mid", "rv_close", "c", "rv_s_sync", "idio"]
LINEAR = ["corr", "beta"]
WINDOW = ["rv_w", "rv_m"]


def corr_k(x: pd.Series, y: pd.Series) -> dict:
    """corr(x_d, y_{d+k}) по строкам; y — ln RV той же бумаги (по строкам индекса x)."""
    return {k: float(x.corr(y.shift(-k))) for k in KS}


def verdict(c: dict, n: int) -> dict:
    """argmax |corr|; ok = argmax == 0; для несовпадения — шумовой разбор: разница |corr_k*| − |corr_0| < 2/√n."""
    a = {k: abs(v) for k, v in c.items() if np.isfinite(v)}
    if len(a) < len(KS):
        return {"argmax": None, "ok": False, "diag": "ряд не определён (NaN) при части сдвигов"}
    k = max(a, key=a.get)
    gap = a[k] - a[0]
    diag = "" if k == 0 else ("слабая связь, разница в пределах шума 2/√n=%.3f" % (2 / np.sqrt(n)) if gap < 2 / np.sqrt(n) else "РАЗБОР: пик вне k=0 сверх шума")
    return {"argmax": k, "ok": k == 0, "diag": diag}


def input_series(d: pd.DataFrame, imo_rv: pd.Series) -> dict:
    """Словарь {имя: (ряд, вид)}: M10-величины бумаги, синхронные с IMOEX и ln RV индекса на датах бумаги."""
    out = {}
    for c in (c for c in LEVEL if c in d):       # у IMOEX нет синхронных с самим собой рядов
        x = d[c]
        out["ln_" + c] = (np.log(x.where(x > 0)), "level")
    out["ln_j1p"] = (np.log1p(d["j"]), "level")
    for c in (c for c in LINEAR if c in d):
        out[c] = (d[c], "level")
    for c in WINDOW:
        out["ln_" + c] = (np.log(d[c].where(d[c] > 0)).diff(), "increment")
    out["ln_imoex_rv"] = (np.log(imo_rv.where(imo_rv > 0)).reindex(d.index), "level")
    return out


def table_one(t: str, d: pd.DataFrame, imo_rv: pd.Series, k1: pd.DataFrame, start: str = "2017-01-03") -> list:
    y = np.log(d["rv"])
    rows = []
    items = input_series(d, imo_rv)
    s = k1.set_index("d")["sigma2"]          # прогноз хранится на дате информации d (как все входные ряды); цель — RV_{d+1}
    items["k1_sigma2@d"] = (np.log(s).reindex(d.index), "level")
    for name, (x, kind) in items.items():
        m = x.index >= pd.Timestamp(start)
        xs, ys = x[m], y[m]
        n = int((xs.notna() & ys.notna()).sum())
        c = corr_k(xs, ys)
        rows.append({"ticker": t, "series": name, "kind": kind, "n": n, **{"corr_k%+d" % k: v for k, v in c.items()}, **verdict(c, n)})
    return rows
