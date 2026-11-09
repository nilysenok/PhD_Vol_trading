"""DE0, G2 и G4 (часть): чистые функции на массивах. l = rv/F − ln(rv/F) − 1 = e^u − u − 1, u = ln(rv/F). Δ = mean l_гибрид − mean l_K2: минус — гибрид лучше.
Окна — по году t⁺: 2018_2021, 2022_2025 (и 2018_2025 для разложения)."""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = {"2018_2021": (2018, 2021), "2022_2025": (2022, 2025), "2018_2025": (2018, 2025)}
YEARS = list(range(2018, 2026))
TOL = 1e-12


def qlike_day(rv, f) -> np.ndarray:
    r = np.asarray(rv, float) / np.asarray(f, float)
    return r - np.log(r) - 1.0


def u_of(rv, f) -> np.ndarray:
    return np.log(np.asarray(rv, float) / np.asarray(f, float))


def wmask(year: np.ndarray, win: str) -> np.ndarray:
    a, b = WINDOWS[win]
    return (year >= a) & (year <= b)


def contrib_years(dl: np.ndarray, year: np.ndarray) -> pd.DataFrame:
    """(i): вклад года в Δ за 2018–2025 = Σ_{t в году} d_t / N; сумма вкладов = Δ (строка `итого`)."""
    m = wmask(year, "2018_2025")
    n = int(m.sum())
    rows = [dict(year=str(y), n=int((m & (year == y)).sum()), mean_delta=float(dl[m & (year == y)].mean()), contrib=float(dl[m & (year == y)].sum() / n)) for y in YEARS]
    rows.append(dict(year="итого", n=n, mean_delta=float(dl[m].mean()), contrib=float(dl[m].sum() / n)))
    return pd.DataFrame(rows)


def identity_years(t: pd.DataFrame) -> float:
    """|Σ вкладов − Δ| для таблицы `contrib_years`."""
    body, tot = t[t["year"] != "итого"], t[t["year"] == "итого"].iloc[0]
    return abs(float(body["contrib"].sum()) - float(tot["contrib"]))


def under_over(rv, f, year: np.ndarray, win: str) -> dict:
    """(iii): QLIKE = under + over, under = Σ_{u>0} l / n, over = Σ_{u≤0} l / n; mean_u — смещение."""
    m = wmask(year, win)
    u, l = u_of(rv, f)[m], qlike_day(rv, f)[m]
    n = int(m.sum())
    return dict(n=n, qlike=float(l.mean()), under=float(l[u > 0].sum() / n), over=float(l[u <= 0].sum() / n), mean_u=float(u.mean()), share_under_days=float((u > 0).mean()))


def identity_under_over(r: dict) -> float:
    return abs(r["under"] + r["over"] - r["qlike"])


def tail_share(dl: np.ndarray, q: float) -> dict:
    """(iv): доля Δ от доли q самых «крайних» дат по |d_t|. Σ top + Σ rest = Σ всех (тождество)."""
    n = len(dl)
    k = max(1, int(np.ceil(q * n)))
    top = np.argsort(-np.abs(dl), kind="stable")[:k]
    sel = np.zeros(n, bool)
    sel[top] = True
    s_top, s_rest, s_all = float(dl[sel].sum()), float(dl[~sel].sum()), float(dl.sum())
    return dict(q=q, n=n, k=k, delta=s_all / n, delta_top_sum=s_top / n, delta_ex_top=s_rest / (n - k), share_of_delta=s_top / s_all if s_all != 0 else float("nan"),
                top_worse=int((dl[sel] > 0).sum()), identity=abs(s_top + s_rest - s_all))


def tail_identity(r: dict) -> float:
    """Σ по крайним + Σ по остальным − Σ всех из полей строки `tail_share`."""
    return abs(r["delta_top_sum"] * r["n"] + r["delta_ex_top"] * (r["n"] - r["k"]) - r["delta"] * r["n"])


def cause_row(rv, f: dict, year: np.ndarray, hyb: str) -> dict:
    """Главный вопрос: преимущество гибрида над K2 adv = q(K2) − q(гибрид) = −Δ_H + gain, gain = q(H) − q(гибрид) = (under_H − under_g) + (over_H − over_g).
    Смена adv между окнами 2018_2021 → 2022_2025 раскладывается на четыре части, сумма = смена; доли — от смены."""
    out = dict(hybrid=hyb)
    parts = {}
    for w in ("2018_2021", "2022_2025"):
        h, k, g = (under_over(rv, f[s], year, w) for s in ("K1", "K2", hyb))
        out["adv_" + w] = k["qlike"] - g["qlike"]
        parts[w] = dict(base=-(h["qlike"] - k["qlike"]), gu=h["under"] - g["under"], go=h["over"] - g["over"])
        out["delta_H_" + w] = h["qlike"] - k["qlike"]
        out["share_closed_" + w] = (h["qlike"] - g["qlike"]) / (h["qlike"] - k["qlike"]) if h["qlike"] != k["qlike"] else float("nan")
    d = out["adv_2022_2025"] - out["adv_2018_2021"]
    out["d_adv"] = d
    for p in ("base", "gu", "go"):
        out["part_" + p] = parts["2022_2025"][p] - parts["2018_2021"][p]
        out["share_" + p] = out["part_" + p] / d if d != 0 else float("nan")
    out["identity"] = abs(out["part_base"] + out["part_gu"] + out["part_go"] - d)
    return out


def acf(x: np.ndarray, lags=range(1, 6)) -> dict:
    x = np.asarray(x, float)
    xc = x - x.mean()
    den = float((xc ** 2).sum())
    return {k: float((xc[k:] * xc[:-k]).sum() / den) for k in lags}


def run_lengths(dl: np.ndarray) -> np.ndarray:
    """Длины серий подряд d_t > 0."""
    out, cur = [], 0
    for v in np.asarray(dl) > 0:
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return np.array(out, int)


def run_summary(dl: np.ndarray) -> dict:
    """Распределение длин серий; ожидание при независимости: p = доля d_t > 0, длина ≥ k встречается в среднем (n_runs)·p^(k−1) раз."""
    r = run_lengths(dl)
    p = float((np.asarray(dl) > 0).mean())
    row = dict(n=len(dl), p_pos=p, n_runs=len(r), mean_len=float(r.mean()) if len(r) else 0.0, max_len=int(r.max()) if len(r) else 0, exp_mean_len=1.0 / (1.0 - p) if p < 1 else float("nan"))
    for k in (1, 2, 3, 4, 5):
        row["n_len_%d" % k] = int((r == k).sum()) if k < 5 else int((r >= 5).sum())
    return row
