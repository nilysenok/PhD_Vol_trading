"""Размер позиции и режим рынка этапа 1 (Ж-047 п. 3, контракт К3). Всё по прошлому.

Решение на закрытии d принимается по данным до 18:50 d, исполняется на d+1. Поэтому «известно в d» —
всё, что случилось не позже 18:50 d; «текущий день» в смысле запрета — день d+1 (исполнения), а в ранге
σ рынка — ещё и сам день d (Н1: ранг среди d−252…d−1).

Размер: w = min(L, σ*/(σ_d·√252)) (обратный σ) или min(L, 1) (равный); σ_d — σ дня сигнала, вес
фиксируется на сделку снаружи (в etap1). L = 1 (❓2: без плеча на позицию).
Режим (True = вход разрешён): «net» — всегда; «ma200» — close IMOEX d > MA200 по d−199…d;
«rank08» — ранг σ рынка d среди d−252…d−1 < 0,8. Нет истории — вход запрещён.
Константы *_LAG — для тестов-мутаций (подмена в памяти): 0/1 — как в контракте.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

END = pd.Timestamp("2022-01-01")
TARGETS = (0.15, 0.20, 0.25)
SIZES = ("obratnyj", "ravnyj")
RANK_WIN = 252
RANK_MAX = 0.8
MA_WIN = 200
MIN_BARS = 30
WEIGHT_LAG = 0   # σ дня сигнала d; −1 = σ дня d+1 (мутация «вес с текущим днём»)
MA_LAG = 0       # окно MA200 кончается в d; −1 = захватывает d+1 (мутация)
RANK_LAG = 1     # окно ранга кончается в d−1; 0 = включает d (мутация, mutations.md № 8)


def razmer_ves(sigma: np.ndarray, kind: str, target: float = 0.20, limit: float = 1.0) -> np.ndarray:
    """Вес позиции по дням сигнала d. sigma — дневная σ, известная в d. σ ≤ 0 или NaN → вес 0."""
    if kind not in SIZES:
        raise ValueError(f"размер {kind!r}: допустимо {SIZES}")
    s = np.asarray(sigma, float)
    n = len(s)
    src = np.full(n, np.nan)
    idx = np.arange(n) - WEIGHT_LAG                # WEIGHT_LAG = −1: берём σ[d+1]
    ok = (idx >= 0) & (idx < n)
    src[ok] = s[idx[ok]]
    good = np.isfinite(src) & (src > 0)
    if kind == "ravnyj":
        w = np.where(good, 1.0, 0.0)
    else:
        w = np.zeros(n)
        w[good] = target / (src[good] * np.sqrt(252.0))
    return np.minimum(w, limit)


def flag_ma200(close: np.ndarray) -> np.ndarray:
    c = np.asarray(close, float)
    n = len(c)
    out = np.zeros(n, bool)
    for i in range(n):
        e = i - MA_LAG                              # последний день окна (включительно)
        if e - MA_WIN + 1 < 0 or e >= n:
            continue
        w = c[e - MA_WIN + 1:e + 1]
        out[i] = bool(np.isfinite(w).all() and c[i] > w.mean())
    return out


def rank_sigma(sig: np.ndarray) -> np.ndarray:
    """Доля значений σ в окне d−252…d−1, строго меньших σ_d. NaN — нет истории."""
    s = np.asarray(sig, float)
    n = len(s)
    out = np.full(n, np.nan)
    for i in range(n):
        e = i - RANK_LAG + 1                        # конец окна, исключительно
        lo = e - RANK_WIN
        if lo < 0 or not np.isfinite(s[i]):
            continue
        w = s[lo:e]
        if np.isfinite(w).all():
            out[i] = float((w < s[i]).mean())
    return out


def flag_rank(sig: np.ndarray, rmax: float = RANK_MAX) -> np.ndarray:
    r = rank_sigma(sig)
    return np.isfinite(r) & (r < rmax)


def _hold_short(flag: np.ndarray, short: np.ndarray) -> np.ndarray:
    """День с < MIN_BARS баров — флаг предыдущего дня (контракт §1.1)."""
    f = flag.copy()
    for i in np.flatnonzero(short):
        f[i] = f[i - 1] if i > 0 else False
    return f


def rezhim_arrays(close: np.ndarray, sig_mkt: np.ndarray, nbars: np.ndarray | None = None) -> dict:
    """{имя: bool по датам} — True = вход разрешён."""
    n = len(close)
    short = np.zeros(n, bool) if nbars is None else (np.asarray(nbars) < MIN_BARS)
    return {"net": np.ones(n, bool),
            "ma200": _hold_short(flag_ma200(close), short),
            "rank08": _hold_short(flag_rank(sig_mkt), short)}


def imoex_daily(path=None) -> pd.DataFrame:
    """Дневной ряд IMOEX до 18:50 из 10-минуток (только < 2022): date, close, sig_mkt, nbars.
    sig_mkt = √(среднее RV за 22 дня), RV дня = Σ r² по 10-мин лог-доходностям внутри дня."""
    import pyarrow.parquet as pq
    from fastbt import daily_1850 as d18
    p = path or (d18.default_main() / "moex_discovery/data/final_v6/candles_10m/external/IMOEX.parquet")
    # схема внешних рядов: begin (строка ISO), open/high/low/close/value/volume; фильтр < 2022 — на чтении
    b = pq.read_table(p, filters=[("begin", "<", "2022-01-01")]).to_pandas()
    b["timestamp"] = pd.to_datetime(b["begin"])
    day = d18.aggregate(b, "IMOEX")
    b = b.sort_values("timestamp")
    m = b["timestamp"].dt.hour * 60 + b["timestamp"].dt.minute
    b = b[(m >= d18.OPEN_MIN) & (m < d18.CUT_MIN)].copy()
    b["date"] = b["timestamp"].dt.normalize()
    b["r2"] = b.groupby("date")["close"].transform(lambda x: np.log(x).diff()) ** 2
    g = b.groupby("date")
    rv = g["r2"].sum()
    out = pd.DataFrame({"close": day.set_index("date")["close"], "rv": rv, "nbars": g.size()}).dropna()
    out["sig_mkt"] = np.sqrt(out["rv"].rolling(22).mean())
    out = out.reset_index().rename(columns={"index": "date"})
    if out["date"].max() >= END:
        raise PermissionError("IMOEX: даты ≥ 2022")
    return out[["date", "close", "sig_mkt", "nbars"]]
