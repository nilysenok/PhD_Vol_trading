"""V7-0, G4: PIT-построитель внутридневных последовательностей X_d (окно дней × 53 слота × каналы) с маской. Только чтение баров; ничего не обучается.
Дни — канонические (`daily_SBER`: основная сессия 10:00–18:40, без выходных 2025+). Слот = (begin − 10:00)/10 минут (0…52) — расписание по часам, не по номеру бара: пропуск бара = слот с маской 0.
Каналы (блок Ж-232, п. 3): r (×100), hl, oc, r2, rs_pos, rs_neg, lvol (ln объёма), vol_rel (объём / медиана ТОГО ЖЕ слота за прошлые сессии, строго до дня), vwap (положение close к дневной VWAP бара, внутри дня), cum_rv, tod (доля сессии).
PIT: окно дня d = последние W сессий ≤ d; нормировка z = (x − μ)/σ по ВСЕМ барам сессий ≤ d (расширяющееся прошлое; для tod и vol_rel — не применяется); сессий < MIN_SESS — признак norm_ok = 0. Бары t⁺ нигде не используются."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily as D

CH = ["r", "hl", "oc", "r2", "rs_pos", "rs_neg", "lvol", "vol_rel", "vwap", "cum_rv", "tod"]
K, SLOTS, T0, T_SPAN, MIN_SESS = len(CH), 53, 600, 520.0, 20
ZCH = np.array([c not in ("vol_rel", "tod") for c in CH])


def bar_table(bars: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Бары основной сессии канонических дней -> строка на бар: day, slot, каналы (до нормировки, vol_rel — NaN)."""
    m = D.main_session(bars).sort_values("begin").reset_index(drop=True)
    m = m[m["begin"].dt.normalize().isin(days)].reset_index(drop=True)
    r = D.intraday_returns(m)["r"].to_numpy(float)
    day = m["begin"].dt.normalize()
    mins = (m["begin"].dt.hour * 60 + m["begin"].dt.minute).to_numpy()
    r2 = (100 * r) ** 2
    cv, cq = m.groupby(day)["value"].cumsum().to_numpy(float), m.groupby(day)["volume"].cumsum().to_numpy(float)
    f = pd.DataFrame({"day": day.to_numpy(), "slot": (mins - T0) // 10, "r": 100 * r, "hl": 100 * np.log(m["high"] / m["low"]), "oc": 100 * np.log(m["close"] / m["open"]), "r2": r2,
                      "rs_pos": np.where(r > 0, r2, 0.0), "rs_neg": np.where(r < 0, r2, 0.0), "lvol": np.log(m["volume"].to_numpy(float)), "vol_rel": np.nan,
                      "vwap": 100 * np.log(m["close"].to_numpy(float) / (cv / cq)), "tod": (mins - T0) / T_SPAN, "volume": m["volume"].to_numpy(float)})
    f["cum_rv"] = f.groupby("day")["r2"].cumsum()
    return f


def slot_median_before(vol: np.ndarray) -> np.ndarray:
    """vol (D, SLOTS), NaN где бара нет -> медиана того же слота по сессиям СТРОГО до дня (expanding, сдвиг на 1)."""
    return pd.DataFrame(vol).expanding(min_periods=1).median().shift(1).to_numpy()


def build_store(bars: pd.DataFrame, days: pd.DatetimeIndex) -> dict:
    """-> store: days, raw (D, SLOTS, K) с NaN, present (D, SLOTS), cumulative n/Σx/Σx² по сессиям (D, K)."""
    days = pd.DatetimeIndex(days)
    f = bar_table(bars, days)
    i = days.get_indexer(pd.DatetimeIndex(f["day"]))
    if (f["slot"].to_numpy() < 0).any() or (f["slot"].to_numpy() >= SLOTS).any():
        raise ValueError("слот вне 0…52: расписание вне объявленного")
    raw = np.full((len(days), SLOTS, K), np.nan)
    present = np.zeros((len(days), SLOTS), bool)
    vol = np.full((len(days), SLOTS), np.nan)
    vol[i, f["slot"].to_numpy()] = f["volume"].to_numpy()
    f["vol_rel"] = f["volume"].to_numpy() / slot_median_before(vol)[i, f["slot"].to_numpy()]
    raw[i, f["slot"].to_numpy()] = f[CH].to_numpy(float)
    present[i, f["slot"].to_numpy()] = True
    x = np.nan_to_num(np.where(present[:, :, None], raw, 0.0))     # vol_rel в первую сессию NaN -> 0 (канал не нормируется z; маска дня сохраняется)
    cn = np.cumsum(present.sum(1))[:, None] * np.ones((1, K))
    return dict(days=days, raw=raw, present=present, cn=cn, cs=np.cumsum(x.sum(1), 0), css=np.cumsum((x * x).sum(1), 0))


def stats_row(store: dict, i: int) -> tuple:
    """Номер строки накопленных сумм для дня-входа i: сессии ≤ d (включая d), не дальше."""
    return i


def norm_stats(store: dict, i: int) -> tuple:
    j = stats_row(store, i)
    n = store["cn"][j]
    mu = store["cs"][j] / n
    sd = np.sqrt(np.maximum(store["css"][j] / n - mu * mu, 0.0))
    return mu, np.where(sd > 0, sd, 1.0)


def window_rows(i: int, W: int) -> np.ndarray:
    """Строки дней окна: последние W сессий ≤ d (i — строка дня d)."""
    return np.arange(i - W + 1, i + 1)


def day_block(store: dict, j: int) -> np.ndarray:
    return store["raw"][j]


def bar_mask(store: dict, j: int) -> np.ndarray:
    """Маска слотов дня j: бар есть. Слот без бара (неполный день, смена расписания) = 0."""
    return store["present"][j]


def x_of(store: dict, i: int, W: int) -> tuple:
    """-> (X (W, SLOTS, K) float, mask (W, SLOTS) bool, day_ok (W,) bool, norm_ok bool). Дни до начала истории — нули с маской 0."""
    mu, sd = norm_stats(store, i)
    X, M, ok = np.zeros((W, SLOTS, K)), np.zeros((W, SLOTS), bool), np.zeros(W, bool)
    for w, j in enumerate(window_rows(i, W)):
        if j < 0 or j >= len(store["days"]):
            continue
        b = np.where(ZCH, (day_block(store, j) - mu) / sd, day_block(store, j))
        M[w], ok[w] = bar_mask(store, j), True
        X[w] = np.where(M[w][:, None] & np.isfinite(b), b, 0.0)
    norm_ok = bool(i + 1 >= MIN_SESS)
    return X, M, ok, norm_ok


def truncate(bars: pd.DataFrame, days: pd.DatetimeIndex, d) -> tuple:
    """Данные, известные на конец дня d: бары с begin ≤ d (до конца суток) и дни ≤ d."""
    end = pd.Timestamp(d) + pd.Timedelta(days=1)
    return bars[bars["begin"] < end], pd.DatetimeIndex(days)[pd.DatetimeIndex(days) <= pd.Timestamp(d)]
