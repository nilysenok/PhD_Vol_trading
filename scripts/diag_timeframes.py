"""Быстрая проверка таймфреймов (Ж-098): 6 ТФ × 3 семейства «голой идеи», train4 2016–2019 (прогрев 2015), 17 тикеров.

Бары — fastbt.bars_agg поверх готовых 10-минуток (candles_10m, чтение строго < 2020-01-01). Отбора нет — карта.
Семейства (сторона — по смыслу, обе против движения кроме тренда): контртренд z=(close−SMA20)/STD20, |z|>2 → против z
(z>2 шорт, z<−2 лонг); тренд — close пробил max high / min low предыдущих 20 баров → по пробою; диапазон — |close−VWAP20|>1,5·dev
(VWAP по типичной цене (h+l+c)/3, dev — взвешенное по объёму СКО) → к VWAP. Окна 20 баров идут сквозь дни.
Режимы: den_1850 — сигнал только на последнем баре дня до 18:50 (рамка Ж-098 п. 6; день/неделя — всегда он);
kazhdyj_bar — сигнал на любом баре (диагностика, куда уходит информация и сколько съедают издержки).
Вход — open следующего бара (для дня/недели и последнего бара дня — open 10:00 d+1). Выход: close бара i+h (h=1,3,10,30
баров того же ТФ) или close дня d+1 («d1»). Избыточная доходность = r − безусловное среднее r (тикер, горизонт, режим);
спред = среднее side·избыточная. Кластерное t — по дню сигнала. Контроль — 100 перемешиваний вектора сигналов (сторона
вместе с сигналом) внутри тикера; ctrl_poz — доля перемешанных спредов ниже фактического. Издержки 0,08 % на круг.
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastbt import bars_agg as ba  # noqa: E402
from fastbt.data import TICKERS  # noqa: E402

MAIN = Path(os.environ.get("TF_MAIN", "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading"))
OUT = Path(__file__).resolve().parent.parent / "docs" / "tables" / "progon_e1" / "timeframes_golaya.csv"
TFS, FAMS, HOR, N, COST, REPS = ("10m", "30m", "1h", "2h", "den", "nedelya"), ("kontrtrend", "trend", "diapazon"), (1, 3, 10, 30), 20, 0.0008, 100
A, B = pd.Timestamp("2016-01-01"), pd.Timestamp("2020-01-01")
STATUS = "разработка train4 2016-2019, до аудита; потолок, не результат"


def frames(ticker: str) -> dict:
    b10 = ba.load_10m(ticker, "2020-01-01", MAIN)
    b10 = b10[b10["date"] >= pd.Timestamp("2015-01-01")]
    dd = ba.daily(b10)
    fr = {k: ba.resample(b10, ba.K[k]) for k in ("10m", "30m", "1h", "2h")}
    fr["den"], fr["nedelya"] = dd, ba.weekly(dd)
    dates = dd["date"].values
    for f in fr.values():
        f["day_idx"] = np.searchsorted(dates, f["date"].values)
    return fr, dd["close"].values


def signal(f: pd.DataFrame, fam: str) -> np.ndarray:
    c = f["close"]
    if fam == "kontrtrend":
        z = (c - c.rolling(N).mean()) / c.rolling(N).std()
        return np.where(z > 2, -1, np.where(z < -2, 1, 0))
    if fam == "trend":
        hi, lo = f["high"].rolling(N).max().shift(1), f["low"].rolling(N).min().shift(1)
        return np.where(c > hi, 1, np.where(c < lo, -1, 0))
    p, v = (f["high"] + f["low"] + c) / 3, f["volume"].astype(float)
    vs = v.rolling(N).sum()
    vw = (p * v).rolling(N).sum() / vs
    dev = np.sqrt(np.maximum((p * p * v).rolling(N).sum() / vs - vw * vw, 0))
    dv = c - vw
    return np.where((dev > 0) & (dv > 1.5 * dev), -1, np.where((dev > 0) & (dv < -1.5 * dev), 1, 0))


def returns(f: pd.DataFrame, dclose: np.ndarray) -> np.ndarray:
    """Матрица (бары × [h1,h3,h10,h30,d1]) сырых доходностей лонга от open следующего бара."""
    n, c = len(f), f["close"].values
    ent = np.append(f["open"].values[1:], np.nan)
    out = np.full((n, len(HOR) + 1), np.nan)
    for j, h in enumerate(HOR):
        if n > h:
            out[: n - h, j] = c[h:] / ent[: n - h] - 1
    di = f["day_idx"].values + 1
    ok = di < len(dclose)
    out[ok, -1] = dclose[di[ok]] / ent[ok] - 1
    return out


def one(f: pd.DataFrame, dclose, fam: str, mode: str, rng) -> tuple:
    side = signal(f, fam)
    dt = f["date"]
    elig = ((dt >= A) & (dt < B)).values
    if mode == "den_1850":
        elig &= (dt != dt.shift(-1)).values
    r = returns(f, dclose)[elig]
    x = r - np.nanmean(r, axis=0)
    s, d = side[elig], dt.values[elig]
    sg = s != 0
    sh_s, sh_c = np.zeros((REPS, x.shape[1])), np.zeros((REPS, x.shape[1]))
    for i in range(REPS):
        sp = s[rng.permutation(len(s))]
        v = sp[sp != 0, None] * x[sp != 0]
        sh_s[i], sh_c[i] = np.nansum(v, 0), (~np.isnan(v)).sum(0)
    return d[sg], s[sg, None] * x[sg], sh_s, sh_c, len(np.unique(d)), len(np.unique(d[sg]))


def worker(args: tuple) -> dict:
    k, ticker = args
    fr, dclose = frames(ticker)
    rng = np.random.default_rng(1000 + k)
    res = {}
    for tf in TFS:
        for fam in FAMS:
            for mode in (("den_1850",) if tf in ("den", "nedelya") else ("den_1850", "kazhdyj_bar")):
                res[(tf, fam, mode)] = one(fr[tf], dclose, fam, mode, rng)
    return res


def cluster_t(days: np.ndarray, x: np.ndarray) -> float:
    g = pd.DataFrame({"d": days, "x": x}).groupby("d")["x"].agg(["sum", "count"])
    m = g["sum"].sum() / g["count"].sum()
    se = np.sqrt(((g["sum"] - g["count"] * m) ** 2).sum()) / g["count"].sum()
    return float(m / se) if se > 0 else float("nan")


def rows(key: tuple, parts: list) -> list:
    tf, fam, mode = key
    d = np.concatenate([p[0] for p in parts])
    x = np.concatenate([p[1] for p in parts])
    sh_s, sh_c = sum(p[2] for p in parts), sum(p[3] for p in parts)
    nd, nds = sum(p[4] for p in parts), sum(p[5] for p in parts)
    out = []
    for j, h in enumerate((*HOR, "d1")):
        ok = ~np.isnan(x[:, j])
        m = float(x[ok, j].mean()) if ok.any() else float("nan")
        sh = sh_s[:, j] / np.maximum(sh_c[:, j], 1)
        out.append(dict(tf=tf, semejstvo=fam, rezhim=mode, gorizont=h, n_sig=int(ok.sum()), sig_v_god=len(d) / 4,
                        dolya_dnej=nds / nd, spred_pct=100 * m, t_klaster=cluster_t(d[ok], x[ok, j]),
                        spred_posle_izderzhek_pct=100 * (m - COST), ctrl_poz=float((sh < m).mean()),
                        ctrl_sred_pct=100 * float(sh.mean()), status=STATUS))
    return out


def main() -> None:
    with Pool(4) as pool:
        per = pool.map(worker, list(enumerate(TICKERS)))
    out = [r for key in per[0] for r in rows(key, [p[key] for p in per])]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).round(4).to_csv(OUT, index=False)
    print(f"{len(out)} строк, {len(TICKERS)} тикеров -> {OUT}")


if __name__ == "__main__":
    main()
