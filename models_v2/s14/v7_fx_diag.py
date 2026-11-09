"""V7-FX, расследование: согласованы ли координаты z энкодеров разных лет? Для Y = 2018…2025, seed 0–4, TCN и CNN (окно 5): корреляция каждой компоненты z энкодера года Y+1 на датах года Y
с сохранённым z энкодера года Y на тех же датах (со знаком, 4×4), плюс перестановка компонент с максимумом Σ|corr|. Только чтение артефактов Stage A; `FX_DIAG.csv` пишет `write()`.
Если у всех компонент corr ≈ +1 по диагонали — рассогласования нет (гипотеза отброшена)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_io as IO, v7_1b_sta as STA

YEARS = tuple(range(2018, 2026))
STATUS = "DEVELOPMENT / диагностика, не оценка; Ж-237 решение [V7] лист FX: рассогласование координат z энкодеров разных лет (корреляции на датах года Y; без сравнений, p и окон 2022+ как критерия)"


def corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """C[i, j] = corr(a[:, i], b[:, j]) (Пирсон, со знаком)."""
    a, b = a - a.mean(0), b - b.mean(0)
    den = np.sqrt((a ** 2).sum(0))[:, None] * np.sqrt((b ** 2).sum(0))[None, :]
    return np.einsum("ni,nj->ij", a, b) / np.where(den > 0, den, np.nan)


def best_perm(c: np.ndarray) -> tuple:
    """Перестановка p (компонента i энкодера Y ↔ компонента p[i] энкодера Y+1) с максимумом Σ|c[i, p[i]]|. -> (p, |c| по перестановке суммарно)."""
    k = c.shape[0]
    best = max(itertools.permutations(range(k)), key=lambda p: np.abs(c[np.arange(k), list(p)]).sum())
    return best, float(np.abs(c[np.arange(k), list(best)]).sum())


def shared(load, kind: str, seed: int, year: int, window: int = STA.WINDOW) -> tuple:
    """z энкодера года Y и энкодера года Y+1 на ОБЩИХ датах года Y. -> (zY, zY1)."""
    a, b = (load(STA.directory() / STA.name(kind, y, seed, window)) for y in (year, year + 1))
    da, db = (pd.DatetimeIndex(x["days"].astype("datetime64[ns]")) for x in (a, b))
    common = da[(da.year == year)].intersection(db)
    return a["z"][da.get_indexer(common)], b["z"][db.get_indexer(common)]


def rows(load=STA.load, kinds=STA.KINDS, seeds=STA.SEEDS, years=YEARS) -> pd.DataFrame:
    out = []
    for kind in kinds:
        for y in years:
            for s in seeds:
                za, zb = shared(load, kind, s, y)
                c = corr_matrix(za, zb)
                p, tot = best_perm(c)
                for i in range(c.shape[0]):
                    out.append(dict(kind=kind, year=y, seed=s, comp=i + 1, n_dates=len(za), corr_same=c[i, i], corr_best_perm=c[i, p[i]], perm_comp=p[i] + 1,
                                    perm_is_identity=bool(list(p) == list(range(c.shape[0]))), sum_abs_perm=tot))
    return pd.DataFrame(out)


def write() -> pd.DataFrame:
    df = rows()
    IO.write_csv(df, "FX_DIAG.csv", "FX: corr(z энкодера Y+1, z энкодера Y) на датах года Y, по компонентам; corr_same — одноимённые компоненты, со знаком; corr_best_perm — по перестановке с max Σ|corr|", status=STATUS)
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """По (kind, comp): медиана и минимум corr_same, доля |corr_same| < 0,9, доля отрицательных, доля нетождественных перестановок."""
    g = df.groupby(["kind", "comp"])
    return pd.DataFrame(dict(median_corr=g["corr_same"].median(), min_corr=g["corr_same"].min(), share_abs_lt_09=g["corr_same"].apply(lambda s: float((s.abs() < 0.9).mean())),
                             share_negative=g["corr_same"].apply(lambda s: float((s < 0).mean())), share_perm_not_identity=g["perm_is_identity"].apply(lambda s: float((~s).mean()))))


if __name__ == "__main__":
    d = write()
    print(summary(d).round(3).to_string())
