"""Л-8: walk-forward по 8 правилам отбора региона и PBO_wf (прогон этапа 1, train4).

OOS — кварталы 2018Q1…2019Q4 (индексы 8…15), общие для всех 8 правил.
Для OOS-квартала q регион выбирается заново по W кварталам строго до q
(`vybor_region` на `kvartaly_idx = q-W … q-1`), его ряд берётся на днях q.
Данных ≥ 2020 модуль не читает: M, rows, daty передаёт вызывающий (T = 1010).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt.metriki_e import sr_dnevnoy
from fastbt.progon_region import ryad_regiona, vybor_region
from fastbt.sweep_drivers import pbo_cscv

PRAVILA = [(W, r, t) for W in (4, 8) for r in (1, 2) for t in ("mediana", "dolya")]
OSNOVNOE = (8, 1, "mediana")  # объявлено до счёта
OOS_KV = tuple(range(8, 16))  # 2018Q1…2019Q4 в нумерации 2016Q1 = 0
SQRT252 = float(np.sqrt(252.0))


def _label(k: int) -> str:
    return f"{2016 + k // 4}Q{k % 4 + 1}"


def kvartal_dnya(daty) -> np.ndarray:
    """Индекс календарного квартала каждого дня: 2016Q1 = 0."""
    d = pd.DatetimeIndex(np.asarray(daty, dtype="datetime64[D]"))
    return (d.year - 2016) * 4 + (d.month - 1) // 3


def _sr(x: np.ndarray) -> float:
    return float("nan") if len(x) < 2 else sr_dnevnoy(x) * SQRT252


def okno_otbora(W: int, q: int) -> list:
    """Кварталы отбора для OOS-квартала q: W штук строго до q."""
    return list(range(q - W, q))


def wf(M, rows, daty, pravilo) -> dict:
    W, radius, tsel = pravilo
    M = np.asarray(M, dtype=float)
    kd = kvartal_dnya(daty)
    if M.shape[0] != len(kd):
        raise ValueError("M и daty разной длины")
    parts, by_q, regiony = [], {}, {}
    for q in OOS_KV:
        okno = okno_otbora(W, q)
        if okno[0] < 0 or okno[-1] >= q:
            raise ValueError("окно отбора вне допустимых кварталов")
        reg = vybor_region(M, rows, daty, kvartaly_idx=okno, radius=radius, tsel=tsel)
        mask = kd == q
        # нет допустимого региона — позиции нет, ряд нулевой (не подмена другим выбором)
        part = np.zeros(int(mask.sum())) if reg is None else ryad_regiona(M, reg["chleny"])[mask]
        parts.append(part)
        by_q[_label(q)] = _sr(part)
        regiony[_label(q)] = None if reg is None else reg["centr"]
    oos = np.concatenate(parts)
    return {"pravilo": pravilo, "oos_ryad": oos, "oos_by_quarter": by_q, "region": regiony}


def tablica(M, rows, daty) -> pd.DataFrame:
    out = []
    for p in PRAVILA:
        r = wf(M, rows, daty, p)
        v = np.array(list(r["oos_by_quarter"].values()), dtype=float)
        out.append({
            "pravilo": p, "W": p[0], "radius": p[1], "tsel": p[2],
            "mediana_oos_kv": float(np.median(v)),
            "dolya_pol_oos_kv": float(np.sum(v > 0) / len(v)),
            "sharpe_oos": _sr(r["oos_ryad"]),
            "region": r["region"],
        })
    return pd.DataFrame(out)


def oos_matrica(M, rows, daty) -> np.ndarray:
    """T_oos × 8: столбец на правило, порядок PRAVILA; общий OOS."""
    return np.column_stack([wf(M, rows, daty, p)["oos_ryad"] for p in PRAVILA])


def pbo_wf(oos: np.ndarray) -> float:
    """CSCV по T_oos × 8 правил; 8 блоков (равные по дням, ≈ 8 OOS-кварталов); выбор argmax."""
    oos = np.asarray(oos, dtype=float)
    if oos.ndim != 2 or oos.shape[1] != len(PRAVILA):
        raise ValueError(f"нужно {len(PRAVILA)} правил, получено {oos.shape}")
    return float(pbo_cscv(oos.tolist(), n_blocks=8)["pbo"])
