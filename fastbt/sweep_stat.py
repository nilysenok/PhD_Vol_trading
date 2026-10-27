"""Статистика движка перебора (Ж-031, Э1-а): периоды, цель, строка метрик, DSR(N), PBO по CSCV.

Периоды (решение владельца 27.10): настройка 2018, контроль 2019, train = оба, validation 2020–2021.
Validation в целевую функцию не входит никогда: `objective` берёт только годы из `train_years`,
а `Periods` не создаётся с пересечением train и validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from fastbt import metriki_e as me

LAM = 0.5            # λ цели: среднее годовых Шарпов − λ·σ по годам
END_ALLOWED = 2021   # ряд только ≤ 2021 (Н6)


@dataclass(frozen=True)
class Periods:
    setup: tuple = (2018,)
    control: tuple = (2019,)
    validation: tuple = (2020, 2021)

    def __post_init__(self):
        s, c, v = set(self.setup), set(self.control), set(self.validation)
        if s & c or s & v or c & v:
            raise ValueError(f"периоды пересекаются: настройка {sorted(s)}, контроль {sorted(c)}, validation {sorted(v)}")
        if max(s | c | v) > END_ALLOWED:
            raise ValueError(f"данные ≥ {END_ALLOWED + 1} запрещены")
        if max(s | c) >= min(v):
            raise ValueError("validation должна идти после train")

    @property
    def train(self) -> tuple:
        return tuple(self.setup) + tuple(self.control)

    def years(self, name: str) -> tuple:
        return {"setup": self.setup, "control": self.control, "train": self.train,
                "validation": self.validation}[name]


def year_slice(df: pd.DataFrame, years) -> pd.DataFrame:
    return df[df.index.year.isin(list(years))]


def yearly_sharpes(df: pd.DataFrame, years) -> dict:
    """{год: Шарп(ddof=1) избыточной}; меньше 2 дней — nan."""
    out = {}
    for y in years:
        x = df.loc[df.index.year == y, "excess"].values
        out[int(y)] = float(me.sharpe(x)) if len(x) > 1 else float("nan")
    return out


def objective(sh_by_year: dict, lam: float = LAM) -> float:
    """Среднее годовых Шарпов − λ·σ по годам (σ с ddof=1; один год — σ = 0)."""
    v = np.array(list(sh_by_year.values()), float)
    if len(v) == 0 or np.isnan(v).any():
        return float("nan")
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return float(v.mean() - lam * sd)


def metrics_row(df: pd.DataFrame, years) -> dict:
    """Стандартная строка метрик на периоде: Шарп, mSR, Сортино, MDD, оборот, доля дней, сделок/год, по годам."""
    q = year_slice(df, years)
    x = q["excess"].values
    ny = max(len(q) / 252.0, 1e-9)
    row = dict(sharpe=float(me.sharpe(x)), msr=float(me.msr(x)), sortino=float(me.sortino_e(x)),
               mdd=float(me.mdd(x)), oborot_y=float(q["turnover"].sum() / ny),
               dni_v_rynke=float((q["exposure"] > 0).mean()),
               sdelok_y=float(q["sides"].sum() / ny) if "sides" in q else float("nan"))
    for y, s in yearly_sharpes(q, years).items():
        row[f"sharpe_{y}"] = s
    return row


def dsr_n(r, sr_points, mult: int = 1) -> float:
    """DSR(N): N = число накопленных проб × mult (mult=2 при повторном взгляде на validation),
    V — дисперсия дневных SR проб. N=1 → SR* = 0 (выбора нет), DSR = PSR."""
    s = np.asarray(sr_points, float)
    n = len(s) * mult
    v = float(np.var(s, ddof=1)) if len(s) > 1 else 0.0
    return me.dsr(r, n, v)


from fastbt.sweep_pbo import PBO_BLOCKS, pbo_cscv  # noqa: E402,F401  (перенесено в sweep_pbo, v1.1)
