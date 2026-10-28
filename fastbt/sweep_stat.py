"""Оценщик метрик ряда (Ж-031, Э1-а): периоды и строка метрик. Цель, DSR, PBO, плато — в `sweep_drivers.py` (Д7).

Периоды (решение владельца 27.10): настройка 2018, контроль 2019, train = оба, validation 2020–2021.
Validation в целевую функцию не входит никогда: `Periods` не создаётся с пересечением train и validation,
а `Sweep` не отдаёт годы validation, пока не вызван режим validation.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fastbt import metriki_e as me

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


def metrics_row(df: pd.DataFrame, years) -> dict:
    """Строка метрик на периоде. sharpe — годовой (ddof=1), sr_d — дневной; T, skew, kurt (сырой, нормаль = 3) — для DSR;
    by_year — {год: Шарп}."""
    q = year_slice(df, years)
    x = q["excess"].values
    ny = max(len(q) / 252.0, 1e-9)
    return dict(sharpe=float(me.sharpe(x)), sr_d=float(me.sr_dnevnoy(x)), msr=float(me.msr(x)),
                sortino=float(me.sortino_e(x)), mdd=float(me.mdd(x)), T=int(len(x)),
                skew=float(me.asimmetriya(x)), kurt=float(me.ekscess(x) + 3.0),
                oborot_y=float(q["turnover"].sum() / ny), dni_v_rynke=float((q["exposure"] > 0).mean()),
                sdelok_y=float(q["sides"].sum() / ny) if "sides" in q else float("nan"),
                by_year=yearly_sharpes(q, years))
