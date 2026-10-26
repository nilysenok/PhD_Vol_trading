"""Режимы fastbt. Ворота: Config.gate() = --leak-mask-old без исправлений (код c22a4b3).

Исправления (решение владельца С1, 25.10: включаются постоянно после ворот):
  fix_c_s12_val — валидация C у S1/S2 по полным позициям A (как у остальных);
  fix_b_median  — медиана σ в B только по прошлому (бары до начала тестового года);
  fix_d_units   — σ D переводится в годовые единицы (·√252), как target/floor/cap;
  fix_c_fill    — нет допустимой конфигурации C → C = A (а не пропуск строк);
  fix_rate / fix_sharpe — ставка ЦБ по решениям и Шарп mean(excess)/std(excess) в МСП.
comm — комиссия отбора на сторону (печати: 0,40%; честный пересчёт: 0,05%).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

FIXES = ("c_s12_val", "b_median", "d_units", "c_fill", "rate", "sharpe")


@dataclass(frozen=True)
class Config:
    mask_old: bool = False
    fix_c_s12_val: bool = True
    fix_b_median: bool = True
    fix_d_units: bool = True
    fix_c_fill: bool = True
    fix_rate: bool = True
    fix_sharpe: bool = True
    comm: float = 0.0005
    msp_comm: float = 0.0005
    years_a: tuple = tuple(range(2020, 2027))
    years_bcd: tuple = tuple(range(2022, 2027))
    corp_actions: bool = True      # data/corporate_actions.csv (в воротах — нет)
    val_start: int = 2020
    approaches: str = "ABCD"          # начало валидации B/C/D (год; в маске [1] — со следующего)
    design: tuple = ()                # варианты дизайна B/C/D (fastbt.design); () — как в коде
    # последний год окон валидации/калибровки (0 — до Y−1, расширяющееся окно, как в разработке).
    # Тест главы 4 (владелец 26.10, 10:05, п. 10/13): дизайны зафиксированы на разработке, на тесте
    # выбора нет — окна B/C/D и фильтра оборота заморожены на 2018–2021 (val_start=2018, val_end=2021)
    val_end: int = 0

    @staticmethod
    def gate() -> "Config":
        return Config(mask_old=True, comm=0.004, corp_actions=False,
                      **{f"fix_{f}": False for f in FIXES})

    def bound(self) -> int:
        """Индекс в _year_bounds для концов окон: [1] — старая утечка, [0] — без неё."""
        return 1 if self.mask_old else 0

    def cal_end_year(self, Y: int) -> int:
        """Год, с 1 января которого кончается окно калибровки для тестового года Y."""
        return Y if not self.val_end else min(Y, self.val_end + 1)

    def val_bounds(self, dates, Y: int) -> tuple:
        """(vs, ve) окна валидации B/C/D для года Y; при val_end пустое окно — ошибка."""
        from fastbt.data import year_bounds
        b = self.bound()
        vs, ve = year_bounds(dates, self.val_start)[b], year_bounds(dates, self.cal_end_year(Y))[b]
        if self.val_end and ve <= vs:
            raise ValueError(f"пустое окно валидации {self.val_start}…{self.val_end} для {Y}")
        return vs, ve

    def to_dict(self) -> dict:
        return asdict(self)


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--leak-mask-old", action="store_true",
                   help="режим ворот: маска [1] и поведение c22a4b3 без исправлений")
    for f in FIXES:
        p.add_argument(f"--fix-{f.replace('_', '-')}", dest=f"fix_{f}", default=None,
                       action=argparse.BooleanOptionalAction)
    p.add_argument("--comm", type=float, default=None, help="комиссия отбора на сторону")


def from_args(a) -> Config:
    base = Config.gate() if a.leak_mask_old else Config()
    kw = base.to_dict()
    for f in FIXES:
        v = getattr(a, f"fix_{f}")
        if v is not None:
            kw[f"fix_{f}"] = v
    if a.comm is not None:
        kw["comm"] = a.comm
    return Config(**kw)
