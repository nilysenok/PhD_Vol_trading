"""Правило качества данных и состав вселенной по месяцам (Ж-114 п. 1, 28.10.2026), одно для всех периодов.

- тикер-день, где в окне [09:50; 18:50) меньше 30 десятиминуток, невалиден (нет сигнала, нет σ за день;
  открытая позиция держится по правилам выхода — это забота стратегии, здесь только флаг);
- решение о составе принимается в первый торговый день месяца по 60 торговым дням ДО него (день решения
  в окно не входит): входит при доле валидных дней >= 95 %, выходит при < 80 %, между — остаётся как был
  (гистерезис); возвращается по тому же правилу. Состояние месяца действует на все его дни, поэтому для
  любого d оно считано по данным не позже d-1;
- IRAO: 01.12.2014 – 28.02.2015 невалиден поверх правила (консолидация), по дням;
- кросс-секция S7–S9: во вселенной минимум 10 тикеров, иначе день пропускается.

Торговый календарь — объединение дат, где у любого из 17 тикеров есть бар окна. Читается готовым
`daily_1850.read_bars` (конец 2020-01-01 исключительно, ≥ 2020 не читается). К стратегиям не подключено.
Запуск: `python3 -m fastbt.vselennaya_kachestvo` пишет таблицы в docs/tables/inv/.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from fastbt import daily_1850 as d18

# Те же 17, что fastbt.data.TICKERS; импорт fastbt.data запускает сборку дневного ряда, поэтому список здесь.
TICKERS = sorted(["AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX",
                  "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "SBER", "TATN", "VTBR"])
END = pd.Timestamp("2020-01-01")
MIN_BARS = 30
WINDOW = 60
ENTER, EXIT = 0.95, 0.80
MIN_TIK = 10
IRAO_BAD = {"IRAO": (pd.Timestamp("2014-12-01"), pd.Timestamp("2015-02-28"))}
OUT = Path("docs/tables/inv")


def _okno(i: int) -> slice:
    """Окно доли для решения в позиции i календаря: 60 дней строго до i."""
    return slice(i - WINDOW, i)


def bars_per_day(tiker: str, main: Path | None = None) -> pd.Series:
    """Число десятиминуток окна [09:50; 18:50) по дням (до 2020-01-01)."""
    p = (main or d18.default_main()) / "moex_discovery" / "data" / "raw" / "candles_10m" / f"{tiker}.parquet"
    b = d18.read_bars(p, END)
    mod = b["timestamp"].dt.hour * 60 + b["timestamp"].dt.minute
    b = b[(mod >= d18.OPEN_MIN) & (mod < d18.CUT_MIN)]
    return b.groupby(b["timestamp"].dt.normalize()).size()


class Kachestvo:
    """counts: DataFrame (индекс — торговые даты по возрастанию, колонки — тикеры, значения — число баров)."""

    def __init__(self, counts: pd.DataFrame):
        self.counts = counts.sort_index()
        self.dates = self.counts.index
        self.valid = pd.DataFrame({t: self._valid_col(t) for t in self.counts.columns})
        self._state, self._dolya = self._members()

    def _valid_col(self, t: str) -> pd.Series:
        ok = self.counts[t] >= MIN_BARS
        if t in IRAO_BAD:
            a, b = IRAO_BAD[t]
            ok = ok & ~((self.dates >= a) & (self.dates <= b))
        return ok

    def _members(self):
        first = self.dates.to_series().groupby(self.dates.to_period("M")).first()
        state = {t: {} for t in self.valid.columns}
        dolya = {t: {} for t in self.valid.columns}
        cur = dict.fromkeys(self.valid.columns, False)
        for m, d0 in first.items():
            i = int(self.dates.get_loc(d0))
            for t in self.valid.columns:
                if i < WINDOW:
                    share = float("nan")
                else:
                    share = float(self.valid[t].iloc[_okno(i)].mean())
                    if share >= ENTER:
                        cur[t] = True
                    elif share < EXIT:
                        cur[t] = False
                state[t][m], dolya[t][m] = cur[t], share
        return state, dolya

    def valid_den(self, tiker: str, d) -> bool:
        d = pd.Timestamp(d)
        return bool(d in self.dates and self.valid[tiker].loc[d])

    def vo_vselennoj(self, tiker: str, d) -> bool:
        return self._state[tiker].get(pd.Timestamp(d).to_period("M"), False)

    def sostav(self, d) -> list[str]:
        return [t for t in self.valid.columns if self.vo_vselennoj(t, d)]

    def den_ok_s7_s9(self, d) -> bool:
        """False — день кросс-секции S7–S9 пропускается (во вселенной меньше MIN_TIK тикеров)."""
        return len(self.sostav(d)) >= MIN_TIK

    def sostav_po_mesyacam(self) -> pd.DataFrame:
        rows = [(str(m), t, int(s), self._dolya[t][m])
                for t in self.valid.columns for m, s in self._state[t].items()]
        df = pd.DataFrame(rows, columns=["mesyac", "tiker", "vo_vselennoj", "dolya_60"])
        return df.sort_values(["mesyac", "tiker"]).reset_index(drop=True)

    def intervaly_vne(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Подряд идущие месяцы вне вселенной (по таблице df, по умолчанию — по всей)."""
        df = self.sostav_po_mesyacam() if df is None else df
        rows = []
        for t, g in df.groupby("tiker"):
            run = None
            for m, s in zip(g["mesyac"], g["vo_vselennoj"]):
                if not s:
                    run = (run[0], m, run[2] + 1) if run else (m, m, 1)
                elif run:
                    rows.append((t, *run))
                    run = None
            if run:
                rows.append((t, *run))
        return pd.DataFrame(rows, columns=["tiker", "s_mesyaca", "po_mesyac", "mesyacev"])


@lru_cache(maxsize=1)
def default(main: Path | None = None) -> Kachestvo:
    cols = {t: bars_per_day(t, main) for t in TICKERS}
    return Kachestvo(pd.DataFrame(cols).fillna(0).astype(int))


def valid_den(tiker: str, d) -> bool:
    return default().valid_den(tiker, d)


def vo_vselennoj(tiker: str, d) -> bool:
    return default().vo_vselennoj(tiker, d)


def sostav_po_mesyacam() -> pd.DataFrame:
    return default().sostav_po_mesyacam()


def main() -> None:
    k = default()
    df = k.sostav_po_mesyacam()
    df = df[(df["mesyac"] >= "2014-08") & (df["mesyac"] <= "2019-12")]
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "vselennaya_po_mesyacam.csv", index=False, float_format="%.4f")
    vne = k.intervaly_vne(df)
    vne.to_csv(OUT / "vselennaya_vne.csv", index=False)
    print(vne.to_string(index=False))


if __name__ == "__main__":
    main()
