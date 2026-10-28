"""Справочный замер Ж-086 п. 4 / Ж-087 п. 5 (только train4, 2016–2019): новая цена выхода против старой.

Для каждого дня каждого тикера: |open бара 10:00 − open первого бара дня| и |open бара 15:00 − close бара 14:55|, в % от старой цены.
Медиана, p95, max — по всем дням и по годам. Плюс доля дней по времени начала первого бара (09:50 / 09:55 / другой).
5-минутки — из 1-минуток окна 09:50–18:50 (как fastbt/bars5m, s_predotkrytiem); чтение — через bars5m._read, обрезка до 2019-12-31.
Выбора нет, определение open 10:00 не меняется. Нет нужного бара в дне — день в метрику не входит (n_dnej показывает).
Запуск: python3 scripts/analysis/zamer_open_10.py <каталог bars1m> [выход.csv]; без аргументов — самопроверка на синтетике.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastbt import bars5m  # noqa: E402

START, END = "2016-01-01", "2019-12-31"                    # train4; данные ≥ 2020 не читаются
OUT = ROOT / "docs" / "tables" / "inv" / "zamer_open_10.csv"
F, T = 9 * 60 + 50, 18 * 60 + 50                           # окно 09:50–18:50
COLS = ["metrika", "god", "n_dnej", "mediana_pct", "p95_pct", "max_pct", "dolya_dnej"]


def chitat(ticker: str, src: Path) -> pd.DataFrame:
    m1 = bars5m._read(ticker, START, END, Path(src))
    if len(m1) and m1["begin"].max() >= pd.Timestamp(END) + pd.Timedelta(days=1):
        raise AssertionError(f"{ticker}: чтение за {END}")
    return m1


def po_dnyam(m1: pd.DataFrame) -> pd.DataFrame:
    """Индекс — дата; first_t (минуты начала первого бара), a = |o10 − o_первого|, c = |o15 − c14:55| (%, NaN — нет бара)."""
    cols = ["first_t", "a", "c"]
    if m1.empty:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([]), dtype=float)
    mod = m1["begin"].dt.hour * 60 + m1["begin"].dt.minute
    m = m1[(mod >= F) & (mod < T)].sort_values("begin")
    b = m.assign(ts=m["begin"].dt.floor("5min")).groupby("ts").agg(open=("open", "first"), close=("close", "last"))
    mt = b.index.hour * 60 + b.index.minute
    day = b.index.normalize()
    first = b.groupby(day).agg(o1=("open", "first"))
    first["first_t"] = pd.Series(mt, index=day).groupby(level=0).first()
    o10 = b["open"][mt == 600].set_axis(day[mt == 600])
    o15 = b["open"][mt == 900].set_axis(day[mt == 900])
    c55 = b["close"][mt == 895].set_axis(day[mt == 895])
    first["a"] = (o10.reindex(first.index) - first["o1"]).abs() / first["o1"] * 100
    first["c"] = (o15.reindex(first.index) - c55.reindex(first.index)).abs() / c55.reindex(first.index) * 100
    return first[cols]


def statistika(x: pd.Series) -> tuple[int, float, float, float]:
    x = x.dropna()
    if x.empty:
        return 0, np.nan, np.nan, np.nan
    return len(x), float(np.median(x)), float(np.percentile(x, 95)), float(x.max())


def tablica(dni: pd.DataFrame) -> pd.DataFrame:
    """dni: po_dnyam всех тикеров, склеенные. Строки: метрика × (все | год); доли первого бара."""
    rows, god = [], dni.index.year
    for name, col in (("open10_vs_pervyj", "a"), ("open15_vs_close1455", "c")):
        for g, sub in [("all", dni)] + [(int(y), dni[god == y]) for y in sorted(set(god))]:
            n, med, p95, mx = statistika(sub[col])
            rows.append([name, g, n, med, p95, mx, np.nan])
    for g, sub in [("all", dni)] + [(int(y), dni[god == y]) for y in sorted(set(god))]:
        for name, mask in (("pervyj_bar_0950", sub["first_t"] == 590), ("pervyj_bar_0955", sub["first_t"] == 595),
                           ("pervyj_bar_drugoj", ~sub["first_t"].isin([590, 595]))):
            rows.append([name, g, len(sub), np.nan, np.nan, np.nan, float(mask.mean()) if len(sub) else np.nan])
    return pd.DataFrame(rows, columns=COLS)


def sobrat(src: Path, tickers: list[str]) -> pd.DataFrame:
    return pd.concat([po_dnyam(chitat(t, src)) for t in tickers]).sort_index()


def tikery(src: Path) -> list[str]:
    return sorted({p.name.split("_")[0] for p in Path(src).glob("*_2016.csv")})


def run(src: Path, out: Path) -> None:
    tk = tikery(src)
    tab = tablica(sobrat(src, tk))
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.round(4).to_csv(out, index=False)
    print(f"тикеров {len(tk)}: {' '.join(tk)}\n{tab.round(4).to_string(index=False)}")


def selftest() -> None:
    idx = pd.date_range("2016-03-15 09:55", "2016-03-15 18:45", freq="5min")
    m1 = pd.DataFrame({"begin": idx, "open": 100.0, "close": 100.0})
    at = lambda hm: m1["begin"] == pd.Timestamp(f"2016-03-15 {hm}")  # noqa: E731
    m1.loc[at("09:55"), "open"], m1.loc[at("10:00"), "open"] = 99.0, 101.0
    m1.loc[at("15:00"), "open"] = 102.0
    d = po_dnyam(m1)
    assert abs(d["a"].iloc[0] - 2 / 99 * 100) < 1e-9 and abs(d["c"].iloc[0] - 2.0) < 1e-9 and d["first_t"].iloc[0] == 595
    t = tablica(d)
    assert t[(t.metrika == "pervyj_bar_0955") & (t.god == "all")]["dolya_dnej"].iloc[0] == 1.0
    print("самопроверка: ок")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    else:
        run(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else OUT)
