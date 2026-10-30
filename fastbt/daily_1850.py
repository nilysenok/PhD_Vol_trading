"""Дневной ряд OHLCV до 18:50 МСК из 10-минуток (Н6, Ж-029, решение владельца 27.10 17:50, вариант (б)).

Информационное множество проекта едино: всё, что видит стратегия в день d, — до 18:50 дня d.
Старый ряд `ohlcv_daily_full.parquet` группировал 10-минутки по календарной дате без фильтра по
времени; с 22.06.2020 (LSRG — с 03.08.2020) в день попадали вечерние бары до 23:40, и close дня
был ценой 23:40, а не 18:50.

Правило: бар входит в день, если его начало лежит в [09:50; 18:50). Утренняя граница 09:50 —
как в `rv_1850_noquar` (moex_iss/src/rv1850.rs, окно W0950: бар предоткрытия 09:50 входит, как в старых
признаках; вечерние отброшены), поэтому до 19.06.2020 ряд совпадает со старым побайтно.
Последний бар до 18:50 — бар с началом 18:40 (он покрывает аукцион закрытия 18:40–18:50; баров
с началом 18:45 в 10-минутках нет; бар 18:50 встречается единично и отброшен). close = close
последнего бара с началом < 18:50, volume = сумма volume таких баров.
Следствие 09:50: утренние бары (06:50–09:40) с 06.12.2021 в день не входят, в отличие от старого ряда.

Ряд идёт с начала 10-минуток (06.2014: разогрев окна валидации 2017 и индикаторов), по 31.12.2021.
Данные ≥ 2022 не читаются: фильтр по timestamp < END идёт на чтении (pyarrow).
Файл кладётся рядом со старым рядом не в moex_volatility, а в data/fastbt/daily/ (в git не идёт);
старый ряд не перезаписывается. Запуск: `python3 -m fastbt.daily_1850 [--out PATH]`.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from fastbt.kalendar_moex import CALENDAR_MOEX, moex_sessions  # noqa: F401  (имена доступны как D.*)

END = pd.Timestamp("2022-01-01")          # исключительно: только 2018–2021
OPEN_MIN = 9 * 60 + 50                    # 09:50 — как rv_1850_noquar
CUT_MIN = 18 * 60 + 50                    # 18:50 — бар с началом ≥ 18:50 в день не входит
END_WF = pd.Timestamp("2026-03-01")       # wf (Ж-136): исключительно, т.е. по 2026-02-28; читать только при отметке владельца
WEEKEND_FROM = pd.Timestamp("2025-01-01")  # с неё день входит только как основная сессия календаря MOEX (Ж-142 п. 5:
#                                            рабочие субботы — да, ДСВД и праздники — нет); до неё правило прежнее
COLS =["ticker", "date", "open", "high", "low", "close", "volume"]
log = logging.getLogger(__name__)


def default_main() -> Path:
    return Path(os.environ.get("FASTBT_MAIN", "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading"))


def aggregate(bars: pd.DataFrame, ticker: str, cut_min: int = CUT_MIN, sessions=None) -> pd.DataFrame:
    """Дневные OHLCV из 10-минуток (колонки timestamp, open, high, low, close, volume).
    Нижняя граница — константа модуля OPEN_MIN (тесты-мутации подменяют её в памяти)."""
    b = bars.sort_values("timestamp")
    mod = b["timestamp"].dt.hour * 60 + b["timestamp"].dt.minute
    b = b[(mod >= OPEN_MIN) & (mod < cut_min)]
    day = b["timestamp"].dt.normalize()
    late = day >= WEEKEND_FROM
    # календарь MOEX читается лениво: только если в данных есть дни ≥ WEEKEND_FROM (train4/v2 файла не требуют; пакет без него)
    out_cal = late & ~day.isin((moex_sessions() if sessions is None else sessions) if late.any() else ())
    if out_cal.any():
        log.info("%s: дней с барами без основной сессии в календаре MOEX (не включены): %d", ticker, day[out_cal].nunique())
    b, day = b[~out_cal], day[~out_cal]
    g =b.groupby(day, sort=True)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum().astype("int64")})
    out.index.name = "date"
    out = out.reset_index()
    out.insert(0, "ticker", ticker)
    return out[COLS]


def read_bars(path: Path, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """end по умолчанию — граница зоны (zona.end(): 2022-01-01, в зоне Z2 — 2025-01-01); за HARD — отказ."""
    from fastbt import zona
    end = zona.end() if end is None else end
    zona.check(end - pd.Timedelta(days=1), "бары")
    t = pq.read_table(path, filters=[("timestamp", "<", end.to_pydatetime())])
    return t.to_pandas()


def build(main: Path | None = None, tickers: list[str] | None = None) -> pd.DataFrame:
    main = main or default_main()
    src = main / "moex_discovery" / "data" / "raw" / "candles_10m"
    if tickers is None:
        from fastbt.data import TICKERS
        tickers = TICKERS
    parts = [aggregate(read_bars(src / f"{t}.parquet"), t) for t in tickers]
    d = pd.concat(parts, ignore_index=True)
    from fastbt import zona
    d = d[d["date"] < zona.end()]
    return d.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_wf(main: Path | None = None, tickers: list[str] | None = None, end: pd.Timestamp = END_WF) -> pd.DataFrame:
    """Ряд для wf (06.2014 – end, исключительно): чтение только при отметке владельца (period_v2.proverit_chtenie).
    Участок < 2022 идентичен build(); после докачки 10-минуток — просто перезапуск."""
    from fastbt import period_v2
    from fastbt.data import TICKERS
    period_v2.proverit_chtenie("wf")
    src = (main or default_main()) / "moex_discovery" / "data" / "raw" / "candles_10m"
    flt = [("timestamp", "<", end.to_pydatetime())]
    parts = [aggregate(pq.read_table(src / f"{t}.parquet", filters=flt).to_pandas(), t) for t in (tickers or TICKERS)]
    d = pd.concat(parts, ignore_index=True)
    return d.sort_values(["ticker", "date"]).reset_index(drop=True)


def calendar_extend(old_days: list, daily: pd.DataFrame) -> list:
    """Календарь: прежние строки без изменений + дни ряда после них; ≥ WEEKEND_FROM — только основные сессии MOEX."""
    old = sorted(pd.Timestamp(x) for x in old_days)
    new = sorted(set(pd.to_datetime(daily["date"])) - set(old))
    new = [d for d in new if d > old[-1]]
    ses = moex_sessions() if any(d >= WEEKEND_FROM for d in new) else frozenset()    # лениво: до 2025 файл не нужен
    new = [d for d in new if not (d >= WEEKEND_FROM and d not in ses)]
    return old + new


def calendar_path(main: Path | None = None) -> Path:
    return (main or default_main()) / "data" / "calendar" / "torgovye_dni_moex.csv"


def out_path(main: Path | None = None) -> Path:
    return (main or default_main()) / "data" / "fastbt" / "daily" / "ohlcv_daily_1850.parquet"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def marker(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def is_ready(path: Path) -> bool:
    """Готов = файл есть, маркер завершения есть и его хэш равен хэшу файла (не просто «существует»)."""
    m = marker(path)
    return path.exists() and m.exists() and m.read_text().strip() == sha256(path)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def ensure(path: Path, main: Path | None = None) -> Path:
    """Собрать ряд, если он не готов (data/fastbt в git не идёт). Запись: временный файл в той же папке,
    os.replace, затем маркер с хэшем; параллельные процессы идут по очереди под flock."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(f".{path.name}.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not is_ready(path):
            marker(path).unlink(missing_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.parquet.tmp")
            build(main).to_parquet(tmp, index=False)
            os.replace(tmp, path)
            _atomic_write(marker(path), sha256(path).encode())
    return path


def check_data_root(root: str | Path) -> Path:
    """--data-root: только копия НОВОГО ряда (ohlcv_daily_1850.parquet, даты ≤ 2021). Старый ряд
    ohlcv_daily_full (вечерняя сессия) — режим «как до Н6», Ж-019 п. 11 его запрещает. Возвращает путь к ряду."""
    r = Path(root)
    f = r if r.is_file() else r / "ohlcv_daily_1850.parquet"
    if f.name == "ohlcv_daily_full.parquet" or (r.is_dir() and (r / "ohlcv_daily_full.parquet").exists()):
        raise SystemExit(f"--data-root {root}: старый ряд ohlcv_daily_full (с вечерней сессией) запрещён (Ж-019 п. 11); "
                         "нужна копия ohlcv_daily_1850.parquet")
    if f.name != "ohlcv_daily_1850.parquet" or not f.exists():
        raise SystemExit(f"--data-root {root}: нет ohlcv_daily_1850.parquet (копии нового ряда)")
    mx = pd.read_parquet(f, columns=["date"])["date"].max()
    if pd.Timestamp(mx) >= END:
        raise SystemExit(f"--data-root {root}: ряд содержит даты ≥ 2022 ({mx}) — это не новый ряд")
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wf", action="store_true", help="ряд до 2026-02-28 + календарь (в --out-каталог data/…)")
    ap.add_argument("--calendar-out", type=Path, default=None)
    a = ap.parse_args()
    if a.wf:
        d = build_wf()
        p = a.out or out_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(p, index=False)
        cal = calendar_extend(list(pd.read_csv(calendar_path())["date"]), d)
        cp = a.calendar_out or p.parents[2] / "calendar" / "torgovye_dni_moex.csv"
        cp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": [x.strftime("%Y-%m-%d") for x in cal]}).to_csv(cp, index=False)
        print(p, sha256(p), cp, d["date"].max().date())
        raise SystemExit(0)
    p = a.out or out_path()
    ensure(p)
    print(p, sha256(p))
