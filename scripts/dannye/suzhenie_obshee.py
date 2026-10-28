"""Общие части сужения test_locked до 2025+ (Ж-112 п. 1): хэши, даты, метаданные, разрешённый вывод.

Значения столбцов нигде не читаются и не печатаются: читается только столбец даты (белый список).
"""
import hashlib
import re
from bisect import bisect_right
from datetime import date
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

MSK = ZoneInfo("Europe/Moscow")
DATE_COLS = ("date", "Date", "datetime", "timestamp", "TRADEDATE", "begin", "d0")
T1_LO, T1_HI, H25_LO = date(2022, 1, 1), date(2024, 12, 31), date(2025, 1, 1)
MANIFEST_COLS = ("path", "size", "mtime", "sha256", "columns", "nrows", "date_col", "dmin", "dmax")
RAZRESHENO = frozenset(MANIFEST_COLS) | {"klass", "h", "cut", "status", "n_pre", "n_t1", "n_h25", "holdout_sha", "proveno",
                                                  "reason", "min_h25", "n_h25_2024"}


def vyvod(**row):
    """Единственный путь печати: только разрешённые ключи и скалярные значения."""
    for k, v in row.items():
        if k not in RAZRESHENO:
            raise ValueError(f"ключ вывода не разрешён: {k}")
        if not isinstance(v, (str, int, float, date, type(None))):
            raise TypeError(f"значение вывода не скаляр: {k}")
    print("\t".join(f"{k}={v}" for k, v in row.items()))


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def sha_tabl(t):
    """sha сериализованной таблицы (без вывода значений)."""
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, t.schema) as w:
        w.write_table(t)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def to_dates(values):
    """Ряд моментов -> даты по МСК."""
    s = pd.to_datetime(pd.Series(values))
    if s.dt.tz is not None:
        s = s.dt.tz_convert(MSK).dt.tz_localize(None)
    return s.dt.date


def find_date_col(names):
    return next((c for c in DATE_COLS if c in names), None)


def dates_of(path, col):
    """Отсортированные уникальные даты столбца даты (только он читается)."""
    return sorted(set(to_dates(pq.read_table(path, columns=[col]).column(0).to_pandas())))


def date_range(pf, col, path):
    """min/max столбца даты по статистике row-group; нет статистики — чтение одного столбца."""
    ix = pf.metadata.schema.names.index(col)
    los, his = [], []
    for i in range(pf.metadata.num_row_groups):
        st = pf.metadata.row_group(i).column(ix).statistics
        if st is None or not st.has_min_max:
            ds_ = dates_of(path, col)
            return ds_[0], ds_[-1]
        los.append(st.min)
        his.append(st.max)
    if not los:
        return None, None
    d = to_dates([min(los), max(his)])
    return d.iloc[0], d.iloc[1]


def meta(path, rel):
    """Строка манифеста: путь, размер, mtime, sha; для parquet — схема, строки, границы даты."""
    p = Path(path)
    row = dict(path=rel, size=p.stat().st_size, mtime=int(p.stat().st_mtime), sha256=sha_file(p),
               columns=None, nrows=None, date_col=None, dmin=None, dmax=None)
    if p.suffix != ".parquet":
        return row
    try:
        pf = pq.ParquetFile(p)
    except (pa.ArrowInvalid, OSError):
        return row
    names = pf.schema_arrow.names
    row.update(columns=",".join(names), nrows=pf.metadata.num_rows, date_col=find_date_col(names))
    if row["date_col"] and pf.metadata.num_rows:
        row["dmin"], row["dmax"] = date_range(pf, row["date_col"], p)
    return row


def h_of(name):
    m = re.search(r"(?<![A-Za-z0-9])h(\d+)(?![0-9])", Path(name).stem)
    return int(m.group(1)) if m else 0


KALENDAR = Path(__file__).resolve().parents[2] / "data/calendar/torgovye_dni_moex.csv"


@lru_cache(maxsize=1)
def kalendar():
    """Торговые дни MOEX <= 2024-12-31 (IMOEX 10м + MCFTR/RGBITR; позже 2024 в файле дат нет)."""
    with open(KALENDAR, encoding="utf-8") as f:
        next(f)
        return tuple(date.fromisoformat(ln.strip()) for ln in f if ln.strip())


def granitsa(dates, h, cal=None):
    """Последний торговый день d0, у которого d0 + h торговых дней <= 31.12.2024 (по календарю MOEX, не по файлу)."""
    cal = [d for d in (cal or kalendar()) if d <= T1_HI]
    i = len(cal) - 1 - h
    return cal[i] if i >= 0 else None


def okno_ok(dates, t1_max, h, cal=None):
    """Независимая проверка: цель строки t1_max (h-й торговый день после неё по календарю) <= 31.12.2024."""
    if t1_max is None:
        return True
    cal = [d for d in (cal or kalendar()) if d <= T1_HI]
    i = bisect_right(cal, t1_max) - 1
    return t1_max <= T1_HI and i >= 0 and i + h < len(cal)


def classify(rel, m, dates):
    """(класс а/б/в/г/д, граница cut, h) по Ш2."""
    if Path(rel).name.startswith("OPENED_"):
        return "д", None, 0
    if not m["date_col"] or not m["nrows"]:
        return "г", None, 0
    h = h_of(rel)
    cut = granitsa(dates, h)
    if m["dmin"] >= H25_LO or cut is None:
        return "б", cut, h
    # h=0: цель строки — она сама, нерабочая дата <= 31.12.2024 (строка 31.12.2024) остаётся открытой
    return ("а" if (T1_HI if h == 0 else cut) >= m["dmax"] else "в"), cut, h


def find_files(koren, vkhod, skip=()):
    out = set()
    for v in vkhod:
        p = Path(koren) / v
        for f in ([p] if p.is_file() else p.rglob("*")):
            rel = f.relative_to(koren).as_posix()
            if f.is_file() and not any(rel.startswith(s.rstrip("/") + "/") for s in skip):
                out.add(rel)
    return sorted(out)


def pishi_tsv(path, rows, cols=MANIFEST_COLS):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")


def chitai_tsv(path):
    with open(path, encoding="utf-8") as f:
        cols = f.readline().rstrip("\n").split("\t")
        return [dict(zip(cols, ln.rstrip("\n").split("\t"))) for ln in f if ln.strip()]
