"""Внешние ряды на 18:50 d: Brent (BR FORTS), USDRUB TOM, RVI (К-24а, контракт #376 §2Б.6, Н6).

Значение дня d — close последнего 10-минутного бара с началом < 18:50 (`daily_1850.aggregate`); вечерняя
сессия и закрытие ICE (после 18:50) в d не входят. Данные ≥ 2022 не читаются: фильтр по `begin` на чтении.
Brent: ближний контракт; на день d активен первый контракт, у которого d < lsttrade − ROLL_DAYS (5) рабочих дней
(по `data/iss_ext/br_contracts.json`; календарь известен заранее — PIT). Склейка по доходностям: ret_d =
ln(close_d / close_prev) внутри ОДНОГО контракта (уровни контрактов не сшиваются, Д4). Дня нет в активном
контракте — строки нет, без ffill; число таких дней — `missing_days`. Рабочие дни — пн–пт (календарь бирж
не используется: граница перехода ±1 день не влияет на склейку по доходностям).
Изменение для z (К-24б): Brent и USDRUB — лог-доходность, RVI — разность.
Константы CUT_MIN, ROLL_DAYS, SPLICE_BY_LEVEL, END — для тестов-мутаций (подмена в памяти).
Запуск: `python3 -m fastbt.vneshnie_1850` — проверка сдвига (Д1) → docs/tables/diag/vneshnie_1850_sdvig.csv.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fastbt import daily_1850 as d18

CUT_MIN = d18.CUT_MIN      # 18:50; 24·60 = вечерние бары в день (мутация)
ROLL_DAYS = 5              # переход на следующий контракт за 5 рабочих дней до lsttrade; 0 = мутация
SPLICE_BY_LEVEL = False    # True = доходность по склеенному уровню (скачок на смене контракта; мутация)
END = pd.Timestamp("2022-01-01")   # умолчание (train/dev); wf — только через `granitsa("wf")` при отметке владельца
IDX = "moex_discovery/data/final_v6/candles_10m/external/"
CUR = "moex_discovery/data/external/moex_iss/currency_10m/"
PATHS = {"rvi": IDX + "RVI.parquet", "usdrub": CUR + "USD000UTSTOM.parquet",
         "cnyrub": CUR + "CNYRUB_TOM.parquet",   # замена USDRUB после обрыва 11.06.2024 (склейка — `vneshnie_usdrub`)
         "imoex": IDX + "IMOEX.parquet"}


def granitsa(period: str | None = None, otmetka=None) -> pd.Timestamp:
    """Граница чтения (строго <). wf: конец периода + 1 день (2026-03-01), без отметки снятого замка — отказ (Ж-136)."""
    if period != "wf":
        return END
    from fastbt import period_v2
    period_v2.proverit_chtenie(period, otmetka=period_v2.OTMETKA if otmetka is None else otmetka)
    return period_v2.PERIODY[period].konec + pd.Timedelta(days=1)
BR_DIR, BR_JSON = "data/iss_ext/bars_10m/", "data/iss_ext/br_contracts.json"
SHIFTS = (-2, -1, 0, 1)


def read_bars(path, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """10-минутки с колонкой timestamp; строки begin ≥ end (по умолчанию 2022-01-01) не читаются."""
    end = END if end is None else end
    typ = pq.ParquetFile(path).schema_arrow.field("begin").type
    lit = end.strftime("%Y-%m-%d") if pa.types.is_string(typ) else end.to_pydatetime()
    b = pq.read_table(path, filters=[("begin", "<", lit)]).to_pandas()
    b["timestamp"] = pd.to_datetime(b["begin"])
    return b


def close_1850(bars: pd.DataFrame, name: str) -> pd.Series:
    d = d18.aggregate(bars, name, cut_min=CUT_MIN)
    return d.set_index("date")["close"].astype(float)


def change(name: str, s: pd.Series) -> pd.Series:
    """Изменение за день: лог-доходность (brent, usdrub), разность (rvi); по соседним наблюдениям ряда."""
    return s.diff() if name == "rvi" else np.log(s).diff()


def roll_date(lsttrade) -> pd.Timestamp:
    d = np.busday_offset(np.datetime64(pd.Timestamp(lsttrade).date()), -ROLL_DAYS, roll="backward")
    return pd.Timestamp(d)


def contract_table(main: Path) -> pd.DataFrame:
    main = resolve_root(main)
    c = pd.DataFrame(json.loads((main / BR_JSON).read_text()))
    c["lsttrade"] = pd.to_datetime(c["lsttrade"])
    c = c.sort_values("lsttrade").reset_index(drop=True)
    c["roll"] = [roll_date(x) for x in c["lsttrade"]]
    return c[["secid", "lsttrade", "roll"]]


def active_secid(dates, table: pd.DataFrame) -> pd.Series:
    """Активный контракт на каждую дату: первый с d < roll. Зависит только от календаря (PIT)."""
    rolls = table["roll"].to_numpy(dtype="datetime64[ns]")
    i = np.searchsorted(rolls, pd.DatetimeIndex(dates).to_numpy(dtype="datetime64[ns]"), side="right")
    sec = table["secid"].to_numpy()
    return pd.Series([sec[k] if k < len(sec) else None for k in i], index=pd.DatetimeIndex(dates))


def splice_brent(per_contract: dict, table: pd.DataFrame) -> pd.DataFrame:
    """per_contract {secid: close 18:50 по дням} → date, secid, close, ret (склейка по доходностям)."""
    own = {k: np.log(v).diff() for k, v in per_contract.items() if len(v)}
    dates = sorted(set().union(*[set(v.index) for v in per_contract.values()]))
    act = active_secid(dates, table)
    rows = [(d, a, per_contract[a].get(d), own[a].get(d)) for d, a in act.items()
            if a in per_contract and d in per_contract[a].index]
    out = pd.DataFrame(rows, columns=["date", "secid", "close", "ret"])
    if SPLICE_BY_LEVEL:
        out["ret"] = np.log(out["close"]).diff()
    return out


def missing_days(brent: pd.DataFrame, calendar) -> pd.Series:
    """Дни календаря (торговые дни ММВБ) без значения Brent, по годам."""
    cal = pd.DatetimeIndex(calendar)
    miss = cal[~cal.isin(pd.DatetimeIndex(brent["date"]))]
    return pd.Series(1, index=miss).groupby(miss.year).sum()


def resolve_root(path) -> Path:
    """Корень репозитория данных (с `data/iss_ext/br_contracts.json`). Принимает корень или его `data/`;
    иначе FileNotFoundError — не тихий пустой ряд."""
    p = Path(path)
    for c in (p, p.parent):
        if (c / BR_JSON).exists():
            return c
    raise FileNotFoundError(f"vneshnie_1850: в {p} (и в его родителе) нет {BR_JSON}; ждётся корень репозитория данных")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def header(main: Path) -> str:
    """Шапка CSV (proverka.md п. 4): статус, коммит, sha256[:12] входов."""
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).parent).stdout.strip()
    brd = main / BR_DIR
    br = hashlib.sha256("".join(f"{p.name}:{_sha(p)};" for p in sorted(brd.glob("BR*.parquet"))).encode()).hexdigest()[:12]
    ins = f"BR(br_contracts.json {_sha(main / BR_JSON)}; BR*.parquet набор {br}), " + ", ".join(
        f"{n.upper()} {_sha(main / PATHS[n])}" for n in ("usdrub", "rvi", "imoex"))
    return ("# статус: диагностика, 2016–2021, до аудита\n"
            f"# коммит кода: {git}\n# хэш входов sha256[:12]: {ins}\n"
            "# максимум на сдвиге 0 к IMOEX — синхронная связь; отсутствие утечки обеспечивается отсечкой 18:50 "
            "и тестом вечернего бара\n")


def load_brent(main: Path) -> pd.DataFrame:
    main = resolve_root(main)
    table = contract_table(main)
    per = {}
    for s in table["secid"]:
        p = main / BR_DIR / f"{s}.parquet"
        if not p.exists():
            p = main / BR_DIR / f"{s.split('_')[0]}.parquet"
        if p.exists():
            b = read_bars(p)
            if len(b):
                per[s] = close_1850(b, s)
    return splice_brent(per, table)


def load_series(name: str, main: Path) -> pd.Series:
    main = resolve_root(main)
    return close_1850(read_bars(main / PATHS[name]), name)


def shift_corr(x: pd.Series, y: pd.Series, shifts=SHIFTS) -> dict:
    """corr(x_d, y_{d+k}) на торговых днях y; x переиндексирован на календарь y (без ffill)."""
    x = x.reindex(y.index)
    return {k: float(x.corr(y.shift(-k))) for k in shifts}


def sdvig_table(main: Path) -> pd.DataFrame:
    """Д1: изменения рядов против самих себя (цель 'self') и против дневной доходности IMOEX ('imoex')."""
    b = load_brent(main).set_index("date")["ret"]
    ser = {"brent": b, "usdrub": change("usdrub", load_series("usdrub", main)),
           "rvi": change("rvi", load_series("rvi", main))}
    imoex = change("imoex", load_series("imoex", main))
    rows = []
    for n, x in ser.items():
        for tgt, y in (("self", x.dropna()), ("imoex", imoex.dropna())):
            xx, yy = x[(x.index >= "2016-01-01")], y[(y.index >= "2016-01-01")]
            rows += [(n, tgt, k, c) for k, c in shift_corr(xx, yy).items()]
    return pd.DataFrame(rows, columns=["ryad", "cel", "sdvig", "corr"])


def max_at_zero(t: pd.DataFrame) -> dict:
    return {(r, c): int(g.loc[g["corr"].abs().idxmax(), "sdvig"]) == 0 for (r, c), g in t.groupby(["ryad", "cel"])}


if __name__ == "__main__":
    main = resolve_root(os.environ.get("VNESHNIE_DATA") or d18.default_main())   # корень данных, только чтение
    t = sdvig_table(main)
    out = Path(__file__).resolve().parents[1] / "docs/tables/diag/vneshnie_1850_sdvig.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header(main) + t.to_csv(index=False, float_format="%.6f"))
    print(t.to_string(index=False))
    print("максимум |corr| на сдвиге 0:", max_at_zero(t))
    br = load_brent(main)
    cal = load_series("usdrub", main).index
    print("дней без Brent по годам (календарь USDRUB TOM):\n", missing_days(br, cal[(cal >= "2015-01-01")]).to_string())
    roll_ret = br[br["secid"] != br["secid"].shift(1)]["ret"].abs()
    print("|ret| в первый день контракта: макс", roll_ret.max(), "; медиана |ret| всех дней", br["ret"].abs().median())
