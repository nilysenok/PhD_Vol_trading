"""W5-A, аудит метки gap (W5_SPEC раздел 4, гейт A.G4): перечень пар окна M с перерывом, не объяснённым одними выходными. Перенос `w5/gaps.py`; ничего, кроме `--write`, не пишет.
Публичное API:
- `N_EXPECTED` = 66; `GAPS` = `docs/otchety/harj_x1/W5_GAPS.csv`; `STOP_2022` = 2022-02-25 (день остановки торгов: пары с таким d быть не должно).
- `odd_pairs(x=None) -> DataFrame`: пары с Δ = (t⁺ − d, дни) − 1 > 0, кроме «d — пятница и Δ = 2». Колонки: d, d_weekday, t_plus, t_plus_weekday, delta_days, nonworking_between
  (календарные дни строго между d и t⁺: `ГГГГ-ММ-ДД Дн`, через `|`). `x` — окно M (`x1_w3_core.load()`), по умолчанию загружается.
- `to_csv_text(df) -> str`, `write(path=GAPS)` (единственный пишущий вызов), `check(path=GAPS) -> list` замечаний, только чтение: файл равен пересчёту по строкам данных, ровно 66 пар,
  нет пары с d = 2022-02-25 (тождество «пары через остановку 2022 нет»), нет пятниц с Δ = 2.
Ручная сверка перечня с производственным календарём — ручной гейт, вне кода.
Запуск: `python3 -W ignore -m models_v2.s16.x1_w5_gaps [--write | --check]`; без флага — печать."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from models_v2.s16 import x1_w3_core as C

GAPS = C.ROOT / "docs/otchety/harj_x1/W5_GAPS.csv"
N_EXPECTED = 66
STOP_2022 = pd.Timestamp("2022-02-25")
HEAD = ["# W5_GAPS — пары окна M с перерывом, не объяснённым одними выходными (d — не пятница или Δ ≠ 2); Δ = (t+ − d) − 1 календарных дней (W5_SPEC раздел 4, гейт A.G4)",
        "# перечень выводит код; сверка с производственным календарём РФ — ручной гейт; генератор models_v2.s16.x1_w5_gaps"]


def odd_pairs(x: pd.DataFrame | None = None) -> pd.DataFrame:
    x = C.load() if x is None else x
    delta = (x["t_plus"] - x["d"]).dt.days - 1
    odd = x[(delta > 0) & ~((x["d"].dt.dayofweek == 4) & (delta == 2))]
    rows = []
    for d, tp in zip(odd["d"], odd["t_plus"]):
        between = [d + pd.Timedelta(days=k) for k in range(1, (tp - d).days)]
        rows.append(dict(d=d.date(), d_weekday=d.day_name()[:3], t_plus=tp.date(), t_plus_weekday=tp.day_name()[:3], delta_days=(tp - d).days - 1,
                         nonworking_between="|".join("%s %s" % (b.date(), b.day_name()[:3]) for b in between)))
    return pd.DataFrame(rows, columns=["d", "d_weekday", "t_plus", "t_plus_weekday", "delta_days", "nonworking_between"])


def to_csv_text(df: pd.DataFrame) -> str:
    return "\n".join(HEAD) + "\n" + df.to_csv(index=False, lineterminator="\n")


def write(path: Path = GAPS) -> int:
    df = odd_pairs()
    path.write_text(to_csv_text(df), encoding="utf-8")
    return len(df)


def _data_lines(text: str) -> list:
    return [ln for ln in text.splitlines() if not ln.startswith("#")]


def check(path: Path = GAPS) -> list:
    """Только чтение. -> список замечаний (пусто = ok)."""
    x = C.load()
    df = odd_pairs(x)
    bad = []
    if len(df) != N_EXPECTED:
        bad.append("пар с перерывом не по выходным %d, ожидалось %d" % (len(df), N_EXPECTED))
    if STOP_2022.date() in set(x["d"].dt.date):
        bad.append("есть пара с d = 2022-02-25 (остановка торгов)")
    if ((df["d_weekday"] == "Fri") & (df["delta_days"] == 2)).any():
        bad.append("в перечне пятница с Δ = 2")
    if not path.exists():
        bad.append("нет файла %s" % path)
    elif _data_lines(path.read_text(encoding="utf-8")) != _data_lines(to_csv_text(df)):
        bad.append("строки данных %s не равны пересчёту" % path.name)
    return bad


if __name__ == "__main__":
    if "--write" in sys.argv:
        print("wrote", GAPS, write(), "pairs")
    elif "--check" in sys.argv:
        b = check()
        print("check %s: пар %d" % ("ok" if not b else "FAIL " + "; ".join(b), len(odd_pairs())))
        sys.exit(1 if b else 0)
    else:
        print(odd_pairs().to_string(index=False))
