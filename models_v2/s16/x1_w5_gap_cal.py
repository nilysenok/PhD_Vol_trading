"""W5-B (Ж-250, Ж-251, Ж-252; `W5_SPEC.md` раздел 11): метка gap^PIT — календарь, известный на закрытии d (бары до d 18:40), и `W5_GAP_PIT.csv`.
Публичное API (чистые функции, ничего не пишут, кроме `write_gap_pit`):
- `CAL` = `docs/otchety/harj_x1/W5_CALENDAR_PIT.csv`; `GAP_PIT` = `docs/otchety/harj_x1/W5_GAP_PIT.csv`; `CUT` = 18:40; `MARKS` — три пометки 11.5.
- `load_calendar(path=CAL) -> DataFrame`: date, kind (`weekday_closed` — только пн–пт; `weekend_trading` — только сб/вс), reason, announced_at (NaT, если пусто; запись без времени — конец дня,
  т. е. известна только со следующего дня), source_url, source_kind, confirmed (bool: yes/true/1/да). Дубль даты, неизвестный kind или kind не по дню недели — SystemExit.
- `canonical_saturdays(path=None) -> set`: субботы/воскресенья ≥ 2018-01-01, которые являются каноническими торговыми днями календаря пар (`data/fwd/models/calendar.parquet`, решение дирижёра Ж-252 п. 1).
- `known(row, d) -> bool`: строка подтверждена и announced_at ≤ d 18:40 (`CUT`); неподтверждённая или без времени объявления — неизвестна.
- `gap_pit(d, cal, canon) -> DataFrame` (d, gap, source, unconfirmed_date): gap_d = 1, если день d + 1 не торговый по расписанию, известному на d 18:40 (11.2):
  (а) сб/вс, кроме канонической торговой субботы, объявленной и подтверждённой до d 18:40 (строки `weekend_trading` не канонические — сессии выходного дня 2025–2026 — день торговым НЕ делают, правило 1 дирижёра;
  неподтверждённая или не объявленная к d торговая суббота — неторговая, gap = 1); (б) будний день `weekday_closed`, подтверждённый и объявленный до d 18:40; неподтверждённый или объявленный позже — gap = 0 (11.3).
  `unconfirmed_date` — дата d + 1, если решение зависело от неподтверждённой или позднее объявленной строки.
- `gap_real(d, tp) -> ndarray`: 1[Δ_d > 0], Δ_d = `x1_ya_feat.delta_days` (по факту t⁺; в модель не входит).
- `cross_check(cal, canon) -> dict`: число строк weekend_trading, сколько из них канонические (оставлено), сколько отброшено правилом 1, канонические субботы без строки календаря,
  строки «рабочая суббота», не канонические по календарю пар.
- `pit_table(x, cal, canon) -> DataFrame`: по всем парам окна M: d, t_plus, delta_days, gap_pit, gap_real, agree, source, unconfirmed_date, note.
- `summary(tbl, cal, canon) -> dict` и `header(tbl, cal, canon) -> str` (шапка CSV: пометки 11.5 + счётчики); `csv_text`, `write_gap_pit(x, cal, canon, path=GAP_PIT)`.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_ya_feat as FE

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "docs/otchety/harj_x1"
CAL, GAP_PIT = REP / "W5_CALENDAR_PIT.csv", REP / "W5_GAP_PIT.csv"
CUT = pd.Timedelta(hours=18, minutes=40)
MARKS = ("# состояние gap выбрано из 24 после анализа истории окна M",
         "# способ оценки β выбран после просмотра 24.02.2022 (Д-2)",
         "# исследовательский результат на раскрытой истории, не новое независимое подтверждение (Ж-250 п. 6)")
KINDS = ("weekday_closed", "weekend_trading")
YES = ("yes", "true", "1", "да", "y")


def read_commented(path: Path) -> pd.DataFrame:
    """CSV без строк, начинающихся с `#` (запятые и `#` внутри комментариев не мешают)."""
    text = "\n".join(ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if not ln.startswith("#"))
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def _announced(s: str):
    s = s.strip()
    if not s:
        return pd.NaT
    t = pd.Timestamp(s)
    return t + pd.Timedelta(hours=23, minutes=59, seconds=59) if len(s) <= 10 else t


def load_calendar(path: Path = CAL) -> pd.DataFrame:
    c = read_commented(path)
    need = ["date", "kind", "reason", "announced_at", "source_url", "source_kind", "confirmed"]
    if list(c.columns) != need:
        raise SystemExit("календарь: колонки %s, ожидались %s" % (list(c.columns), need))
    c["date"] = pd.to_datetime(c["date"])
    c["announced_at"] = pd.to_datetime([_announced(s) for s in c["announced_at"]])
    c["confirmed"] = c["confirmed"].str.strip().str.lower().isin(YES)
    wk = c["date"].dt.dayofweek.to_numpy()
    if c["date"].duplicated().any() or not c["kind"].isin(KINDS).all():
        raise SystemExit("календарь: дубль даты или неизвестный kind")
    if ((c["kind"] == "weekday_closed") & (wk >= 5)).any() or ((c["kind"] == "weekend_trading") & (wk < 5)).any():
        raise SystemExit("календарь: kind не соответствует дню недели")
    return c.sort_values("date").reset_index(drop=True)


def canonical_saturdays(path: Path | None = None) -> set:
    p = path or ROOT / "data/fwd/models/calendar.parquet"
    d = pd.to_datetime(pd.read_parquet(p)["date"])
    return set(d[(d.dt.dayofweek >= 5) & (d >= pd.Timestamp("2018-01-01"))])


def known(row, d) -> bool:
    return bool(row["confirmed"]) and pd.notna(row["announced_at"]) and row["announced_at"] <= pd.Timestamp(d) + CUT


def _one(d, rows: dict, canon: set) -> tuple:
    n = d + pd.Timedelta(days=1)
    r = rows.get(n)
    if n.dayofweek >= 5:
        if r is None or r["kind"] != "weekend_trading":
            return 1, "выходной (нет строки календаря)", pd.NaT
        if n not in canon:
            return 1, "выходной: сессия выходного дня не канонический день (правило 1)", pd.NaT
        if known(r, d):
            return 0, "рабочая суббота объявлена: %s %s" % (r["announced_at"], r["source_url"]), pd.NaT
        return 1, "рабочая суббота не подтверждена к d: считается неторговой (11.3)", n
    if r is None or r["kind"] != "weekday_closed":
        return 0, "", pd.NaT
    if known(r, d):
        return 1, "%s; объявлено %s; %s" % (r["reason"], r["announced_at"], r["source_url"]), pd.NaT
    return 0, "будний неторговый день не подтверждён к d: gap = 0 (11.3)", n


def gap_pit(d, cal: pd.DataFrame, canon: set) -> pd.DataFrame:
    rows = {r["date"]: r for _, r in cal.iterrows()}
    out = [_one(pd.Timestamp(x), rows, canon) for x in pd.DatetimeIndex(d)]
    g = pd.DataFrame(out, columns=["gap", "source", "unconfirmed_date"])
    g.insert(0, "d", pd.DatetimeIndex(d))
    return g


def gap_real(d, tp) -> np.ndarray:
    return (FE.delta_days(d, tp) > 0).astype(int)


def cross_check(cal: pd.DataFrame, canon: set) -> dict:
    w = cal[cal["kind"] == "weekend_trading"]
    kept = w[w["date"].isin(canon)]
    work = set(w.loc[w["reason"].str.startswith("рабочая суббота"), "date"])
    return dict(n_rows=len(w), n_kept=len(kept), n_dropped_rule1=len(w) - len(kept), canon_without_row=sorted(canon - set(w["date"])),
                working_not_canonical=sorted(work - canon), sets_equal=work == set(kept["date"]) and canon == set(kept["date"]))


def pit_table(x: pd.DataFrame, cal: pd.DataFrame, canon: set) -> pd.DataFrame:
    g = gap_pit(x["d"], cal, canon)
    real = gap_real(x["d"], x["t_plus"])
    t = pd.DataFrame(dict(d=x["d"].dt.date, t_plus=pd.DatetimeIndex(x["t_plus"]).date, delta_days=FE.delta_days(x["d"], x["t_plus"]).astype(int), gap_pit=g["gap"].to_numpy(int), gap_real=real))
    t["agree"] = (t["gap_pit"] == t["gap_real"]).astype(int)
    t["source"] = g["source"].to_numpy()
    t["unconfirmed_date"] = [str(v.date()) if pd.notna(v) else "" for v in g["unconfirmed_date"]]
    t["note"] = np.where(t["agree"] == 1, "", np.where(t["gap_real"] == 1, "только по факту t⁺: нет объявленного к d перерыва", "только по календарю: t⁺ = d + 1, перерыв не наступил"))
    return t


def summary(tbl: pd.DataFrame, cal: pd.DataFrame, canon: set) -> dict:
    cc = cross_check(cal, canon)
    return dict(n=len(tbl), n_pit=int(tbl["gap_pit"].sum()), n_real=int(tbl["gap_real"].sum()), n_disagree=int((tbl["agree"] == 0).sum()),
                n_unconfirmed_days=int((~cal["confirmed"]).sum()), n_unconfirmed_pairs=int((tbl["unconfirmed_date"] != "").sum()), **cc)


def header(tbl: pd.DataFrame, cal: pd.DataFrame, canon: set) -> str:
    s = summary(tbl, cal, canon)
    lines = list(MARKS) + ["# пар %d; gap^PIT = 1: %d; gap^real = 1: %d; расхождений %d (все перечислены в строках с agree = 0)" % (s["n"], s["n_pit"], s["n_real"], s["n_disagree"]),
                           "# неподтверждённых дней календаря: %d; пар, где решение зависело от неподтверждённой строки: %d" % (s["n_unconfirmed_days"], s["n_unconfirmed_pairs"]),
                           "# строк weekend_trading: %d; канонических торговых суббот оставлено: %d; отброшено правилом 1 (сессии выходного дня, Ж-252): %d" % (s["n_rows"], s["n_kept"], s["n_dropped_rule1"]),
                           "# сверка множеств: «рабочая суббота» в календаре и канонические субботы календаря пар совпадают: %s; канонические без строки: %s; «рабочая суббота» не каноническая: %s" % (
                               "да" if s["sets_equal"] else "НЕТ", [str(v.date()) for v in s["canon_without_row"]] or "нет", [str(v.date()) for v in s["working_not_canonical"]] or "нет")]
    return "\n".join(lines) + "\n"


def csv_text(tbl: pd.DataFrame, cal: pd.DataFrame, canon: set) -> str:
    return header(tbl, cal, canon) + tbl.to_csv(index=False, lineterminator="\n")


def write_gap_pit(x: pd.DataFrame, cal: pd.DataFrame, canon: set, path: Path = GAP_PIT) -> int:
    t = pit_table(x, cal, canon)
    Path(path).write_text(csv_text(t, cal, canon), encoding="utf-8")
    return len(t)
