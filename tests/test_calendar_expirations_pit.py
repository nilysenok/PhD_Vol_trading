"""PIT экспирации MIX-3.22 (Д3): 17.03 -> 24.03 (15.03) -> 30.03 (23.03) -> 29.03 (24.03.2022).

Источник: docs/issledovaniya/moex_perenos_ekspiracii_2022.md (moex.com n45304, n45520, n45596).
Читатель — d73_features.load_events/features. Мутации — только в памяти."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import d73_features as d73  # noqa: E402

CAL = ROOT / "data" / "calendar"
OZHIDAEMO = [("2022-03-14", "2022-03-17"), ("2022-03-15", "2022-03-24"), ("2022-03-16", "2022-03-24"),
             ("2022-03-22", "2022-03-24"), ("2022-03-23", "2022-03-30"), ("2022-03-24", "2022-03-29"),
             ("2022-03-28", "2022-03-29"), ("2022-03-29", "2022-06-16")]


def _plan(ev: pd.DataFrame, t: str) -> list[str]:
    """Даты марта 2022 после t, известные на t (то, что отдаёт features)."""
    t = pd.Timestamp(t)
    k = ev[ev["known_from"] <= t]
    if "known_to" in k:
        k = k[k["known_to"].isna() | (k["known_to"] > t)]
    return [str(x.date()) for x in k["date"] if x > t and x.month in (3, 6) and x.year == 2022][:1]


def _ev() -> pd.DataFrame:
    return d73.load_events(str(CAL), ("expirations",))["expirations"]


def test_plan_na_datu_t() -> None:
    ev = _ev()
    for t, ozh in OZHIDAEMO:
        assert _plan(ev, t) == [ozh], t


def test_ne_dve_versii_srazu() -> None:
    """На любую дату t известна ровно одна версия экспирации марта 2022."""
    ev = _ev()
    for t in pd.date_range("2022-01-03", "2022-03-29"):
        k = ev[(ev["known_from"] <= t) & (ev["date"].dt.month == 3) & (ev["date"].dt.year == 2022)]
        k = k[k["known_to"].isna() | (k["known_to"] > t)]
        assert len(k) == 1, t.date()


def test_features_cherez_chitatel() -> None:
    """dte_expirations на 23.03: до 30.03 = 7 дней; на 24.03: до 29.03 = 5; на 16.03: до 24.03 = 8."""
    ev = {"expirations": _ev()}
    assert d73.features("2022-03-16", 22, ev)["dte_expirations"] == 8
    assert d73.features("2022-03-23", 22, ev)["dte_expirations"] == 7
    assert d73.features("2022-03-24", 22, ev)["dte_expirations"] == 5


def test_do_2022_pit_ne_menyaetsya() -> None:
    """Все версии, кроме мартовских 2022, без known_to; known_from ≤ 2021 только у строк ≤ 2021."""
    d = pd.read_csv(CAL / "expirations.csv", parse_dates=["date", "known_from", "known_to"])
    assert d[d["date"] < "2022-03-01"]["known_to"].isna().all()
    assert len(d[d["date"] < "2022-03-01"]) == 32
    assert d[d["known_to"].notna()]["date"].astype(str).str[:7].eq("2022-03").all()


def test_mutatsiya_known_from_nachalo_goda_lovitsya() -> None:
    ev = _ev()
    ev.loc[ev["date"] == "2022-03-29", "known_from"] = pd.Timestamp("2022-01-01")
    ev.loc[ev["date"] == "2022-03-29", "known_to"] = pd.NaT
    ozh = dict(OZHIDAEMO)
    assert any(_plan(ev, t) != [o] for t, o in ozh.items())
    assert _ploxo(ev)


def _ploxo(ev: pd.DataFrame) -> bool:
    """Две версии одновременно на какой-то дате — признак утечки."""
    for t in pd.date_range("2022-03-15", "2022-03-28"):
        k = ev[(ev["known_from"] <= t) & (ev["date"].dt.month == 3) & (ev["date"].dt.year == 2022)]
        k = k[k["known_to"].isna() | (k["known_to"] > t)]
        if len(k) != 1:
            return True
    return False


def test_mutatsiya_chitatel_bez_known_to_lovitsya() -> None:
    """Читатель, игнорирующий known_to, видит две даты сразу — тест это ловит."""
    ev = _ev().drop(columns="known_to")
    k = ev[(ev["known_from"] <= pd.Timestamp("2022-03-16")) & (ev["date"].dt.month == 3) & (ev["date"].dt.year == 2022)]
    assert len(k) > 1
