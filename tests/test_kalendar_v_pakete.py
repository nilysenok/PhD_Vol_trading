"""Календарь MOEX в пакете переноса: ≤ 2021 файла календаря не читает (лениво), ≥ 2025 без файла — отказ, в манифест ≤ 2021
не входит, для wf едет правилом усечения (накладка docs/issledovaniya/, как свод Д8).
Мутация: чтение не ленивое (граница WEEKEND_FROM сдвинута в прошлое) — тест «train4 без файла» падает."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

from fastbt import daily_1850 as D
from fastbt import kalendar_moex as K

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/server"))
import usech_pravila  # noqa: E402

REL = "docs/issledovaniya/2026-10-30_kalendar_moex_2025_2026.csv"


def _bars(days):
    ts = [pd.Timestamp(d) + pd.Timedelta(hours=11, minutes=m) for d in days for m in (0, 10)]
    n = len(ts)
    return pd.DataFrame({"timestamp": ts, "open": [1.0] * n, "high": [2.0] * n, "low": [0.5] * n, "close": [1.5] * n, "volume": [10] * n})


@pytest.fixture
def bez_kalendarya(monkeypatch, tmp_path):
    monkeypatch.setattr(K, "CALENDAR_MOEX", tmp_path / "net.csv")


def test_train4_bez_fajla_kalendarya_rabotaet(bez_kalendarya):
    d = D.aggregate(_bars(["2021-12-29", "2021-12-30"]), "X")
    assert [str(x.date()) for x in d["date"]] == ["2021-12-29", "2021-12-30"]


def test_calendar_extend_do_2025_bez_fajla_rabotaet(bez_kalendarya):
    out = D.calendar_extend(["2021-12-28"], D.aggregate(_bars(["2021-12-29"]), "X"))
    assert [str(x.date()) for x in out] == ["2021-12-28", "2021-12-29"]


def test_dannye_ot_2025_bez_fajla_otkaz(bez_kalendarya):
    with pytest.raises(FileNotFoundError):
        D.aggregate(_bars(["2025-01-09"]), "X")
    with pytest.raises(FileNotFoundError):
        D.calendar_extend(["2024-12-30"], pd.DataFrame({"date": [pd.Timestamp("2025-01-09")]}))


def test_mutatsiya_chtenie_ne_lenivoe_lovitsya(bez_kalendarya, monkeypatch):
    """Нелениво = календарь нужен и на 2021: сдвиг границы в прошлое роняет train4-сценарий без файла."""
    monkeypatch.setattr(D, "WEEKEND_FROM", pd.Timestamp("2000-01-01"))
    with pytest.raises(FileNotFoundError):
        D.aggregate(_bars(["2021-12-29"]), "X")


def test_kalendar_v_pravilah_wf_i_ne_v_manifeste_2021():
    (r,) = [p for p in usech_pravila.PRAVILA if p[0] == REL]
    nak = re.search(r"NAKLADKA='([^']+)'", (REPO / "scripts/server/sobrat_perenos.sh").read_text(encoding="utf-8")).group(1)
    assert re.match(nak, r[0]) and r[2] == ("date",)
    man = (REPO / "docs/server/manifest_dannyh_2021.tsv").read_text(encoding="utf-8")
    assert "kalendar_moex" not in man, "календарь 2025–2026 в манифесте ≤ 2021"
    assert (REPO / REL).is_file() and pd.read_csv(REPO / REL)["date"].min() > "2021-12-31"
