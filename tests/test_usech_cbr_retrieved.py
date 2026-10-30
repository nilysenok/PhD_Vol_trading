"""Усечение cbr.csv (Ж-131 п. 2): retrieved_at (2026-10-25 — тоже дата позже границы) обнуляется намеренно;
все остальные колонки совпадают с исходником, формат строк не меняется сверх обнуления этой колонки.
Мутация: retrieved_at не обнуляется -> тест падает."""
import csv
import datetime as dt
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_s = importlib.util.spec_from_file_location("usech_2021_cbr", REPO / "scripts/server/usech_2021.py")
U = importlib.util.module_from_spec(_s)
_s.loader.exec_module(U)
SRC = REPO / "data/calendar/cbr.csv"


@pytest.mark.parametrize("cut", [dt.date(2021, 12, 31), dt.date(2026, 2, 28)])
def test_cbr_vse_kolonki_krome_retrieved_at_sovpadayut(tmp_path, monkeypatch, cut):
    monkeypatch.setattr(U, "CUT", cut)
    dst = tmp_path / "cbr.csv"
    U.usech_csv(SRC, dst, U.KAL, ("retrieved_at",))
    src_rows = list(csv.reader(SRC.read_text(encoding="utf-8-sig").splitlines()))
    h, ri = src_rows[0], src_rows[0].index("retrieved_at")
    di, ki = h.index("date"), h.index("known_from")
    ozhid = [r for r in src_rows[1:] if r and dt.date.fromisoformat(r[di]) <= cut and dt.date.fromisoformat(r[ki]) <= cut]
    got = dst.read_text(encoding="utf-8").splitlines()
    assert ozhid and len(got) == len(ozhid) + 1 and got[0] == ",".join(h)
    for line, r in zip(got[1:], ozhid):
        pustaya = r[:ri] + [""] + r[ri + 1:]
        assert next(csv.reader([line])) == pustaya                 # значения всех колонок те же, retrieved_at пуста
        assert line == ",".join(pustaya)                           # формат: без кавычек и правок чисел сверх обнуления
