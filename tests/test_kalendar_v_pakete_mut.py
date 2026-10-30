"""Поведение правила усечения календаря MOEX на настоящем файле (тестировщик PR #765): ≤ 2021 — ни одной строки,
wf (до 2026-02-28) — строки только до границы, колонка accessed очищена. Мутации: граница не применяется; календарь
целиком попадает в усечённую копию ≤ 2021; правило без колонки дат."""
from __future__ import annotations

import csv
import datetime as dt
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/server"))
import usech_2021 as U  # noqa: E402
import usech_pravila  # noqa: E402

REL = "docs/issledovaniya/2026-10-30_kalendar_moex_2025_2026.csv"


def _pravilo():
    (r,) = [p for p in usech_pravila.PRAVILA if p[0] == REL]
    return r


def _usech(monkeypatch, tmp_path, cut):
    monkeypatch.setattr(U, "CUT", dt.date.fromisoformat(cut))
    r = _pravilo()
    dst = tmp_path / "k.csv"
    st = U.usech_csv(REPO / REL, dst, r[2], r[3])
    return st, list(csv.DictReader(dst.open(encoding="utf-8")))


def test_do_2021_ni_odnoj_stroki(monkeypatch, tmp_path):
    st, rows = _usech(monkeypatch, tmp_path, "2021-12-31")
    assert rows == [] and st.stroki == 0 and st.otbrosheno > 0


def test_wf_stroki_do_granitsy_i_accessed_pust(monkeypatch, tmp_path):
    st, rows = _usech(monkeypatch, tmp_path, "2026-02-28")
    assert len(rows) > 300 and max(r["date"] for r in rows) == "2026-02-28" and st.otbrosheno == 0
    assert all(r["accessed"] == "" for r in rows)
    _, rows = _usech(monkeypatch, tmp_path, "2025-06-30")
    assert rows and max(r["date"] for r in rows) <= "2025-06-30"
