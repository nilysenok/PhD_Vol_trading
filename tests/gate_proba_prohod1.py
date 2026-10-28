"""Сквозной гейт пробы прохода 1: 30 случайных точек Соболя (seed 20261028, из первых 5000) на каждую S1–S6, реальные данные train4.
Запуск: FASTBT_MAIN=<копия> FASTBT_BARS1M=<D25/bars1m> python3 -m pytest tests/gate_proba_prohod1.py
Вне автосбора (имя gate_*): нужны 1-минутки D25; без них — красный, не skip. Ошибок (status oshibka) — ноль."""
import random

import pytest

from fastbt import bars5m
from fastbt import progon_sobol as ps

ZNAEM = ("filtr div: условие не определено на данных",)


@pytest.mark.parametrize("sid", ps.SIDS)
def test_skvoznoj_30_tochek_bez_isklyuchenij(sid):
    if not bars5m.SRC.exists():
        pytest.fail(f"{bars5m.SRC}: нет 1-минуток D25 — задать FASTBT_BARS1M (гейт не пропускается)")
    pts = ps.tochki(ps.load_kub(sid), 0, 5000)
    idx = random.Random(20261028).sample(range(len(pts)), 30)
    sr, bad, st = ps.sreda(sid), [], {"ok": 0, "ne_schitana": 0}
    for j in idx:
        i, pt = pts[j]
        r, _, _ = ps.schitat_tochku(sr, sid, i, pt, "test", True, True)
        if r["status"] == "ok" or (r["status"] == "ne_schitana" and r["prichina"] in ZNAEM):
            st[r["status"]] += 1
        else:
            bad.append((i, r["status"], r["prichina"]))
    print(sid, st)
    assert not bad, bad
