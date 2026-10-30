"""Мутации тестировщика PR #782 (не пойманные тестами автора): (б) защита `rang_nan_u_vseh` включена и для m_sily = 1;
(в) число сделок `ns` берётся после веса (нулевой excess → «0 сделок»), а не из sdelok_y строки ledger.
(б) — на данных train4 (skip без них); (в) — синтетика."""
from __future__ import annotations

import numpy as np
import pytest

from fastbt import eval_v5, eval_v5_osi as eo, progon_morris_v6 as pm, progon_morris_v6_status as st


def _est_dannye() -> bool:
    from fastbt import daily_1850
    return daily_1850.out_path().exists()


@pytest.mark.skipif(not _est_dannye(), reason="нет дневного ряда 18:50 в FASTBT_MAIN")
def test_m_sily_1_s_rangom_bezubytka_ne_otbrasyvaetsya_zashchitoj(monkeypatch):
    """Тренд S3 (3 тикера), m_sily = 1, безубыток от ранга, rang_tip srednee: ранг нужен (rk ≠ None), но вес не от него — защита молчит."""
    from test_eval_v5 import _ocenshchiki, _sp5
    _, e5 = _ocenshchiki("S3", 3, semya="v5")
    pt = {**eo.yakor("S3"), **_sp5("S3"), "m_sily": "1", "rang_tip": "srednee", "bezubytok": "rang:1", "stop_nach": "atr:2.0"}
    vyzovy = []
    orig = eval_v5.rang_nan_u_vseh
    monkeypatch.setattr(eval_v5, "rang_nan_u_vseh", lambda *a: vyzovy.append(1) or orig(*a))
    r, _, _ = e5.evaluate_v2(pt, "train4")
    assert e5.r5.rang and e5.r5.m_sily == "1", "точка не включает ранг при m_sily = 1 — тест не доказывает"
    assert not vyzovy, "защита вызвана при m_sily = 1"
    assert r["status"] == "ok", (r["status"], r.get("prichina"))


class _Evr:
    def __init__(self, r, ex):
        self.r, self.ex = r, ex

    def evaluate_v2(self, pt, period):
        return self.r, self.ex, None


def test_ns_bere_sdelki_iz_ledger_a_ne_posle_vesa():
    """excess тождественно 0 (вес 0), сделки в ledger есть: ns = 5, а не 0 → точка ves0, не bez_sdelok."""
    sr = type("S", (), {"daty": np.arange("2018-01-01", "2018-01-04", dtype="datetime64[D]"),
                        "evr": _Evr({"status": "ok", "sdelok_y": {"2018": 5}}, np.zeros(3))})()
    y, s, ns = pm.ocenka(sr, {}, "train4")
    assert s == "ok" and ns == 5.0
    assert st.razdelit_nol([True], [ns]) == {"bez_sdelok": 0, "ves0": 1}
