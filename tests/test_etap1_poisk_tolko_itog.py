# -*- coding: utf-8 -*-
"""--tolko-itog (tolko_itog): досчитывает наружные точки k_s/R (0,75; 0,5) и τ = 2, 3 края сетки; они входят в N. Синтетика."""
from __future__ import annotations

import pandas as pd
import pytest

from fastbt import etap1_poisk as P
from fastbt import etap1_poisk_vorota as V
from test_etap1_poisk import AXES, SETKA, World

NET = "net"
AX = {**AXES, "k_s": [1.0, 2.0, 3.0, NET], "R": [1.0, 2.0, 3.0, NET], "tau": [1, 2, 3, NET]}
SETKA_E = {**SETKA, "axes": AX, "yakor": {**SETKA["yakor"], "a": 6, "b": 0, "k_s": 1.0, "R": 1.0, "tau": 1}}


class Rec(World):
    """Записывает точки, которые попросили посчитать; краевая точка (k_s = R = τ = 1) чуть лучше (0,08 < порога плато 0,1)."""
    pts: list

    def evaluate(self, p, period):
        self.pts = getattr(self, "pts", [])
        self.pts.append(V.pkey(p))
        r = super().evaluate(p, period)
        bonus = 0.08 if (p["k_s"], p["R"], p["tau"]) == (1.0, 1.0, 1) else 0.0
        return dict(r, msr_by_year={y: v + bonus for y, v in r["msr_by_year"].items()})


def prepare(tmp_path):
    P.run_strategy("S9", Rec().evaluate, lambda p: True, out=tmp_path, run_id="1", setka=SETKA_E)
    w = Rec()
    row = P.tolko_itog("S9", w.evaluate, lambda p: True, out=tmp_path, run_id="2", setka=SETKA_E)
    return row, w, pd.read_csv(tmp_path / "S9_itog.csv", comment="#")


def test_tolko_itog_dosschityvaet_naruzhnye_k_s_R_tau_2_3_i_N(tmp_path):
    row, w, d = prepare(tmp_path)
    assert len(d) == 2 and row["p_k_s"] == 1.0 and row["p_R"] == 1.0 and row["p_tau"] == 1      # выбрана краевая точка
    pt = {k: row[f"p_{k}"] for k in AX}
    need = [{**pt, "k_s": v} for v in (0.75, 0.5)] + [{**pt, "R": v} for v in (0.75, 0.5)] + [{**pt, "tau": v} for v in (2, 3)]
    assert {V.pkey(q) for q in need} <= set(w.pts)                                                # досчитаны
    assert row["kray_setki"] == "k_s R tau"
    assert d.N_unikalnykh.iloc[1] == len(set(w.pts))              # Ж-051: N = все уникальные точки, без двойного счёта (раньше old + added)
    assert d.N_unikalnykh.iloc[1] >= d.N_do_dosheta.iloc[1]


def test_mutaciya_bez_rasshireniya_tolko_itog_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "OUTER", {})                                                           # наружные точки не считаются
    monkeypatch.setattr(V, "one_sided", lambda led, pt, a, vals: True)                            # τ = 2, 3 не считаются
    with pytest.raises(AssertionError):
        test_tolko_itog_dosschityvaet_naruzhnye_k_s_R_tau_2_3_i_N(tmp_path)


def test_popravka_N_dopisyvaet_stroku_i_ne_perezapisyvaet(tmp_path):
    _, _, d = prepare(tmp_path)
    before = (tmp_path / "S9_itog.csv").read_text(encoding="utf-8")
    P.popravka_N("S9", 1500, out=tmp_path, run_id="fix")
    after = (tmp_path / "S9_itog.csv").read_text(encoding="utf-8")
    assert after.startswith(before) and len(after) > len(before)                                   # старые строки байт в байт
    e = pd.read_csv(tmp_path / "S9_itog.csv", comment="#")
    assert len(e) == len(d) + 1 and e.N_unikalnykh.iloc[-1] == 1500 and e.N_dsr.iloc[-1] == 1500
    assert e.N_unikalnykh.iloc[:-1].tolist() == d.N_unikalnykh.tolist()


def test_mutaciya_ledger_ne_vosstanovlen_do_doschyota_lovitsya(tmp_path, monkeypatch):
    """Ж-051: старт с пустого Ledger (Моррис не воспроизведён) завышал N; теперь это ошибка, а не молчаливое +20…+40."""
    P.run_strategy("S9", Rec().evaluate, lambda p: True, out=tmp_path, run_id="1", setka=SETKA_E)
    monkeypatch.setattr(P.sd, "morris_screen", lambda *a, **k: {"effects": {}, "n_evaluations": 0})
    with pytest.raises(ValueError, match="Ledger не восстановлен"):
        P.tolko_itog("S9", Rec().evaluate, lambda p: True, out=tmp_path, run_id="2", setka=SETKA_E)
