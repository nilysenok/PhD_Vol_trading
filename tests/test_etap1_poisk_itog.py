# -*- coding: utf-8 -*-
"""itog без кандидата: тот же набор колонок, дозапись не падает, вердикт «нет кандидата». Синтетика."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from fastbt import etap1_poisk as P
from test_etap1_poisk import SETKA, World


class NanWorld(World):
    """Целевая нигде не определена -> кандидата нет."""

    def evaluate(self, p, period):
        r = super().evaluate(p, period)
        return dict(r, msr_by_year={y: float("nan") for y in r["msr_by_year"]})


def two_runs(tmp_path):
    P.run_strategy("S9", World().evaluate, lambda p: True, out=tmp_path, run_id="1", setka=SETKA)
    return P.run_strategy("S9", NanWorld().evaluate, lambda p: True, out=tmp_path, run_id="2", setka=SETKA)


def test_itog_bez_kandidata_te_zhe_kolonki_i_dozapis(tmp_path):
    row = two_runs(tmp_path)
    d = pd.read_csv(tmp_path / "S9_itog.csv", comment="#")
    assert len(d) == 2 and d.verdikt.iloc[1] == "не дошла: нет допустимой точки" and d.verdikt.iloc[0] != d.verdikt.iloc[1]
    assert math.isnan(row["obj_min_msr"]) and math.isnan(row["plateau"]) and math.isnan(row["p_a"]) and math.isnan(row["ok"])
    assert row["N_dsr"] == row["N_unikalnykh"]


def test_mutaciya_kolonki_bez_kandidata_otlichayutsya_lovitsya(tmp_path, monkeypatch):
    orig = P.append_csv

    def staryy_bag(path, rows):                                      # строка без кандидата теряет колонки точки и ворот
        if "нет допустимой точки" in str(rows[0].get("verdikt")):
            rows = [{k: v for k, v in rows[0].items() if not k.startswith("p_") and k not in P.GATE_COLS}]
        orig(path, rows)
    monkeypatch.setattr(P, "append_csv", staryy_bag)
    with pytest.raises(ValueError):
        two_runs(tmp_path)
