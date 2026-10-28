"""Разбор проходов 1+2: колонка активности (справочно) и параметр префикса. Синтетика, без сети и без ledger.
Мутации: доля дней считается по всем дням (нули как движение); ключи sdelok_y со строковыми годами теряются."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import prohod1_razbor as RZ  # noqa: E402


def _daty():
    return pd.bdate_range("2016-01-01", "2019-12-31").to_numpy().astype("datetime64[D]")


def test_aktivnost_dolya_i_sdelki():
    d = _daty()
    x = np.zeros(len(d))
    god = d.astype("datetime64[Y]").astype(int) + 1970
    x[(god == 2017) & (np.arange(len(d)) % 2 == 0)] = 0.01          # 2017: половина дней с движением, остальные годы нули
    row = dict(klyuch_sha="k", sdelok_y={"2016": 10, "2017": 40, "2018": 0, "2019": 30})
    a = RZ.aktivnost("S1", "t", row, x, d)
    assert a["sdelok_2017"] == 40 and a["sdelok_2018"] == 0
    assert abs(a["sdelok_v_god_sred"] - 20.0) < 1e-12
    assert abs(a["dolya_dnej_2017"] - 0.5) < 0.01 and a["dolya_dnej_2016"] == 0.0
    assert 0.1 < a["dolya_dnej_s_dvizheniem"] < 0.15, "нулевые дни не движение (мутация: доля по всем дням даст 1,0)"


def test_aktivnost_bez_sdelok_ne_ronyaet():
    d = _daty()
    a = RZ.aktivnost("S1", "t", dict(klyuch_sha="k"), np.ones(len(d)), d)
    assert np.isnan(a["sdelok_v_god_sred"]) and a["dolya_dnej_s_dvizheniem"] == 1.0


def test_svod_prefiks(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(RZ, "TAB", tmp_path)
    js = {t: [dict(S="S1", a=1)] for t in RZ.TABLICY}
    (tmp_path / "S1.json").write_text(json.dumps(js), encoding="utf-8")
    RZ.svod(tmp_path, "prohod12")
    assert (tmp_path / "prohod12_aktivnost.csv").exists() and not (tmp_path / "prohod1_aktivnost.csv").exists()
