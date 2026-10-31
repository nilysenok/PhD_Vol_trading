"""Лист s8_s12_boj (31.10), продолжение: Б4 ist7 доходит до EvaluatorV7 из progon_v7, панель сечения покрывает дни рядов оценщика; G6 — реальный прогон
драйвера S8, S10 на train4 (нужны данные основной копии, без них пропуск). Синтетика без сети.
python3 -m pytest tests/test_s8_s12_boj_2.py -q"""
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import eval_v7, period_v2
from fastbt import eval_v7_ist as E


# ---------------- Б4: ist7 в EvaluatorV7 из progon_v7 ----------------
def _ocenshchik(monkeypatch, sid, period="polnaya"):
    import fastbt.etap1 as et
    import fastbt.istochniki_v6 as i6
    import fastbt.progon_sobol as ps
    import fastbt.progon_v7 as p7
    got = {}
    monkeypatch.setattr(eval_v7, "EvaluatorV7", lambda *a, **kw: got.update(kw) or got)
    monkeypatch.setattr(et, "base_point", lambda s: {})
    monkeypatch.setattr(i6, "zagruzit", lambda per, tik: "ist6")
    monkeypatch.setattr(E, "zagruzit", lambda per, tik, **kw: types.SimpleNamespace(ist7=lambda: "ist7_" + per) if kw.get("dev") == (per == "train4") else 1 / 0)
    inp = types.SimpleNamespace(sk=types.SimpleNamespace(tickers=["AAA"]))
    with p7.podmeny({}, 100, lambda *a: []):
        ps.ocenshchik(sid, inp, p7.rezhim7(1, None, period=period))
    return got


@pytest.mark.parametrize("sid,est", [("S8", True), ("S9", True), ("S10", True), ("S11", True), ("S12", True), ("S1", False), ("S6", False)])
def test_b4_ist7_podayotsya_tolko_semyam_v7(monkeypatch, sid, est):
    kw = _ocenshchik(monkeypatch, sid, "train4")
    assert (kw.get("ist7") == "ist7_train4") is est and (est or "ist7" not in kw)


def test_b4_mutatsiya_ist7_ne_podayotsya_lovitsya(monkeypatch):
    import fastbt.progon_v7 as p7
    monkeypatch.setattr(p7, "SEMEJSTVA_IST7", ())
    with pytest.raises(AssertionError):
        test_b4_ist7_podayotsya_tolko_semyam_v7(monkeypatch, "S10", True)


def test_b4_zagruzit_train4_chitaet_do_2020_i_panel_s_nachala_ryadov(monkeypatch):
    """train4 читается только с dev=True (без него — отказ, как раньше); панель — с первой даты рядов оценщика (wf.progrev), не с progrev train4."""
    seen = {}
    monkeypatch.setattr(E, "sobrat", lambda raw, im, karta, tikery, nachalo, konec, **kw: seen.update(nachalo=nachalo, konec=konec, **kw) or "ok")
    f = types.SimpleNamespace(end=pd.Timestamp("2020-01-01"), akcii=lambda t: None,
                              indeks=lambda n: pd.DataFrame({"timestamp": [pd.Timestamp("2016-01-04 10:00")], "open": [1.0], "high": [1.0],
                                                             "low": [1.0], "close": [1.0], "volume": [1]}))
    with pytest.raises(NotImplementedError):
        E.zagruzit("train4", ["AAA"], fajly=f)
    assert E.zagruzit("train4", ["AAA"], fajly=f, dev=True) == "ok"
    assert seen["nachalo"] == period_v2.PERIODY["train4"].progrev and seen["konec"] == pd.Timestamp("2020-01-01")
    assert seen["ryady_s"] == period_v2.PERIODY["wf"].progrev                      # панель покрывает все дни рядов оценщика (с 2014-06)


def _panel_dni(ryady_s):
    import test_eval_v7_ist as tv
    s = E.sobrat(tv._raw(), tv._im(), tv.KARTA, tv.TK, "2018-01-01", "2019-01-01", korp=tv._korp0, mnozh=tv._mn1, ryady_s=ryady_s)
    return s.panel


def test_b4_panel_pokryvaet_dni_ryadov_ocenshchika_do_pervoj_10minutki():
    pn = _panel_dni("2017-11-01")
    daty = pd.DatetimeIndex(pn.daty)
    assert daty[0] == pd.Timestamp("2017-11-01") and pn.c.shape[0] == len(daty) == pn.sigma.shape[0] == pn.ok.shape[0]
    assert pn.c.loc[:"2018-01-05"].isna().all().all() and not pn.ok[daty < "2018-01-08"].any()          # до данных: NaN и допуск входа закрыт
    assert not np.array_equal(daty, pd.DatetimeIndex(_panel_dni(None).daty))                              # без ryady_s панель начиналась позже (KeyError «вне панели»)


# ---------------- G6: реальный прогон драйвера (нужны данные основной копии; без них — пропуск) ----------------
DANNYE = Path(__import__("fastbt.daily_1850", fromlist=["x"]).default_main()) / "moex_discovery/data/raw/candles_10m/AFLT.parquet"


@pytest.mark.skipif(not DANNYE.exists(), reason="нет candles_10m основной копии")
def test_g6_progon_pilot_S8_S10_train4(tmp_path):
    import json

    import fastbt.progon_v7 as p7
    rc = p7.main(["--kub", "v7", "--period", "train4", "--pilot", "--semeystva", "S8,S10", "--n1", "2", "--procs", "1", "--out", str(tmp_path)])
    z = json.loads((tmp_path / p7.ZAMER).read_text(encoding="utf-8"))
    assert rc == 0 and set(z) == {"S8", "S10"}
    for s in z:
        assert not z[s]["sboy"] and z[s]["strok"] >= 1 and z[s]["sek_na_tochku_p50"] > 0
        rows = [json.loads(x) for p in sorted((tmp_path / s).glob("p[0-9]*/rows.jsonl")) for x in p.read_text(encoding="utf-8").splitlines()]
        assert all(r["status"] == "ok" and sum(r["sdelok_y"].values()) > 0 for r in rows), s          # без ошибок и со сделками
