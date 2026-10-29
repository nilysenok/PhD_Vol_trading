"""Ж-126/Ж-127 п. 1: сигнальные оси S1–S3 из daily_upper_bound.py (6a7a4636) — полные, без точек и итогов.

Мутации: добавить ключ точки/итога (best, sharpe, params, tochki, rezultat) — проверка обязана его найти;
сузить ось или испортить obem_signal — тест обязан упасть.
"""
import copy
import math
from pathlib import Path

import pytest
import yaml

D = Path(__file__).resolve().parents[1] / "fastbt/designs/zh126"
SIDS = ["S1", "S2", "S3"]
ZAPRET = ("best", "sharpe", "params", "tochki", "rezultat")

# Списки источника, переписанные из daily_upper_bound.py @ 6a7a4636 (строки 349–353, 367).
MA_TYPES = ["sma", "ema", "wma", "dema", "hull", "kama"]
WINDOWS = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 80, 100, 120, 150, 200]
STD_DEVS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
ATR_DEVS = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
BB_STDS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
DC_WINDOWS = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 80, 100, 120]


def _y(sid: str) -> dict:
    return yaml.safe_load((D / f"zh126_{sid}.yaml").read_text(encoding="utf-8"))


def dlina(os_: dict) -> int:
    if "znacheniya" in os_:
        return len(os_["znacheniya"])
    return math.floor((os_["do"] - os_["ot"]) / os_["shag"] + 1e-9) + 1


def obem(d: dict) -> int:
    """Произведение длин осей; значение-родитель зависимой оси размножается на её длину."""
    zav = d.get("zavisimye_osi") or []
    itog = 1
    for imya, os_ in d["signal"].items():
        deti = [z for z in zav if z["pri"]["os"] == imya]
        if not deti:
            itog *= dlina(os_)
            continue
        summa = 0
        for v in os_["znacheniya"]:
            k = 1
            for z in deti:
                if z["pri"]["znachenie"] == v:
                    k *= dlina(z)
            summa += k
        itog *= summa
    return itog


def zapreshchennye(obj, put="") -> list:
    nashli = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(t in str(k).lower() for t in ZAPRET):
                nashli.append(f"{put}/{k}")
            nashli += zapreshchennye(v, f"{put}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            nashli += zapreshchennye(v, f"{put}[{i}]")
    return nashli


@pytest.mark.parametrize("sid", SIDS)
def test_yaml_chitaetsya(sid):
    d = _y(sid)
    assert d["strategiya"] == sid
    assert d["istochnik"]["kommit"] == "6a7a4636"
    assert d["istochnik"]["fajl"] == "moex_volatility/scripts/daily_upper_bound.py"
    assert d["obshchee"] == "zh126_obshchee.yaml"
    assert d["filtry_tipa"]["filtry"]
    assert d["signal"]


@pytest.mark.parametrize("sid", SIDS)
def test_u_kazhdoj_osi_istochnik_stroki(sid):
    d = _y(sid)
    for imya, os_ in list(d["signal"].items()) + [(z["os"], z) for z in d["zavisimye_osi"]]:
        assert str(os_.get("istochnik_stroki", "")).strip(), imya
        assert "znacheniya" in os_ or {"ot", "do", "shag"} <= set(os_), imya


@pytest.mark.parametrize("sid", SIDS)
def test_obem_signal_ravno_proizvedeniyu(sid):
    d = _y(sid)
    assert d["obem_signal"] == obem(d)


def test_obem_chisla():
    assert [_y(s)["obem_signal"] for s in SIDS] == [6 * 19 * 2 * 13, 6 * 19 * 13, 17]
    assert _y("S1")["obem_signal_s_rasshireniem"] == 7 * 19 * 2 * 13
    assert _y("S2")["obem_signal_s_rasshireniem"] == 7 * 19 * 13
    for s in ("S1", "S2"):
        assert "tema" in _y(s)["rasshireniya"]["ma_tip"]
        assert "tema" not in _y(s)["signal"]["ma_tip"]["znacheniya"]


def test_zavisimaya_os_razmnozhaet_tolko_svoe_znachenie():
    d = {"signal": {"a": {"znacheniya": ["x", "y"]}, "b": {"znacheniya": [1, 2, 3]}},
         "zavisimye_osi": [{"os": "c", "pri": {"os": "a", "znachenie": "x"}, "ot": 1, "do": 2, "shag": 0.5}]}
    assert obem(d) == (3 + 1) * 3


@pytest.mark.parametrize("sid", SIDS)
def test_net_klyuchej_tochek_i_itogov(sid):
    assert zapreshchennye(_y(sid)) == []


@pytest.mark.parametrize("klyuch", ZAPRET + ("ex_sharpe", "best_params"))
def test_mutaciya_zapreshchennyj_klyuch_lovitsya(klyuch):
    m = copy.deepcopy(_y("S1"))
    m["signal"]["okno_ma"][klyuch] = 20
    assert zapreshchennye(m), klyuch


def test_mutaciya_obem_i_suzhenie_lovyatsya():
    m = copy.deepcopy(_y("S2"))
    m["obem_signal"] += 1
    assert m["obem_signal"] != obem(m)
    m = copy.deepcopy(_y("S1"))
    m["signal"]["okno_ma"]["znacheniya"].remove(200)
    assert m["signal"]["okno_ma"]["znacheniya"] != WINDOWS


def test_znacheniya_kak_v_istochnike():
    s1, s2, s3 = (_y(s)["signal"] for s in SIDS)
    assert s1["ma_tip"]["znacheniya"] == MA_TYPES == s2["ma_tip"]["znacheniya"]
    assert s1["okno_ma"]["znacheniya"] == WINDOWS == s2["okno_ma"]["znacheniya"]
    assert s1["tip_otkloneniya"]["znacheniya"] == ["std", "atr"]
    assert s1["velichina_otkloneniya"]["znacheniya"] == sorted(set(STD_DEVS) | set(ATR_DEVS))
    assert s1["velichina_otkloneniya"]["spiski_istochnika"] == {"STD_DEVS": STD_DEVS, "ATR_DEVS": ATR_DEVS}
    assert s2["shirina_polos_std"]["znacheniya"] == BB_STDS
    assert s3["okno_donchiana"]["znacheniya"] == DC_WINDOWS
    assert s3["okno_donchiana"]["suzheno_v_istochnike"] is True
