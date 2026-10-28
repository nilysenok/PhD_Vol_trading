# -*- coding: utf-8 -*-
"""Ж-084 п. 2: основная и справочная строки (fastbt/progon_osnovnaya.py). Синтетика; каждая мутация обязана ронять проверку."""
import csv
import gzip
import itertools

import numpy as np
import pytest

from fastbt import ledger_v2 as L
from fastbt import progon_osnovnaya as P
from fastbt import progon_region as RG
from fastbt import reestr_otbora_v1 as V1
from fastbt import reestr_progon as R

T, KOD, VHODY = 4, "K1", "V1"
MAN = {"T": T, "kod": KOD, "vhody": VHODY, "kod_slit": True}
KVS = [f"{g}Q{q}" for g in (2016, 2017, 2018, 2019) for q in range(1, 5)]
TB = {"S1": False, "S5": True}  # синтетическая таблица: у S1 выхода по сигналу нет (только τ), у S5 есть


def _proverit_pravilo(fn):
    """Правило по параметрам точки; fn(klyuch, sid, tablica) -> bool."""
    assert fn({"t_vyhod": "close"}, "S5", TB) is False       # S5: на close возможен выход по сигналу
    assert fn({"t_vyhod": "close"}, "S1", TB) is True        # S1: на close только τ
    assert fn({"t_vyhod": "open2"}, "S5", TB) is True
    for t in ("1500", "1700", 1500, "1700.0"):
        assert fn({"t_vyhod": t}, "S1", TB) is False         # 1500 / 1700 справочные у любой S
    assert fn({"pt": {"t_vyhod": "close"}}, "S5", TB) is False


def test_tablica_pravila_po_S():
    _proverit_pravilo(P.osnovnaya)
    assert P.VYHOD_PO_SIGNALU == {s: True for s in P.SIDS}   # в коде выход по сигналу включён у всех S1–S6
    assert all(P.osnovnaya({"t_vyhod": "close"}, s) is False for s in P.SIDS)
    assert all(P.osnovnaya({"t_vyhod": "open2"}, s) is True for s in P.SIDS)
    with pytest.raises(ValueError):
        P.osnovnaya({"t_vyhod": "0930"}, "S1")
    with pytest.raises(ValueError):
        P.osnovnaya({"t_vyhod": "close"})                    # без стратегии для close ответа нет


def test_mut_1500_osnovnaya(monkeypatch):
    monkeypatch.setattr(P, "OSNOVNYE_T", ("open2", "1500"))
    with pytest.raises(AssertionError):
        _proverit_pravilo(P.osnovnaya)


def _po_ishodam(klyuch, sid, tablica):
    return klyuch.get("dolya_signal", 0.0) == 0.0            # мутация: по исходам сделок, а не по параметрам


def test_ne_zavisit_ot_ishodov_sdelok():
    for dolya in (0.0, 0.7):
        row = {"strategiya": "S5", "klyuch": {"t_vyhod": "close"}, "dolya_signal": dolya}
        assert P.osnovnaya_stroki(row) is False
        row["klyuch"]["t_vyhod"] = "open2"
        assert P.osnovnaya_stroki(row) is True
    with pytest.raises(AssertionError):
        _proverit_pravilo(_po_ishodam)


def _kub(spravochnye_mu):
    """Оси a, b, c × t_vyhod: open2 — основные (пик в центре), close (S5) и 1500 — справочные со средним выше любого основного."""
    ts = ("open2", "close", "1500")
    pts = [dict(zip("abc", v), t_vyhod=t) for t in ts for v in itertools.product(range(5), repeat=3)]
    rows = [{"pt": p, "strategiya": "S5", "sdelok_y": {"2016": 40, "2017": 40}} for p in pts]
    rng, cols, daty = np.random.default_rng(1), [], []
    for k in range(16):
        d = np.array(["2016-01-01"], "datetime64[M]")[0] + 3 * k
        daty.append(d.astype("datetime64[D]") + np.arange(20))
        mu = np.array([0.02 * np.exp(-sum((p[a] - 2) ** 2 for a in "abc") / 2) if p["t_vyhod"] == "open2" else spravochnye_mu for p in pts])
        cols.append(mu[None, :] - 0.002 + rng.normal(0, 0.005, (20, len(pts))))
    return np.vstack(cols), rows, np.concatenate(daty)


def _proverit_region_i_pbo(M, rows, daty):
    osn = {j for j, r in enumerate(rows) if r["pt"]["t_vyhod"] == "open2"}
    res = RG.vybor_region(M, rows, daty)
    assert res is not None and set(res["chleny"]) <= osn    # регион: только основные столбцы
    assert np.isnan(res["tstolb"][sorted(set(range(len(rows))) - osn)]).all() and len(res["tstolb"]) == len(rows)
    assert RG.select_fn_region(rows, daty)(M.T, None, daty) in osn   # выбор под CSCV/PBO: тоже только основные


def test_region_i_pbo_ne_vidyat_spravochnye():
    _proverit_region_i_pbo(*_kub(0.03))


def test_mut_spravochnyj_stolbec_v_matrice(monkeypatch):
    monkeypatch.setattr(RG, "maska", lambda rows: np.ones(len(rows), bool))
    with pytest.raises(AssertionError):
        _proverit_region_i_pbo(*_kub(0.03))


def _row(idx, sid, t_vyhod):
    pt = {"i": idx, "t_vyhod": t_vyhod}
    kl = L.klyuch(sid, pt, "train4", 0.0004, None)
    yrs = {str(y): 0.5 for y in (2016, 2017, 2018, 2019)}
    kv = {k: float(idx + n) for n, k in enumerate(KVS)}
    return {"v": "2.0", "format": "etap1-ledger", "strategiya": sid, "klyuch": kl, "klyuch_sha": L.klyuch_sha(kl, KOD, VHODY),
            "kod": KOD, "vhody": VHODY, "kod_slit": True, "period": "train4", "komissiya": 0.0004, "fdr": None, "status": "ok",
            "prichina": None, "istochnik": "sobol1", "sloy": "fast", "by_year": yrs, "sdelok_y": {k: 10.0 for k in yrs},
            "msr_by_year": dict(yrs), "by_quarter": kv, "mediana_kv": 1.5, "dolya_pol_kv": 0.5,
            "by_quarter_zaem": {"5": kv, "10": kv}, "flagi": {"vyrozhdena": False, "kray": [], "short_shag1_kray": False}, "t_sek": 0.1}


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "VYHOD", str(tmp_path / "progon.csv.gz"))
    monkeypatch.setitem(V1.VYHOD, "train4", str(tmp_path / "A_v1.csv"))
    pi = L.Pisatel(tmp_path / "led", "S5", 1, MAN)
    for i, t in enumerate(("close", "open2", "1500", "1700")):
        pi.add(_row(i, "S5", t), np.full(T, float(i)), np.zeros((17, T)))
    pi.close()
    return tmp_path / "led", tmp_path / "progon.csv.gz"


def _proverit_reestr(root, gz):
    assert R.dopisat_iz_ledger(root, "S5") == 4
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4                                    # ничего не удалено (otbor_statistika п. 4)
    assert {json_t(r): r["osnovnaya"] for r in rows} == {"close": "0", "open2": "1", "1500": "0", "1700": "0"}


def json_t(r):
    import json
    return json.loads(r["klyuch"])["t_vyhod"]


def test_reestr_pishet_vse_i_kolonku(ledger):
    _proverit_reestr(*ledger)
    assert R.kolonki()[-1] == "osnovnaya"


def test_mut_reestr_vykinul_spravochnye(ledger, monkeypatch):
    orig = L.Chitatel.rows
    monkeypatch.setattr(L.Chitatel, "rows", lambda s, status="ok": [r for r in orig(s, status) if P.osnovnaya_stroki(r)])
    with pytest.raises(AssertionError):
        _proverit_reestr(*ledger)


def test_chitatel_matrica_i_N_tolko_osnovnye(ledger):
    root, _ = ledger
    ch = L.Chitatel(root, "S5")
    rs = P.rows_osnovnye(ch)
    assert len(rs) == 1 and ch.matrica(rs).shape == (T, 1) and float(ch.matrica(rs)[0, 0]) == 1.0
    assert P.chisla_N(ch.rows()) == {"N_osnovnye": 1, "N_spravochnye": 3}
    assert P.osnovnaya_stroki({"strategiya": "S5", "klyuch": {"i": 1}}) is None   # оси нет: не классифицируется
