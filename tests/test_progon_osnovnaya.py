# -*- coding: utf-8 -*-
"""Ж-086: основная и справочная строки (fastbt/progon_osnovnaya.py). Синтетика; каждая мутация обязана ронять проверку.

Переписано тестировщиком (П-7, Ж-081) под Ж-086 вместо Ж-084 п. 2. Что изменилось и почему:
- правило «close справочная только при выходе по сигналу» (таблица по S) заменено правилом Ж-086 п. 1: close — справочная всегда,
  основные t_vyhod ∈ {open2, 1500, 1700}, vhod = rynok справочный, пока исполняется по close d (RYNOK_ISPOLNENIE = close_d);
- 1500 / 1700 теперь основные (были справочными): мутация «1500 справочная» ловится;
- строка без оси t_vyhod или vhod — ValueError, а не «остаётся в выборе» (закрывает дыру отступления Ж-084);
- якорь A = s9 (istochnik из ISTOCHNIKI_YAKORYA) не входит в отбор, регион, PBO: отдельный столбец-якорь в матрице;
- добавлены пути WF, LOYO, LOTO (шпион на vybor_region модулей progon_wf и progon_ustojch).
Мутации: close основная; 1500 справочная; rynok по close основная; фильтр по исходам; справочный столбец в PBO / регионе / WF /
LOYO / LOTO; якорь в регионе; реестр без справочных; chisla_N по всем строкам.
"""
import csv
import gzip
import itertools
import json

import numpy as np
import pytest

from fastbt import ledger_v2 as L
from fastbt import progon_osnovnaya as P
from fastbt import progon_region as RG
from fastbt import progon_ustojch as U
from fastbt import progon_wf as PW
from fastbt import reestr_otbora_v1 as V1
from fastbt import reestr_progon as R

T, KOD, VHODY = 4, "K1", "V1"
MAN = {"T": T, "kod": KOD, "vhody": VHODY, "kod_slit": True}
KVS = [f"{g}Q{q}" for g in (2016, 2017, 2018, 2019) for q in range(1, 5)]


def _proverit_pravilo(fn):
    """Правило по параметрам точки; fn(klyuch) -> bool."""
    for v in ("0.5", "rynok"):
        assert fn({"t_vyhod": "close", "vhod": v}) is False          # close x справочная при любом входе
    for t in ("open2", "1500", "1700", 1500, "1700.0"):
        assert fn({"t_vyhod": t, "vhod": "0.5"}) is True             # open2, 15:00, 17:00 + лимитный вход — основные
        assert fn({"t_vyhod": t, "vhod": "rynok"}) is True           # rynok по open d+1 основной (RYNOK_ISPOLNENIE = open_d1)
    assert fn({"pt": {"t_vyhod": "open2", "vhod": "1.0"}}) is True


def test_pravilo_po_parametram():
    _proverit_pravilo(P.osnovnaya)
    assert P.RYNOK_ISPOLNENIE == "open_d1"


def test_rynok_close_d_ostayotsya_spravochnym(monkeypatch):
    monkeypatch.setattr(P, "RYNOK_ISPOLNENIE", "close_d")        # подмена одной константой: rynok по close d — справочный
    assert P.osnovnaya({"t_vyhod": "open2", "vhod": "rynok"}) is False
    assert P.osnovnaya({"t_vyhod": "close", "vhod": "rynok"}) is False


def test_osi_obyazatelny():
    for pt in ({"vhod": "0.5"}, {"t_vyhod": "open2"}, {}, {"pt": {"i": 1}}):
        with pytest.raises(ValueError, match="нет оси"):
            P.osnovnaya(pt)
    with pytest.raises(ValueError):
        P.osnovnaya({"t_vyhod": "0930", "vhod": "0.5"})
    with pytest.raises(ValueError, match="нет оси"):
        P.maska([{"klyuch": {"t_vyhod": "open2", "vhod": "0.5"}}, {"klyuch": {"t_vyhod": "open2"}}])


def _mut_close(k):
    return True if P._pt(k)["t_vyhod"] == "close" else P.osnovnaya(k)


def _mut_1500(k):
    return False if P._tok(P._pt(k)["t_vyhod"]) == "1500" else P.osnovnaya(k)


def _mut_rynok(k):
    return False if str(P._pt(k)["vhod"]) == "rynok" else P.osnovnaya(k)     # rynok снова справочный (исполнение по close d)


def _po_ishodam(k):
    return k.get("dolya_signal", 0.0) == 0.0                     # мутация: по исходам сделок, а не по параметрам


@pytest.mark.parametrize("mut", [_mut_close, _mut_1500, _mut_rynok, _po_ishodam], ids=["close", "1500", "rynok_close", "ishody"])
def test_mutacii_pravila_lovyatsya(mut):
    with pytest.raises(AssertionError):
        _proverit_pravilo(mut)


def test_ne_zavisit_ot_ishodov_sdelok():
    for dolya in (0.0, 0.7):
        assert P.osnovnaya_stroki({"strategiya": "S5", "klyuch": {"t_vyhod": "close", "vhod": "0.5"}, "dolya_signal": dolya}) is False
        assert P.osnovnaya_stroki({"strategiya": "S5", "klyuch": {"t_vyhod": "open2", "vhod": "0.5"}, "dolya_signal": dolya}) is True


def test_yakor_ne_osnovnoj_a_sobol_osnovnye():
    kl = {"t_vyhod": "open2", "vhod": "0.5"}
    assert P.osnovnaya_stroki({"klyuch": kl, "istochnik": P.ISTOCHNIKI_YAKORYA[0]}) is False
    for ist in ("sobol1", "sobol2"):                             # реально пишет progon_sobol (ist = sobol1 / sobol2)
        assert P.osnovnaya_stroki({"klyuch": kl, "istochnik": ist}) is True


# --- матрица: 125 точек × варианты (t_vyhod, vhod, istochnik); основные — первые два, остальные справочные или якорь
VAR = [("open2", "0.5", "sobol1", 0.02), ("1700", "0.5", "sobol1", 0.02),     # основные, пик в центре
       ("close", "0.5", "sobol1", 0.03), ("close", "rynok", "sobol1", 0.03),  # справочные: среднее выше любого основного
       ("open2", "0.25", P.ISTOCHNIKI_YAKORYA[0], 0.04)]                      # якорь: среднее ещё выше


def _kub():
    pts = list(itertools.product(range(5), repeat=3))
    rows, mu = [], []
    for t, v, ist, m in VAR:
        for p in pts:
            rows.append({"pt": {**dict(zip("abc", p)), "t_vyhod": t, "vhod": v}, "strategiya": "S5", "istochnik": ist,
                         "sdelok_y": {"2016": 40, "2017": 40}})
            mu.append(m * np.exp(-sum((x - 2) ** 2 for x in p) / 2) if m == 0.02 else m)
    rng, cols, daty = np.random.default_rng(1), [], []
    for k in range(16):
        d = np.array(["2016-01-01"], "datetime64[M]")[0] + 3 * k
        daty.append(d.astype("datetime64[D]") + np.arange(20))
        cols.append(np.array(mu)[None, :] - 0.002 + rng.normal(0, 0.005, (20, len(rows))))
    return np.vstack(cols), rows, np.concatenate(daty)


def _osn(rows):
    return {j for j, r in enumerate(rows) if r["istochnik"] == "sobol1" and r["pt"]["t_vyhod"] in ("open2", "1700")
            and r["pt"]["vhod"] == "0.5"}


def _shpion(monkeypatch, modul):
    """Записывает chleny каждого результата vybor_region, вызванного модулем (WF, LOYO, LOTO)."""
    got, orig = [], modul.vybor_region

    def spy(*a, **kw):
        res = orig(*a, **kw)
        if res is not None:
            got.append(set(res["chleny"]))
        return res
    monkeypatch.setattr(modul, "vybor_region", spy)
    return got


def _proverit_region_i_pbo(M, rows, daty):
    osn = _osn(rows)
    res = RG.vybor_region(M, rows, daty)
    assert res is not None and set(res["chleny"]) <= osn    # регион: ни справочных, ни якоря
    assert np.isnan(res["tstolb"][sorted(set(range(len(rows))) - osn)]).all() and len(res["tstolb"]) == len(rows)
    assert RG.select_fn_region(rows, daty)(M.T, None, daty) in osn   # выбор под CSCV/PBO: тоже только основные


def test_region_i_pbo_ne_vidyat_spravochnye_i_yakor():
    _proverit_region_i_pbo(*_kub())


def _vse_true(rows):
    return np.ones(len(rows), bool)


def test_mut_spravochnyj_stolbec_v_regione_i_pbo(monkeypatch):
    monkeypatch.setattr(RG, "maska", _vse_true)
    with pytest.raises(AssertionError):
        _proverit_region_i_pbo(*_kub())


def test_mut_yakor_v_regione(monkeypatch):
    monkeypatch.setattr(P, "ISTOCHNIKI_YAKORYA", ())
    with pytest.raises(AssertionError):
        _proverit_region_i_pbo(*_kub())


def _wf(M, rows, daty, monkeypatch):
    got = _shpion(monkeypatch, PW)
    PW.wf(M, rows, daty, PW.OSNOVNOE)
    return got


def test_wf_ne_vidit_spravochnye(monkeypatch):
    M, rows, d = _kub()
    got = _wf(M, rows, d, monkeypatch)
    assert got and all(g <= _osn(rows) for g in got)


def test_mut_spravochnyj_stolbec_v_wf(monkeypatch):
    monkeypatch.setattr(RG, "maska", _vse_true)
    M, rows, d = _kub()
    got = _wf(M, rows, d, monkeypatch)
    with pytest.raises(AssertionError):
        assert got and all(g <= _osn(rows) for g in got)


def test_loyo_loto_ne_vidyat_spravochnye(monkeypatch):
    M, rows, d = _kub()
    got = _shpion(monkeypatch, U)
    U.loyo(M, rows, d)
    n_loyo = len(got)
    tik = (M.T[:, None, :] + np.random.default_rng(2).normal(0, 0.001, (M.shape[1], 2, M.shape[0]))).astype(np.float32)
    U.loto(tik, rows, d, ["T0", "T1"])
    assert n_loyo >= 4 and len(got) > n_loyo and all(g <= _osn(rows) for g in got)


def test_mut_spravochnyj_stolbec_v_loyo_loto(monkeypatch):
    monkeypatch.setattr(RG, "maska", _vse_true)
    M, rows, d = _kub()
    got = _shpion(monkeypatch, U)
    U.loyo(M, rows, d)
    with pytest.raises(AssertionError):                  # LOYO
        assert got and all(g <= _osn(rows) for g in got)
    got.clear()
    tik = (M.T[:, None, :] + np.random.default_rng(2).normal(0, 0.001, (M.shape[1], 2, M.shape[0]))).astype(np.float32)
    U.loto(tik, rows, d, ["T0", "T1"])
    with pytest.raises(AssertionError):                  # LOTO
        assert got and all(g <= _osn(rows) for g in got)


def _row(idx, sid, t_vyhod, vhod="0.5"):
    pt = {"i": idx, "t_vyhod": t_vyhod, "vhod": vhod}
    kl = L.klyuch(sid, pt, "train4", 0.0004, None)
    yrs = {str(y): 0.5 for y in (2016, 2017, 2018, 2019)}
    kv = {k: float(idx + n) for n, k in enumerate(KVS)}
    return {"v": "2.0", "format": "etap1-ledger", "strategiya": sid, "klyuch": kl, "klyuch_sha": L.klyuch_sha(kl, KOD, VHODY),
            "kod": KOD, "vhody": VHODY, "kod_slit": True, "period": "train4", "komissiya": 0.0004, "fdr": None, "status": "ok",
            "prichina": None, "istochnik": "sobol1", "sloy": "fast", "by_year": yrs, "sdelok_y": {k: 10.0 for k in yrs},
            "msr_by_year": dict(yrs), "by_quarter": kv, "mediana_kv": 1.5, "dolya_pol_kv": 0.5,
            "by_quarter_zaem": {"5": kv, "10": kv}, "flagi": {"vyrozhdena": False, "kray": [], "short_shag1_kray": False}, "t_sek": 0.1}


OZhID = {("close", "0.5"): "0", ("open2", "0.5"): "1", ("1500", "0.5"): "1", ("1700", "0.5"): "1", ("close", "rynok"): "0", ("open2", "rynok"): "1"}


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "VYHOD", str(tmp_path / "progon.csv.gz"))
    monkeypatch.setitem(V1.VYHOD, "train4", str(tmp_path / "A_v1.csv"))
    pi = L.Pisatel(tmp_path / "led", "S5", 1, MAN)
    for i, (t, v) in enumerate(OZhID):
        pi.add(_row(i, "S5", t, v), np.full(T, float(i)), np.zeros((17, T)))
    pi.close()
    return tmp_path / "led", tmp_path / "progon.csv.gz"


def _proverit_reestr(root, gz):
    assert R.dopisat_iz_ledger(root, "S5") == len(OZhID)
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(OZhID)                           # ничего не удалено (otbor_statistika п. 4)
    kl = [json.loads(r["klyuch"]) for r in rows]
    assert {(k["t_vyhod"], k["vhod"]): r["osnovnaya"] for k, r in zip(kl, rows)} == OZhID


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
    assert len(rs) == 4 and ch.matrica(rs).shape == (T, 4)
    assert sorted(float(c) for c in ch.matrica(rs)[0]) == [1.0, 2.0, 3.0, 5.0]
    assert P.chisla_N(ch.rows()) == {"N_osnovnye": 4, "N_spravochnye": 2}


def test_mut_chisla_N_po_vsem_strokam(ledger, monkeypatch):
    ch = L.Chitatel(ledger[0], "S5")
    monkeypatch.setattr(P, "maska", lambda rows: np.ones(len(rows), bool))
    with pytest.raises(AssertionError):
        assert P.chisla_N(ch.rows()) == {"N_osnovnye": 4, "N_spravochnye": 2}
