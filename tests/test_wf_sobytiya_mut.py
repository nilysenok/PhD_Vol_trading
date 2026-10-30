"""Мутации листа wf-sobytiya на настоящем коде (тестировщик PR #753): каждая обязана ронять проверку автора.

а) событие видно до «известно с»; б) ставка 20.02.2026 = 16,0 (ffill / вступление в день решения);
в) closed_not_event применяется как множитель; г) сборка без wf захватывает >= 2022. Сеть не нужна.
Проверки автора вызываются как функции из tests/test_wf_sobytiya.py (загрузка по пути, не `from tests.`).
"""
import importlib.util
import sys
from datetime import date, timedelta  # noqa: F401  (timedelta нужен exec-мутантам)
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

_s = importlib.util.spec_from_file_location("twf_avtor", ROOT / "tests" / "test_wf_sobytiya.py")
T = importlib.util.module_from_spec(_s)
_s.loader.exec_module(T)
SRC = Path(T.pit.__file__).read_text(encoding="utf-8")


@pytest.fixture()
def wf_csv(tmp_path, monkeypatch):
    p = tmp_path / "events.csv"
    p.write_text(T.SE.sobrat_wf(), encoding="utf-8")
    monkeypatch.setattr(T.pit, "END", date(2026, 3, 1))      # _pit_ok читает строки через настоящий pit._rows
    return p


def _kopiya(old=None, new=None):
    m = ModuleType("mut_pit")
    m.__file__ = T.pit.__file__
    src = SRC
    if old is not None:
        assert old in SRC, old
        src = SRC.replace(old, new, 1)
    exec(compile(src, "mut", "exec"), m.__dict__)
    m.timedelta = timedelta
    m.END = date(2026, 3, 1)                       # замок wf снят
    return m


# ---- а) PIT ----

def test_a_kopiya_bez_mutacii_prohodit(wf_csv):
    assert T._pit_ok(_kopiya().sobytiya_na, wf_csv)


@pytest.mark.parametrize("old,new", [
    ('r["data_obyavleniya"] <= d\n', 'True\n'),                                              # известность не проверяется
    ('r["data_obyavleniya"] <= d\n', 'r["data_obyavleniya"] <= d + timedelta(days=1)\n'),    # d+1 вместо d
    ('r["data_obyavleniya"] <= d\n', 'r["data_sobytiya"] <= d + timedelta(days=vpered)\n'),  # по дате события
], ids=["bez_proverki", "d_plyus_1", "po_sobytiyu"])
def test_a_mutatsiya_lovitsya(wf_csv, old, new):
    assert not T._pit_ok(_kopiya(old, new).sobytiya_na, wf_csv)


# ---- б) ставка ----

def test_b_ffill_16_na_20_fevralya_lovitsya(monkeypatch):
    monkeypatch.setattr(T.PS, "stavka_na", lambda dni, reshenia: [16.0] * len(dni))
    assert T._stavka_20_fevralya(T._posle(T._staryj_sint())) == 16.0
    with pytest.raises(AssertionError):
        T.test_stavka_20_fevralya_15_5()


def test_b_vstuplenie_v_den_resheniya_lovitsya(monkeypatch):
    orig = T.PS.stavka_na
    dni_resh = pd.read_csv(T.PS.CSV, parse_dates=["data_resheniya"])["data_resheniya"]
    monkeypatch.setattr(T.PS, "stavka_na", lambda dni, r: orig(dni, r.assign(data_vstupleniya=dni_resh.values)))
    assert T._posle(T._staryj_sint()).set_index("date")["value"]["2026-02-13"] == 15.5   # решение ещё не в силе
    with pytest.raises(AssertionError):
        T.test_stavka_20_fevralya_15_5()


def test_b_konec_ryada_ranshe_lovitsya(monkeypatch):
    monkeypatch.setattr(T.PS, "KONEC", pd.Timestamp("2026-02-20"))
    with pytest.raises(AssertionError):
        T.test_stavka_20_fevralya_15_5()


# ---- в) closed_not_event ----

def _primenyaet_zakrytye(ticker):
    ca = pd.read_csv(T.fdata.CORP, parse_dates=["date"])
    return ca[(ca["ticker"] == ticker) & ca["status"].isin(["applied", "closed_not_event"])]


@pytest.mark.parametrize("tk", ["MOEX", "MTSS", "OGKB", "VTBR"])
def test_v_zakrytye_kak_mnozhitel_lovitsya(tmp_path, monkeypatch, tk):
    monkeypatch.setattr(T.fdata, "corp_actions", _primenyaet_zakrytye)
    with pytest.raises(AssertionError):
        T.test_zakrytye_question_ne_dayut_mnozhitel(tmp_path, monkeypatch, tk)


# ---- г) сборка без wf ----

def test_g_granitsa_sborki_sdvinuta_lovitsya(monkeypatch):
    monkeypatch.setattr(T.SE, "END", "2027-01-01")
    with pytest.raises((AssertionError, ValueError)):      # ValueError: сборка wf сама отказывает (нижняя граница = END)
        T.test_do_2022_pobajtovo_prezhnij(T.SE.sobrat_wf())


def _main_v_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(T.SE, "VYHOD", tmp_path / "out.csv")
    T.SE.main()
    return (tmp_path / "out.csv").read_bytes()


def _prefiks():
    """Файл data/events.csv целиком и его префикс <= 2021 (сборка по умолчанию)."""
    fajl = (ROOT / "data" / "events.csv").read_bytes()
    return fajl, fajl[:len(T.SE.sobrat().encode())]


def test_g_main_po_umolchaniyu_daet_prezhnij_fajl(tmp_path, monkeypatch):
    """Было: == весь файл. Стало: режим по умолчанию = префикс <= 2021 текущего файла, строго короче целого."""
    fajl, pref = _prefiks()
    out = _main_v_tmp(tmp_path, monkeypatch)
    assert out == pref and len(out) < len(fajl)


def test_g_main_na_sobrat_wf_lovitsya(tmp_path, monkeypatch):
    """Мутация: режим по умолчанию собирает wf (346 строк 2022+) — вывод перестаёт быть префиксом <= 2021."""
    fajl, pref = _prefiks()
    wf = T.SE.sobrat_wf()
    monkeypatch.setattr(T.SE, "sobrat", lambda: wf)
    assert _main_v_tmp(tmp_path, monkeypatch) != pref
