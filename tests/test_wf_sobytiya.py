"""Лист wf-sobytiya: события 2022–2026 (point-in-time), ставка ЦБ до 2026-02-28, корпсобытия question закрыты.

Сеть не нужна. Мутации: «событие видно до даты известно с», «ставка ffill за решением 13.02 (16,0 на 20.02)»,
«закрытая строка применяется множителем».
"""
import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _modul(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


SE = _modul("sobrat_events_wf", "scripts/sobrat_events.py")
PS = _modul("prodlit_stavku_wf", "scripts/dannye/prodlit_stavku.py")
from fastbt import data as fdata  # noqa: E402
from fastbt import sobytiya_pit as pit  # noqa: E402

KONEC_WF = date(2026, 3, 1)
RATE = ROOT / "moex_discovery" / "data" / "external" / "macro" / "raw" / "key_rate_cbr.parquet"


@pytest.fixture(scope="module")
def wf_text():
    return SE.sobrat_wf()


@pytest.fixture()
def wf_csv(tmp_path, wf_text, monkeypatch):
    p = tmp_path / "events.csv"
    p.write_text(wf_text, encoding="utf-8")
    monkeypatch.setattr(pit, "END", KONEC_WF)     # замок 2022+ снят (Ж-136): читаем до 2026-02-28
    return p


def test_do_2022_pobajtovo_prezhnij(wf_text):
    staryj = SE.sobrat()
    assert wf_text.startswith(staryj)
    fajl = (ROOT / "data" / "events.csv").read_text(encoding="utf-8")
    assert wf_text == fajl                       # файл в data/ = wf-сборка (было: = сборка по умолчанию, ≤ 2021)
    assert staryj == fajl[:len(staryj)]          # сборка по умолчанию = префикс файла
    assert wf_text.count("\n") - staryj.count("\n") == 346


def test_sobytiya_2023_est(wf_csv):
    r = pit.sobytiya_na("2023-02-10", "SBER", path=wf_csv)
    assert [x["tip"] for x in r] == ["cbr_zasedanie_klyuchevaya_stavka"]
    forts = [x for x in pit._rows(wf_csv) if x["tip"] == "forts_expiraciya" and x["data_sobytiya"].year == 2023]
    assert len(forts) == 68


def _pit_ok(fn, path):
    """PIT: строка не видна накануне даты объявления и видна в день объявления (по всем строкам 2022+)."""
    for r in pit._rows(path):
        a, s = r["data_obyavleniya"], r["data_sobytiya"]
        if s.year < 2022 or a == s:
            continue
        d = a - timedelta(days=1)
        vp = (s - d).days
        for x in fn(d, r["ticker"] if r["ticker"] != "*" else None, vp, path):
            if x["data_obyavleniya"] == a and x["data_sobytiya"] == s and x["tip"] == r["tip"]:
                return False
        if not any(x["data_obyavleniya"] == a and x["data_sobytiya"] == s and x["tip"] == r["tip"]
                   for x in fn(a, r["ticker"] if r["ticker"] != "*" else None, (s - a).days, path)):
            return False
    return True


def _mut_vidno_do_izvestno(d, ticker, vpered, path):
    d = date.fromisoformat(str(d)[:10])
    return [r for r in pit._rows(path) if d <= r["data_sobytiya"] <= d + timedelta(days=vpered)
            and (ticker is None or r["ticker"] in ("*", ticker))]


def test_pit_ne_narushen(wf_csv):
    assert _pit_ok(lambda d, t, v, p: pit.sobytiya_na(d, t, v, p), wf_csv)


def test_mutatsiya_sobytie_vidno_do_izvestno_s_lovitsya(wf_csv):
    assert not _pit_ok(_mut_vidno_do_izvestno, wf_csv)


def test_wf_sobrat_otkazyvaet_pri_obyavlenii_pozzhe_sobytiya(monkeypatch, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("ticker,tip,data_obyavleniya,data_sobytiya,istochnik,accessed,note\n"
                   "*,x,2023-05-02,2023-05-01,u,2026-10-30,n\n", encoding="utf-8")
    monkeypatch.setattr(SE, "VHOD_WF", bad)
    with pytest.raises(ValueError, match="объявление позже"):
        SE.sobrat_wf()


# ---- ставка ----

def _stavka_20_fevralya(seriya):
    return float(seriya.loc[seriya["date"] == pd.Timestamp("2026-02-20"), "value"].iloc[0])


def _posle(staryj):
    return PS.prodlit(staryj, PS.chitat_reshenia())


def _staryj_sint():
    return pd.DataFrame({"date": pd.bdate_range("2026-01-30", "2026-02-06"), "value": 16.0, "ticker": "K"})


def test_stavka_20_fevralya_15_5():
    n = _posle(_staryj_sint())
    assert _stavka_20_fevralya(n) == 15.5
    v = n.set_index("date")["value"]
    assert v["2026-02-13"] == 16.0 and v["2026-02-16"] == 15.5
    assert n["date"].max() == pd.Timestamp("2026-02-27") and pd.Timestamp("2026-02-23") not in set(n["date"])


def test_mutatsiya_stavka_ffill_za_resheniem_lovitsya():
    ffill = _staryj_sint()
    dni = pd.bdate_range("2026-02-09", "2026-02-27")
    ffill = pd.concat([ffill, pd.DataFrame({"date": dni, "value": 16.0, "ticker": "K"})], ignore_index=True)
    assert _stavka_20_fevralya(ffill) == 16.0 != 15.5


def test_stavka_do_resheniya_ne_ffill_nazad():
    with pytest.raises(ValueError, match="нет решения"):
        PS.stavka_na(pd.DatetimeIndex(["2024-01-01"]), PS.chitat_reshenia())


@pytest.mark.skipif(not RATE.exists(), reason="нет parquet ставки")
def test_stavka_real_do_06_02_pobajtovo():
    s = pd.read_parquet(RATE)
    n = _posle(s) if s["date"].max() < PS.KONEC else s
    pd.testing.assert_frame_equal(n.iloc[:len(s)].reset_index(drop=True), s.reset_index(drop=True))
    assert _stavka_20_fevralya(n) == 15.5 and n["date"].max() == pd.Timestamp("2026-02-27")


# ---- корпсобытия ----

def _korp_s_zakrytymi(tmp_path, status):
    ca = pd.read_csv(ROOT / "data" / "corporate_actions.csv")
    klyuchi = {("MOEX", "2025-07-10"), ("MTSS", "2022-09-21"), ("MTSS", "2025-07-07"),
               ("OGKB", "2025-11-05"), ("VTBR", "2025-07-11")}
    q = pd.Series([(t, d) in klyuchi for t, d in zip(ca["ticker"], ca["date"])])
    assert q.sum() == 5
    ca.loc[q, "status"] = status
    if status == "applied":
        ca.loc[q, "factor"] = 0.87
    p = tmp_path / "corporate_actions.csv"
    ca.to_csv(p, index=False)
    return p


@pytest.mark.parametrize("tk", ["MOEX", "MTSS", "OGKB", "VTBR"])
def test_zakrytye_question_ne_dayut_mnozhitel(tmp_path, monkeypatch, tk):
    monkeypatch.setattr(fdata, "CORP", _korp_s_zakrytymi(tmp_path, "closed_not_event"))
    ev = fdata.corp_actions(tk)
    assert ev.empty


def test_mutatsiya_zakrytaya_stroka_primenena_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(fdata, "CORP", _korp_s_zakrytymi(tmp_path, "applied"))
    ev = fdata.corp_actions("MTSS")
    assert len(ev) == 2 and (ev["factor"] == 0.87).all()      # такая ошибка видна: тест выше упал бы


# ---- граница wf в загрузчике: только при отметке снятого замка ----

def _bez_otmetki(monkeypatch):
    import functools
    from fastbt import period_v2 as pv
    monkeypatch.setattr(pv, "proverit_chtenie", functools.partial(pv.proverit_chtenie, otmetka=Path("/net/otmetki.json")))


def _wf_chitaet(mod, p):
    r = mod.sobytiya_na("2023-02-10", "SBER", path=p, period="wf")
    return [x["tip"] for x in r] == ["cbr_zasedanie_klyuchevaya_stavka"]


def test_wf_s_otmetkoj_chitaet_2022_2026(tmp_path, wf_text):
    p = tmp_path / "e.csv"
    p.write_text(wf_text, encoding="utf-8")
    assert _wf_chitaet(pit, p)
    assert pit.sobytiya_na("2026-02-13", path=p, period="wf")
    with pytest.raises(PermissionError):
        pit.sobytiya_na("2026-03-01", path=p, period="wf")


def test_wf_bez_otmetki_otkaz_i_prochie_periody_END(tmp_path, wf_text, monkeypatch):
    p = tmp_path / "e.csv"
    p.write_text(wf_text, encoding="utf-8")
    _bez_otmetki(monkeypatch)
    with pytest.raises(PermissionError):
        pit.sobytiya_na("2023-02-10", "SBER", path=p, period="wf")
    with pytest.raises(PermissionError):
        pit.sobytiya_na("2023-02-10", "SBER", path=p)
    with pytest.raises(PermissionError):
        pit.sobytiya_na("2023-02-10", "SBER", path=p, period="train4")


def test_mutatsiya_granitsa_wf_bez_otmetki_lovitsya(tmp_path, wf_text, monkeypatch):
    from types import ModuleType
    src = Path(pit.__file__).read_text(encoding="utf-8")
    old = "    period_v2.proverit_chtenie(\"wf\")\n"
    assert old in src
    m = ModuleType("mut")
    m.__file__ = pit.__file__
    exec(compile(src.replace(old, "", 1), "mut", "exec"), m.__dict__)
    p = tmp_path / "e.csv"
    p.write_text(wf_text, encoding="utf-8")
    _bez_otmetki(monkeypatch)
    with pytest.raises(PermissionError):
        pit.sobytiya_na("2023-02-10", "SBER", path=p, period="wf")      # настоящий код отказывает
    assert _wf_chitaet(m, p)                                            # мутант читает без отметки -> проверка выше его ловит
