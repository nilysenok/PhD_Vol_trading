"""Свод сверки дивидендов по Д8 (scripts/div_svod.py) и ось дивидендов v6 на годных строках. Синтетика, сеть и данные не нужны."""
import csv
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from fastbt import dividendy, istochniki_v6 as iv, sobytiya_e1

_p = Path(__file__).resolve().parent.parent / "scripts" / "div_svod.py"
_spec = importlib.util.spec_from_file_location("div_svod_skript", _p)
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)

HDR = ["ticker", "record_date", "announce_date", "dps_rub", "src_tablica", "source2_url", "record_date_2", "announce_date_2",
       "dps_rub_2", "tip_istochnika", "verified", "primechanie"]


def _r(verified="yes", rec="2018-07-06", ann="2018-05-30", dps="10", rec2=None, ann2="", dps2="", src="2017_2021", t="AAA"):
    return dict(zip(HDR, [t, rec, ann, dps, src, "", rec2 if rec2 is not None else rec, ann2, dps2 or dps, "", verified, ""]))


def _zapis(papka, name, rows):
    with open(papka / name, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HDR)
        w.writeheader()
        w.writerows(rows)


def _svod(tmp_path, k1, k2=None):
    _zapis(tmp_path, "2026-10-30_div_sverka_A.csv", k1)
    if k2:
        _zapis(tmp_path, "2026-10-31_div_sverka2_A.csv", k2)
    return {(x["ticker"], x["record_date"]): x for x in ds.svod(tmp_path)}


def test_yes_godna(tmp_path):
    x = _svod(tmp_path, [_r()])[("AAA", "2018-07-06")]
    assert x["godna"] == 1 and x["announce_pit"] == "2018-05-30" and x["prichina"] == ""


def test_rashozhdenie_sovpali_reestr_i_summa_announce_pozdnyaya(tmp_path):
    x = _svod(tmp_path, [_r("rashozhdenie", ann="2018-05-30", ann2="2018-06-04", dps="10", dps2="10.05")])[("AAA", "2018-07-06")]
    assert x["godna"] == 1 and x["announce_pit"] == "2018-06-04"
    y = _svod(tmp_path, [_r("rashozhdenie", ann="2018-06-10", ann2="2018-06-04")])[("AAA", "2018-07-06")]
    assert y["announce_pit"] == "2018-06-10"


def test_rashozhdenie_summy_i_reestra_ne_godny(tmp_path):
    x = _svod(tmp_path, [_r("rashozhdenie", dps="10", dps2="11")])[("AAA", "2018-07-06")]
    assert x["godna"] == 0 and x["prichina"] == "summa_ne_sovpala"
    y = _svod(tmp_path, [_r("rashozhdenie", rec2="2018-07-09")])[("AAA", "2018-07-06")]
    assert y["godna"] == 0 and y["prichina"] == "reestr_ne_sovpal"


def test_net_istochnika_i_pustaya_summa_ne_godny(tmp_path):
    assert _svod(tmp_path, [_r("net_istochnika")])[("AAA", "2018-07-06")]["godna"] == 0
    y =_svod(tmp_path, [{**_r("rashozhdenie"), "dps_rub_2": ""}])[("AAA", "2018-07-06")]
    assert y["godna"] == 0


def test_net_vyplat_ne_idet_v_svod_i_net_daty_obyavleniya(tmp_path):
    s = _svod(tmp_path, [_r(rec="", src="2015_2016"), _r(ann="")])
    assert list(s) == [("AAA", "2018-07-06")] and s[("AAA", "2018-07-06")]["godna"] == 0


def test_krug2_perekryvaet_krug1_po_klyuchu(tmp_path):
    k1 = [_r("net_istochnika")]
    k2 = [_r("yes", ann="2018-05-31")]
    x = _svod(tmp_path, k1, k2)[("AAA", "2018-07-06")]
    assert x["godna"] == 1 and x["announce_pit"] == "2018-05-31"
    # другой src_tablica — другой ключ, круг 2 его не трогает
    s = ds.chitat(tmp_path)
    assert len(s) == 1


def test_svod_ne_chitaet_sam_sebya_i_zapis_chitaetsya(tmp_path):
    _zapis(tmp_path, "2026-10-30_div_sverka_A.csv", [_r()])
    ds.zapisat(ds.svod(tmp_path), tmp_path / "div_sverka_svod.csv")
    assert len(ds.svod(tmp_path)) == 1


def test_realnyj_svod_213_godnyh_iz_247_a_do_2021_146_iz_173():
    s = ds.svod()
    assert (len(s), sum(x["godna"] for x in s)) == (247, 213)
    do21 = [x for x in s if x["record_date"] <= "2021-12-31"]
    assert (len(do21), sum(x["godna"] for x in do21)) == (173, 146)      # прежние числа сохранены
    assert ds.SVOD.read_text(encoding="utf-8") == _tekst(s)
    # круг 2 перекрывает круг 1 на реальных файлах
    d = {(x["ticker"], x["record_date"]): x for x in s}
    assert sum(x["godna"] for x in s if x["ticker"] == "PHOR") > 2  # после круга 1 было 2
    v = d[("VTBR", "2021-07-15")]
    assert v["godna"] == 0 and "reestr" in v["prichina"]
    assert d[("RTKM", "2019-07-04")]["godna"] == 0


def _tekst(rows):
    import io
    b = io.StringIO()
    w = csv.DictWriter(b, fieldnames=ds.COLS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return b.getvalue()


# ---------------- ось событий: только годные строки, дата объявления = announce_pit ----------------

def _svod_csv(tmp_path, rows):
    p = tmp_path / "svod.csv"
    ds.zapisat(rows, p)
    return p


@pytest.fixture
def kal(monkeypatch):
    cal = pd.bdate_range("2018-01-01", "2018-12-31").to_numpy("datetime64[ns]")
    monkeypatch.setattr(dividendy, "_calendar", lambda: cal)
    return pd.DatetimeIndex(cal)


def _sv(rec, ann, godna, t="AAA"):
    return {"ticker": t, "record_date": rec, "dps_rub": "10", "announce_pit": ann, "godna": godna, "prichina": "" if godna else "x"}


def test_os_poluchaet_tolko_godnye(tmp_path, kal):
    p = _svod_csv(tmp_path, [_sv("2018-07-06", "2018-05-30", 1), _sv("2018-09-07", "2018-08-01", 0)])
    d = sobytiya_e1.load_div_svod(p)
    assert list(d["record_date"]) == [pd.Timestamp("2018-07-06")] and d["announce_date"].iloc[0] == pd.Timestamp("2018-05-30")
    b = sobytiya_e1.div_block(kal, d, "AAA")
    assert kal[b].tolist() == [pd.Timestamp("2018-07-04")]       # T+2 по барам, блок на last_cum


def test_os_announce_pit_pozdnyaya_ne_blokiruet_zadnim_chislom(tmp_path, kal):
    d = sobytiya_e1.load_div_svod(_svod_csv(tmp_path, [_sv("2018-07-06", "2018-07-05", 1)]))
    assert not sobytiya_e1.div_block(kal, d, "AAA").any()      # объявлено после last_cum 04.07 — день не блокируется


def test_net_svoda_os_zakryta(tmp_path):
    assert sobytiya_e1.load_div_svod(tmp_path / "net.csv") is None


def test_sobytiya_s_div_verified_daet_div(tmp_path, kal):
    d = sobytiya_e1.load_div_svod(_svod_csv(tmp_path, [_sv("2018-07-06", "2018-05-30", 1)]))
    cbr = pd.DataFrame({"date": [], "known_from": []})
    assert sobytiya_e1.sobytiya(kal, ["AAA"], cbr, d, True)["div"]["AAA"].sum() == 1
    assert "div" not in sobytiya_e1.sobytiya(kal, ["AAA"], cbr, d, False)


# ---------------- мутации ----------------

def test_mutaciya_announce_pit_rannyaya_lovitsya(tmp_path, monkeypatch):
    r = _r("rashozhdenie", ann="2018-05-30", ann2="2018-06-04")
    assert ds.stroka(r)["announce_pit"] == "2018-06-04"
    monkeypatch.setattr(ds, "_pozdnyaya", lambda daty: min(d for d in daty if d))
    assert ds.stroka(r)["announce_pit"] == "2018-05-30"      # мутант (ранняя дата) даёт иное значение: проверка выше его роняет


def test_mutaciya_net_istochnika_v_os_lovitsya(tmp_path, kal, monkeypatch):
    p = _svod_csv(tmp_path, [_sv("2018-07-06", "2018-05-30", 1), _sv("2018-09-07", "2018-08-01", 0)])
    assert len(sobytiya_e1.load_div_svod(p)) == 1
    real = pd.read_csv

    def bez_filtra(path, **kw):
        return real(path, **kw).assign(godna=1)
    monkeypatch.setattr(pd, "read_csv", bez_filtra)
    assert len(sobytiya_e1.load_div_svod(p)) == 2             # мутант пропускает негодную строку — тест выше падает на нём


def test_mutaciya_krug2_ne_perekryvaet_lovitsya(tmp_path, monkeypatch):
    k1, k2 = [_r("net_istochnika")], [_r("yes")]
    assert _svod(tmp_path, k1, k2)[("AAA", "2018-07-06")]["godna"] == 1
    monkeypatch.setattr(ds, "sorted", lambda it, key=None: list(reversed(sorted(it, key=key))), raising=False)
    assert _svod(tmp_path, k1, k2)[("AAA", "2018-07-06")]["godna"] == 0


class _Fajly(iv.Fajly):
    def __init__(self):
        self.end, self.main = pd.Timestamp("2022-01-01"), None

    def indeks(self, name):
        return None

    def brent(self):
        return {}, pd.DataFrame({"secid": [], "lsttrade": [], "roll": []})

    def kalendar(self):
        return pd.DataFrame({"date": [], "known_from": []}), sobytiya_e1.load_div_svod(sobytiya_e1.SVOD_CSV)


def test_smena_svoda_menyaet_klyuch_kesha(tmp_path, kal, monkeypatch):
    f = _Fajly()
    h = []
    for rows in ([_sv("2018-07-06", "2018-05-30", 1)], [_sv("2018-07-06", "2018-05-31", 1)], [_sv("2018-07-06", "2018-05-30", 1), _sv("2018-09-07", "2018-08-01", 1)]):
        monkeypatch.setattr(sobytiya_e1, "SVOD_CSV", _svod_csv(tmp_path, rows))
        h.append(iv.sha_ryadov(f, []))
    assert len(set(h)) == 3
    assert f.div_otfiltrovan() is True


def test_mutaciya_sha_bez_div_lovitsya(tmp_path, kal, monkeypatch):
    f = _Fajly()
    monkeypatch.setattr(_Fajly, "ryady", lambda self, t: [(n, x) for n, x in iv.Fajly.ryady(self, t) if n != "div"])
    h = []
    for ann in ("2018-05-30", "2018-05-31"):
        monkeypatch.setattr(sobytiya_e1, "SVOD_CSV", _svod_csv(tmp_path, [_sv("2018-07-06", ann, 1)]))
        h.append(iv.sha_ryadov(f, []))
    assert h[0] == h[1]                                        # мутант слеп к своду; настоящий тест выше требует различия


def test_net_svoda_div_verified_false(tmp_path, monkeypatch):
    monkeypatch.setattr(sobytiya_e1, "SVOD_CSV", tmp_path / "net.csv")
    assert _Fajly().div_otfiltrovan() is False
