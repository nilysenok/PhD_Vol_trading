"""Л3.3 (Ж-146 п. 3, Ж-093): драйвер `scripts/analysis/a_shtrih_v7.py`, только синтетика, прогона на данных нет.
G1 шапка таблицы полна; G2 вход знаковый (|позиция| из ledger — отказ); G3 данные ≥ 2020 не читаются;
G4 строки (основная, 2 контроля, аблация 1→4) считаются движком книги; G5 доля дисперсии рынка = R².
Тест написан ДО кода: на базе падает (нет модуля)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import a_shtrih_v7_kniga as K
from fastbt import progon_urovni as pu
from test_a_shtrih_v7_kniga import _mk

_p = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "a_shtrih_v7.py"
_s = importlib.util.spec_from_file_location("a_shtrih_v7_drajver", _p)
D = importlib.util.module_from_spec(_s)
_s.loader.exec_module(D)

STROKI = ["osnovnaya", "kontrol_ravnyj_risk", "kontrol_bez_hedzha", "ablaciya_1", "ablaciya_2", "ablaciya_3", "ablaciya_4"]


def _vhod(T=120, seed=3, **kw):
    d = _mk(S=3, N=4, T=T, seed=seed, nan=False)
    d["poz"] = np.where(d["poz"] == 0, 1.0, d["poz"])             # в синтетике есть и лонги, и шорты
    d.update(daty=np.asarray(pd.bdate_range("2016-01-04", periods=T), dtype="datetime64[D]"), znak="znakovyj",
             istochnik="centry_ryady")
    d.update(kw)
    return d


class _Zhurnal(dict):
    """Словарь, помнящий, какие ключи прочитаны: проверка порядка «даты раньше данных»."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.chitano = []

    def __getitem__(self, k):
        self.chitano.append(k)
        return super().__getitem__(k)


# ---- G2: вход знаковый
@pytest.mark.parametrize("izm", [
    {"znak": "modul"}, {"znak": None}, {"istochnik": "ledger"}, {"poz": None},
])
def test_g2_vhod_ne_znakovyj_otkaz(izm):
    d = _vhod()
    d.update(izm)
    if izm.get("znak", 1) is None:
        d.pop("znak")
    with pytest.raises(ValueError):
        D.proverit_vhod(d)


def test_g2_pozicii_bez_edinogo_shorta_otkaz_kak_modul_iz_ledger():
    d = _vhod()
    d["poz"] = np.abs(d["poz"])                                    # |позиция| при метке «знаковый»: всё равно отказ
    with pytest.raises(ValueError, match="шорт|знак"):
        D.proverit_vhod(d)


def test_g2_znakovyj_vhod_prohodit():
    D.proverit_vhod(_vhod())


# ---- G3: ≥ 2020 не читается
def test_g3_data_2020_otkaz_i_massivy_ne_chitayutsya():
    d = _vhod()
    daty = d["daty"].copy()
    daty[-1] = np.datetime64("2020-01-02")
    z = _Zhurnal({**d, "daty": daty})
    with pytest.raises(ValueError, match="2020"):
        D.zagruzit(z)
    assert not ({"X", "poz", "imoex", "mix"} & set(z.chitano))


def test_g3_granica_31_12_2019_prohodit_a_1_1_2020_net():
    d = _vhod(T=2)
    d["daty"] = np.array(["2019-12-30", "2019-12-31"], dtype="datetime64[D]")
    D.proverit_vhod(d)
    d["daty"] = np.array(["2019-12-31", "2020-01-01"], dtype="datetime64[D]")
    with pytest.raises(ValueError):
        D.proverit_vhod(d)


def test_g3_zagruzit_chitaet_daty_pervymi():
    z = _Zhurnal(_vhod())
    D.zagruzit(z)
    assert z.chitano[0] in ("daty", "znak", "istochnik") and "daty" in z.chitano[:3]


# ---- G4: строки
def test_g4_nabor_strok_i_kolonki():
    d = _vhod()
    rez = D.schitat(d)
    assert list(rez) == STROKI
    tab = D.tablica(d, rez)
    assert list(tab.index) == STROKI and list(tab.columns) == list(D.KOLONKI)
    for k in ("sharpe", "msr", "sortino", "mdd", "kalmar", "sharpe_ned", "msr_ned", "sharpe_norm", "ekspoziciya",
              "cena_prostoev", "dolya_rynka", "kom_tik_god", "kom_hedzh_god"):
        assert k in tab.columns, k


def test_g4_osnovnaya_ravna_ablaciya_4_pobajtno_i_ravna_dvizhku_knigi():
    d = _vhod()
    rez = D.schitat(d)
    ref = K.kniga_ryada(d["X"], d["poz"], d["sektora"], d["imoex"], d["mix"], d["ks"], vesa="paritet")
    assert rez["osnovnaya"].ryad.tobytes() == ref.ryad.tobytes() == rez["ablaciya_4"].ryad.tobytes()


def test_g4_ablaciya_1_ravna_a_shtrih_ravnyj_bez_targeta_pobajtno():
    d = _vhod()
    konf = next(c for c in pu.konfiguracii() if c["rol"] == "ravnyj_bez_targeta")
    ref = pu.a_shtrih({f"S{k}": d["X"][k] for k in range(3)}, konf, pd.DatetimeIndex(d["daty"]))
    assert rez_bytes(D.schitat(d)["ablaciya_1"]) == ref.tobytes()


def rez_bytes(r):
    return r.ryad.tobytes()


def test_g4_kontroli_otlichayutsya_ot_osnovnoj_tolko_svoim_flagom():
    d = _vhod(T=300, seed=5)
    rez = D.schitat(d)
    assert not rez["kontrol_bez_hedzha"].h.any() and rez["osnovnaya"].h.any()
    bez = K.kniga_ryada(d["X"], d["poz"], d["sektora"], d["imoex"], d["mix"], d["ks"], vesa="paritet", hedzh=False)
    assert rez["kontrol_bez_hedzha"].ryad.tobytes() == bez.ryad.tobytes()
    rr = K.kniga_ryada(d["X"], d["poz"], d["sektora"], d["imoex"], d["mix"], d["ks"], vesa="ravnyj_risk")
    assert rez["kontrol_ravnyj_risk"].ryad.tobytes() == rr.ryad.tobytes()


def test_g4_ablaciya_dobavlyaet_po_odnomu_urovnyu():
    d = _vhod(T=300, seed=6)
    r = D.schitat(d)
    assert r["ablaciya_2"].a.tobytes() != r["ablaciya_1"].a.tobytes()          # +паритет: веса другие
    assert not r["ablaciya_2"].h.any() and r["ablaciya_3"].h.any()              # +нейтрализация (хедж) — на шаге 3
    assert (r["ablaciya_3"].m == 1).all() and (r["ablaciya_3"].f == 1).all()   # на шаге 3 таргета и выключателя ещё нет
    assert not (r["ablaciya_4"].m == 1).all()


def test_g4_komissiya_v_stroke_net_nulya_pri_vklyuchennyh_masshtabah():
    tab = D.tablica(_vhod(T=300, seed=7), D.schitat(_vhod(T=300, seed=7)))
    assert tab.loc["osnovnaya", "kom_tik_god"] > 0 and tab.loc["osnovnaya", "kom_hedzh_god"] > 0
    assert tab.loc["ablaciya_1", "kom_tik_god"] == 0 and tab.loc["ablaciya_1", "kom_hedzh_god"] == 0


# ---- G5: доля дисперсии рынка
def test_g5_dolya_rynka_izvestnyj_otvet():
    rng = np.random.default_rng(1)
    im = rng.normal(0, 0.01, 500)
    assert D.dolya_rynka(2 * im, im) == pytest.approx(1.0, abs=1e-12)
    assert D.dolya_rynka(-3 * im, im) == pytest.approx(1.0, abs=1e-12)           # R², а не корреляция: знак не влияет
    assert D.dolya_rynka(rng.normal(0, 0.01, 500), im) < 0.05
    mix = 0.6 * im + 0.8 * im.std() * rng.normal(0, 1, 500)
    assert D.dolya_rynka(mix, im) == pytest.approx(np.corrcoef(mix, im)[0, 1] ** 2, abs=1e-12)


def test_g5_dolya_rynka_propuski_i_pustota():
    im = np.array([0.01, np.nan, -0.02, 0.03, 0.0])
    r = np.array([0.02, 0.01, np.nan, 0.06, 0.0])
    assert np.isfinite(D.dolya_rynka(r, im))
    assert np.isnan(D.dolya_rynka(np.array([np.nan, np.nan]), np.array([0.1, 0.2])))
    assert np.isnan(D.dolya_rynka(np.zeros(10), np.arange(10.0)))                  # нулевая дисперсия ряда -> nan, не 0


def test_g5_dolya_do_i_posle_v_tablice():
    d = _vhod(T=300, seed=8)
    tab = D.tablica(d, D.schitat(d))
    assert tab.loc["ablaciya_1", "dolya_rynka"] == pytest.approx(
        D.dolya_rynka(D.schitat(d)["ablaciya_1"].ryad, d["imoex"]), abs=1e-15)


# ---- G1: шапка
def test_g1_shapka_polna_i_pishetsya_v_csv(tmp_path):
    d = _vhod()
    tab = D.tablica(d, D.schitat(d))
    sh = D.shapka(d)
    for k in D.POLYA_SHAPKI:
        assert sh.get(k) not in (None, ""), k
    assert "0,04" in sh["komissiya"] and sh["status"].startswith("разработка") and "до аудита" in sh["status"]
    assert len(sh["hash_vhodov"]) == 64 and len(sh["hash_koda"]) == 64
    p = D.zapisat(tab, sh, tmp_path)
    txt = Path(p).read_text(encoding="utf-8")
    assert txt.startswith("# ") and all(f"# {k}:" in txt for k in D.POLYA_SHAPKI)
    back = pd.read_csv(p, comment="#", index_col=0)
    assert list(back.index) == STROKI


def test_g1_hash_vhodov_menyaetsya_pri_podmene_lyubogo_vhoda():
    d = _vhod()
    h0 = D.shapka(d)["hash_vhodov"]
    for k in ("X", "poz", "imoex", "mix", "ks", "sektora"):
        d2 = {**d, k: np.asarray(d[k], float).copy() if k != "sektora" else d[k].copy()}
        d2[k].flat[0] = d2[k].flat[0] + 1
        assert D.shapka(d2)["hash_vhodov"] != h0, k
