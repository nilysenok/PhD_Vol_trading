"""Д-2 доп. (Ж-151): Моррис по числу сделок (вердикт «частота»/«сигнал») и сравнение μ* сигнала со стопом и размером. Синтетика с известным ответом."""
from __future__ import annotations

import json

import numpy as np
import pytest

from fastbt import diag_v6_t1_chastota as ch
from fastbt import morris_v6 as mv

OSI = ["sa", "sb", "stop_nach", "razmer_tip"]
GR = {"sa": ("signal", "signal"), "sb": ("signal", "signal"), "stop_nach": ("vyhod_risk", "x"), "razmer_tip": ("vyhod_risk", "x")}
NOM = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
POR = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
NS = [0, 0, 40, 0, 0, 0, 0, 80, 0, 0]                          # сделок за 4 года; sa: обе точки пары без сделок, sb: 0 → 40 / 0 → 80
Y = [0, 0, 0, 0.2, 0.1, 0, 0, 0, 0.3, 0.1]                      # точка 2: сделки есть, Y = 0 (вес 0) — mSR «плоский»


def _plan(sid, r, mnozh):
    return NOM, POR, OSI, 10


@pytest.fixture()
def vhod(tmp_path):
    root = tmp_path / "morris"
    d = root / "S1"
    d.mkdir(parents=True)
    y = np.array([[v] * 3 for v in Y], dtype=float)
    E = mv.effekty(y[NOM, 0], NOM, POR, (y == 0).all(axis=1)[NOM])
    n = np.isfinite(E).sum(0)
    mu = np.where(n > 0, np.nan_to_num(np.abs(E)).sum(0) / np.maximum(n, 1), np.nan)
    s = ["# r: 2", "# mnozh: 10", "# tochek: 10", "os,gruppa,novaya,okno,n_akt,mu_star,sigma,U95,zamorozit,prichina"]
    for i, o in enumerate(OSI):
        s += [f"{o},x,False,{okno},{n[i]},{mu[i]},0,0,False,\"a, b\"" for okno in mv_okna()]
    (d / "morris_S1.csv").write_text("\n".join(s) + "\n", encoding="utf-8")
    (d / "zamorozka.yaml").write_text("v: 1\n", encoding="utf-8")
    (d / "morris_y.jsonl").write_text(json.dumps({"a": 0, "b": 10, "y": y.tolist(), "ns": NS}) + "\n", encoding="utf-8")
    g = tmp_path / "gruppy.csv"
    g.write_text("sid,os,razdel_kuba,gruppa,spornaya,prichina\n" + "".join(f"S1,{o},x,{gg},net,\n" for o, (gg, _) in GR.items()), encoding="utf-8")
    return root, g


def mv_okna():
    return ("<=2016", "<=2017", "<=2018")


def _stroka(df, os):
    return df[df.os == os].iloc[0]


def test_izvestnyj_otvet_chastota_i_signal(vhod):
    df = ch.tablica_chastota(*vhod, plan_fn=_plan)
    assert list(df.os) == ["sa", "sb"]                             # только сигнальные оси с «не измерено»
    a, b = _stroka(df, "sa"), _stroka(df, "sb")
    assert (a.n_shagov, a.dolya_par_bez_sdelok, a.mu_sdelki, a.med_sdelok_god, a.verdikt) == (2, 1.0, 0.0, 0.0, "частота")
    assert (b.n_shagov, b.dolya_par_bez_sdelok, b.mu_sdelki, b.med_sdelok_god, b.verdikt) == (2, 0.0, 15.0, 15.0, "сигнал")
    assert np.isnan(b.mu_sharpe)                                   # Шарп плоский: обе точки пары с Y = 0 — не эффект


def test_mutaciya_verdikt_bez_proverki_sdelok_lovitsya(vhod, monkeypatch):
    monkeypatch.setattr(ch, "verdikt", lambda med: "сигнал")
    df = ch.tablica_chastota(*vhod, plan_fn=_plan)
    assert _stroka(df, "sa").verdikt != "частота"                  # на правке — «частота»; мутация расходится с известным ответом


def test_verdikt_granica_i_net_shagov():
    assert ch.verdikt(ch.KRIT_SDELOK_GOD - 0.01) == "частота"
    assert ch.verdikt(ch.KRIT_SDELOK_GOD) == "сигнал"
    assert ch.verdikt(float("nan")) == "нет шагов"


def test_plan_ne_sovpal_s_csv_oshibka(vhod):
    def plan_drugoj(sid, r, mnozh):
        return NOM, POR[:, ::-1], OSI, 10                           # порядок осей в траекториях иной → n_akt не те
    with pytest.raises(ValueError, match="план Морриса не совпал"):
        ch.tablica_chastota(*vhod, plan_fn=plan_drugoj)


def test_chislo_tochek_ne_sovpalo_oshibka(vhod):
    with pytest.raises(ValueError, match="точек"):
        ch.tablica_chastota(*vhod, plan_fn=lambda sid, r, m: (NOM, POR, OSI, 11))


def test_sravnenie_mu_signala_stopa_razmera(vhod):
    df = ch.sravnenie(*vhod)
    r = df.iloc[0]
    assert len(df) == 1 and r.sid == "S1"
    assert (r.ocen_signal, r.mu_signal_sr) == (0, 0.0)             # у sa, sb нет ни одного активного шага с Шарпом
    assert (r.os_stop_max, r.mu_stop_max) == ("stop_nach", pytest.approx(0.25))     # |0.2 − 0| и |0.3 − 0| на шаг
    assert (r.os_razmer_max, r.mu_razmer_max) == ("razmer_tip", pytest.approx(0.15))  # |0.1 − 0.2| и |0.1 − 0.3|


def test_zapis_pisat_edinicy_v_shapku(vhod, tmp_path):
    p = tmp_path / "o" / "t.csv"
    ch.zapisat(ch.sravnenie(*vhod), p)
    tekst = p.read_text(encoding="utf-8")
    assert tekst.startswith("# μ* = среднее |Δ Шарп дн.| на один активный шаг") and "умолчание дирижёра" in tekst
