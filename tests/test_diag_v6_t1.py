"""Д-2 (Ж-145): таблица (1) на синтетике с известным ответом + мутация «группа по колонке gruppa CSV Морриса»."""
from __future__ import annotations

import pytest

from fastbt import diag_v6_t1 as t1

# (ось, колонка gruppa CSV Морриса (раздел куба), группа владельца, n_akt, μ* по окнам ≤2016/≤2017/≤2018)
S1 = [("a", "filtr", "signal", 20, (0.3, 0.2, 0.1)), ("b", "signal", "kontekst", 25, (0.1, 0.1, 0.1)),
      ("c", "vyhod", "vyhod_risk", 20, (0.4, 0.4, 0.4)), ("d", "vyhod", "vyhod_risk", 5, (0.9, 0.9, 0.9)),
      ("e", "vhod", "ispolnenie", 0, (None, None, None))]
S2 = [("x", "signal", "signal", 20, (0.01, 0.01, 0.01)), ("y", "vhod", "ispolnenie", 20, (0.5, 0.5, 0.5)),
      ("z", "vyhod", "vyhod_risk", 20, (0.4, 0.4, 0.4))]
OKNA = ("<=2016", "<=2017", "<=2018")


def _zapisat(root, sid, osi, ne_ocen):
    d = root / sid
    d.mkdir(parents=True)
    s = ['# "v": "1"', "# kod: a, b", "os,gruppa,novaya,okno,n_akt,mu_star,sigma,U95,zamorozit,prichina"]
    for os, kol, _g, n, mus in osi:
        for okno, mu in zip(OKNA, mus):
            s.append(f"{os},{kol},False,{okno},{n},{'nan' if mu is None else mu},0.0,0.0,False,\"x, y\"")
    (d / f"morris_{sid}.csv").write_text("\n".join(s) + "\n", encoding="utf-8")
    (d / "zamorozka.yaml").write_text("v: 1\nne_ocenyeny_net_sdelok:\n" + "".join(f"- {o}\n" for o in ne_ocen) + "mertvye_ne_ocenyeny: []\n", encoding="utf-8")


@pytest.fixture()
def vhod(tmp_path):
    root = tmp_path / "morris"
    _zapisat(root, "S1", S1, ["e"])
    _zapisat(root, "S2", S2, [])
    g = tmp_path / "gruppy.csv"
    g.write_text("# komm, s zapyatoj\nsid,os,razdel_kuba,gruppa,spornaya,prichina\n" + "".join(
        f"{sid},{o},{k},{gg},net,\n" for sid, osi in (("S1", S1), ("S2", S2)) for o, k, gg, *_ in osi), encoding="utf-8")
    return root, g


def _stroka(df, sid, okno, gruppa):
    return df[(df.sid == sid) & (df.okno == okno) & (df.gruppa == gruppa)].iloc[0]


def test_izvestnyj_otvet_summy_doli_ne_izmereno(vhod):
    df, _ = t1.tablica_1(*vhod)
    assert len(df) == 2 * 3 * 4
    s = _stroka(df, "S1", "<=2016", "signal")
    assert (s.osej, s.izmereno, s.sum_mu, s.max_mu) == (1, 1, 0.3, 0.3)
    k = _stroka(df, "S1", "<=2016", "kontekst")
    assert k.sum_mu == pytest.approx(0.1) and k.dolya == pytest.approx(0.1 / 0.8)
    v = _stroka(df, "S1", "<=2016", "vyhod_risk")
    # d: n_akt 5 < 20 — не измерена и не входит в Σμ* (0.9 не прибавлено)
    assert (v.osej, v.izmereno, v.ne_izmereno, v.sum_mu, v.dolya) == (2, 1, 1, 0.4, pytest.approx(0.5))
    i = _stroka(df, "S1", "<=2016", "ispolnenie")
    assert (i.osej, i.izmereno, i.ne_izmereno, i.ne_ocenyeny, i.sum_mu, i.dolya) == (1, 0, 1, 1, 0.0, 0.0)
    assert _stroka(df, "S1", "<=2018", "signal").sum_mu == pytest.approx(0.1)
    assert df[(df.sid == "S1") & (df.okno == "<=2016")].dolya.sum() == pytest.approx(1.0)


def test_vyvod_slaboe_mesto_tolko_s2(vhod):
    _, v = t1.tablica_1(*vhod)
    # S1: сигнал 0,1 ≥ 0,05, ранг 2 из 4 (ничья с контекстом, порядок групп) — ок; S2: сигнал 0,01 — последний и max < 0,05
    assert v.startswith("слабое место:") and v.endswith("Sk: S2")
    assert "S1:" not in v and "max μ* сигнала < 0,05" in v and "не в двух верхних" in v


def test_vyvod_ne_slaboe_mesto(tmp_path, vhod):
    root, g = vhod
    import shutil
    shutil.rmtree(root / "S2")
    assert t1.tablica_1(root, g)[1] == "не слабое место"


def test_os_ne_v_gruppah_oshibka(vhod):
    root, g = vhod
    g.write_text(g.read_text().replace("S1,a,filtr,signal,net,\n", ""))
    with pytest.raises(ValueError, match="S1.a"):
        t1.tablica_1(root, g)


def test_pustoj_koren_oshibka(tmp_path):
    with pytest.raises(ValueError):
        t1.tablica_1(tmp_path)


def test_mutaciya_gruppa_po_kolonke_csv_lovitsya(vhod, monkeypatch):
    """G2.2: группа берётся из колонки `gruppa` CSV Морриса (раздел куба) — итог обязан разойтись с известным ответом."""
    eta, _ = t1.tablica_1(*vhod)
    monkeypatch.setattr(t1, "_gruppa", lambda gr, sid, os, razdel: {"filtr": "kontekst", "vyhod": "vyhod_risk", "vhod": "ispolnenie"}.get(razdel, razdel))
    mut, _ = t1.tablica_1(*vhod)
    assert _stroka(eta, "S1", "<=2016", "signal").sum_mu == pytest.approx(0.3)
    assert _stroka(mut, "S1", "<=2016", "signal").sum_mu == pytest.approx(0.1)
    assert not eta.equals(mut)
