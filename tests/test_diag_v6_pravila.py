"""Лист Д-0 диагностики v6 train4: правила до данных. Мутации: ось выкинута из CSV; порог изменён; топ-10 по иному ключу."""
import csv
import math
from pathlib import Path

import pytest

from fastbt import canon_v6, diag_v6_pravila as D

ROOT = Path(__file__).resolve().parents[1]
# Пороги из толкования [31.10 00:12], п. (в) — независимая копия (литералы), не импорт из модуля.
TOLKOVANIE = dict(sdelok_v_god_mediana_min=10, nuli_dolya_max=0.5, plato_dolya_min=0.3, kontrol_percentil=95, odna_storona_dolya_max=0.8,
                  odna_storona_let_min=2, bez_luchshih_dnej_n=10, bez_luchshih_dnej_dolya_min=0.5, prochee_dolya_max=0.3)


def chitat():
    with D.CSV.open(encoding="utf-8") as f:
        return [tuple(r.values()) for r in csv.DictReader(l for l in f if not l.startswith("#"))]


def test_csv_pokryvaet_vse_osi_rovno_odin_raz():
    D.proverit_pokrytie(chitat())


def test_csv_ravno_generatoru_i_shapka():
    assert [r[:5] for r in chitat()] == [(s, o, r_, g, "da" if sp else "net") for s, o, r_, g, sp, _ in D.stroki_osej()]
    t = D.CSV.read_text(encoding="utf-8").splitlines()
    assert t[0].startswith("# статус: разработка") and t[1].startswith("# kod:") and t[2].startswith("# vhody:")
    assert t[2].endswith(D.sha_vhodov())


def test_mutaciya_os_vykinuta_iz_csv_lovitsya():
    rows = chitat()
    for i in (0, len(rows) // 2, len(rows) - 1):
        with pytest.raises(ValueError):
            D.proverit_pokrytie(rows[:i] + rows[i + 1:])


def test_mutaciya_dubl_i_neizvestnaya_gruppa_lovyatsya():
    rows = chitat()
    with pytest.raises(ValueError):
        D.proverit_pokrytie(rows + [rows[0]])
    with pytest.raises(ValueError):
        D.proverit_pokrytie([(*rows[0][:3], "prochee", *rows[0][4:])] + rows[1:])


def test_gruppy_po_planu():
    g = {s: D.gruppa_osi(s, r) for s, r in dict(adx_th="filtr", k_e="vhod", tejk="vyhod", dlit="kontekst", tf="kontekst", sigma_ist="ispolnenie",
                                                  pauza_okno="ispolnenie", ed_lim="ispolnenie", z_entry="signal", relf_k="kontekst").items()}
    assert g == dict(adx_th="kontekst", k_e="ispolnenie", tejk="vyhod_risk", dlit="signal", tf="signal", sigma_ist="vyhod_risk",
                     pauza_okno="kontekst", ed_lim="ispolnenie", z_entry="signal", relf_k="kontekst")
    assert all({x[3] for x in chitat() if x[1] == o} == {"vyhod_risk"} for o in ("bezubytok", "pir", "smena", "stop_v_plyus"))  # в S3/S4 раздел другой


def test_porogi_ravny_tolkovaniyu():
    assert D.PRAVILA == TOLKOVANIE
    assert {k for k in D.PREDLOZHENIYA} and all(isinstance(v, str) and v for v in D.PREDLOZHENIYA.values())


def test_mutaciya_porog_izmenyon_lovitsya():
    for k, v in TOLKOVANIE.items():
        assert dict(D.PRAVILA, **{k: v + 1}) != TOLKOVANIE


def test_tolkovanie_soderzhit_te_zhe_porogi():
    t = (ROOT / "docs/resheniya/tolkovaniya.md").read_text(encoding="utf-8")
    blok = t[t.index("[31.10 00:12 по `date`, класс B] Диагностика"):]
    blok = blok[:blok.index("\n")]
    for frag in ("< 10", "> 50 %", "< 30 %", "95-м перцентиле", "> 80 %", "10 лучших дней", "вдвое", "> 30 %"):
        assert frag in blok, frag


def _rows():
    rows = [dict(klyuch_n=i, status="ok", rol="sobol", msr=float(i), sharpe=-1.0 - i) for i in range(15)]
    return rows + [dict(klyuch_n=99, status="ok", rol="sobol", msr=float("nan"), sharpe=9.5),
                   dict(klyuch_n=98, status="oshibka", rol="sobol", msr=50.0, sharpe=9.0),
                   dict(klyuch_n=97, status="ok", rol="kontrol", msr=60.0, sharpe=9.0),
                   dict(klyuch_n=96, status="ok", rol="sobol", msr=-0.5, sharpe=-9.0),
                   dict(klyuch_n=95, status="ok", rol="sobol", msr=None, sharpe=-0.1),
                   dict(klyuch_n=94, status="ok", rol="sobol", msr=None, sharpe=float("nan"))]


def test_top10_po_msr_zapas_sharpe_pomechen_bez_vorot():
    t = D.top10(_rows())
    # 99: msr не определён -> sharpe 9,5 (между msr 10 и 9, с пометкой); остальные по msr 14…6
    assert [r["klyuch_n"] for r in t] == [14, 13, 12, 11, 10, 99, 9, 8, 7, 6]
    assert [r["klyuch_n"] for r in t if r["pometka_sharpe"]] == [99]
    assert 94 not in [r["klyuch_n"] for r in t] and 98 not in [r["klyuch_n"] for r in t] and 97 not in [r["klyuch_n"] for r in t]
    # ворот нет: msr 3, sharpe -0,1 (пометка), msr -0,5 — все в списке
    assert [(r["klyuch_n"], r["pometka_sharpe"]) for r in D.top10([r for r in _rows() if r["klyuch_n"] in (96, 95, 3)])] == [(3, False), (95, True), (96, False)]


def test_mutaciya_ranzhirovanie_po_drugomu_polyu_lovitsya(monkeypatch):
    ozhid = [r["klyuch_n"] for r in D.top10(_rows())]
    for pole in ("sharpe", "msr_ned"):
        monkeypatch.setitem(D.TOP10, "kluch", pole)
        assert [r["klyuch_n"] for r in D.top10(_rows())] != ozhid, pole


def test_top10_tie_po_klyuchu():
    rows = [dict(klyuch_n=k, status="ok", rol="sobol", msr=1.0) for k in (5, 2, 9)]
    assert [r["klyuch_n"] for r in D.top10(rows)] == [2, 5, 9]


def test_vse_sidy_v_csv():
    assert {r[0] for r in chitat()} == set(canon_v6.SIDS)
