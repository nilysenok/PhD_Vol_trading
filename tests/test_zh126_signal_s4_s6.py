"""Ж-127 п. 1: сигнальные оси S4–S6 (zh126_S{k}.yaml) — полнота, объём, без точек.

Проверки: yaml читается; у каждой оси есть istochnik_stroki; obem_signal = произведение
длин осей (с учётом зависимых); списки совпадают со строками источника на 6a7a4636
(ничего не сужено); нет ключей точек/результатов. Мутации — в конце файла.
"""
import ast
import copy
import re
import subprocess
from math import prod
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
YAML = str(ROOT / "fastbt/designs/zh126/zh126_{k}.yaml")
STRATEGII = ["S4", "S5", "S6"]
ZAPRET = ("best", "sharpe", "params", "tochki", "rezultat")


def zagruzit(k):
    return yaml.safe_load(Path(YAML.format(k=k)).read_text(encoding="utf-8"))


def obem(d):
    """Произведение длин независимых осей; зависимая ось {os, ot, znacheniya_po}
    заменяет множитель родителя суммой длин по его значениям."""
    zav = d.get("zavisimye_osi") or []
    roditeli = {z["ot"] for z in zav}
    nez = [len(o["znacheniya"]) for n, o in d["signal"].items() if n not in roditeli]
    mnozh = []
    for r in roditeli:
        po_r = [z for z in zav if z["ot"] == r]
        mnozh.append(sum(prod(len(z["znacheniya_po"].get(v, [None])) for z in po_r)
                         for v in d["signal"][r]["znacheniya"]))
    return prod(nez) * prod(mnozh)


def zapreshchennye_klyuchi(obj, put=""):
    najdeno = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(z in str(k).lower() for z in ZAPRET):
                najdeno.append(f"{put}/{k}")
            najdeno += zapreshchennye_klyuchi(v, f"{put}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            najdeno += zapreshchennye_klyuchi(v, f"{put}[{i}]")
    return najdeno


@pytest.fixture(scope="module")
def istochnik():
    d = zagruzit("S4")["istochnik"]
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{d['kommit']}:{d['fajl']}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"коммит источника недоступен: {r.stderr.strip()}")
    return r.stdout.splitlines()


def spiski_iz_strok(d, stroki_istochnika):
    """Для каждой оси: списки-литералы на одиночных строках из istochnik_stroki."""
    for imya, os_ in d["signal"].items():
        for kusok in str(os_["istochnik_stroki"]).split(","):
            kusok = kusok.strip()
            if not kusok.isdigit():
                continue
            m = re.search(r"\[[^\[\]]*\]", stroki_istochnika[int(kusok) - 1])
            if m:
                yield imya, kusok, ast.literal_eval(m.group(0))


@pytest.mark.parametrize("k", STRATEGII)
def test_yaml_chitaetsya_i_format(k):
    d = zagruzit(k)
    assert d["strategiya"] == k
    assert d["istochnik"]["kommit"] == "6a7a4636"
    assert d["istochnik"]["fajl"] == "moex_volatility/scripts/daily_upper_bound.py"
    assert d["obshchee"] == "zh126_obshchee.yaml"
    assert isinstance(d["zavisimye_osi"], list)
    assert d["filtry_tipa"] and d["tip_filtrov"] in ("trend", "kontrtrend")
    assert d["signal"]


@pytest.mark.parametrize("k", STRATEGII)
def test_u_kazhdoj_osi_stroki_istochnika(k):
    for imya, os_ in zagruzit(k)["signal"].items():
        assert str(os_.get("istochnik_stroki", "")).strip(), imya
        assert os_["znacheniya"], imya
        assert len(set(map(str, os_["znacheniya"]))) == len(os_["znacheniya"]), imya


@pytest.mark.parametrize("k", STRATEGII)
def test_obem_ravno_proizvedeniyu_osej(k):
    d = zagruzit(k)
    assert d["obem_signal"] == obem(d)


def test_obem_po_strategiyam():
    assert [zagruzit(k)["obem_signal"] for k in STRATEGII] == [225, 12, 180]


@pytest.mark.parametrize("k", STRATEGII)
def test_net_klyuchej_tochek_i_rezultatov(k):
    assert zapreshchennye_klyuchi(zagruzit(k)) == []


@pytest.mark.parametrize("k", STRATEGII)
def test_spiski_sovpadayut_s_istochnikom(k, istochnik):
    d = zagruzit(k)
    sverki = list(spiski_iz_strok(d, istochnik))
    imena = {o for o, v in d["signal"].items() if v.get("imya_v_istochnike")}
    assert {s[0] for s in sverki} == imena  # каждая ось с именем сверена со строкой
    for imya, stroka, spisok in sverki:
        assert d["signal"][imya]["znacheniya"] == spisok, f"{k}.{imya}, строка {stroka}"


# ── мутации: проверка обязана падать ──

def test_mutaciya_klyuch_rezultata_lovitsya():
    for klyuch in ZAPRET:
        d = copy.deepcopy(zagruzit("S4"))
        d["signal"]["mult"][klyuch] = 1.0
        assert zapreshchennye_klyuchi(d), klyuch


def test_mutaciya_suzhenie_osi_lovitsya(istochnik):
    d = copy.deepcopy(zagruzit("S6"))
    d["signal"]["porog"]["znacheniya"] = d["signal"]["porog"]["znacheniya"][:-1]
    rashozhdeniya = [i for i, _, s in spiski_iz_strok(d, istochnik)
                     if d["signal"][i]["znacheniya"] != s]
    assert rashozhdeniya == ["porog"]
    assert d["obem_signal"] != obem(d)


def test_mutaciya_zavisimaya_os_menyaet_obem():
    d = copy.deepcopy(zagruzit("S5"))
    d["zavisimye_osi"] = [{"os": "x", "ot": "uroven",
                           "znacheniya_po": {1: [0, 1], 2: [0], 3: [0, 1, 2]}}]
    assert obem(d) == 4 * 1 * (2 + 1 + 3)
    assert d["obem_signal"] != obem(d)
