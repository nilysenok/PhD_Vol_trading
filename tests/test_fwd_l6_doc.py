"""Л6: отчёт `fwd_doc` только из таблиц: статус первой строкой, абзац раздела 10, подменённое число (№ 125/140), «лучше/хуже» и знак Δ,
«подтверждено» только при p Холма < 0,05 (№ 158), числа не вписаны в текст генератора (AST). Синтетика, без новых данных."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd
import pytest

from models_v2.s17 import fwd_doc as D
from models_v2.s17 import fwd_eval as E
from test_fwd_l6_eval import fake_stats, make_frame

SRC = Path(D.__file__)


@pytest.fixture(scope="module")
def world():
    fr = make_frame(n=150)
    t = E.all_tables(fr)
    t["checks"] = pd.DataFrame([dict(check="календарь", status="ok", detail="дней 150"), dict(check="RV", status="↺", detail="расхождение")])
    power = pd.DataFrame(dict(kind=["istoricheskij"] * 3, window=["2018_2025", "2023_2025", "2018_2025_bez_2022"], hyp=["X1<K2"] * 3, n_new=[150] * 3, block=[10] * 3, power_holm7=[0.004, 0.038, 0.005]))
    tabs = {k: v for k, v in t.items() if k != "pairs"}
    return fr, tabs, power, D.render(tabs, power)


def test_status_pervoj_strokoj_i_abzac_razdela_10(world):
    fr, tabs, power, text = world
    first = text.splitlines()[0]
    a, b = fr["t_plus"].iloc[0].date(), fr["t_plus"].iloc[-1].date()
    assert first == f"Статус: проверка на новом периоде {a}…{b}, одно открытие по протоколу; X1, ZB заморожены до открытия."
    assert "не имеют на нём независимого подтверждения" in text and "после просмотра" in text and "Повторных открытий этого периода нет" in text
    assert f"после просмотра {str(a)[:7]}…{str(b)[5:7]}" in text
    assert D.verify(text, tabs, power) == []
    assert "0,0" in text and "X1<K2" not in text  # мощность по POWER.csv показана для фактического n; в тексте — «окно: значение»


def test_vsegda_sem_strok_v_poryadke_i_bez_luchshego(world):
    _, tabs, power, text = world
    heads = re.findall(r"(?m)^\| (H\d) \|", text)
    assert heads == ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]
    assert "лучший гибрид" not in text.lower()
    assert D.verify(text + "\nЛучший гибрид на новом периоде — X1.", tabs, power)  # фраза запрещена


def test_mutatsiya_podmenennoe_chislo_lovitsya(world):
    _, tabs, power, text = world
    nums = re.findall(r"(?<![\w.])[-+]?\d+,\d+", text)
    assert nums
    bad = text.replace(nums[len(nums) // 2], "1,23", 1)
    assert bad != text and D.verify(bad, tabs, power)
    bad2 = text.replace("Число пар: ", "Число пар: 1", 1)
    assert D.verify(bad2, tabs, power)


def test_mutatsiya_luchshe_khuzhe_i_znak_delta_lovyatsya(world):
    _, tabs, power, text = world
    assert " лучше " in text
    flipped = re.sub(r"(?m)^(- H\d: \w+) лучше", r"\1 хуже", text, count=1)
    assert flipped != text and D.verify(flipped, tabs, power)
    t2 = dict(tabs)
    fam = tabs["family"].copy()
    fam["delta_pct"] = -fam["delta_pct"]  # знак Δ не из QLIKE
    t2["family"] = fam
    assert any("Δ" in m for m in D.verify(D.render(t2, power), t2, power))


def test_podtverzhdeno_tolko_pri_p_holma_menshe_005_v_otchete(world, monkeypatch):
    fr, tabs, power, _ = world
    monkeypatch.setattr(E, "pair_stats", fake_stats([0.04, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    t = dict(tabs, family=E.family_table(fr))  # p = 0,04 < 0,05, p Холма = 0,28
    text = D.render(t, power)
    assert not re.search(r"(?m)^- H\d: .*; Подтверждено", text) and "; Подтверждено на новом" not in text and D.verify(text, t, power) == []
    mut = t["family"].copy()
    mut.loc[0, "verdict"] = E.CONF
    t3 = dict(t, family=mut)
    assert D.verify(D.render(t3, power), t3, power)
    monkeypatch.setattr(E, "pair_stats", fake_stats([0.004, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    t4 = dict(tabs, family=E.family_table(fr))
    assert "; Подтверждено на новом" in D.render(t4, power)


NAME = re.compile(r"\b[A-Za-z_]{1,12}\d{1,2}\b")  # имена: X1, ZB0, H7, cfg8, столбец power_holm7


def digit_literals(src: str) -> list:
    """Строковые константы (кроме докстрок) с цифрами вне имён вроде X1, ZB0, H7 — числа в тексте генератора."""
    tree = ast.parse(src)
    doc = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
           and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant) and isinstance(n.body[0].value.value, str)}
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc and re.search(r"\d", NAME.sub("", n.value)):
            bad.append(n.value)
    return bad


def test_chisla_ne_vpisany_v_tekst_generatora_ast():
    assert digit_literals(SRC.read_text(encoding="utf-8")) == []
    mutant = SRC.read_text(encoding="utf-8") + '\nTEXT = "Δ лучше на 1,5 %"\n'
    assert digit_literals(mutant) == ["Δ лучше на 1,5 %"]


def test_modul_ne_chitaet_dannye_i_ne_obuchaet():
    banned = {"train_encoder", "fit_pca_past", "train_missing", "read_parquet", "read_csv", "open"}
    for mod in (SRC, Path(E.__file__)):
        tree = ast.parse(mod.read_text(encoding="utf-8"))
        used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        assert not (banned & (used | imported)), mod.name


def test_pustye_tablicy_otchet_stroitsya():
    fr = make_frame(n=1)
    t = {k: v for k, v in E.all_tables(fr).items() if k != "pairs"}
    text = D.render(t)
    assert D.verify(text, t) == [] and text.splitlines()[0].startswith("Статус: проверка на новом периоде ")
