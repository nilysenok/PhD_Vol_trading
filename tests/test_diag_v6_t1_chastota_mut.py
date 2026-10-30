"""Мутации к fastbt/diag_v6_t1_chastota.py (тестировщик PR #864, Ж-151): подложенная ошибка подставляется в исходник модуля и исполняется копией."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fastbt import diag_v6_t1_chastota as ch

NOM = np.array([[0, 1], [2, 3], [4, 5]])
POR = np.zeros((3, 1), dtype=int)
NS = np.array([0, 0, 0, 0, 0, 100.0])          # пары: (0,0), (0,0), (0,100) → медиана max = 0 («частота»), среднее max/год = 8.3 (> порога)


def _mutant(star: str, novoe: str):
    """Копия модуля с одной заменой в исходнике; замена обязана найтись."""
    src = Path(ch.__file__).read_text(encoding="utf-8")
    assert star in src, "мутируемая строка не найдена — тест устарел"
    ns: dict = {"__name__": "mut_chastota", "__file__": ch.__file__}
    exec(compile(src.replace(star, novoe), ch.__file__, "exec"), ns)
    return ns


def test_osnova_mediana_chastota():
    o = ch.os_sdelki(NOM, POR, NS)[0]
    assert (o["dolya_par_bez_sdelok"], o["med_sdelok_god"]) == (pytest.approx(2 / 3), 0.0)
    assert ch.verdikt(o["med_sdelok_god"]) == "частота"


def test_mutaciya_porog_po_srednemu_vmesto_mediany_lovitsya():
    m = _mutant("np.median(np.maximum(a, b)[m])", "np.mean(np.maximum(a, b)[m])")
    o = m["os_sdelki"](NOM, POR, NS)[0]
    assert m["verdikt"](o["med_sdelok_god"]) == "сигнал"            # мутант расходится с известным ответом
    assert ch.verdikt(ch.os_sdelki(NOM, POR, NS)[0]["med_sdelok_god"]) == "частота"


def test_mutaciya_para_bez_sdelok_esli_hot_odna_tochka_lovitsya():
    m = _mutant("((a == 0) & (b == 0))", "((a == 0) | (b == 0))")
    assert m["os_sdelki"](NOM, POR, NS)[0]["dolya_par_bez_sdelok"] == 1.0
    assert ch.os_sdelki(NOM, POR, NS)[0]["dolya_par_bez_sdelok"] == pytest.approx(2 / 3)


def test_mutaciya_Y_ne_togo_okna_lovitsya():
    """Y другого окна/метрики (столбец 2 вместо 0 в сверке с n_akt CSV): правка проходит, мутант падает ValueError."""
    nom, por = np.array([[0, 1, 2]]), np.array([[0, 1]])
    Y = np.array([[0.1, 0, 0.5], [0.3, 0, 0.5], [np.nan, 0, 0.9]])  # окно ≤2016: шаг 0 активен, у шага 1 нет Y; столбец 2: оба шага конечны
    import pandas as pd
    df = pd.DataFrame({"okno": ["<=2016"] * 2, "os": ["a", "b"], "n_akt": [1, 0]})
    ch.soglasovat(nom, por, Y, df, ["a", "b"])
    m = _mutant("mv.effekty(Y[nom, 0]", "mv.effekty(Y[nom, 2]")
    with pytest.raises(ValueError, match="не совпал"):
        m["soglasovat"](nom, por, Y, df, ["a", "b"])


def test_ns_chitaetsya_iz_jsonl_po_tochkam(tmp_path):
    p = tmp_path / "y.jsonl"
    p.write_text(json.dumps({"a": 0, "b": 2, "y": [[0, 0, 0]] * 2, "ns": [3, 4]}) + "\n" + json.dumps({"a": 2, "b": 3, "y": [[0, 0, 0]], "ns": [9]}) + "\n", encoding="utf-8")
    Y, ns = ch._y_ns(p)
    assert list(ns) == [3, 4, 9] and Y.shape == (3, 3)
