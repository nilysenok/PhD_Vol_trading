"""Паритет рисков не должен останавливаться на шуме float у оптимума (Л3.3-прогон, находка на реальных центрах S1–S12, train4).
Матрица — LW-ковариация 12 рядов Sk за 252 дня до 2016-10-28 (дисперсии ~1e-6, есть отрицательные корреляции): до правки линейный
поиск с критерием «f не растёт» застревал на невязке 1,9e-8 и `paritet` бросал «не сошёлся». Мутация: возврат к старому критерию."""
from pathlib import Path

import numpy as np
import pytest

from fastbt import a_shtrih_v7 as A

S = np.load(Path(__file__).with_name("fixtures_a_shtrih_v7_lw_stall.npy"))


def razbros(w):
    rc = w * (S @ w)
    return np.max(np.abs(rc - rc.mean())) / rc.mean()


def test_paritet_shoditsya_na_matrice_s_zastryavaniem():
    w = A.paritet(S)
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()
    assert razbros(w) < 1e-8


def test_vesa_dnya_ne_padaet_na_etoj_matrice():
    rng = np.random.default_rng(7)
    L = np.linalg.cholesky(S)
    ex = (L @ rng.normal(size=(12, 600))).T                        # ряды с той же ковариацией
    w, fl = A.vesa_dnya(ex, 500)
    assert fl == "ok" and w.sum() == pytest.approx(1.0)


def test_mut_staryj_kriterij_padaet(monkeypatch):
    """Старый критерий (только убывание f) на этой матрице бросает ValueError — тест на мутацию: подмена `err` бесконечностью."""
    import fastbt.a_shtrih_v7 as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    staryj = src.replace("(f(y - t * step) > f(y) and err(y - t * step) >= err(y))", "f(y - t * step) > f(y)")
    assert staryj != src
    ns: dict = {"__name__": "staryj_a7"}
    exec(compile(staryj, "staryj_a7", "exec"), ns)               # noqa: S102 — код репозитория с подложенной ошибкой
    with pytest.raises(ValueError, match="не сошёлся"):
        ns["paritet"](S)
