"""Л5: мощность для n_new (fwd_power). Перенос `power.py` архитектора не меняет расчёт: n = 179, блок 10 воспроизводит строки прежнего POWER.csv (сохранён как POWER_pre.csv);
n_new = длина календаря; известный ответ на синтетике (сильный эффект → мощность ≈ 1, обратный знак → ≈ 0); мутации: n из прежнего диапазона вместо n_new, знак разности перепутан, окно 2022+.
Новых данных не читает (только архив t⁺ <= 2026-02-27). Сети нет."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from models_v2.s17 import fwd_power as PW

PRE = PW.OUT.with_name("POWER_pre.csv")
HAVE = PRE.exists() and PW.OUT.exists()


@pytest.fixture(scope="module")
def hist():
    L, y, shas = PW.load()
    return L, y, shas


def test_perenos_vosproizvodit_pryezhnij_power_csv_pri_n179(hist):
    L, y, _ = hist
    pre = pd.read_csv(PRE, comment="#")
    E, base = PW.E_of(L, (y >= 2018) & (y <= 2025))
    ph, p1 = PW.power(E, 179, PW.BLOCK_MAIN)
    ref = pre[(pre.kind == "istoricheskij") & (pre.window == "2018_2025") & (pre.n_new == 179) & (pre.block == 10)]
    assert len(ref) == 7 and np.allclose(ph, ref["power_holm7"].to_numpy(), atol=1e-9) and np.allclose(p1, ref["power_bez_popravki"].to_numpy(), atol=1e-9)
    assert np.allclose(base, ref["base_qlike"].to_numpy(), rtol=1e-5)
    other = PW.power(E, 170, PW.BLOCK_MAIN)[0]
    assert not np.array_equal(other, ph) or np.all(ph == 0)                                    # мутация «n прежнего диапазона вместо n_new» отличима (если мощность не нулевая)


def test_izvestnyj_otvet_silnyj_effekt_i_obratnyj_znak():
    rng = np.random.default_rng(1)
    E = rng.normal(0.6, 1.0, size=(800, 7))                                                    # A лучше B: e = L_B − L_A > 0
    assert PW.power(E, 180, 10)[0].min() > 0.99
    assert PW.power(-E, 180, 10)[0].max() < 0.01                                               # перепутанный знак — нулевая мощность
    z = rng.normal(0.0, 1.0, size=(800, 7))
    assert PW.power(z, 180, 10)[1].max() < 0.12                                                # без эффекта размер ≈ 5 % (без поправки)


def test_n_new_ravno_dline_kalendarya_i_n_bolshe_istorii_otkaz(hist):
    assert PW.n_new_of_calendar() == 180
    with pytest.raises(SystemExit):
        PW.power(np.zeros((100, 7)), 180, 10)


def test_okno_arhiv_do_2026_02_27(hist):
    L, y, _ = hist
    assert len(L["X1"]) == 2018 and y.max() == 2026                                            # t⁺ последней архивной пары — 2026-02-27, окно 2022+ не входит в расчёт эффекта
    w = (y >= 2018) & (y <= 2025)
    assert w.sum() == 1980 and not np.any(y[w] >= 2026)


@pytest.mark.skipif(not HAVE, reason="POWER.csv / POWER_pre.csv не записаны")
def test_power_csv_dlya_n180_i_pre_sohranen():
    h = PW.OUT.read_text(encoding="utf-8").splitlines()[0]
    df = pd.read_csv(PW.OUT, comment="#")
    assert set(df["n_new"]) == {180} and "n_new = 180" in h and set(df["hyp"]) == {f"{a}<{b}" for a, b in PW.PAIRS} and len(df) == 3 * 2 * 7 + 2 * 11 * 7
    assert hashlib.sha256(PRE.read_bytes()).hexdigest().startswith("ef6596206052237d")       # прежний файл (n = 170/179/185) не изменён
    assert set(pd.read_csv(PRE, comment="#")["n_new"]) == {170, 179, 185}
    assert df[(df.kind == "istoricheskij") & (df.block == 10)]["power_holm7"].between(0, 1).all()
