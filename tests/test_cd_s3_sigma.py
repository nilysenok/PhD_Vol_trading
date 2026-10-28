"""Затвор по рангу σ для S3 (cd_s3_sigma): без заглядывания вперёд, знак, пустая лента, пропуск сделки целиком. Синтетика, без данных.
Мутация: ранг σ̂ по d+1 (ряд сдвинут на −1) — тест без заглядывания обязан падать."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import cd_s3_sigma as C  # noqa: E402


def _ryad(n: int = 400, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).uniform(1, 2, n)


def zatvor_po_d_plus_1(arr):
    """Мутант: σ̂ дня d+1 вместо дня d."""
    return C.zatvor(np.r_[np.asarray(arr, float)[1:], np.nan])


def bez_zaglyadyvaniya(zat, arr, k: int = 300) -> bool:
    """Маска дней < k не зависит от значений с индекса k и далее."""
    x = np.asarray(arr, float).copy()
    y = x.copy()
    y[k:] = y[k:] * 100.0 + 5.0
    return bool(np.array_equal(zat(x)[:k], zat(y)[:k]))


def test_zatvor_ne_zaglyadyvaet_vperyod():
    assert bez_zaglyadyvaniya(C.zatvor, _ryad())


def test_mutatsiya_rang_po_d_plus_1_lovitsya():
    assert not bez_zaglyadyvaniya(zatvor_po_d_plus_1, _ryad())


def test_znak_vysokaya_sigma_zakryta():
    x = np.r_[_ryad(300), 10.0, 0.01]                      # день 300 — максимум истории, день 301 — минимум
    a = C.zatvor(x)
    assert not a[300] and a[301] and a[:126].all() and (~a[126:300]).any()


def test_pustaya_lenta_i_korotkaya_istoriya():
    assert len(C.zatvor(np.array([]))) == 0
    assert C.zatvor(np.arange(100.0)).all()                 # < 126 значений: ранг не определён, затвор открыт
    assert np.isnan(C.rang(np.full(300, np.nan), 299))


def test_porog_p67():
    x = np.r_[np.arange(251.0), 167.0]                      # 167 значений строго ниже из 252 -> 0,66 < 0,67
    assert C.zatvor(x)[-1]
    x[-1] = 170.0                                           # 170/252 = 0,675 >= 0,67 -> закрыт
    assert not C.zatvor(x)[-1]


def test_sig_fn_propuskaet_vhod_ne_zakryvaet_poziciyu():
    pos = np.array([0, 1, 1, 1, 0, 1, 1, 0.0])
    allow = np.array([1, 0, 1, 1, 1, 1, 1, 1], bool)         # вход в день 1 запрещён, вход в день 5 разрешён
    out = C.sig_fn(lambda t, d: allow)("T", pos, np.arange(8))
    assert out.tolist() == [0, 0, 0, 0, 0, 1, 1, 0]
    inside = C.sig_fn(lambda t, d: np.array([1, 1, 0, 0, 0, 0, 0, 0], bool))("T", pos, np.arange(8))
    assert inside.tolist() == [0, 1, 1, 1, 0, 0, 0, 0]       # позиция, вошедшая до закрытия затвора, держится


def test_sig_fn_dlina():
    with pytest.raises(ValueError):
        C.sig_fn(lambda t, d: np.ones(3, bool))("T", np.zeros(5), np.arange(5))
