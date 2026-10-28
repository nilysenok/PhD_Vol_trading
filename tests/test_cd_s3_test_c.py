"""cd_s3_test_c: затвор C1 W504 p70 (TEST_C) — без будущего, сторона S3, сверка с s9, мутация «ранг по d+1» красная."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import cd_s3_test_c as T  # noqa: E402


def _ryad(n=900, seed=1):
    return np.abs(np.random.default_rng(seed).normal(1.0, 0.3, n)) + 0.01


def _allow_d_plus_1(arr, p=0.70):
    """Мутация: ранг значения d+1 (прогноз «завтра») против окна, оканчивающегося днём d."""
    arr = np.asarray(arr, float)
    out = np.ones(len(arr), bool)
    for i in range(len(arr) - 1):
        w = arr[max(i - T.W + 1, 0):i + 1]
        w = w[np.isfinite(w)]
        if len(w) >= T.MINP:
            out[i] = (w < arr[i + 1]).sum() / len(w) > 1 - p
    return out


def _zavisit_ot_budushchego(fn, arr, i=700) -> bool:
    a, b = arr.copy(), arr.copy()
    b[i + 1:] = b[i + 1:][::-1] * 3.0
    return bool(fn(a)[i] != fn(b)[i]) or bool((fn(a)[:i + 1] != fn(b)[:i + 1]).any())


def test_zatvor_ne_chitaet_budushchee():
    assert not _zavisit_ot_budushchego(T.allow_c1, _ryad())


def test_mutatsiya_rang_po_d_plus_1_krasnaya():
    arr = _ryad()
    assert _zavisit_ot_budushchego(_allow_d_plus_1, arr)          # мутант читает d+1 — тест на будущее его ловит
    assert (T.allow_c1(arr) != _allow_d_plus_1(arr)).any()        # и маска отличается от правильной


def test_okno_ne_vklyuchaet_tekushchiy_bar_i_storona_s3():
    a = np.ones(600)
    a[400] = 5.0        # всплеск: все 504 прошлых значения строго ниже -> ранг 1 -> торгуем
    a[401] = 0.1        # падение: ранг 0 (низкая σ̂) -> затвор закрыт
    m = T.allow_c1(a)
    assert m[400] and not m[401]
    assert T.rang(a, 400) == 1.0 and T.rang(a, 401) == 0.0


def test_nan_i_korotkoe_okno_otkryty():
    a = _ryad(300)
    a[:100] = np.nan
    m = T.allow_c1(a)
    assert np.isnan([T.rang(a, i) for i in (50, 200, 299)]).all()    # < 252 значений
    assert m.all()


def test_granitsa_p70_porog_030():
    a = np.arange(1.0, 701.0)                 # растёт: ранг каждого нового значения = 1
    a[600] = a[600 - 504:600][int(0.30 * 504)]      # значение на уровне ~30-го перцентиля окна
    r = T.rang(a, 600)
    assert abs(r - 0.30) < 0.01 and T.allow_c1(a)[600] == (r > 0.30)


def test_sverka_s_s9():
    pytest.importorskip("numba")
    arr = _ryad(1100, 7)
    arr[:200] = np.nan
    assert (T.allow_c1(arr) == T.mask_s9(arr)).all()


def test_sig_fn_poz_umnozhaet_i_proveryaet_dlinu():
    fn = T.sig_fn_poz(lambda t, d: np.array([True, False, True]))
    assert fn("X", np.array([1.0, 1.0, -1.0]), [0, 1, 2]).tolist() == [1.0, 0.0, -1.0]
    with pytest.raises(ValueError):
        fn("X", np.ones(4), [0, 1, 2, 3])
