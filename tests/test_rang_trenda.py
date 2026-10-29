"""Р-4 / Ф-ранг (29.10): вес сделки тренда S3/S4 (стоп-заявка) — по рангу ДНЯ СТАРТА ТЕКУЩЕГО СИГНАЛА (eval_v5.rang_starta).
Мутации: (1) ранг d_sig как было (тождество); (2) ранг дня входа (заглядывание: d+1 вместо старта); (3) перенос через конец сигнала.
python3 -m pytest tests/test_rang_trenda.py -q"""
import numpy as np
import pytest

from fastbt import eval_v5
from fastbt.eval_v5 import rang_starta

NAN = float("nan")
#            день:  0    1    2    3    4    5    6    7    8    9
SIG = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0])
RG = np.array([NAN, 0.7, NAN, NAN, NAN, 0.9, 0.4, NAN, 0.2, NAN])   # ранг определён только в днях старта


def test_ves_po_rangu_dnya_starta_poka_signal_aktiven():
    r = rang_starta(SIG, RG)
    assert r[1:4].tolist() == [0.7, 0.7, 0.7]          # вечера s…s+k — ранг r
    assert r[6] == 0.4 and r[7] == 0.4                 # новый старт — ранг r'
    assert r[8] == 0.2 and r[9] == 0.2                 # смена знака — новый старт


def test_ves_ne_perenositsya_cherez_konec_signala():
    r = rang_starta(SIG, RG)
    assert np.isnan(r[0]) and np.isnan(r[4]) and np.isnan(r[5])   # день 5: ранг 0,9 есть, но сигнала нет → не размер


def test_start_bez_ranga_ne_zanimaet_chuzhoj():
    r = rang_starta(np.array([1.0, 1.0, 0.0, 1.0, 1.0]), np.array([0.5, NAN, NAN, NAN, NAN]))
    assert r[:2].tolist() == [0.5, 0.5] and np.isnan(r[3]) and np.isnan(r[4])


def test_kontrtrend_start_pobajtno_prezhnij():
    """Контртренд: d_sig = день старта → в день старта значение то же, что ранг без переноса."""
    start = (SIG != 0) & (SIG != np.r_[0.0, SIG[:-1]])
    assert np.array_equal(rang_starta(SIG, RG)[start], RG[start], equal_nan=True)


# --- мутации: каждая даёт ответ, отличный от правильного ---
def _mut_dsig(sig, rg):
    return np.asarray(rg, float)


def _mut_vhod(sig, rg):
    return np.r_[np.asarray(rg, float)[1:], NAN]


def _mut_cherez_konec(sig, rg):
    out, cur = np.full(len(sig), NAN), NAN
    for d in range(len(sig)):
        cur = rg[d] if np.isfinite(rg[d]) else cur
        out[d] = cur
    return out


@pytest.mark.parametrize("mut", [_mut_dsig, _mut_vhod, _mut_cherez_konec])
def test_mutatsii_lovyatsya(mut):
    assert not np.array_equal(mut(SIG, RG), rang_starta(SIG, RG), equal_nan=True)
    assert mut is not eval_v5.rang_starta
