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


# --- Р-4 (Ж-137 п. 7б): тренд × m_sily ≠ 1 × rang_tip = tikernyj несовместимо ---
def _pt(**izm):
    return {"stop_nach": "sigma:2.0", "m_sily": "kvadr", "rang_tip": "tikernyj", **izm}


@pytest.mark.parametrize("sid", ["S3", "S4"])
def test_kanon_vybrasyvaet_trend_tikernyj_s_rangom(sid):
    from fastbt import canon_v5, canon_v6
    assert canon_v5.canon_v5(sid, _pt()) is None and canon_v6.canon_v6(sid, _pt()) is None
    assert canon_v5.canon_v5(sid, _pt(rang_tip="obshchiy")) is not None             # другие типы ранга живы
    assert canon_v5.canon_v5(sid, _pt(m_sily="1"))["rang_tip"] == "tikernyj"        # без ранга — свёртка, не выброс
    assert canon_v5.canon_v5("S1", _pt()) is not None                               # контртренд не затронут


def test_N_do_vybrosa_vybros_uchten():
    from fastbt import canon_v5
    pts, sch = canon_v5.tochki_v5(canon_v5.load_kub("S3"), 0, 512)
    assert sch["vybrosheno"] > 0 and len(pts) + sch["vybrosheno"] == 512
    assert not any(p["m_sily"] != "1" and p["rang_tip"] == "tikernyj" for _, p in pts)


def test_ocenshchik_otkazyvaet_trend_tikernyj():
    from fastbt import canon_v5
    from fastbt.eval_v5_osi import razbor
    pts, _ = canon_v5.tochki_v5(canon_v5.load_kub("S3"), 0, 64)
    with pytest.raises(ValueError, match="несовместимо"):
        razbor("S3", {**pts[0][1], "m_sily": "kvadr", "rang_tip": "tikernyj"}, eval_v5.SIGNAL["S3"])
    razbor("S3", {**pts[0][1], "m_sily": "kvadr", "rang_tip": "obshchiy"}, eval_v5.SIGNAL["S3"])


def test_mutaciya_sochetanie_ne_vybrasyvaetsya_lovitsya(monkeypatch):
    from fastbt import canon_v5
    monkeypatch.setattr(canon_v5, "TREND", set())            # правило выключено
    assert canon_v5.canon_v5("S3", _pt()) is not None        # тест выше (is None) на такой мутации падает
