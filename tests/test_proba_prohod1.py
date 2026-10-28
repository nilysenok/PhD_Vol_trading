"""Проба прохода 1 (17:04, --tochek 20): 89 ошибок из 120 точек — две причины, обе под тестом.
(а) filtr ≠ net/cbr/div: ряды `sk._td[t]` (с 2014-06, 1909 дней) против локальной сетки окна (1261) — broadcast (filtr_e1_rynok);
(б) t_vyhod ≠ close при proverka = zakrytie: движок отвергает (exec_e._check_params) — схлопывание в canon_v2.
Сквозной гейт на реальных данных — tests/gate_proba_prohod1.py."""
from types import SimpleNamespace

import numpy as np
import pytest

from fastbt import filtr_e1_rynok as fr
from fastbt import ledger_v2 as lm

N, S0, T_TD = 300, 40, 400          # локальная сетка 300 дней с позиции 40 в полном ряду из 400


def _inp(seed=0):
    rng = np.random.default_rng(seed)
    dts = (np.datetime64("2014-06-09") + np.arange(T_TD)).astype("datetime64[ns]")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, T_TD)))
    td = SimpleNamespace(dates=dts, close=c, high=c * 1.01, low=c * 0.99, volume=rng.uniform(1e3, 2e3, T_TD))
    sig = {k: {"X": np.full(N, 0.02)} for k in ("rv5", "rv22", "rv66")}
    inp = SimpleNamespace(sk=SimpleNamespace(_td={"X": (td, {})}), sigma=sig, sobytiya={})
    return inp, td, dts[S0:S0 + N]


@pytest.mark.parametrize("imya", ["szhatie", "obyom", "rv_perevorot"])
def test_filtr_na_lokalnoj_setke_ne_padaet_i_ravent_srezu_polnogo(imya):
    inp, td, dates = _inp()
    a = fr.razresheno(imya, "X", inp, np.ones(N), dates)       # до правки: ValueError broadcast (400,) (300,)
    assert a.shape == (N,)
    inp2 = SimpleNamespace(sk=SimpleNamespace(_td={"X": (td, {})}), sobytiya={},
                           sigma={k: {"X": np.full(T_TD, 0.02)} for k in ("rv5", "rv22", "rv66")})
    full = fr.razresheno(imya, "X", inp2, np.ones(T_TD), td.dates)      # те же причинные фильтры на полном ряду
    assert np.array_equal(a, full[S0:S0 + N])


def test_post_shok_i_rel_sila_na_lokalnoj_setke(monkeypatch):
    inp, _, dates = _inp(1)
    monkeypatch.setattr(fr, "_imoex", lambda i, d: (np.zeros(len(d)), np.zeros(len(d))))
    for imya in ("post_shok", "rel_sila_imoex"):
        assert fr.razresheno(imya, "X", inp, np.ones(N), dates).shape == (N,)


def test_mutaciya_dyra_v_setke_lovitsya():
    """В сетке дыра (нет одного дня полного ряда): молча считать нельзя — отказ."""
    inp, _, dates = _inp(2)
    with pytest.raises(ValueError, match="не совпали"):
        fr.razresheno("szhatie", "X", inp, np.ones(N - 1), np.delete(dates, 100))


def test_canon_t_vyhod_ne_close_perevodit_zakrytie_v_bary():
    pt = dict(k_s=2.0, proverka="zakrytie", t_vyhod=1700, vhod=0.5)
    assert lm.canon_v2("S1", pt)["proverka"] == "bars5m"        # до правки: zakrytie → движок ValueError
    for tv in (1500, "open2"):
        assert lm.canon_v2("S1", {**pt, "t_vyhod": tv})["proverka"] == "bars5m"
    assert lm.canon_v2("S1", {**pt, "t_vyhod": "close"})["proverka"] == "zakrytie"      # close — оставляем
    assert lm.canon_v2("S1", {**pt, "proverka": "bars5m", "t_vyhod": 1700})["proverka"] == "bars5m"
    c = lm.canon_v2("S1", pt)
    assert lm.canon_v2("S1", c) == c                           # идемпотентно
    assert lm.klyuch("S1", pt, "train4", 0.0004, None) == lm.klyuch("S1", {**pt, "proverka": "bars5m"}, "train4", 0.0004, None)
