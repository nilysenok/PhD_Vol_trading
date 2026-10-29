"""К-3: к1 по префиксу Соболя в `svod_server_v5` (синтетика, счёта и сети нет).
Мутация (mutations.md, тема «ключ/выборка»): отбор по sobol_i убран (к1 по всему пулу) — тест усечения падает."""
import numpy as np
import pytest

from fastbt import kalibr_v5 as K, kontroli_v5 as kv, svod_server_v5 as SV
from fastbt.kontroli_v5 import sintez_pul
from test_svod_server_v5 import DATY, OSI_S, chitat, zapisat

N = 40


def pul_(seed=3):
    return sintez_pul(seed, DATY)


def test_prefiks_ravny_pulu_kogda_vse_v_prefikse():
    pul = pul_()
    sh, idx = SV.k1_prefiks(pul, list(range(len(pul.tochki))), n=10**6)
    assert abs(sh - kv.k1(pul)) <= 1e-12 and idx == list(range(len(pul.tochki)))


def test_usechenie_srednee_tolko_tochek_prefiksa():
    pul = pul_()
    n_all = len(pul.tochki)
    assert n_all > 4
    k = n_all // 2
    sh, idx = SV.k1_prefiks(pul, list(range(n_all)), n=k)
    ex = pul.ex[:k][:, :, np.isin(kv._god(pul), kv.GODY)].mean((0, 1))
    assert idx == list(range(k)) and abs(sh - kv.sharpe(ex)) <= 1e-12
    assert abs(sh - kv.k1(pul)) > 1e-9                      # префикс ≠ весь пул (данные не вырождены)


def test_ne_sobol_tochki_ne_vhodyat():
    pul = pul_()
    s = [None if i % 2 else i for i in range(len(pul.tochki))]
    _, idx = SV.k1_prefiks(pul, s, n=10**6)
    assert idx == [i for i in s if i is not None]


def test_pustoj_prefiks_i_nesovpadenie_dliny_otkaz():
    pul = pul_()
    with pytest.raises(ValueError):
        SV.k1_prefiks(pul, [10**6] * len(pul.tochki), n=10)
    with pytest.raises(ValueError):
        SV.k1_prefiks(pul, [0], n=10)


def test_mutaciya_bez_otbora_po_sobol_i_lovitsya():
    pul = pul_()
    k = len(pul.tochki) // 2
    sh, _ = SV.k1_prefiks(pul, list(range(len(pul.tochki))), n=k)
    assert abs(sh - kv.k1(pul)) > 1e-9                      # мутант (k1 по пулу) дал бы равенство и упал здесь


def test_svod_pishet_k1_prefiks_s_seed_i_indeksami(tmp_path):
    zapisat(tmp_path / "r" / "sobol1_seed1", seed=1)
    out = tmp_path / "out"
    SV.svod(tmp_path / "r", out, OSI_S, 19, n_prefiks=N)
    shapka, rows = chitat(out / "k1_prefiks.csv")
    pul = K.pul_iz_ledger(tmp_path / "r" / "sobol1_seed1", "S1", OSI_S["S1"])
    assert len(rows) == 1 and abs(float(rows[0]["k1_sharpe"]) - kv.k1(kv.podpul(pul, np.arange(len(pul.tochki)) < N))) <= 1e-6
    assert int(rows[0]["tochek_prefiksa"]) == min(N, len(pul.tochki))
    assert any("seed=" in h and "индексы 0.." in h for h in shapka)
