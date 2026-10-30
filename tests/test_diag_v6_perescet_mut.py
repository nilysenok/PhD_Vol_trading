"""Тестировщик PR #880: мутации Д-4 (знак позиций, комиссия журнала, комиссия сделки). Реальный ledger — как в test_diag_v6_perescet."""
import copy
import os
from pathlib import Path

import numpy as np
import pytest

from fastbt import diag_v6_perescet as d, progon_sobol as ps, progon_v6 as v6

L = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/server_2021/train4_v6_31-10/ledger")
DANNYE = (L / "train4_v6_s1" / "S4" / "manifest.json").exists() and os.environ.get("FASTBT_MAIN") not in (None, "")
pytestmark = pytest.mark.skipif(not DANNYE, reason="нужны забор ledger и FASTBT_MAIN=<теневой корень>")
SID, I = "S4", 162850


def _zam():
    return v6.zamorozka(L / "train4_morris", [SID], lambda s: ps.kub_put(s, ps.Rezhim("v6", "train4", "seed", False, None)))[0][2]


@pytest.fixture(scope="module")
def baza(tmp_path_factory):
    return d.perescet(SID, "seed", I, L / "train4_v6_s1", _zam(), tmp_path_factory.mktemp("k"))


def test_shorty_otricatelny_i_dlinnye_polozhitelny(baza):
    tp = baza["tik_poz"]
    assert (tp < 0).any() and (tp > 0).any()                      # № 20: шорт отрицателен


def test_komissiya_zhurnala_polozhitelna(baza):
    assert all(t["komissiya"] >= 0 for t in baza["zhurnal"]) and sum(t["komissiya"] for t in baza["zhurnal"]) > 0


def test_mutaciya_pozicii_bez_znaka_lovitsya(baza, monkeypatch, tmp_path):
    orig = d._seti

    def bez_znaka(sr):
        df, inv, eng, rs, keep, col = orig(sr)
        rs2 = []
        for r in rs:
            r2 = copy.copy(r)
            r2.trades = [copy.copy(t) for t in r.trades]
            for t in r2.trades:
                t.side = abs(t.side)
            rs2.append(r2)
        return df, inv, eng, rs2, keep, col
    monkeypatch.setattr(d, "_seti", bez_znaka)
    mut = d.perescet(SID, "seed", I, L / "train4_v6_s1", _zam(), tmp_path)
    assert not (mut["tik_poz"] < 0).any()                         # мутация видна свойством «шорт отрицателен»
    assert (baza["tik_poz"] < 0).any()


def test_mutaciya_komissiya_zhurnala_nol_lovitsya(baza, monkeypatch, tmp_path):
    orig = d.zhurnal
    monkeypatch.setattr(d, "zhurnal", lambda sr, comm: orig(sr, 0.0))
    mut = d.perescet(SID, "seed", I, L / "train4_v6_s1", _zam(), tmp_path)
    assert abs(mut["sum_zhurnal"] - mut["sum_x"]) > 1e-9           # G4.4 обязан падать
    assert abs(baza["sum_zhurnal"] - baza["sum_x"]) < 1e-12


def test_msr_nedelnyj_iz_metriki_ned(baza):
    from fastbt import metriki_e as me
    ref = me.metriki_ned(baza["x"], baza["daty"])["msr_ned"]
    assert baza["msr_ned"] == pytest.approx(ref, nan_ok=True) and np.isfinite(ref)


@pytest.mark.xfail(strict=True, reason="ДЕФЕКТ #880: журнал считает только родителей и теряет/удваивает g при лестнице/пирамидинге: "
                                       "S1 i=2275 Σжурнал−Σx = −9,8e-3 (S2, S5, S6 тоже); G4.4 держится лишь без лестниц (S3, S4)")
def test_g44_na_tochke_s_lestnicej_S1(tmp_path):
    r = d.perescet("S1", "seed", 2275, L / "train4_v6_s1", v6.zamorozka(L / "train4_morris", ["S1"], lambda s: ps.kub_put(
        s, ps.Rezhim("v6", "train4", "seed", False, None)))[0][2], tmp_path)
    assert r["sum_zhurnal"] == pytest.approx(r["sum_x"], abs=1e-12)
