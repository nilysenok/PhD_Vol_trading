"""Д-4 (Ж-145): пересчёт точки куба v6 на Маке. Гейты G4.1–G4.5. Тесты на реальном ledger — только если есть забор
train4_v6_31-10 и FASTBT_MAIN на теневой корень (иначе skip); в сеть не ходят."""
import os
from pathlib import Path

import numpy as np
import pytest

from fastbt import diag_v6_perescet as d, progon_morris_v6_polnyj as pm, progon_sobol as ps, progon_v6 as v6

L = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/server_2021/train4_v6_31-10/ledger")
HAVE = (L / "train4_v6_s1" / "S4" / "manifest.json").exists()
DANNYE = HAVE and os.environ.get("FASTBT_MAIN") not in (None, "")
zab = pytest.mark.skipif(not HAVE, reason="нет забора ledger train4_v6_31-10")


def _zam(sid):
    return v6.zamorozka(L / "train4_morris", [sid], lambda s: ps.kub_put(s, ps.Rezhim("v6", "train4", "seed", False, None)))[0][2]


def test_sovp_nan_v_raznyh_mestah_i_sdvig():
    a = np.array([1.0, np.nan, 2.0])
    assert d._sovp(a, a.copy()) == 0.0
    assert d._sovp(a, np.array([1.0, 0.0, 2.0])) == float("inf")      # nan не на том месте — не «совпало»
    assert d._sovp(a, a + 1e-6) == pytest.approx(1e-6)                # дискретное строго: Δ виден, допуск не подгоняется


@zab
def test_g41_klyuch_sovpadaet_na_shesti_tochkah():
    for sid in ("S1", "S2", "S3", "S4", "S5", "S6"):
        i = min(r["sobol_i"] for r in d._chitatel(str(L / "train4_v6_s1"), sid).rows())
        pt, rz, r0 = d.proverit_klyuch(sid, "seed", i, L / "train4_v6_s1", _zam(sid))
        assert r0["sobol_i"] == i


@zab
def test_g43_mutaciya_sobol_i_plus_odin_lovitsya(monkeypatch):
    sid = "S4"
    i = min(r["sobol_i"] for r in d._chitatel(str(L / "train4_v6_s1"), sid).rows())
    orig = pm.tochki_polnye
    monkeypatch.setattr(d.pm, "tochki_polnye", lambda s, z, n0, n1, seed, kub=None: orig(s, z, n0 + 1, n1 + 1, seed, kub))
    with pytest.raises(ValueError, match="klyuch не совпал"):
        d.proverit_klyuch(sid, "seed", i, L / "train4_v6_s1", _zam(sid))


@pytest.mark.skipif(not DANNYE, reason="нужны забор ledger и FASTBT_MAIN=<теневой корень>")
def test_g42_g44_g45_perescet_s4(tmp_path):
    sid = "S4"
    i = min(r["sobol_i"] for r in d._chitatel(str(L / "train4_v6_s1"), sid).rows())
    r = d.perescet(sid, "seed", i, L / "train4_v6_s1", _zam(sid), tmp_path)
    assert r["d_port"] <= 1e-9                                         # G4.2: суммы кварталов port Мак ↔ сервер
    assert len(r["zhurnal"]) > 0                                       # не вырожденная точка: сделки есть
    assert r["sum_zhurnal"] == pytest.approx(r["sum_x"], abs=1e-12)    # G4.4
    bad = r["sum_zhurnal"] + 1e-4 * 0.0004 / 17                        # мутация: лишняя половина комиссии → не сходится
    assert abs(bad - r["sum_x"]) > 1e-12
    assert np.isfinite(r["msr_dn"]) and np.isfinite(r["msr_ned"])      # G4.5: дневной и недельный рядом
    assert (tmp_path / f"{sid}_seed_{i:06d}.npz").exists()
