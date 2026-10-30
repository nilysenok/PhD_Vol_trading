"""Починка голых идей S7–S12 (после #878): корпсобытия в драйвере, одно плацебо на ряд, к2 без прогрева. Синтетика, без сети."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis")]
import golaya_v7 as gv  # noqa: E402
from fastbt import kross_v7 as kv, progon_kontrol as pk  # noqa: E402
from test_golaya_v7 import TK, bary  # noqa: E402
from test_golaya_v7_anomalii import FAKTOR, _s_konsolidaciej  # noqa: E402


def _smen(S):
    return int((np.diff(S.astype(int), axis=0) != 0).sum())


@pytest.fixture(scope="module")
def D():
    raw, im = bary()
    return gv.iz_barov(raw, im, "train4", TK)


def test_konsolidaciya_100_k_1_ne_daet_pribyli(tmp_path, monkeypatch):
    """Лонг через день консолидации: без поправки ≈ +99 на единицу цены входа, с поправкой — обычный ход; цены без скачка ×100."""
    raw, im, t, data = _s_konsolidaciej(tmp_path, monkeypatch)
    D = gv.iz_barov(raw, im, "train4", TK)
    i, j = int(np.flatnonzero(D["daty"] == data)[0]), TK.index(t)
    for k in ("o", "cr", "c"):
        x = D[k][t].to_numpy()
        assert abs(np.log(x[i] / x[i - 1])) < 0.2, k
    sig = np.zeros((len(D["daty"]), len(TK)), np.int8)
    sig[i - 5:i + 2, j] = 1
    (_, _, e, x, s, br, *_), = gv.schet(D, sig, vhod="d1", tau=None, sost=True)["tr"]
    assert D["daty"][e] < data <= D["daty"][x] and s == 1
    assert abs(br) < 0.5, f"валовая доходность сделки {br:.1f} на единицу цены входа (×{FAKTOR:.0f} без поправки)"


def test_placebo_s8_s9_ne_razduvaet_oborot(D):
    for sid in ("S8", "S9"):
        c = _smen(gv.signaly(D, sid))
        p = [_smen(gv.signaly(D, sid, ("placebo", gv.SEED + s))) for s in range(3)]
        assert c > 0 and max(p) <= 2 * c, (sid, c, p)


def test_placebo_s7_oborot_ne_vyshe_centra_kratno(D):
    c = _smen(gv.signaly(D, "S7"))
    p = [_smen(gv.signaly(D, "S7", ("placebo", gv.SEED + s))) for s in range(3)]
    assert max(p) <= 2 * c, (c, p)


def test_placebo_zavisit_ot_seed(D):
    a, b = (gv.signaly(D, "S8", ("placebo", gv.SEED + s)) for s in (0, 1))
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("sid", ["S7", "S8", "S9"])
def test_k2_secheniya_ne_zahodit_v_progrev(D, sid):
    """Доля дней окна со скором после к2 не меньше, чем у центра (сдвиг только внутри окна); лаг 0 — центр."""
    c = gv.CENTR[sid]
    sk, w = kv.rebal(kv.panel(D["c"], D["im"]["CLOSE"], D["metki"], sid), D["daty"], c["rezhim"]), D["win"]
    assert np.array_equal(gv.signaly(D, sid, ("k2", 0)), gv.signaly(D, sid))
    for lag in pk.k2_lags(gv.N_POVT)[:3]:
        sh = gv.skor_k2(D, sk, lag)
        assert np.isfinite(sh[w]).any(axis=1).mean() >= np.isfinite(sk[w]).any(axis=1).mean() - 1e-9, lag
        assert np.array_equal(sh[~w], sk[~w], equal_nan=True)
        assert not np.array_equal(sh[w], sk[w], equal_nan=True)
