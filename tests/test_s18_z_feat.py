"""S-Z-IMPL-0 (Ж-258, разд. 8): склад z — синтетика с известным ответом и мутации (сдвиг, пропуск 2026 F, файл). Без сети, без записи в `data/`."""
import torch  # noqa: F401  (первым: конфликт OpenMP, класс № 152)

import io

import numpy as np
import pandas as pd
import pytest

from models_v2.s17 import fwd_store as FS
from models_v2.s17 import fwd_z as FZ
from models_v2.s17.fwd_src import LAST_SESSION
from models_v2.s18 import z_feat as F

SESS = pd.bdate_range("2024-01-01", periods=700)
DM = SESS[100:].delete(np.arange(0, 600, 7))      # D_M: часть сессий пропущена, строки ≠ сессии
YEARS = F.YEARS


def _stub(monkeypatch):
    """z энкодера Y на сетке = 10000·(Y − 2018) + номер сессии + 0,1·seed: значение однозначно говорит, откуда оно взято."""
    def z_of_year(kind, s, index, y, load=None, window=1):
        return np.tile((np.arange(len(index)) + 0.1 * s + 10000.0 * (y - 2018))[:, None], (1, 4))
    monkeypatch.setattr(F.CFG, "z_of_year", z_of_year)


def _frames(drop=()):
    idx = SESS[SESS >= "2026-01-01"].delete([i for i in drop])
    return {s: pd.DataFrame(np.tile((np.arange(len(idx)) + 0.5 + s)[:, None], (1, 4)), index=idx, columns=list("abcd")) for s in F.SEEDS}


def _expected(shift):
    pos, out = SESS.get_indexer(DM), {}
    for y in YEARS:
        out[y] = {}
        for s in F.SEEDS:
            e = np.full((len(DM), 4), np.nan)
            for i, p in enumerate(pos):
                if p - shift >= 0 and DM[i] < pd.Timestamp(y + 1, 1, 1) and y != 2026:
                    e[i] = p - shift + 0.1 * s + 10000.0 * (y - 2018)
            out[y][s] = e
    return out


def _same(a, b):
    return all(np.array_equal(a[y][s], b[y][s], equal_nan=True) for y in YEARS if y != 2026 for s in F.SEEDS)


def _grids(monkeypatch, frames=None):
    _stub(monkeypatch)
    return F.zgrids(SESS, _frames() if frames is None else frames)


def test_zmaps_zshift_known_answer(monkeypatch):
    zg = _grids(monkeypatch)
    assert _same(F.zmaps(zg, DM, SESS), _expected(0))
    assert _same(F.zshift(zg, DM, SESS), _expected(252))
    assert np.isnan(F.zshift(zg, DM, SESS)[2025][0][np.flatnonzero(SESS.get_indexer(DM) < 252)]).all()     # позиции < 252 — NaN, без переноса


def test_mutation_np_roll_caught(monkeypatch):
    """Ж-257: сдвиг через np.roll подмешивает конец сетки в начало."""
    zg = _grids(monkeypatch)
    monkeypatch.setattr(F, "shift_rows", lambda g, pos, shift=252: np.roll(F.take(g, pos), shift, axis=0))
    assert not _same(F.zshift(zg, DM, SESS), _expected(252))


def test_mutation_shift_by_dm_rows_caught(monkeypatch):
    """Сдвиг на 252 строки D_M вместо 252 сессий: расходится там, где в D_M есть пропуски."""
    zg = _grids(monkeypatch)
    zm = F.zmaps(zg, DM, SESS)
    monkeypatch.setattr(F, "zshift", lambda g, dm, sess: {y: {s: np.vstack([np.full((252, 4), np.nan), zm[y][s][:-252]]) for s in F.SEEDS} for y in YEARS})
    assert not _same(F.zshift(zg, DM, SESS), _expected(252))


def test_shift_rows_and_take_known_answer():
    g = np.repeat(np.arange(300.0)[:, None], 4, axis=1)
    pos = np.array([0, 100, 251, 252, 253, 299])
    want = np.array([np.nan, np.nan, np.nan, 0.0, 1.0, 47.0])[:, None] * np.ones(4)
    assert np.array_equal(F.shift_rows(g, pos), want, equal_nan=True)
    assert np.array_equal(F.take(g, [-1, 0, 5]), np.array([np.nan, 0.0, 5.0])[:, None] * np.ones(4), equal_nan=True)
    assert np.isnan(F.take(g, [-300])).all()


def _gap_ok(zg) -> bool:
    miss = SESS.get_indexer(SESS[SESS >= "2026-01-01"][[10, 11]])
    return bool(np.isnan(zg[2026][0][miss]).all() and np.isfinite(zg[2026][0][miss[0] - 1]).all() and np.isnan(zg[2025][0][np.asarray(SESS >= "2026-01-01")]).all())


def test_gap_2026_is_nan_not_filled(monkeypatch):
    """Д4: пропуск z 2026 F остаётся NaN, после 31.12.Y — NaN."""
    assert _gap_ok(_grids(monkeypatch, _frames(drop=[10, 11])))


def test_mutation_ffill_2026_caught(monkeypatch):
    monkeypatch.setattr(F, "_grid_2026", lambda fr, sess: pd.DataFrame(fr.to_numpy(float), index=fr.index).reindex(sess).ffill().to_numpy(float))
    assert not _gap_ok(_grids(monkeypatch, _frames(drop=[10, 11])))


def test_frames_2026_cut_is_last_session_and_per_seed(monkeypatch):
    calls = []
    monkeypatch.setattr(FS, "build_store", lambda cut, src: calls.append(("store", cut)) or "st")
    monkeypatch.setattr(FS, "build_xw5", lambda st: "xw")
    monkeypatch.setattr(FZ, "build_z", lambda cut, src, s, st, xw: calls.append(("z", cut, s)) or pd.DataFrame({"d": SESS[:3], "z0": [1.0, 2, 3], "z1": 0.0, "z2": 0.0, "z3": 0.0}))
    out = F.frames_2026(src=object())
    assert LAST_SESSION == pd.Timestamp("2026-11-10") and sorted(out) == list(F.SEEDS) and out[0].index.name == "d"
    assert {c[1] for c in calls} == {LAST_SESSION} and [c[2] for c in calls if c[0] == "z"] == list(F.SEEDS)


def test_pack_unpack_roundtrip_bitwise_with_nan():
    rng = np.random.default_rng(0)
    dm = DM[:5]
    mk = lambda: {y: {s: np.where(rng.random((5, 4)) < 0.3, np.nan, rng.normal(size=(5, 4))) for s in F.SEEDS} for y in YEARS}
    zm, zs = mk(), mk()
    a = F.pack(zm, zs, dm)
    assert int(a["v"]) == 1 and len(a["hashes"]) == 45 and a["z"].shape == (9, 5, 5, 4)
    buf = io.BytesIO()
    np.savez(buf, **a)
    buf.seek(0)
    z2, s2, d2 = F.unpack(buf)
    assert _same_all(zm, z2) and _same_all(zs, s2) and (d2 == dm).all()
    assert not _same_all(zm, s2)                                         # zm и zs не перепутаны


def _same_all(a, b):
    return all(np.array_equal(a[y][s], b[y][s], equal_nan=True) for y in YEARS for s in F.SEEDS)
