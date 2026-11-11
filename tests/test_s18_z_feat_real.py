"""S-Z-IMPL-0: склад z на реальных артефактах Stage A (читаются только локальные файлы) против независимого чтения; мутации класса № 146 (энкодер года d + 1, z пары от энкодера её года)."""
import torch  # noqa: F401  (первым: конфликт OpenMP, класс № 152)

import numpy as np
import pandas as pd
import pytest

from models_v2.s14 import v7_1b_sta as STA
from models_v2.s17.fwd_src import ARCHIVE_END
from models_v2.s18 import z_feat as F
from models_v2.s18 import z_feat_chk as CHK

Y1 = F.YEARS[:-1]


@pytest.fixture(scope="module")
def world():
    sess, dm = F.sessions(), F.dm_dates()
    empty = {s: pd.DataFrame(np.empty((0, 4)), index=pd.DatetimeIndex([]), columns=list("abcd")) for s in F.SEEDS}
    return sess, dm, empty


def _truth(sess, y, s):
    """Независимо: файл энкодера года y, z его дат ≤ 31.12.y на сетке сессий."""
    a = STA.load(STA.directory() / STA.name(F.KIND, y, s))
    days = pd.DatetimeIndex(a["days"].astype("datetime64[ns]"))
    pos, use = sess.get_indexer(days), np.asarray(days < pd.Timestamp(y + 1, 1, 1))
    z = np.full((len(sess), 4), np.nan)
    z[pos[use & (pos >= 0)]] = np.asarray(a["z"], float)[use & (pos >= 0)]
    return z


def _grids_ok(zg, sess) -> bool:
    return all(np.array_equal(zg[y][s], _truth(sess, y, s), equal_nan=True) for y in Y1 for s in F.SEEDS)


def test_hashes45_and_load_checked():
    h = F.hashes45()
    assert len(h) == 45 and all(STA.content_hash(F.load_checked(STA.directory() / k)) == v for k, v in h.items())


def test_load_checked_refuses_wrong_hash(monkeypatch):
    k = STA.name(F.KIND, 2020, 0)
    bad = {**F.hashes45(), k: "0" * 64}
    monkeypatch.setattr(F, "hashes45", lambda: bad)
    with pytest.raises(ValueError, match="content_hash"):
        F.load_checked(STA.directory() / k)


def test_zgrids_equal_independent_reading_and_dm_maps(world):
    sess, dm, empty = world
    zg = F.zgrids(sess, empty)
    assert _grids_ok(zg, sess)
    zm, zs, pos = F.zmaps(zg, dm, sess), F.zshift(zg, dm, sess), sess.get_indexer(dm)
    for y in Y1:
        for s in F.SEEDS:
            t = _truth(sess, y, s)
            assert np.array_equal(zm[y][s], t[pos], equal_nan=True)
            late = np.asarray(dm >= pd.Timestamp(y + 1, 1, 1))
            want = np.array([t[p - 252] if p >= 252 else [np.nan] * 4 for p in pos])
            want[late] = np.nan
            assert np.array_equal(zs[y][s], want, equal_nan=True)
            assert np.isnan(zm[y][s][late]).all() and np.isnan(zs[y][s][late]).all()
            assert np.isfinite(zm[y][s][~late]).all() and np.isfinite(zs[y][s][~late & (pos >= 252)]).all()    # дыр нет (Д4)


def test_shift_is_by_sessions_not_by_dm_rows(world):
    """Строки D_M ≠ сессии: сдвиг на 252 строки D_M даёт другой z почти везде (расходятся даты)."""
    sess, dm, empty = world
    zg = F.zgrids(sess, empty)
    pos = sess.get_indexer(dm)
    n_same = int((sess[pos[252:] - 252] == dm[:-252]).sum())
    assert n_same < len(dm) - 252
    zs, zm = F.zshift(zg, dm, sess)[2025][0], F.zmaps(zg, dm, sess)[2025][0]
    assert not np.array_equal(zs[252:], zm[:-252], equal_nan=True)


def test_mutation_encoder_of_year_plus_one_caught(world, monkeypatch):
    sess, _, empty = world
    monkeypatch.setattr(F.CFG, "enc_year", lambda y: y + 1)
    assert not _grids_ok(F.zgrids(sess, empty), sess)


def test_mutation_pair_z_from_encoder_of_its_own_year_caught(world, monkeypatch):
    sess, dm, empty = world
    orig = F.CFG.z_of_year

    def own_year(kind, seed, index, year, load=STA.load, window=STA.WINDOW):
        out = np.full((len(index), 4), np.nan)
        for yy in range(2018, year + 1):
            m = np.asarray(index.year == yy)
            out[m] = orig(kind, seed, index, yy, load, window)[m]
        return out
    monkeypatch.setattr(F.CFG, "z_of_year", own_year)
    zg = F.zgrids(sess, empty)
    assert not _grids_ok(zg, sess)
    assert not np.array_equal(F.zmaps(zg, dm, sess)[2025][0], _truth(sess, 2025, 0)[sess.get_indexer(dm)], equal_nan=True)


def test_every_year_uses_own_encoder_file(world):
    sess, _, empty = world
    z = F.zgrids(sess, empty)
    assert not np.array_equal(z[2020][0], z[2021][0], equal_nan=True) and not np.array_equal(z[2025][0], z[2025][1], equal_nan=True)
    assert np.isnan(z[2019][0][np.asarray(sess >= pd.Timestamp(2020, 1, 1))]).all()


def test_pit_check_ok_and_mutation_future_in_f_caught(monkeypatch):
    """Проверка PIT 2026 F (cut = d против cut = 2026-11-10): честный путь — пусто; склад с «будущим» в строке F-даты — ловится."""
    assert CHK.check_pit() == []
    real = F.frames_2026

    def leaky(src=None):
        fr = real(src)
        for s in fr:
            fr[s].loc[fr[s].index > ARCHIVE_END] += 1e-9
        return fr
    monkeypatch.setattr(F, "frames_2026", leaky)
    assert CHK.check_pit() != []
