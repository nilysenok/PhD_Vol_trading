"""S-Z-IMPL-2 (Ж-258), тестировщик: контроль S-RCKZs — сдвиг z назад на 252 сессии по сетке сессий SBER БЕЗ переноса (spec разд. 3.3; Ж-257) и порядок «реестр → прогноз» в `run`.
Мутации: сдвиг с переносом (`np.roll`), сдвиг по строкам D_M, нет сдвига; контроль на реальном `zmaps.npz` против самого ряда z; `run` без реестра."""
import torch  # noqa: F401  (первым, класс № 152)

import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import z_feat as ZF
from models_v2.s18 import z_walk as ZW

SH = ZF.SHIFT


def synth():
    """сетка 600 сессий, D_M — каждая вторая сессия с 100-й (дыры); z строки = номер сессии + 0,1·j: известный ответ."""
    sess = pd.bdate_range("2018-01-01", periods=600)
    dm = sess[100::2]
    g = np.arange(600)[:, None] + 0.1 * np.arange(4)[None, :]
    return {y: {s: g.copy() for s in ZF.SEEDS} for y in ZF.YEARS}, dm, sess


def expected(dm, sess, y):
    pos = sess.get_indexer(dm) - SH
    out = np.full((len(dm), 4), np.nan)
    ok = pos >= 0
    out[ok] = pos[ok, None] + 0.1 * np.arange(4)[None, :]
    out[np.asarray(dm >= pd.Timestamp(y + 1, 1, 1))] = np.nan
    return out


def shift_ok() -> bool:
    zg, dm, sess = synth()
    zs = ZF.zshift(zg, dm, sess)
    return all(np.array_equal(zs[y][s], expected(dm, sess, y), equal_nan=True) for y in ZF.YEARS for s in ZF.SEEDS)


def test_shift_252_sessions_on_session_grid_no_transfer():
    assert shift_ok()
    zg, dm, sess = synth()
    zs = ZF.zshift(zg, dm, sess)[2018][0]
    assert np.isnan(zs[sess.get_indexer(dm) < SH]).all() and np.isfinite(zs[sess.get_indexer(dm) >= SH]).any()      # нет сессий ≥ 252 раньше ⇒ NaN, не перенос


def test_mut_shift_with_wraparound_np_roll_is_caught(monkeypatch):
    monkeypatch.setattr(ZF, "shift_rows", lambda zg, pos, shift=SH: np.roll(ZF.take(zg, pos), shift, axis=0))
    assert not shift_ok()
    monkeypatch.setattr(ZF, "shift_rows", lambda zg, pos, shift=SH: zg[(np.asarray(pos) - shift) % len(zg)])
    assert not shift_ok()


def test_mut_shift_by_d_m_rows_instead_of_sessions_is_caught(monkeypatch):
    def rows(zg, pos, shift=SH):
        z = ZF.take(zg, pos)
        out = np.full_like(z, np.nan)
        out[shift:] = z[:-shift]
        return out
    monkeypatch.setattr(ZF, "shift_rows", rows)
    assert not shift_ok()


def test_mut_no_shift_is_caught(monkeypatch):
    monkeypatch.setattr(ZF, "shift_rows", lambda zg, pos, shift=SH: ZF.take(zg, pos))
    assert not shift_ok()


def real_control_ok(zm, zs, dm, sess) -> bool:
    """zs на дате D_M = z сессии pos − 252 сетки SBER: где эта сессия есть в D_M — равна zm той даты побитно; NaN ровно там, где нет сессии или z года."""
    pos, ok = sess.get_indexer(dm), True
    for y in zm:
        z0, z1 = zm[y][0], zs[y][0]
        for i in np.flatnonzero((pos >= SH) & np.asarray(dm < pd.Timestamp(y + 1, 1, 1)))[::7]:     # строки с датой ≥ 1.1.(Y+1) у артефакта года Y обнулены по построению
            j = dm.get_indexer([sess[pos[i] - SH]])[0]
            ok &= j < 0 or np.array_equal(z1[i], z0[j], equal_nan=True)
    return bool(ok)


def test_real_zmaps_control_is_session_shift_and_mutations_are_caught():
    zm, zs, dm = ZF.unpack(ZF.ZMAPS)
    sess = ZF.sessions()
    assert real_control_ok(zm, zs, dm, sess)
    assert any(np.isfinite(zs[y][0]).all(1).any() for y in zs)
    roll = {y: {s: np.roll(zm[y][s], SH, axis=0) for s in zm[y]} for y in zm}              # перенос
    rows = {y: {s: np.vstack([np.full((SH, 4), np.nan), zm[y][s][:-SH]]) for s in zm[y]} for y in zm}      # сдвиг по строкам D_M
    assert not real_control_ok(zm, roll, dm, sess) and not real_control_ok(zm, rows, dm, sess) and not real_control_ok(zm, zm, dm, sess)


def refuses(match, monkeypatch, trials) -> bool:
    orig = ZW.ZR.require
    monkeypatch.setattr(ZW.ZR, "require", lambda cfg: orig(cfg, trials))
    monkeypatch.setattr(ZW.RC, "make_ctx", lambda arch: (_ for _ in ()).throw(RuntimeError("дошли до счёта")))
    try:
        ZW.run("rckz", write=False)
    except SystemExit as e:
        return match in str(e)
    except RuntimeError:
        return False
    return False


def test_run_refuses_without_registry_row_and_mutation_without_require_is_caught(monkeypatch, tmp_path):
    t = tmp_path / "ALL_TRIALS.csv"
    t.write_text("id,name,formula,params,status,registered_at\nS-RZ,a,b,c,d,12.11.2026 01:00:00\n", encoding="utf-8")
    assert refuses("нет в реестре", monkeypatch, t)                      # S-RCKZ в реестре нет
    with monkeypatch.context() as mp:
        mp.setattr(ZW.ZR, "require", lambda cfg, trials=None: None)
        assert not refuses("нет в реестре", mp, t)


def test_run_refuses_when_overwriting_output_that_differs(monkeypatch, tmp_path):
    """существующий выход с иным содержимым — отказ записи; тот же пересчёт — перезапись допустима; пишется только в `out` (боевой Z/ не трогается)."""
    box = {"df": pd.DataFrame(dict(a=[1.0, 2.0]))}
    monkeypatch.setattr(ZW.RC, "make_ctx", lambda arch: None)
    monkeypatch.setattr(ZW, "load_z", lambda: (None, None))
    monkeypatch.setattr(ZW, "walk", lambda *a, **k: [])
    monkeypatch.setattr(ZW, "assemble", lambda ctx, parts: box["df"])
    monkeypatch.setattr(ZW.ZR, "check", lambda *a, **k: [])
    monkeypatch.setattr(ZW.ZR, "require", lambda cfg, trials=None: None)
    ZW.run("rz", True, tmp_path)
    assert (tmp_path / "rz.parquet").exists() and not (tmp_path / "rz.parquet.tmp").exists()
    ZW.run("rz", True, tmp_path)
    box["df"] = pd.DataFrame(dict(a=[1.0, 2.0000001]))
    with pytest.raises(SystemExit, match="перезапись отказана"):
        ZW.run("rz", True, tmp_path)
    assert pd.read_parquet(tmp_path / "rz.parquet")["a"].tolist() == [1.0, 2.0]
