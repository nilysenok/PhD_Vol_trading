"""S-Z-EVAL (Ж-258), тестировщик: общие синтетические входы для `tests/test_s18_z_eval*.py` и `test_s18_z_doc*.py`.
Никакой записи в боевые выходы; бутстрап — на малом B (`small_boot`), константы спецификации остаются боевыми (B = 10 000 в `S.B` не трогается)."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import z_eval_cfg as C

NB = 200
Z, ZL = "S-RCKZ", "X1_R(S-RCKZ)"
LAYERS = ["X1_R(S-RCK)", "X1_R(S-RCKZ)", "X1_R(S-RZ)", "X1_R(S-RCKZs)"]
ALL = list(C.SHOWN) + ["S-RCKm", "S-RCKc"] + LAYERS


def calendar(n=C.N_D) -> pd.DatetimeIndex:
    """n дат строго между 2018-01-04 и 2026-11-10 (границы входят), по сетке рабочих дней — как D_M."""
    b = pd.bdate_range(C.FIRST, C.LAST)
    return b[np.unique(np.round(np.linspace(0, len(b) - 1, n)).astype(int))]


def patch_boot(mp):
    """Бутстрап на B = NB: и расчёт, и независимый пересчёт `z_eval_chk` идут через `S.stat_idx`; кэш `idx_for` чистится."""
    orig = S.stat_idx
    mp.setattr(S, "stat_idx", lambda n, B_=NB, mean_block=S.MEAN_BLOCK, seed=S.SEED: orig(n, NB, mean_block, seed))
    S.idx_for.cache_clear()


@pytest.fixture
def small_boot(monkeypatch):
    patch_boot(monkeypatch)
    yield
    monkeypatch.undo()
    S.idx_for.cache_clear()


def world(seed=0, n=C.N_D, gain=None):
    """rv, tp, F: lnF = ln rv − e·(1 − g): ряд тем лучше, чем больше g (сдвиг к истине). gain — g по имени ряда (по умолчанию слабый выигрыш S-RCKZ)."""
    rng = np.random.default_rng(seed)
    tp = calendar(n)
    ln_rv = rng.normal(0.0, 0.5, n)
    e0 = rng.normal(0.0, 0.4, n)
    g = {"S-RCKZ": 0.10, "S-RCKZs": 0.04, "S-RZ": 0.02}
    g.update(gain or {})
    F = {}
    for k in ALL:
        gk = g.get(k, 0.0) if not k.startswith("X1_R") else g.get(k[5:-1], 0.0)
        F[k] = np.exp(ln_rv - (e0 + 0.03 * rng.normal(0, 1, n)) * (1.0 - gk))
    return np.exp(ln_rv), tp, F


def ql(rv, f):
    r = np.asarray(rv, float) / np.asarray(f, float)
    return r - np.log(r) - 1.0


def losses(rv, F):
    return {k: ql(rv, v) for k, v in F.items()}


def test_calendar_has_dm_bounds_and_size():
    t = calendar()
    assert len(t) == C.N_D and str(t.min().date()) == C.FIRST and str(t.max().date()) == C.LAST and t.is_monotonic_increasing and t.is_unique
    assert (t > pd.Timestamp("2026-02-27")).any() and (t.year == 2018).any()
