"""S-RCD-EVAL (Ж-259), тестировщик: общие синтетические входы для `tests/test_s18_rd_eval*.py` и `test_s18_rd_doc*.py`.
Никакой записи в боевые выходы; бутстрап — малый B (`patch_boot`), боевая константа `S.B` не трогается. Календарь собран так, что общее активное окно (d ≥ 2020-08-03) = 1549 дат, первая t⁺ окна 2020-08-04."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import rd_eval_cfg as C
from models_v2.s18 import rd_eval_dc as DC
from models_v2.s18 import rd_eval_gd as GD
from models_v2.s18 import rd_eval_tab as T

NB = 200
R, CDF, RCK = "S-RCD", "S-CDF", "S-RCK"
BASE_SERIES = ["K1", "K2", "K3", "B_mix", "X1", "W5-G"]
ALL = BASE_SERIES + list(C.SERIES) + list(C.LAYERS)


def calendar() -> pd.DatetimeIndex:
    a = pd.bdate_range("2018-01-04", "2020-07-31")
    b = pd.bdate_range("2020-08-04", C.LAST)
    ia, ib = np.unique(np.round(np.linspace(0, len(a) - 1, 649)).astype(int)), np.unique(np.round(np.linspace(0, len(b) - 1, 1549)).astype(int))
    ib[-1] = len(b) - 1
    return a[ia].append(b[ib])


def patch_boot(mp):
    orig = S.stat_idx
    mp.setattr(S, "stat_idx", lambda n, B_=NB, mean_block=S.MEAN_BLOCK, seed=S.SEED: orig(n, NB, mean_block, seed))
    S.idx_for.cache_clear()


@pytest.fixture
def small_boot(monkeypatch):
    patch_boot(monkeypatch)
    yield
    monkeypatch.undo()
    S.idx_for.cache_clear()


def world(gain=None, seed=0, noise=0.03):
    """lnF = ln rv − e·(1 − g): чем больше g (скаляр или массив по датам), тем лучше ряд. Слой X1_V(S-CDF) и т. п. берёт g своего ряда."""
    rng = np.random.default_rng(seed)
    tp = calendar()
    n = len(tp)
    assert n == C.N_D
    ln_rv, e0 = rng.normal(0.0, 0.5, n), rng.normal(0.0, 0.4, n)
    g = dict(gain or {})
    F = {}
    for k in ALL:
        own = C.LAYERS.get(k, k)
        F[k] = np.exp(ln_rv - (e0 + noise * rng.normal(0, 1, n)) * (1.0 - g.get(own, 0.0)))
    rv = np.exp(ln_rv)
    d = tp - pd.offsets.BDay(1)
    fr = lambda: pd.DataFrame({"F_fr": F["W5-G"] * np.exp(rng.normal(0, 0.01, n)), "C": F[R] * 1.01, "cfg8": F[R]})
    frames = {R: fr(), CDF: fr(), RCK: pd.DataFrame({"C": F[RCK] * 1.0})}
    return rv, tp, d, F, frames


def null_seeds(scale: float) -> tuple:
    """Нулевое распределение: 10 шумов, Δ_null = scale·linspace(−1,1) → |Δ| от ≈0,1·scale до scale."""
    v = scale * np.linspace(-1.0, 1.0, 10)
    s = pd.DataFrame({"kind": "seed", "seed": range(10), "qlike_rel_pct": v, "share_theta_nonzero": 0.5})
    summ = pd.DataFrame([dict(kind="summary", item=it, stat=n, value=float(f(x))) for it, x in (("qlike_rel_pct", v), ("share_theta_nonzero", np.full(10, 0.5))) for n, f in (("mean", np.mean), ("std", lambda a: np.std(a, ddof=1)), ("q05", lambda a: np.percentile(a, 5)), ("q95", lambda a: np.percentile(a, 95)))])
    return s, summ


def inputs(gain=None, scale=0.5, N=858, N_wide=3421, noise=0.03):
    rv, tp, d, F, frames = world(gain, noise=noise)
    return dict(rv=rv, tp=tp, d=d, F=F, frames=frames, N=N, N_wide=N_wide, null=null_seeds(scale))


def synth_t(inp: dict) -> dict:
    """Те же функции, что `rd_eval.compute`; params / sel — заглушки (проверяющий их не читает)."""
    rv, tp, F = inp["rv"], inp["tp"], inp["F"]
    ls = T.losses(rv, F)
    spa, act = T.spa_blocks(ls), DC.act_mask(inp)
    tb, st = T.table(rv, tp, F, ls), T.stats(rv, tp, ls, spa, inp["N"], inp["N_wide"], act)
    yrs, cnc = GD.years(tp, ls), GD.conc(tp, ls)
    g = GD.guard(yrs, cnc)
    dummy = pd.DataFrame({"a": [1.0]})
    return dict(table=tb, stats=st, decomp=DC.decomp(inp, ls, inp["null"]), guard=g, conc=cnc, years=yrs, spa=T.spa_table(spa), mcs=dummy, power=T.power_table(tp, ls), params=dummy, sel=dummy, dec=GD.dec_rows(tb, g, st), identity=GD.identity_gap(yrs))


def copy_t(t):
    return {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in t.items()}


def test_calendar_has_dm_bounds_active_window_and_size():
    t = calendar()
    d = t - pd.offsets.BDay(1)
    assert len(t) == C.N_D and str(t.min().date()) == C.FIRST and str(t.max().date()) == C.LAST and t.is_monotonic_increasing and t.is_unique
    m = np.asarray(d >= pd.Timestamp(C.ACTIVE_FROM))
    assert int(m.sum()) == 1549 and str(t[m].min().date()) == "2020-08-04" and (t.year == 2022).any() and (t.year == 2018).any()
