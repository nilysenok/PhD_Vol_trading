"""A5: правило эпсилон — только прошлое; на реальной сетке RV > 0 и граница не срабатывает."""
import numpy as np

from models_v2.s5 import common, frame, lossck, pit


def test_eps_ne_zavisit_ot_budushchego():
    rng = np.random.default_rng(0)
    rv = np.exp(rng.normal(size=300))
    e0 = pit.eps_floor(rv)
    rv2 = rv.copy()
    rv2[200:] = 1e-9   # порча после i = 199
    e1 = pit.eps_floor(rv2)
    assert np.array_equal(e0[:200], e1[:200])


def test_mut_eps_po_vsem_dannym_lovitsya():
    rng = np.random.default_rng(0)
    rv = np.exp(rng.normal(size=300))
    rv2 = rv.copy()
    rv2[200:] = 1e-9
    glob = lambda x: pit.EPS_K * np.full(len(x), x.min())  # noqa: E731
    assert not np.array_equal(glob(rv)[:200], glob(rv2)[:200])


def test_real_set_rv_polozhitelna_i_granitsa_ne_srabatyvaet():
    t = lossck.eps_check(common.controls(), frame.daily_p1()).iloc[0]
    assert t.rv_all_positive and t.min_rv_target > 0 and t.min_s2_b2 > 0 and t.min_s2_harj > 0
    assert t.n_rv_below_eps == 0 and t.n_s2_b2_below_eps == 0 and t.n_s2_harj_below_eps == 0
