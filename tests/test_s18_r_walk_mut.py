"""S-R-IMPL-2: обязательные мутации прохода R (мутации.md № 30, 144, 164; спецификация разд. 6) — каждая обязана ловиться детектором r_walk_syn / боевой проверкой."""
import numpy as np

from models_v2.s16 import x1_w5_gap as W5
from models_v2.s16 import x1_ya_walk as YA
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W
from models_v2.s18 import r_walk_chk as K
from models_v2.s18 import r_walk_syn as S

ORIG_DECIDE, ORIG_BLOCKS, ORIG_ELL = W.decide, W.blocks, RL.ell


def test_honest_detectors_are_clean():
    assert S.rule_problems() == [] and S.block_problems() == [] and S.pool_problems() == []


def test_mut_closed_plus_one_ck_pool_is_caught(monkeypatch):
    monkeypatch.setattr(YA, "closed_date", lambda tp, d: d)
    assert S.pool_problems() != []


def test_mut_closed_plus_one_a_pool_is_caught(monkeypatch):
    monkeypatch.setattr(RL, "closed_rows", lambda tp, d0: np.flatnonzero(np.asarray(tp, "datetime64[ns]") <= np.datetime64(d0) + np.timedelta64(1, "D") * 3))
    assert S.pool_problems() != []


def test_mut_theta_zero_absent_from_grid_is_caught(monkeypatch):
    def mut(G, rule):
        w = {lam: sum(g < 0 for g in G[lam]) for lam in W.LAMS}
        return max(W.LAMS, key=lambda lam: (w[lam], lam)), w
    monkeypatch.setattr(W, "decide", mut)
    assert S.rule_problems() != [] and S.zero_problems() != []


def test_mut_rule_one_of_three_is_caught(monkeypatch):
    monkeypatch.setattr(W, "MIN_W", 1)
    assert S.rule_problems() != []


def test_mut_rule_by_mean_g_is_caught(monkeypatch):
    def mut(G, rule):
        m = {lam: float(np.mean(G[lam])) for lam in W.LAMS}
        best = min(m, key=m.get)
        return (best if m[best] < 0 else 0.0), {lam: sum(g < 0 for g in G[lam]) for lam in W.LAMS}
    monkeypatch.setattr(W, "decide", mut)
    assert S.rule_problems() != []


def test_mut_tie_smaller_lambda_is_caught(monkeypatch):
    def mut(G, rule):
        lam, w = ORIG_DECIDE(G, rule)
        ok = [k for k in W.LAMS if w[k] >= W.MIN_W]
        return (min(k for k in ok if w[k] == max(w[j] for j in ok)) if ok and rule == "blocks" else lam), w
    monkeypatch.setattr(W, "decide", mut)
    assert S.rule_problems() != []


def test_mut_validation_in_clipped_loss_is_caught(monkeypatch):
    monkeypatch.setattr(RL, "ell", lambda rv, lnf: np.minimum(ORIG_ELL(rv, lnf), 0.3))
    assert S.val_problems() != []


def test_mut_training_includes_validation_block_is_caught(monkeypatch):
    def mut(P, nb=W.NB):
        return [(np.asarray(P)[: len(P) - W.VAL * (nb - b - 1)], V) for b, (T, V) in enumerate(ORIG_BLOCKS(P, nb))]
    monkeypatch.setattr(W, "blocks", mut)
    assert S.block_problems() != []


def test_mut_different_inclusion_is_caught(monkeypatch):
    monkeypatch.setattr(W, "MIN_POOL", 500)
    assert S.incl_problems() != []


def test_mut_layers_over_cfg8_instead_of_c_r_is_caught(monkeypatch):
    orig = W.CL.layers
    monkeypatch.setattr(W.CL, "layers", lambda x, c, gap: orig(x, x["C"].to_numpy(float), gap))
    assert S.layer_problems(K.ctx_of("CK")) != []


def test_mut_w5_over_x1_instead_of_x1_r_is_caught(monkeypatch):
    orig = W.CL.layers

    def mut(x, c, gap):
        lay = orig(x, c, gap)
        lay["F"] = W5.walk(x["d"], x["t_plus"], x["rv"], x["F"], gap, base_name="X1_V")["F"].to_numpy()
        return lay
    monkeypatch.setattr(W.CL, "layers", mut)
    assert S.layer_problems(K.ctx_of("CK")) != []
