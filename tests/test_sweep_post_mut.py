"""T1 (Ж-036): подложенные ошибки (monkeypatch) в `fastbt/sweep_post.py`; модуль не правится.
Детектор на правильном коде True (`test_sweep_post.py`), на мутации — False. Запуск: FASTBT_MAIN=$PWD pytest ..."""
import copy
import types

import numpy as np
import pytest

from fastbt import sweep_post as sp
from fastbt import unit as unit_mod
from test_sweep_post import (ctx, ok_cache_not_mutated, ok_old_grid_node_byte_equal, ok_q_no_lookahead,  # noqa: F401
                             ok_rollback, ok_weaker_raises, q_ref)


@pytest.mark.parametrize("upper", ["le", "all"])
def test_mutp_01_q_sily_s_tekushchim_dnem_ili_budushchim(monkeypatch, upper):
    assert ok_q_no_lookahead(sp.strength_quantile_ok) and ok_q_no_lookahead(q_ref("lt"))
    monkeypatch.setattr(sp, "strength_quantile_ok", q_ref(upper))       # окно idx <= t / по всем сигналам
    assert not ok_q_no_lookahead(sp.strength_quantile_ok)


def test_mutp_01b_q_sily_sdvig_na_tekushchiy_den_v_build_view(monkeypatch, ctx):  # noqa: F811
    """Мутация на уровне вида: порог квантиля считается по силе, сдвинутой вперёд на 1 бар (решение t видит t+1)."""
    from test_sweep_post import SIG, rm_of
    u, args = ctx["u"], (ctx["td"], ctx["ind"])
    p = {**SIG, **rm_of(u, 0), "q_sily": 0.75}
    good, _ = sp.build_view("S1", u, *args, p, ctx["ev"].sig_names)
    orig = sp.strength_quantile_ok
    monkeypatch.setattr(sp, "strength_quantile_ok", lambda sig, st, q: orig(sig, np.r_[st[1:], st[-1]], q))
    bad, _ = sp.build_view("S1", u, *args, p, ctx["ev"].sig_names)
    assert not np.array_equal(good.sigs, bad.sigs)


def test_mutp_02_maska_pishetsya_v_kesh(monkeypatch, ctx):  # noqa: F811
    assert ok_cache_not_mutated(ctx)
    monkeypatch.setattr(sp, "copy", types.SimpleNamespace(copy=lambda x: x))          # вид = сам закешированный Unit
    saved = dict(vars(ctx["u"]))
    try:
        assert not ok_cache_not_mutated(ctx)
    finally:
        vars(ctx["u"]).update(saved)                                                      # не оставляем порчу в фикстуре


def test_mutp_03_oslablenie_poroga_molcha_ignoriruetsya(monkeypatch, ctx):  # noqa: F811
    assert ok_weaker_raises(ctx)
    base = {"vf_porog": 0.5, "adx_min": 20, "vol_regime_max": 1.2, "obyom_mnozh_sma20": 1.0, "svechi_min_iz5": 3}
    orig = sp.entry_masks

    def clamped(td, ind, m):                       # ослабление подтягивается к зашитому порогу вместо VneV1
        return orig(td, ind, {k: (base[k] if (k in base and (v == "off" or (v > base[k]) == (k == "vol_regime_max") and v != base[k])) else v)
                              for k, v in m.items() if k != "obyom_nizhe_sma5"})
    assert not ok_weaker_raises(ctx, clamped)


def test_mutp_05_otkat_spiskov_ne_srabatyvaet(monkeypatch, ctx):  # noqa: F811
    assert ok_rollback(ctx)
    real = unit_mod
    saved = (real.PIV_COMBOS, real.VWAP_DEVS, real.TREND_TN)

    class NoRestore:                                # второе присваивание (откат) теряется
        def __init__(self):
            object.__setattr__(self, "n", {})

        def __getattr__(self, k):
            return getattr(real, k)

        def __setattr__(self, k, v):
            self.n[k] = self.n.get(k, 0) + 1
            if self.n[k] == 1:
                setattr(real, k, v)
    monkeypatch.setattr(sp, "unit_mod", NoRestore())
    try:
        assert not ok_rollback(ctx)
    finally:
        real.PIV_COMBOS, real.VWAP_DEVS, real.TREND_TN = saved


def test_mutp_06_rm_znacheniya_ne_primenyayutsya(monkeypatch, ctx):  # noqa: F811
    assert ok_old_grid_node_byte_equal(ctx)
    orig = sp.split

    def no_rm(sid, p, names):
        rtype, rm, mask, ex = orig(sid, p, names)
        return rtype, dict(sp.RM_DEF[rtype]), mask, ex          # RM из точки потерян, всегда зашитые по умолчанию
    monkeypatch.setattr(sp, "split", no_rm)
    assert not ok_old_grid_node_byte_equal(ctx)
