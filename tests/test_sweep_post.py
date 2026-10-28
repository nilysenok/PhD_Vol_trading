"""T1 (Ж-036, #308): пост-сигнальный слой оценщика Sk — `fastbt/sweep_post.py`. Реальные данные: S1, SBER, 2018–2021, ≤ 60 с.
Детекторы `ok_*` — True, если свойство выполнено; мутации — `test_sweep_post_mut.py`.
Запуск: FASTBT_MAIN=$PWD pytest tests/test_sweep_post*.py (хук требует свою копию)."""
import copy
import threading
from unittest import mock

import numpy as np
import pytest

from fastbt import sweep_post as sp
from fastbt import unit as unit_mod
from fastbt.batch import trades_one
from fastbt.stats import pos_from_trades
from fastbt.sweep import Sweep
from fastbt.sweep_spec import Spec

SIG = {"ma_window": 15, "z_entry": 1.5, "z_exit": 0.3}


@pytest.fixture(scope="module")
def ctx():
    sk = pytest.importorskip("fastbt.sweep_sk")
    ev = sk.SkEvaluator("S1", ["SBER"])
    u = ev.signals(SIG)["SBER"]
    td, ind = ev._td["SBER"]
    return dict(ev=ev, u=u, td=td, ind=ind)


def rm_of(u, i):
    return {k: v for k, v in u.rm[i].items() if not k.startswith("_")}


def q_ref(upper):
    """Эталон квантиля силы; upper: 'lt' — прошлые бары (верно), 'le' — с текущим, 'all' — с будущим (мутации)."""
    def f(sig, strength, q):
        ok = np.ones(len(sig), bool)
        if q <= 0:
            return ok
        idx = np.flatnonzero(sig != 0)
        for t in idx:
            hi = {"lt": idx < t, "le": idx <= t, "all": idx >= 0}[upper]
            past = idx[(idx >= t - sp.QWIN) & hi]
            if len(past) >= sp.QMIN:
                ok[t] = strength[t] >= np.quantile(strength[past], q)
        return ok
    return f


def ok_q_no_lookahead(fn) -> bool:
    """(а) бар t не входит в свой порог: прошлые 1..20, текущая 10,4 < медианы 10,5 -> ok=False;
    (б) решение бара t не зависит от силы на барах > t."""
    s = np.r_[np.arange(1, 21, dtype=float), 10.4]
    first = not fn(np.ones(21, np.int8), s, 0.5)[20] and bool(fn(np.ones(21, np.int8), s, 0.5)[:20].all())
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=300), rng.normal(size=300)
    b[:150] = a[:150]
    sig = np.ones(300, np.int8)
    return first and bool(np.array_equal(fn(sig, a, 0.5)[:150], fn(sig, b, 0.5)[:150]))


def ok_cache_not_mutated(ctx) -> bool:
    u = ctx["u"]
    s0, n0, rm0 = u.sigs.copy(), u.nxt.copy(), repr(u.rm)
    p = {**SIG, **rm_of(u, 0)}
    v, _ = sp.build_view("S1", u, ctx["td"], ctx["ind"], {**p, "adx_min": 30, "q_sily": 0.5}, ctx["ev"].sig_names)
    tight = ((v.sigs != 0) <= (s0 != 0)).all() and (v.sigs != 0).sum() < (s0 != 0).sum()
    same = np.array_equal(u.sigs, s0) and np.array_equal(u.nxt, n0) and repr(u.rm) == rm0 and v.sigs is not u.sigs
    v2, _ = sp.build_view("S1", u, ctx["td"], ctx["ind"], p, ctx["ev"].sig_names)       # другая маска -> исходный результат
    return bool(tight and same and np.array_equal(v2.sigs, s0))


WEAK = [{"vf_porog": 0.4}, {"adx_min": 15}, {"vol_regime_max": 1.5}, {"obyom_mnozh_sma20": 0.5},
        {"svechi_min_iz5": 2}, {"vf_porog": "off"}, {"adx_min": "off"}, {"obyom_nizhe_sma5": "off"}]
TIGHT = [{"vf_porog": 0.6}, {"adx_min": 25}, {"vol_regime_max": 1.0}, {"obyom_mnozh_sma20": 1.2}, {"svechi_min_iz5": 4}]


def ok_weaker_raises(ctx, fn=None) -> bool:
    fn = fn or sp.entry_masks
    for m in WEAK:
        try:
            fn(ctx["td"], ctx["ind"], m)
        except sp.VneV1:
            continue
        return False
    return True


def ok_rollback(ctx) -> bool:
    """Списки RM-комбинаций откатываются после вызова и при исключении внутри; замок отпущен."""
    orig = (unit_mod.PIV_COMBOS, unit_mod.VWAP_DEVS, unit_mod.TREND_TN)
    u, args = ctx["u"], ("S1", ctx["u"], ctx["td"], ctx["ind"], {**SIG, **rm_of(ctx["u"], 0), "sl_mult": 2.2}, ctx["ev"].sig_names)
    sp.build_view(*args)
    ok = (unit_mod.PIV_COMBOS, unit_mod.VWAP_DEVS, unit_mod.TREND_TN) == orig and all(
        a is b for a, b in zip((unit_mod.PIV_COMBOS, unit_mod.VWAP_DEVS, unit_mod.TREND_TN), orig))
    with mock.patch.object(type(u), "_build_rm", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            sp.build_view(*args)
    now = (unit_mod.PIV_COMBOS, unit_mod.VWAP_DEVS, unit_mod.TREND_TN)
    return bool(ok and all(a is b for a, b in zip(now, orig)) and not sp.LOCK.locked())


def ok_old_grid_node_byte_equal(ctx, split=None) -> bool:
    u, ev = ctx["u"], ctx["ev"]
    for i in (0, 40, 80, len(u.rm) - 1):
        v, ex = sp.build_view("S1", u, ctx["td"], ctx["ind"], {**SIG, **rm_of(u, i)}, ev.sig_names)
        te, tx, td = trades_one(u, 0, u.rmP[i], u.trend, end=u.n)
        old = pos_from_trades(te, tx, td, len(te), u.n)
        if sp.positions(v, ex, u.n).tobytes() != old.tobytes():
            return False
    return True


# ---------- тесты ----------
def test_q_sily_ne_zaglyadyvaet_i_bar_ne_vkhodit_v_svoy_porog():
    assert ok_q_no_lookahead(sp.strength_quantile_ok)
    rng = np.random.default_rng(1)
    sig, s = (rng.random(400) < 0.5).astype(np.int8), rng.normal(size=400)
    for q in (0.5, 0.67, 0.75):
        assert np.array_equal(sp.strength_quantile_ok(sig, s, q), q_ref("lt")(sig, s, q))
    assert sp.strength_quantile_ok(sig, s, 0.0).all()


def test_q_sily_menshe_QMIN_proshlykh_filtra_net():
    sig = np.ones(sp.QMIN, np.int8)
    assert sp.strength_quantile_ok(sig, -np.arange(sp.QMIN, dtype=float), 0.9).all()


def test_q_sily_na_dannykh_tolko_ubiraet_signaly(ctx):
    u = ctx["u"]
    v, _ = sp.build_view("S1", u, ctx["td"], ctx["ind"], {**SIG, **rm_of(u, 0), "q_sily": 0.75}, ctx["ev"].sig_names)
    assert ((v.sigs != 0) <= (u.sigs != 0)).all() and (v.sigs != 0).sum() < (u.sigs != 0).sum()
    assert np.array_equal(v.sigs[v.sigs != 0], u.sigs[v.sigs != 0])        # знак не меняется


def test_maska_ne_mutiruet_kesh_signalov(ctx):
    assert ok_cache_not_mutated(ctx)


def test_klyuch_kesha_ne_zavisit_ot_masok_i_rm(ctx):
    sw = Sweep(Spec(obj={"kind": "Sk", "sid": "S1"}, base=dict(SIG)), ctx["ev"])
    for extra in ({}, {"adx_min": 30}, {"q_sily": 0.5, "sl_mult": 2.0}, {"max_hold": 7}):
        sw.evaluate(extra, "setup")
    assert sw.cache.misses == 1 and sw.cache.hits >= 3
    sw.evaluate({"z_entry": 2.0}, "setup")                                 # сигнальный параметр -> новый ключ
    assert sw.cache.misses == 2


def test_oslablenie_zashitogo_poroga_vne_v1(ctx):
    assert ok_weaker_raises(ctx)
    for m in TIGHT:
        sp.entry_masks(ctx["td"], ctx["ind"], m)                           # ужесточение не поднимает


def test_zashitye_porogi_ne_menyayut_signal(ctx):
    u = ctx["u"]
    base = {"vf_porog": 0.5, "adx_min": 20, "vol_regime_max": 1.2, "obyom_mnozh_sma20": 1.0, "svechi_min_iz5": 3}
    v, _ = sp.build_view("S1", u, ctx["td"], ctx["ind"], {**SIG, **rm_of(u, 0), **base}, ctx["ev"].sig_names)
    lo, sh = sp.entry_masks(ctx["td"], ctx["ind"], base)
    assert lo.shape == sh.shape == (u.n,)
    # S1 зашитых порогов не имеет, поэтому маски на «зашитых» значениях только ужесточают (заявление #308
    # «не меняют сигнал» проверяемо на семьях с зашитыми порогами — S2–S6, вне этого теста)
    assert ((v.sigs != 0) <= (u.sigs != 0)).all()


def test_otkat_spiskov_rm_v_finally(ctx):
    assert ok_rollback(ctx)


def test_uzel_staroy_setki_pobaytno_kak_staryy_put(ctx):
    assert ok_old_grid_node_byte_equal(ctx)


def test_rm_znachenie_vne_setki_menyaet_pozicii(ctx):
    u = ctx["u"]
    a = sp.positions(*sp.build_view("S1", u, ctx["td"], ctx["ind"], {**SIG, **rm_of(u, 0)}, ctx["ev"].sig_names), u.n)
    b = sp.positions(*sp.build_view("S1", u, ctx["td"], ctx["ind"], {**SIG, **rm_of(u, 0), "sl_mult": 0.77, "max_hold": 3},
                                    ctx["ev"].sig_names), u.n)
    assert not np.array_equal(a, b)


def test_neizvestnyy_parametr_KeyError(ctx):
    with pytest.raises(KeyError):
        sp.build_view("S1", ctx["u"], ctx["td"], ctx["ind"], {**SIG, "net_takogo": 1}, ctx["ev"].sig_names)
