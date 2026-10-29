"""Лист V6-3: ось asim (v6_obertka.primenit_asim). asim=0 = v5 побайтно, известный ответ на подставной семье,
узлы сетки, край, long = итерация 26 на уровне сигнала, мутации (проверочная функция обязана упасть)."""
import numpy as np
import pytest

from fastbt.indicators import compute_base
from fastbt.signals import v5_s12, v5_s34, v5_s56
from fastbt.signals import v6_obertka as W
from fastbt.signals.interface import TickerData

T = 1500
RNG = np.random.default_rng(5)
_t = np.arange(T)
C = 100 * np.exp(np.cumsum(RNG.normal(0, 0.02, T)) + 0.4 * np.sin(_t / 35))
H = C * (1 + RNG.uniform(0, 0.015, T))
L = C * (1 - RNG.uniform(0, 0.015, T))
O = np.r_[C[0], C[:-1]]
V = RNG.uniform(1e3, 5e3, T)
PT = {
    "S1": dict(ma_tip="sma", dev_tip="std", ma_window=20, z_entry=1.0, z_exit="net"),
    "S2": dict(ma_tip="sma", bb_window=20, bb_std=1.5, podtv="none"),
    "S3": dict(okno=10, podtv="none"),
    "S4": dict(period=10, mnozh=1.0),
    "S5": dict(pivot_type="classic", pivot_period="den", uroven=2, atr_filtr=0, filter_set="net"),
    "S6": dict(vwap_window=10, dev_mult=0.5, dir_bias="OFF", filter_set="net"),
}
GEN = {"S1": v5_s12.generate_s1, "S2": v5_s12.generate_s2, "S3": v5_s34.generate_s3_v5,
       "S4": v5_s34.generate_s4_v5, "S5": v5_s56.generate5, "S6": v5_s56.generate6}


def mk():
    lr = np.r_[np.log(C[1:] / C[:-1]), 0.0]
    td = TickerData("X", np.arange(T).astype("datetime64[D]").astype("datetime64[ns]"), O, H, L, C, V, lr)
    return td, compute_base(C, H, L, O, V)


@pytest.mark.parametrize("sid", sorted(PT))
def test_asim0_ravno_v5_pobaytno(sid):
    td, ind = mk()
    f = lambda p: GEN[sid](td, ind, [p]).sig[0]          # noqa: E731
    s = f(PT[sid])
    assert (s > 0).sum() > 5 and (s < 0).sum() > 5
    out, kray = W.primenit_asim(sid, f, PT[sid], 0)
    assert out.dtype == s.dtype and out.tobytes() == s.tobytes() and not kray


@pytest.mark.parametrize("sid", sorted(PT))
def test_long_bez_shortov(sid):
    td, ind = mk()
    f = lambda p: GEN[sid](td, ind, [p]).sig[0]          # noqa: E731
    s = f(PT[sid])
    out, _ = W.primenit_asim(sid, f, PT[sid], "long")
    assert out.tobytes() == np.maximum(s, 0).tobytes() and (out > 0).any() and not (out < 0).any()


@pytest.mark.parametrize("sid", ["S1", "S2", "S3", "S4", "S6"])
def test_shorty_strozhe_pri_plyus_dva(sid):
    td, ind = mk()
    f = lambda p: GEN[sid](td, ind, [p]).sig[0]          # noqa: E731
    base = f(PT[sid])
    out, kray = W.primenit_asim(sid, f, PT[sid], 2)
    assert not kray and (out > 0).sum() <= (base > 0).sum() and 0 < (out < 0).sum() < (base < 0).sum()


def test_porog_shorta_uzly_i_kray():
    p = PT["S1"]
    assert [W.porog_shorta("S1", p, n)[0]["z_entry"] for n in (-2, -1, 0, 1, 2)] == [0.5, 0.75, 1.0, 1.25, 1.5]
    assert W.porog_shorta("S1", dict(p, z_entry=5.0), 1) == (dict(p, z_entry=5.0), True)
    assert W.porog_shorta("S1", dict(p, z_entry=0.3), -2)[0]["z_entry"] == 0.3
    assert W.porog_shorta("S5", PT["S5"], 1)[0]["uroven"] == 3
    with pytest.raises(ValueError):
        W.porog_shorta("S1", dict(p, z_entry=1.1), 1)
    with pytest.raises(ValueError):
        W.primenit_asim("S1", lambda q: np.zeros(3, np.int8), p, 3)


# ---- известный ответ на подставной семье + мутации ----------------------------------------------

ZERO = np.zeros(6, np.int8)
SIG = {1.0: np.array([1, 1, 0, -1, 0, 0], np.int8), 1.25: np.array([0, -1, 0, 0, -1, 0], np.int8)}


def sig_fn(p):
    return SIG.get(p["z_entry"], ZERO).copy()


def _proverka(primenit):
    out, kray = primenit("S1", sig_fn, dict(z_entry=1.0), 1)
    assert out.tolist() == [1, 0, 0, 0, -1, 0] and not kray    # шорт дня 3 (порог лонга) снят, день 1 — конфликт → 0
    out0, _ = primenit("S1", sig_fn, dict(z_entry=1.0), 0)
    assert out0.tolist() == SIG[1.0].tolist()


def test_izvestnyy_otvet_na_podstavnoy_semye():
    _proverka(W.primenit_asim)


def _mut(shorty=None, konflikt_lonu=False, znachenie=False):
    def f(sid, fn, p, asim):
        base = fn(p)
        ps = p if shorty == "porog_longa" else (dict(p, z_entry=p["z_entry"] + asim) if znachenie
                                                else W.porog_shorta(sid, p, asim)[0])
        sh = fn(ps)
        lg, s = base > 0, sh < 0
        out = np.where(lg & ~s, 1, np.where(s & ~lg, -1, 0))
        if konflikt_lonu:
            out = np.where(lg, 1, out)
        return out.astype(np.int8), False
    return f


@pytest.mark.parametrize("mut", [dict(shorty="porog_longa"), dict(konflikt_lonu=True), dict(znachenie=True)],
                         ids=["porog_longa_k_shortu", "konflikt_lonu", "sdvig_v_znacheniyah"])
def test_mutant_padaet_na_proverke(mut):
    with pytest.raises(AssertionError):
        _proverka(_mut(**mut))
