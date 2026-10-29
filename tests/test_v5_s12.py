"""Тесты S1@v5 / S2@v5 (лист Л2): побайтно = v4 на сетке v4, причинность (префикс), мутация «MA с баром t+1»,
подтверждения S2 против формул 6a7a4636, net-выход, ATR-порог на известном ответе."""
import numpy as np
import pytest

from fastbt.indicators import compute_base
from fastbt.ma_tipy import TIPY
from fastbt.signals import s12 as v4
from fastbt.signals import v5_s12 as v5
from fastbt.signals.interface import TickerData, expand_grid

T = 700
RNG = np.random.default_rng(11)
C = 100 * np.exp(np.cumsum(RNG.normal(0, 0.02, T)))
H = C * (1 + RNG.uniform(0, 0.01, T))
L = C * (1 - RNG.uniform(0, 0.01, T))
O = np.r_[C[0], C[:-1]]
V = RNG.uniform(1e3, 5e3, T)


def mk(n=T):
    c, h, lo, o, v = C[:n], H[:n], L[:n], O[:n], V[:n]
    lr = np.r_[np.log(c[1:] / c[:-1]), 0.0]
    td = TickerData("X", np.arange(n).astype("datetime64[D]").astype("datetime64[ns]"), o, h, lo, c, v, lr)
    return td, compute_base(c, h, lo, o, v)


def ind_vse_razresheno(ind, n):
    """v4 зашивает фильтры входа; здесь они пропускают всё — тогда v4 = «фильтры выкл.» v5."""
    out = dict(ind)
    out.update(vol_regime=np.zeros(n), red_count5=np.full(n, 3.0), green_count5=np.full(n, 3.0),
               vol_sma5=np.full(n, np.nan), vf_pass=np.ones(n, bool))
    return out


def sravnit(a, b):
    assert np.array_equal(a.sig, b.sig) and np.array_equal(a.exit_l, b.exit_l)
    assert np.array_equal(a.exit_s, b.exit_s)
    assert a.strength.tobytes() == b.strength.tobytes()


def test_s1_sma_std_ravno_v4_na_setke_v4():
    td, ind = mk()
    ind4 = ind_vse_razresheno(ind, T)
    g4 = expand_grid(v4.GRID_S1)
    g5 = [dict(ma_tip="sma", dev_tip="std", **p) for p in g4]
    ref = v4.S1.generate(td, ind4, g4)
    assert ref.sig.any() and ref.exit_l.any()
    sravnit(v5.generate_s1(td, ind, g5), ref)


def test_s2_sma_none_ravno_v4_na_setke_v4():
    td, ind = mk()
    ind4 = ind_vse_razresheno(ind, T)
    g4 = expand_grid(v4.GRID_S2)
    g5 = [dict(ma_tip="sma", podtv="none", **p) for p in g4]
    ref = v4.S2.generate(td, ind4, g4)
    assert ref.sig.any()
    sravnit(v5.generate_s2(td, ind, g5), ref)


@pytest.mark.parametrize("tip", TIPY)
def test_prichinnost_prefiks_s1_s2(tip):
    """Прогон на [0, t) = префикс прогона на [0, T) — для всех 7 типов MA."""
    t = 500
    td, ind = mk()
    tdp, indp = mk(t)
    g1 = [dict(ma_tip=tip, ma_window=20, dev_tip=d, z_entry=1.0, z_exit=0.5) for d in ("std", "atr")]
    g2 = [dict(ma_tip=tip, bb_window=20, bb_std=1.0, podtv="none")]
    for f, g in ((v5.generate_s1, g1), (v5.generate_s2, g2)):
        a, b = f(td, ind, g), f(tdp, indp, g)
        assert a.sig[:, :t].any()
        assert np.array_equal(a.sig[:, :t], b.sig) and np.array_equal(a.exit_l[:, :t], b.exit_l)


@pytest.mark.parametrize("tip", ["sma", "ema", "kama"])
def test_mutaciya_ma_s_barom_t_plyus_1_lovitsya(tip, monkeypatch):
    """Подложить MA[t+1] вместо MA[t]: тест префикса обязан упасть."""
    t = 500
    td, ind = mk()
    tdp, indp = mk(t)
    g = [dict(ma_tip=tip, ma_window=20, dev_tip="std", z_entry=0.5, z_exit=0.5)]
    real = v5._ma
    monkeypatch.setattr(v5, "_ma", lambda x, tp, n: np.r_[real(x, tp, n)[1:], np.nan])
    a, b = v5.generate_s1(td, ind, g), v5.generate_s1(tdp, indp, g)
    assert not (np.array_equal(a.sig[:, :t], b.sig) and np.array_equal(a.exit_l[:, :t], b.exit_l))


def test_z_vyhod_net_net_signalnogo_vykhoda():
    td, ind = mk()
    g = [dict(ma_tip="sma", ma_window=20, dev_tip="std", z_entry=1.5, z_exit="net"),
         dict(ma_tip="sma", ma_window=20, dev_tip="std", z_entry=1.5, z_exit=0.5)]
    r = v5.generate_s1(td, ind, g)
    assert not r.exit_l[0].any() and not r.exit_s[0].any()
    assert r.exit_l[1].any()
    assert np.array_equal(r.sig[0], r.sig[1])          # вход от z_exit не зависит


def test_porog_v_atr_izvestnyj_otvet():
    """Вход, когда close < MA − porog·ATR; на синтетике с известным ATR и MA."""
    n = 300
    c = np.full(n, 100.0)
    c[250] = 90.0                                       # провал на 10
    h, lo, o = c + 1.0, c - 1.0, c.copy()               # ATR14 = 2 (кроме окна провала)
    v = np.full(n, 1e3)
    td = TickerData("S", np.arange(n).astype("datetime64[D]").astype("datetime64[ns]"), o, h, lo, c, v,
                    np.zeros(n))
    ind = compute_base(c, h, lo, o, v)
    g = [dict(ma_tip="sma", ma_window=20, dev_tip="atr", z_entry=z, z_exit="net") for z in (1.0, 20.0)]
    r = v5.generate_s1(td, ind, g)
    assert r.sig[0, 250] == 1 and r.sig[1, 250] == 0
    assert (r.sig[0, :250] == 0).all()


def test_porog_atr_ne_ravno_porogu_std():
    td, ind = mk()
    g = [dict(ma_tip="sma", ma_window=20, dev_tip=d, z_entry=1.0, z_exit=0.5) for d in ("std", "atr")]
    r = v5.generate_s1(td, ind, g)
    assert not np.array_equal(r.sig[0], r.sig[1])


# --- подтверждения S2: независимая реализация apply_confirmation@6a7a4636 (grid_scan_test.py:285-300) ---
def src_conf(sig, tip, rsi, vol, vs20):
    if tip in ("none", "close_confirm"):
        return sig
    out = sig.copy()
    nanr = np.isnan(rsi)
    if tip == "rsi30":
        out[(sig == 1) & (nanr | (rsi > 30))] = 0
        out[(sig == -1) & (nanr | (rsi < 70))] = 0
    elif tip == "rsi20":
        out[(sig == 1) & (nanr | (rsi > 20))] = 0
        out[(sig == -1) & (nanr | (rsi < 80))] = 0
    else:
        vs = np.where(np.isnan(vs20) | (vs20 < 1), 1.0, vs20)
        out[vol / vs < 1.0] = 0
    return out


@pytest.mark.parametrize("pod", v5.PODTV)
def test_podtverzhdenie_s2_po_formulam_istochnika(pod):
    td, ind = mk()
    base = v5.generate_s2(td, ind, [dict(ma_tip="ema", bb_window=10, bb_std=0.5, podtv="none")]).sig[0]
    r = v5.generate_s2(td, ind, [dict(ma_tip="ema", bb_window=10, bb_std=0.5, podtv=pod)]).sig[0]
    rsi = v5.rsi_wilder(td.close)
    assert np.array_equal(r, src_conf(base, pod, rsi, td.volume, ind["vol_sma20"]))
    if pod in ("rsi30", "rsi20", "volume"):
        assert (r != base).any() and (r[r != 0] == base[r != 0]).all()   # только гасит, не создаёт


def test_podtverzhdenie_rsi_nan_gasit_signal():
    sig = np.array([1, -1, 1], np.int8)
    rsi = np.array([np.nan, np.nan, 10.0])
    out = v5.podtverzhdenie(sig, "rsi30", rsi, np.ones(3), np.ones(3))
    assert out.tolist() == [0, 0, 1]
    assert sig.tolist() == [1, -1, 1]                  # вход не изменён


def test_podtverzhdenie_neizvestnoe_otklonyaetsya():
    with pytest.raises(ValueError):
        v5.podtverzhdenie(np.zeros(3, np.int8), "adx", np.ones(3), np.ones(3), np.ones(3))


def test_mutaciya_podtv_rsi_vmesto_none_lovitsya():
    """Если подтверждение не применяется (мутация), сверка с формулой падает."""
    td, ind = mk()
    base = v5.generate_s2(td, ind, [dict(ma_tip="sma", bb_window=10, bb_std=0.5, podtv="none")]).sig[0]
    rsi = v5.rsi_wilder(td.close)
    assert not np.array_equal(base, src_conf(base, "rsi20", rsi, td.volume, ind["vol_sma20"]))


def test_netochnyj_tip_ma_otklonyaetsya():
    td, ind = mk()
    with pytest.raises(ValueError):
        v5.generate_s1(td, ind, [dict(ma_tip="tema2", ma_window=20, z_entry=1.0, z_exit=0.5)])
