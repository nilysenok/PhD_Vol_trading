"""Л2.3: ось otnositelno ∈ {net, sektor, imoex} в EvaluatorV7 для S1–S4 (fastbt/eval_v7_otn.py). Синтетика, без сети, данные < 2020.

G1 net = путь v6 побайтно (сигналы, ключ кэша); G2 известный ответ: тикер = сектор → относительный ряд 0, сигналов нет; тикер = сектор + тренд →
сигнал по тренду; G3 M-d1: фон дня d+1 / дня d−1 в сигнале дня d ловится; G4 исполнение по реальным ценам тикера (мутация: по относительным);
G5 otnositelno в ключе кэша и ledger (Д5); G6 двойной фон → ValueError. Мутации — «чек падает на подмене» (pytest.raises(AssertionError))."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import ledger_v2
from fastbt import otnositelno_v7 as ot
from fastbt.eval_v6 import EvaluatorV6
from fastbt.eval_v6_ist import IstochnikiV6
from fastbt.eval_v7 import EvaluatorV7, Ist7
from fastbt.eval_v7_otn import OtnIst
from fastbt.indicators import compute_base
from fastbt.signals import v5_s12, v5_s34  # noqa: F401  (регистрация семей v5)
from fastbt.signals.interface import TickerData

T, TIK = 420, ["A", "B", "C", "D"]
DATES = pd.bdate_range("2016-01-04", periods=T)
KARTA = {"A": "X", "B": "X", "C": "Y"}                      # A,B — сектор X; C один в Y (→ IMOEX); D без сектора (→ IMOEX)
SP = {"S1": dict(ma_window=20, z_entry=1.0, z_exit=0.0), "S3": dict(okno=20, podtv="none"), "S4": dict(period=10, mnozh=2.0),
      "S2": dict(bb_window=20, bb_std=1.5, podtv="none")}


def _ryady(seed=7):
    r = np.random.default_rng(seed)
    c = pd.DataFrame(100 * np.exp(np.cumsum(r.normal(0, 0.02, (T, 4)), axis=0)), index=DATES, columns=TIK)
    return c, pd.Series(1000 * np.exp(np.cumsum(r.normal(0, 0.01, T))), index=DATES)


C0, IM0 = _ryady()


def mk_td(c, t):
    cc = c[t].to_numpy()
    o = np.r_[cc[0], cc[:-1]] * 1.001
    h, lo, v = np.maximum(o, cc) * 1.004, np.minimum(o, cc) * 0.996, np.full(len(cc), 1e4)
    return TickerData(t, c.index.to_numpy().astype("datetime64[ns]"), o, h, lo, cc, v, np.r_[np.log(cc[1:] / cc[:-1]), 0.0]), compute_base(cc, h, lo, o, v)


def ev(sid="S3", pt=None, c=C0, imoex=IM0, karta=KARTA, cls=EvaluatorV7):
    e = object.__new__(cls)
    e.sid, e.proxy, e._fk = sid, sid, ()
    e.sk, e.v6, e.ist, e._rel, e._dan = SimpleNamespace(_td={t: mk_td(c, t) for t in c}), {}, IstochnikiV6(), {}, {}
    e.r5 = SimpleNamespace(filtr={}, nov={"vhod_tip": "stop_zayavka"}, sp={})
    if cls is EvaluatorV7:
        e.ist7 = Ist7(otn=OtnIst(imoex, karta))
        e._otn_razbor(pt or {})
    return e


def sig(e, t="A"):
    return e._signals5(SP[e.sid])[t].sigs


# ---------------- G1: net = v6 ----------------
@pytest.mark.parametrize("sid", ["S1", "S2", "S3", "S4"])
def test_g1_net_ravno_v6_pobajtno(sid):
    for pt in ({}, {"otnositelno": "net"}):
        a, b = ev(sid, cls=EvaluatorV6)._signals5(SP[sid]), ev(sid, pt)._signals5(SP[sid])
        for t in TIK:
            for f in ("sigs", "exl", "exs", "strength"):
                assert getattr(a[t], f).tobytes() == getattr(b[t], f).tobytes(), (t, f)
        assert (a["A"].sigs != 0).sum() > 0
    assert ev(sid, {"otnositelno": "net"})._fkey == ev(sid, cls=EvaluatorV6)._fkey == ()


def test_g1_os_ubrana_iz_tochki_dlya_v6():
    e = ev("S3")
    assert e._otn_razbor({"otnositelno": "imoex", "tau": 3}) == {"tau": 3} and e.otn == "imoex"


# ---------------- G2: известный ответ ----------------
def _pary(trend=0.0):
    c = C0.copy()
    c["A"] = c["B"] * np.exp(trend * np.arange(T))             # A ≡ B (+ тренд); сектор A — только B
    return c


@pytest.mark.parametrize("sid", ["S1", "S3"])
def test_g2_tiker_ravno_sektoru_signalov_net(sid):
    e = ev(sid, {"otnositelno": "sektor"}, c=_pary())
    assert not sig(e).any()                                     # r_rel ≡ 0 → close_rel константа
    assert sig(ev(sid, c=_pary())).any()                        # тот же тикер без оси сигналы даёт (шум B)


def test_g2_tiker_plyus_trend_signal_po_trendu():
    s = sig(ev("S3", {"otnositelno": "sektor"}, c=_pary(0.002)))
    assert (s > 0).sum() > 100 and not (s < 0).any()
    assert not sig(ev("S3", {"otnositelno": "sektor"}, c=_pary(-0.002))).clip(min=0).any()


def test_g2_imoex_i_sektor_raznye_ryady():
    assert sig(ev("S1", {"otnositelno": "imoex"})).tobytes() != sig(ev("S1", {"otnositelno": "sektor"})).tobytes()


def test_g2_net_kolleg_fon_imoex_s_flagom():
    ss = ev("S3", {"otnositelno": "sektor"})._signals5(SP["S3"])
    assert ss["C"].ss.extra[ot.FLAG][1:].all() and ss["D"].ss.extra[ot.FLAG][1:].all()      # C один в секторе, D без сектора → фон IMOEX, флаг
    assert not ss["A"].ss.extra[ot.FLAG].any()


# ---------------- G3: M-d1 ----------------
D0 = 300


def _prichinnost():
    """Сигнал дня ≤ D0 не зависит от IMOEX и тикеров после D0 (фон d−1…d, не d+1)."""
    c2, im2 = C0.copy(), IM0.copy()
    c2.iloc[D0 + 1:, 1:] *= 1.3
    im2.iloc[D0 + 1:] *= 0.7
    for rezhim in ("imoex", "sektor"):
        a, b = sig(ev("S1", {"otnositelno": rezhim})), sig(ev("S1", {"otnositelno": rezhim}, c=c2, imoex=im2))
        assert a[:D0 + 1].tobytes() == b[:D0 + 1].tobytes(), rezhim
        assert (a[:D0 + 1] != 0).any()


def _sdvig(k):
    real = ot.fon_ryad

    def f(*a, **kw):
        fon, fl = real(*a, **kw)
        return fon.shift(k).bfill().ffill(), fl
    return f


def test_g3_prichinnost_fona():
    _prichinnost()


def test_g3_mutaciya_fon_dnya_d_plyus_1_lovitsya(monkeypatch):
    monkeypatch.setattr(ot, "fon_ryad", _sdvig(-1))
    with pytest.raises(AssertionError):
        _prichinnost()


@pytest.mark.parametrize("k", [1, -1])
def test_g3_mutaciya_sdvig_fona_lovitsya_izvestnym_otvetom(monkeypatch, k):
    monkeypatch.setattr(ot, "fon_ryad", _sdvig(k))
    with pytest.raises(AssertionError):
        assert not sig(ev("S3", {"otnositelno": "sektor"}, c=_pary())).any()


# ---------------- G4: исполнение по реальным ценам ----------------
def _ispolnenie_real(e):
    for t, u in e._signals5(SP[e.sid]).items():
        td = e.sk._td[t][0]
        assert u.close.tobytes() == td.close.tobytes() and u.high.tobytes() == td.high.tobytes() and u.low.tobytes() == td.low.tobytes()


def test_g4_ispolnenie_po_realnym_cenam():
    for r in ("sektor", "imoex"):
        _ispolnenie_real(ev("S3", {"otnositelno": r}))


def test_g4_mutaciya_ispolnenie_po_otnositelnoj_cene_lovitsya():
    class Mut(EvaluatorV7):
        def _signals5(self, sp):
            from fastbt.unit import Unit
            fon = self._otn_fon()[0]
            return {t: Unit(t, self.sid, ot.td_rel(td, fon[t].reindex(pd.DatetimeIndex(td.dates)).to_numpy()), ind, super(Mut, self)._signals5(sp)[t].ss)
                    for t, (td, ind) in self.sk._td.items()}
    e = ev("S3", {"otnositelno": "sektor"})
    e.__class__ = Mut
    with pytest.raises(AssertionError):
        _ispolnenie_real(e)


# ---------------- G5: ключ (Д5) ----------------
def _klyuchi():
    ks = [ev("S3", {"otnositelno": r})._fkey for r in ("net", "sektor", "imoex")]
    ks += [ev("S3", {"otnositelno": "imoex"}, imoex=IM0 * 1.01)._fkey, ev("S3", {"otnositelno": "imoex"}, karta={"A": "Z"})._fkey]
    assert len(set(ks)) == 5 and ks[0] == ()
    kl = [ledger_v2.klyuch_n("S3", ledger_v2.klyuch("S3", {"okno": 20, "otnositelno": r}, "train4", 0.0004, None)) for r in ("net", "sektor", "imoex")]
    assert len(set(kl)) == 3


def test_g5_klyuch_soderzhit_os():
    _klyuchi()


def test_g5_mutaciya_os_ne_v_klyuche_lovitsya(monkeypatch):
    monkeypatch.setattr(EvaluatorV7, "_otn_key", lambda self: ())
    with pytest.raises(AssertionError):
        _klyuchi()


# ---------------- G6: отказы ----------------
def test_g6_dvojnoj_fon_valueerror():
    for r in ("sektor", "imoex"):
        with pytest.raises(ValueError, match="rel_signal"):
            ev("S3", {"otnositelno": r, "rel_signal": "sektor"})
    ev("S3", {"otnositelno": "sektor", "rel_signal": "syroy"})
    ev("S3", {"otnositelno": "net", "rel_signal": "sektor"})       # без оси фон v6 — путь v6, не наше дело


def test_g6_ne_s1_s4_net_istochnika_nepravilnoe_znachenie():
    with pytest.raises(ValueError, match="S1–S4"):
        ev("S5", {"otnositelno": "sektor"})
    with pytest.raises(ValueError, match="otnositelno"):
        ev("S3", {"otnositelno": "foo"})
    e = ev("S3")
    e.ist7 = Ist7()
    with pytest.raises(ValueError, match="IMOEX"):
        e._otn_razbor({"otnositelno": "imoex"})
    with pytest.raises(ValueError):                                # IMOEX не покрывает даты — без подстановок (Д4)
        sig(ev("S3", {"otnositelno": "imoex"}, imoex=IM0.iloc[:200]))
