"""E1 куба v7 (Ж-153): реестр `v7` и `EvaluatorV7(EvaluatorV6)` (fastbt/eval_v7.py), синтетика. Реальные точки ledger — test_eval_v7_real.py.
G1 «v7 без осей v7 = v6 побайтно» (сигналы, ключ); G2 S10/S11 через оценщик = голая идея (fastbt.golaya_v7_isp.ispolnit — исполнитель
scripts/analysis/golaya_v7.py) по знаку и датам сделок; G3 M-d1 ловится; G4 ключ кэша/ledger включает семью v7.
Контракт SignalSet: exit S10–S12 — состояние (флаг пропущенного сигнала не закрывает позицию нового); strength S8/S9 в прогреве = 0.
S7 в реестре v7 нет (Ж-153 п. 2)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import eval_v7, ledger_v2
from fastbt.eval_v6 import EvaluatorV6, IstochnikiV6
from fastbt.eval_v7 import EvaluatorV7
from fastbt.golaya_v7_isp import ispolnit
from fastbt.indicators import compute_base
from fastbt.signals import v5_s12, v5_s34, v7_impuls, v7_kross  # noqa: F401  (регистрация семей v5)
from fastbt.signals.interface import WARMUP, SignalSet, TickerData, get_family

SP = dict(ma_tip="sma", ma_window=20, dev_tip="std", z_entry=1.0, z_exit=0.5)
N = 700
P_D1, P_T = dict(tau=3, vhod="d1", T="17:00"), dict(tau=3, vhod="T", T="17:00")      # у d1 T не влияет, но входит в точку (v7_impuls._tochka)
DNI = {310: 1, 311: -1, 313: 1, 330: -1, 331: 1, 350: 1}      # день решения → сторона; 311 — внутри удержания; 313 — день выхода (d1: взят, T: нет)
# взятые сигналы, τ = 3: d1 — 310, 313, 330, 350 (вход i+1 после выхода); T — 310, 330, 350 (313 ещё в позиции); 311, 331 пропущены всегда
VZYATO = {"d1": ([1, 1, -1, 1], [313, 316, 333, 353]), "T": ([1, -1, 1], [313, 333, 353])}


def _td(n=N, seed=7):
    r = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(r.normal(0, 0.015, n)))
    o = c * (1 + r.normal(0, 0.003, n))
    h, l, v = np.maximum(o, c) * 1.004, np.minimum(o, c) * 0.996, r.uniform(1e6, 2e6, n)
    lr = np.r_[np.log(c[1:] / c[:-1]), 0.0]
    return TickerData("T", pd.bdate_range("2018-01-02", periods=n).values, o, h, l, c, v, lr, {}), compute_base(c, h, l, o, v)


def _ev(cls, sid="S1", **kw):
    e = object.__new__(cls)
    e.sid, e.sk, e.v6, e.ist, e._rel, e._dan = sid, SimpleNamespace(_td={"T": _td()}), {}, IstochnikiV6(), {}, {}
    e.r5 = SimpleNamespace(filtr={}, nov={"vhod_tip": "stop_zayavka"}, sp=SP)
    if cls is EvaluatorV7:
        e.ist7, e.proxy, e._fk = kw.get("ist7") or eval_v7.Ist7(), eval_v7.PROXY.get(sid, sid), ()
    return e


def _ryad(shift=0):
    s = pd.Series(0.0, index=pd.DatetimeIndex(_td()[0].dates))
    for i, z in DNI.items():
        s.iloc[i + shift] = z
    return s


def _ist7(patch, shift=0):
    patch.setattr(v7_impuls.I, "signal_dnya", lambda b10, sid, t_sig="18:30": _ryad(shift))
    patch.setattr(v7_impuls.I, "sila_dnya", lambda b10, sid, t_sig="18:30": _ryad(shift).abs() + 1.0)
    return eval_v7.Ist7(b10={"T": pd.DataFrame({"date": pd.DatetimeIndex([])})})


def _ss(patch, sid, p, shift=0):
    return _ev(EvaluatorV7, sid, ist7=_ist7(patch, shift))._signals5(p)["T"].ss


def _sdelki(sig, exl, exs, vhod, tau):
    td = _td()[0]
    r = ispolnit(sig, td.open, td.close, td.open * 1.001 if vhod == "T" else None, vhod, tau, exl, exs, dates=td.dates, rate=np.zeros(td.n))
    return [(s, e, x) for s, e, x, _, _ in r["sdelki"]]


def _ravno_goloy(patch, p, sid="S10", shift=0):
    """Сделки (знак, вход, выход) по сигналу и флагам оценщика = голая идея: сигнал + τ, флаги обёртки НЕ подаются."""
    ss = _ss(patch, sid, p, shift)
    gol = _sdelki(_ryad().to_numpy().astype(np.int8), None, None, p["vhod"], p["tau"])
    fl = (ss.exit_l[0], ss.exit_s[0]) if p["vhod"] == "d1" else (None, None)    # T: флаг дня k = выход по p_T[k] (раньше close) — это слой Л6.3/V7-6, не E1
    return _sdelki(ss.sig[0], *fl, p["vhod"], p["tau"]) == gol


# ---------------- G1: v7 без осей v7 = v6 ----------------
def test_g1_bez_osej_v7_signaly_ravny_v6_pobaytno():
    s6, s7 = _ev(EvaluatorV6)._signals5(SP)["T"], _ev(EvaluatorV7)._signals5(SP)["T"]
    assert (s6.sigs != 0).sum() > 20
    for a in ("sigs", "exl", "exs", "strength"):
        assert getattr(s6, a).tobytes() == getattr(s7, a).tobytes(), a


def test_g1_klyuch_s1_s6_ravnyj_klyuchu_v6():
    for sid in ("S1", "S3", "S6"):
        e6, e7 = _ev(EvaluatorV6, sid), _ev(EvaluatorV7, sid)
        e6._fkey = e7._fkey = (("adx_th", 25), ("v6", (("dlit", 2),)))
        assert e6._fkey == e7._fkey and e7._k7() == ()


def test_g1_reestr_v7_s8_s12_est_s7_net_s1_s6_iz_v5():
    for sid in ("S8", "S9", "S10", "S11", "S12"):
        assert callable(get_family(sid, setka="v7"))
    for sid in ("S7", "S11x"):
        with pytest.raises(KeyError):
            get_family(sid, setka="v7")                  # Ж-153 п. 2: S7 в v7 не входит
    for sid in ("S1", "S3", "S4"):
        assert get_family(sid, setka="v7") is get_family(sid, setka="v5")
    assert set(eval_v7.SEMYI_V7) == {"S8", "S9", "S10", "S11", "S12"}


def test_g1_polnyj_put_s8_s12_otkaz_a_ne_molchalivyj_v6():
    with pytest.raises(ValueError, match="сигнальных осей"):          # S8–S12 без сигнальных осей — отказ, не молчаливый путь v6
        _ev(EvaluatorV7, "S10").evaluate_v2({}, "train4")


# ---------------- G2: S10/S11 через оценщик = голая идея ----------------
@pytest.mark.parametrize("sid", ["S10", "S11"])
@pytest.mark.parametrize("p", [P_D1, P_T], ids=["d1", "T"])
def test_g2_semya_cherez_ocenshchik_ravna_golaj_idee(monkeypatch, sid, p):
    ss = _ss(monkeypatch, sid, p)
    assert np.array_equal(ss.sig[0], _ryad().to_numpy().astype(np.int8)) and (ss.sig[0] != 0).sum() == len(DNI)
    assert _ravno_goloy(monkeypatch, p, sid)
    znaki, vyhody = VZYATO[p["vhod"]]
    assert [s for s, _, _ in _sdelki(ss.sig[0], None, None, p["vhod"], p["tau"])] == znaki
    assert np.flatnonzero(ss.exit_l[0]).tolist() == vyhody and np.array_equal(ss.exit_l, ss.exit_s) and ss.split


def test_g2_konec_lenty_bez_vyhoda(monkeypatch):
    monkeypatch.setattr(eval_v7, "_cepochka", eval_v7._cepochka)
    sig = np.zeros(N, np.int8)
    sig[N - 2] = 1
    assert not eval_v7._cepochka(sig, 3, "d1").any() and not eval_v7._cepochka(sig, 3, "T").any()


# ---------------- G3: M-d1 и контракт exit ----------------
def test_g3_mutaciya_d1_signal_zavtra_lovitsya(monkeypatch):
    assert _ravno_goloy(monkeypatch, P_D1)
    assert not _ravno_goloy(monkeypatch, P_D1, shift=1)       # сигнал дня d+1 подан как сигнал дня d — сделки другие


def test_g3_mutaciya_flagi_propushchennyh_lovitsya(monkeypatch):
    monkeypatch.setattr(eval_v7, "_cepochka", lambda s, tau, vhod: np.isin(np.arange(len(s)), np.flatnonzero(s) + tau))
    assert not _ravno_goloy(monkeypatch, P_D1)                # флаг пропущенного 311 (→314) закрыл бы позицию нового, вошедшую в 314


def test_g3_vhod_d1_ne_ran_she_d_plus_1(monkeypatch):
    ss = _ss(monkeypatch, "S11", P_D1)
    for _, e, _ in _sdelki(ss.sig[0], ss.exit_l[0], ss.exit_s[0], "d1", 3):
        assert ss.sig[0, e - 1] != 0                          # вход — на следующий день после сигнала (Д2)


# ---------------- strength S8/S9 в прогреве ----------------
def _stub_sech(td, ind, grid):
    sig = np.ones((len(grid), td.n), np.int8)
    return SignalSet(sig, sig != 1, sig != -1, np.full(sig.shape, 2.5), list(grid), split=True)


@pytest.mark.parametrize("sid", ["S8", "S9"])
def test_strength_progrev_nol_kak_sig(monkeypatch, sid):
    monkeypatch.setattr(v7_kross, "semya", lambda s, pn: _stub_sech)
    td, ind = _td()
    ss = get_family(sid, setka="v7")(eval_v7.Ist7(panel=object()))(td, ind, [dict(porog=0.5)])
    assert not ss.strength[:, :WARMUP].any() and (ss.strength[:, WARMUP:] == 2.5).all() and (ss.sig == 1).all()
    monkeypatch.setattr(eval_v7, "_obnulit_progrev", lambda s: s)             # мутация «прогрев не обнулён» ловится
    assert get_family(sid, setka="v7")(eval_v7.Ist7(panel=object()))(td, ind, [dict(porog=0.5)]).strength[:, :WARMUP].any()


def test_semyi_trebuyut_istochnik():
    for sid, kw in (("S10", {}), ("S12", {"panel": object()}), ("S9", {}), ("S8", {"b10": {"T": 1}})):
        with pytest.raises(ValueError, match="Ist7"):
            get_family(sid, setka="v7")(eval_v7.Ist7(**kw))


# ---------------- G4: ключ кэша / ledger ----------------
def test_g4_klyuch_soderzhit_semyu_i_istochnik(monkeypatch):
    def lenta(x):
        return eval_v7.Ist7(b10={"T": pd.DataFrame({"date": pd.DatetimeIndex(["2018-01-02"]), "close": [x]})})
    a = _ev(EvaluatorV7, "S10", ist7=lenta(1.0))
    ks = {x._fkey for x in (a, _ev(EvaluatorV7, "S11", ist7=lenta(1.0)), _ev(EvaluatorV7, "S10", ist7=lenta(2.0)), _ev(EvaluatorV7, "S10"))}
    assert len(ks) == 4 and a._fkey == _ev(EvaluatorV7, "S10", ist7=lenta(1.0))._fkey      # семья и подмена ленты — другой ключ (Д5, № 7)
    assert _ev(EvaluatorV7, "S1")._fkey == () == _ev(EvaluatorV7, "S5")._fkey
    monkeypatch.setattr(EvaluatorV7, "_k7", lambda self: ())                    # мутация «ключ без семьи v7»
    assert _ev(EvaluatorV7, "S10", ist7=lenta(1.0))._fkey == _ev(EvaluatorV7, "S11", ist7=lenta(2.0))._fkey


def test_g4_ledger_klyuch_n_semya_i_tau():
    kl = {"tau": "3.0", "vhod": "d1", "T": "18:30"}
    ks = {ledger_v2.klyuch_n(s, k) for s in ("S10", "S11") for k in (kl, {**kl, "tau": "2.0"}, {**kl, "vhod": "T"})}
    assert len(ks) == 6
