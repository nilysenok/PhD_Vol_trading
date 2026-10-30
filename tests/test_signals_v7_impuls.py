"""Л1.5 fastbt/signals/v7_impuls.py: семьи S10–S12 с интерфейсом S1–S6 (Ж-146). Только синтетика.

Мутации (каждая ловится): сигнал по цене/бару после T — test_mutaciya_cena_posle_T_*, test_vhod_T_ne_vidit_bary_posle_T;
значение дня d+1 в сигнале d — test_mutaciya_d_plus_1_*; vhod=d1 ≠ Л1.2 (18:30) — test_d1_raven_signal_dnya_pobajtno;
выход не через τ дней — test_exit_cherez_tau_dnej; тикер вне ленты → нули — test_tiker_vne_lenty_keyerror.
"""
import numpy as np
import pandas as pd
import pytest

from fastbt import bars_agg as ba
from fastbt import impuls_v7 as I
from fastbt.signals import interface as IF
from fastbt.signals import v7_impuls as V

ND = 260
START = "2016-01-04"
DAYS = pd.bdate_range(START, periods=ND)
DAY = 230                      # день пробоя (после WARMUP = 200)
S_POSLE_17 = range(36, 48)     # 16:00–18:00: пробой 2ч-бара, закрытого после 17:00 (T=17:00 его не видит)


def b10_iz(cena, ndays=ND, start=START):
    days = pd.bdate_range(start, periods=ndays)
    rows = [(d, s, cena(i, s)) for i, d in enumerate(days) for s in range(ba.SLOTS)]
    df = pd.DataFrame(rows, columns=["date", "slot", "p"])
    for k in ("open", "high", "low", "close"):
        df[k] = df["p"]
    df["volume"] = 100.0
    return df.drop(columns="p")


def proboj(slots, day=DAY, cena=110.0, start=START, ndays=ND):
    return b10_iz(lambda i, s: cena if (i == day and s in slots) else 100.0, ndays, start)


def sluchaynyj(seed=1):
    rng = np.random.default_rng(seed)
    p = 100 + np.cumsum(rng.normal(0, 0.3, ND * ba.SLOTS))
    b = b10_iz(lambda i, s: 0.0)
    b["close"] = p
    b["open"] = np.r_[p[:1], p[:-1]]
    b["high"] = np.maximum(b["open"], b["close"]) + rng.uniform(0, 0.2, len(b))
    b["low"] = np.minimum(b["open"], b["close"]) - rng.uniform(0, 0.2, len(b))
    return b


def td_iz(ticker="AAA", ndays=ND, start=START):
    d = pd.bdate_range(start, periods=ndays).to_numpy("datetime64[ns]")
    c = np.full(ndays, 100.0)
    return IF.TickerData(ticker, d, c, c, c, c, np.full(ndays, 1e3), np.zeros(ndays))


def tochka(vhod="d1", T="18:00", tau=1):
    return {"tau": tau, "vhod": vhod, "T": T}


def gen(b, sid="S11", pts=None, ticker="AAA"):
    pts = pts or [tochka()]
    return V.semya(sid, {ticker: b})(td_iz(ticker), {}, pts)


def test_forma_signalset_i_warmup():
    ss = gen(sluchaynyj(), pts=[tochka(), tochka("T", "17:00", 2), tochka("T", "18:40", 3)])
    assert isinstance(ss, IF.SignalSet) and ss.k == 3 and ss.params[1]["T"] == "17:00"
    assert ss.sig.shape == ss.exit_l.shape == ss.exit_s.shape == ss.strength.shape == (3, ND)
    assert ss.sig.dtype == np.int8 and ss.exit_l.dtype == bool and ss.exit_s.dtype == bool
    assert ss.strength.dtype == np.float64 and ss.extra == {} and ss.split is True
    assert not ss.sig[:, :IF.WARMUP].any() and not ss.exit_l[:, :IF.WARMUP].any()
    assert ss.sig.any()


def test_semyi_s10_s11_s12_vse_stroyatsya():
    b = sluchaynyj(2)
    for sid in ("S10", "S11", "S12"):
        assert gen(b, sid).sig.shape == (1, ND)


def test_d1_raven_signal_dnya_pobajtno():
    """G2 (выкл.): vhod=d1, центр S11 → sig == Л1.2 signal_dnya(…, 18:30) по датам td."""
    b = sluchaynyj(3)
    for T in ("17:00", "18:40"):                  # при d1 T не влияет
        ss = gen(b, "S11", [tochka("d1", T)])
        ref = I.signal_dnya(b, "S11", "18:30").reindex(pd.DatetimeIndex(DAYS)).fillna(0).astype(np.int8).to_numpy()
        ref[:IF.WARMUP] = 0
        assert ref.any() and np.array_equal(ss.sig[0], ref)


def test_known_answer_proboj_vverh_i_vniz():
    for cena, znak in ((110.0, 1), (90.0, -1)):
        ss = gen(proboj(S_POSLE_17, cena=cena), pts=[tochka("T", "18:00")])
        assert ss.sig[0, DAY] == znak and np.count_nonzero(ss.sig[0]) == 1
        assert ss.strength[0, DAY] > 0 and ss.strength[0, DAY - 1] == 0


def test_mutaciya_cena_posle_T_lovitsya():
    """M-T: пробой на баре, закрытом после T=17:00, при vhod=T сигнала не даёт; при T=18:00 даёт."""
    b = proboj(S_POSLE_17)
    assert gen(b, pts=[tochka("T", "17:00")]).sig[0, DAY] == 0
    assert gen(b, pts=[tochka("T", "18:00")]).sig[0, DAY] == 1
    # мутант «сигнал по t_sig=18:30 при vhod=T» отличался бы:
    assert I.signal_dnya(b, "S11", "18:30").loc[DAYS[DAY]] == 1


def test_vhod_T_ne_vidit_bary_posle_T():
    """Любая правка баров с началом ≥ T (слот ≥ 42 для 17:00) не меняет sig."""
    b = sluchaynyj(4)
    c = b.copy()
    m = c["slot"] >= 42
    for k in ("open", "high", "low", "close"):
        c.loc[m, k] = c.loc[m, k] * 1.5 + 7
    p = [tochka("T", "17:00")]
    assert np.array_equal(gen(b, pts=p).sig, gen(c, pts=p).sig)
    assert gen(b, pts=p).sig.any()


def test_mutaciya_d_plus_1_lovitsya():
    """M-d1: правка дня d+1 и позже не меняет sig до дня d включительно."""
    b = sluchaynyj(5)
    c = b.copy()
    m = c["date"] > DAYS[DAY]
    for k in ("open", "high", "low", "close"):
        c.loc[m, k] = c.loc[m, k] * 2.0
    for p in ([tochka()], [tochka("T", "18:40")]):
        a, bb = gen(b, pts=p).sig, gen(c, pts=p).sig
        assert np.array_equal(a[:, :DAY + 1], bb[:, :DAY + 1])


def test_exit_cherez_tau_dnej():
    for tau in (1, 2, 3):
        ss = gen(proboj(S_POSLE_17, cena=110.0), pts=[tochka("T", "18:00", tau)])
        assert np.flatnonzero(ss.exit_l[0]).tolist() == [DAY + tau] == np.flatnonzero(ss.exit_s[0]).tolist()


def test_exit_za_koncom_lenty_ne_padaet():
    ss = V.semya("S11", {"AAA": proboj(S_POSLE_17, day=ND - 1)})(td_iz(), {}, [tochka("T", "18:00", 3)])
    assert ss.sig[0, ND - 1] == 1 and not ss.exit_l[0].any()


def test_tiker_vne_lenty_keyerror():
    with pytest.raises(KeyError):
        V.semya("S11", {"BBB": proboj(S_POSLE_17)})(td_iz("AAA"), {}, [tochka()])


@pytest.mark.parametrize("pt", [tochka("x"), tochka("T", "17:30"), tochka(tau=4), tochka(tau=0)])
def test_nedopustimaya_os_valueerror(pt):
    with pytest.raises(ValueError):
        gen(proboj(S_POSLE_17), pts=[pt])


def test_neizvestnyj_sid_valueerror():
    with pytest.raises(ValueError):
        V.semya("S9", {})


def test_chtenie_ne_dalshe_2020():
    b = proboj(S_POSLE_17, start="2019-03-01")
    with pytest.raises(ValueError):
        V.semya("S11", {"AAA": b})(td_iz(start="2019-03-01"), {}, [tochka()])


def test_stroka_net_v_lente_dayot_nol():
    """День td без 10-минуток — сигнал 0, не NaN и не перенос."""
    b = proboj(S_POSLE_17)
    b = b[b["date"] != DAYS[DAY]]
    assert gen(b, pts=[tochka("T", "18:00")]).sig[0, DAY] == 0
