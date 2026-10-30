"""Л2.1 (Ж-146 п. 2): ось otnositelno ∈ {net, sektor, imoex} для S1–S4. Только синтетика, без сети, данные < 2020 не читаются.

G1: net = исходные td и SignalSet побайтно. G2 (M-d1): сектор/IMOEX дня d в сигнале d ловится. G3: Д1 (максимум на lag 0).
G4: «нет коллег → IMOEX» записан флагом. Известный ответ: close_rel = close0 · Π(1 + r_i − r_b), O/H/L дня — тем же множителем."""
import numpy as np
import pandas as pd
import pytest

from fastbt import otnositelno_v7 as ot
from fastbt import rel_ryad
from fastbt.indicators import compute_base
from fastbt.signals import v5_s12, v5_s34  # noqa: F401  (регистрация семей v5)
from fastbt.signals.interface import TickerData, get_family

T = 420
TIK = ["A", "B", "C", "D"]
RNG = np.random.default_rng(7)
DATES = pd.bdate_range("2016-01-04", periods=T)
C = pd.DataFrame(100 * np.exp(np.cumsum(RNG.normal(0, 0.02, (T, 4)), axis=0)), index=DATES, columns=TIK)
IM = pd.Series(1000 * np.exp(np.cumsum(RNG.normal(0, 0.01, T))), index=DATES)
METKI = np.array([["X", "X", "Y", ""]] * T, dtype=object)         # A,B — сектор X; C — один в Y; D — без сектора
GRIDS = {"S1": [dict(ma_window=20, z_entry=1.0, z_exit=0.0), dict(ma_window=10, z_entry=1.5, z_exit="net")],
         "S2": [dict(bb_window=20, bb_std=1.5, podtv="none"), dict(bb_window=10, bb_std=1.0, podtv="none")],
         "S3": [dict(okno=20, podtv="none"), dict(okno=10, podtv="none")],
         "S4": [dict(period=10, mnozh=2.0), dict(period=14, mnozh=1.5)]}


def mk_td(c: pd.DataFrame, tik="A"):
    cc = c[tik].to_numpy()
    o = np.r_[cc[0], cc[:-1]] * 1.001
    h, lo = np.maximum(o, cc) * 1.004, np.minimum(o, cc) * 0.996
    v = np.full(len(cc), 1e4)
    lr = np.r_[np.log(cc[1:] / cc[:-1]), 0.0]
    td = TickerData(tik, c.index.to_numpy().astype("datetime64[ns]"), o, h, lo, cc, v, lr)
    return td, compute_base(cc, h, lo, o, v)


def sravnit(a, b):
    assert np.array_equal(a.sig, b.sig) and np.array_equal(a.exit_l, b.exit_l) and np.array_equal(a.exit_s, b.exit_s)
    assert a.strength.tobytes() == b.strength.tobytes()


def test_net_td_i_signaly_pobajtno():
    td, ind = mk_td(C)
    assert ot.primenit(td, "net", None) is td and ot.primenit(td, "net", None).digest() == td.digest()
    for sid, grid in GRIDS.items():
        fn = get_family(sid, "v5")
        ref = fn(td, ind, grid)
        assert ref.sig.any()
        got = ot.semya(fn, "net", None)(td, ind, grid)
        sravnit(got, ref)
        assert got.extra == ref.extra            # у net нет ни флага, ни добавок


def test_sektor_znanyj_otvet():
    fon, flag = ot.fon_ryad(C, IM, METKI, "sektor")
    rA, rB = C["A"].pct_change(), C["B"].pct_change()
    assert np.allclose(fon["A"].to_numpy()[1:], rB.to_numpy()[1:])          # коллега по сектору один — B
    td, _ = mk_td(C)
    rel = ot.td_rel(td, fon["A"].to_numpy())
    exp = C["A"].iloc[0] * np.cumprod(np.r_[1.0, 1 + (rA - rB).to_numpy()[1:]])
    assert np.allclose(rel.close, exp, rtol=1e-12)
    m = rel.close / td.close
    for x, y in ((rel.open, td.open), (rel.high, td.high), (rel.low, td.low)):
        assert np.allclose(x, y * m, rtol=1e-12)
    assert np.array_equal(rel.volume, td.volume) and rel.ticker == "A"
    assert np.allclose(rel.log_ret[:-1], np.log(rel.close[1:] / rel.close[:-1])) and rel.log_ret[-1] == 0


def test_imoex_znanyj_otvet():
    fon, flag = ot.fon_ryad(C, IM, METKI, "imoex")
    assert not flag.to_numpy().any()
    assert np.allclose(fon["C"].to_numpy()[1:], IM.pct_change().to_numpy()[1:])
    td, _ = mk_td(C, "C")
    rel = ot.td_rel(td, fon["C"].to_numpy())
    exp = C["C"].iloc[0] * np.cumprod(np.r_[1.0, 1 + (C["C"].pct_change() - IM.pct_change()).to_numpy()[1:]])
    assert np.allclose(rel.close, exp, rtol=1e-12)


def test_g4_net_kolleg_flag_zapisan():
    fon, flag = ot.fon_ryad(C, IM, METKI, "sektor")
    rm = IM.pct_change().to_numpy()
    for t in ("C", "D"):                                  # один в секторе / без сектора → IMOEX с флагом, не молча
        assert flag[t].iloc[1:].all() and np.allclose(fon[t].to_numpy()[1:], rm[1:])
    assert not flag["A"].any() and not flag["B"].any()
    td, ind = mk_td(C, "C")
    ss = ot.semya(get_family("S3", "v5"), "sektor", fon, flag)(td, ind, GRIDS["S3"])
    assert ss.extra["otn_imoex_vmesto_sektora"].all()


def test_den_kolleg_nan_den_imoex_s_flagom():
    c = C.copy()
    c.loc[c.index[100], "B"] = np.nan                    # в день 100 у A нет коллег
    fon, flag = ot.fon_ryad(c, IM, METKI, "sektor")
    assert flag["A"].iloc[100] and np.isclose(fon["A"].iloc[100], IM.pct_change().iloc[100])
    assert not flag["A"].iloc[50]


def prefiks_neizmenen(fon_fn, t0=250):
    """Решение ≤ t0 не должно меняться от рядов после t0 (сектор/IMOEX дня d+1 не должен попасть в d)."""
    c2, im2 = C.copy(), IM.copy()
    c2.iloc[t0 + 1:, :] *= 1.7
    im2.iloc[t0 + 1:] *= 0.5
    out = []
    for c, im in ((C, IM), (c2, im2)):
        fon, _ = fon_fn(c, im, METKI, "sektor")
        td, ind = mk_td(c)
        rel = ot.td_rel(td, fon["A"].to_numpy())
        ss = ot.semya(get_family("S1", "v5"), "sektor", fon)(td, ind, GRIDS["S1"])
        out.append((rel.close[:t0 + 1], ss.sig[:, :t0 + 1]))
    return np.allclose(out[0][0], out[1][0]) and np.array_equal(out[0][1], out[1][1])


def fon_mut_d1(c, imoex, metki, rezhim):
    """Мутация M-d1: фон дня d берётся с доходностью дня d+1."""
    fon, flag = ot.fon_ryad(c, imoex, metki, rezhim)
    return fon.shift(-1), flag


def test_m_d1_prefiks_i_mutaciya_lovitsya():
    assert prefiks_neizmenen(ot.fon_ryad)
    assert not prefiks_neizmenen(fon_mut_d1)


def test_g3_d1_max_na_lag0_i_mutaciya():
    fon, _ = ot.fon_ryad(C, IM, METKI, "sektor")
    td, _ = mk_td(C)
    rel = ot.td_rel(td, fon["A"].to_numpy())
    df = pd.DataFrame({"date": C.index, "close": rel.close})
    ref = C["A"].pct_change() - fon["A"]
    res = rel_ryad.proverka_d1(df, ref)
    assert max(res, key=res.get) == 0
    with pytest.raises(ValueError):
        rel_ryad.proverka_d1(df, ref.shift(-1))


def test_nan_fon_ne_podmenyaetsya_molcha():
    td, _ = mk_td(C)
    fon = np.full(T, np.nan)
    with pytest.raises(ValueError):
        ot.td_rel(td, fon)
    with pytest.raises(ValueError):
        ot.fon_ryad(C, IM, METKI, "rynok")
    with pytest.raises(ValueError):
        ot.primenit(td, "rynok", None)


def test_nulevoj_fon_daet_td_pobajtno():
    td, _ = mk_td(C)
    rel = ot.td_rel(td, np.zeros(T))
    assert np.allclose(rel.close, td.close, rtol=1e-12) and np.allclose(rel.high, td.high, rtol=1e-12)


def test_metki_iz_csv_karta_pit():
    m = ot.metki_iz_csv(ot.KARTA_CSV, DATES, ["AFLT", "ZZZZ"])
    assert (m[:, 0] == "MOEXTN").all() and (m[:, 1] == "").all()
