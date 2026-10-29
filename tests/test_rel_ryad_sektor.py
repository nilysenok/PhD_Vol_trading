"""V6-1b: rel_ryad — сектор PIT, откат на IMOEX с флагом, режимы rel_signal, Д1. Синтетика; мутации ловятся."""
import numpy as np
import pandas as pd
import pytest

from fastbt import daily_1850 as d18
from fastbt import rel_ryad as R
from fastbt.e1_rel_sektor import sektor_pit

BPD = 54  # баров 09:50…18:40 в дне


def _bars(days=8, seed=1, sigma=0.002, p0=100.0):
    rng = np.random.default_rng(seed)
    ts = []
    for d in pd.bdate_range("2021-03-01", periods=days):
        ts += list(pd.date_range(d + pd.Timedelta(hours=9, minutes=50), d + pd.Timedelta(hours=18, minutes=40), freq="10min"))
    n = len(ts)
    c = p0 * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    o = np.r_[p0, c[:-1]]
    return pd.DataFrame({"timestamp": ts, "open": o, "high": np.maximum(o, c) * 1.0005,
                         "low": np.minimum(o, c) * 0.9995, "close": c, "volume": rng.integers(10, 1000, n)})


def _day(i, days=8):
    return pd.bdate_range("2021-03-01", periods=days)[i]


@pytest.fixture
def mk():
    return {"T": _bars(seed=1), "A": _bars(seed=2, p0=50.0), "B": _bars(seed=3, p0=70.0), "M": _bars(seed=4, p0=3000.0)}


def test_ves_period_odin_sektor_daet_td_rel_pobaytno(mk):
    rows = [(pd.Timestamp("2021-01-01"), "A")]
    got, fl = R.td_rel_sektor(mk["T"], mk["M"], {"A": mk["A"]}, rows, "T")
    pd.testing.assert_frame_equal(got, R.td_rel(mk["T"], mk["A"], "T"), check_exact=True)
    assert (fl["flag"] == "").all() and (fl["benchmark"] == "A").all()


def test_otkat_na_rynok_pobaytno_i_s_flagom(mk):
    got, fl = R.td_rel_sektor(mk["T"], mk["M"], {"A": mk["A"]}, [], "T")
    pd.testing.assert_frame_equal(got, R.td_rel(mk["T"], mk["M"], "T"), check_exact=True)
    assert (fl["flag"] == R.F_DO_PERVOJ).all() and (fl["benchmark"] == "IMOEX").all() and len(fl) == 8


def test_flagi_net_sektora_i_net_indeksa_ne_molchat(mk):
    for name, flag in ((None, R.F_NET_SEKTORA), ("MOEXTN", R.F_NET_INDEKSA)):
        got, fl = R.td_rel_sektor(mk["T"], mk["M"], {"A": mk["A"]}, [(pd.Timestamp("2021-01-01"), name)], "T")
        assert (fl["flag"] == flag).all()
        pd.testing.assert_frame_equal(got, R.td_rel(mk["T"], mk["M"], "T"), check_exact=True)


def test_pit_sostav_so_sleduyushchej_rebalansirovki_lovitsya():
    rows = [(pd.Timestamp("2021-03-03"), "A"), (pd.Timestamp("2021-03-05"), "B")]
    have = {"A": 1, "B": 1}
    assert R.indeks_dnya(rows, "2021-03-02", have) == (None, R.F_DO_PERVOJ)
    assert R.indeks_dnya(rows, "2021-03-03", have) == ("A", "")
    assert R.indeks_dnya(rows, "2021-03-04", have) == ("A", "")   # мутация «следующая ребалансировка» даст B
    assert R.indeks_dnya(rows, "2021-03-05", have) == ("B", "")


def test_smena_benchmarka_sshivaetsya_po_nochnomu_perehodu_novogo(mk):
    rows = [(_day(3), "A")]                          # дни 0–2 — IMOEX, с дня 3 — сектор A
    got, fl = R.td_rel_sektor(mk["T"], mk["M"], {"A": mk["A"]}, rows, "T")
    assert list(fl["benchmark"][:4]) == ["IMOEX"] * 3 + ["A"]
    t, a = d18.aggregate(mk["T"], "T"), d18.aggregate(mk["A"], "A")
    want = (t["open"][3] / t["close"][2]) * (a["close"][2] / a["open"][3])
    assert got["open"][3] / got["close"][2] == pytest.approx(want, rel=1e-12)
    # внутри дня 3 — как у чистого сектора, с постоянным множителем
    pure = R.td_rel(mk["T"], mk["A"], "T")
    ratio = got["close"][3:] / pure["close"][3:]
    assert ratio.to_numpy() == pytest.approx(ratio.iloc[0], rel=1e-12)


def test_dni_do_smeny_ne_zavisyat_ot_budushchego_benchmarka(mk):
    rows = [(_day(5), "A")]
    full, _ = R.td_rel_sektor(mk["T"], mk["M"], {"A": mk["A"]}, rows, "T")
    cut = {k: v[v["timestamp"] < _day(5)] for k, v in mk.items()}
    part, _ = R.td_rel_sektor(cut["T"], cut["M"], {"A": cut["A"]}, rows, "T")
    pd.testing.assert_frame_equal(full.iloc[:5].reset_index(drop=True), part.reset_index(drop=True), check_exact=True)


def test_rezhimy(mk):
    pd.testing.assert_frame_equal(R.td_rel_rezhim("syroy", mk["T"], "T")[0], d18.aggregate(mk["T"], "T"), check_exact=True)
    pd.testing.assert_frame_equal(R.td_rel_rezhim("rynok", mk["T"], "T", mk["M"])[0], R.td_rel(mk["T"], mk["M"], "T"), check_exact=True)
    td, fl = R.td_rel_rezhim("sektor", mk["T"], "T", mk["M"], {"A": mk["A"]}, [(pd.Timestamp("2021-01-01"), "A")])
    pd.testing.assert_frame_equal(td, R.td_rel(mk["T"], mk["A"], "T"), check_exact=True)
    assert fl is not None
    with pytest.raises(ValueError):
        R.td_rel_rezhim("beta", mk["T"], "T", mk["M"])


@pytest.mark.parametrize("rezhim", ["sektor", "rynok"])
def test_bez_imoex_ne_stanovitsya_syroy(mk, rezhim):
    with pytest.raises(ValueError):
        R.td_rel_rezhim(rezhim, mk["T"], "T", None, {"A": mk["A"]}, [(pd.Timestamp("2021-01-01"), "A")])


def test_pit_csv_realnye_flagi():
    pit = sektor_pit()
    have = {"MOEXOG": 1, "MOEXFN": 1}                # MOEXTN, MOEXRE в данных нет
    assert R.indeks_dnya(pit["AFLT"], "2016-06-01", have) == (None, R.F_DO_PERVOJ)
    assert R.indeks_dnya(pit["AFLT"], "2018-06-01", have) == (None, R.F_NET_INDEKSA)
    assert R.indeks_dnya(pit["LSRG"], "2018-06-01", have) == (None, R.F_NET_SEKTORA)
    assert R.indeks_dnya(pit["LSRG"], "2017-03-19", have)[1] == R.F_DO_PERVOJ
    assert R.indeks_dnya(pit["LSRG"], "2017-03-20", have)[1] == R.F_NET_SEKTORA


def _ref(t, i):
    a, b = d18.aggregate(t, "x"), d18.aggregate(i, "y")
    return pd.Series(np.log(a["close"].to_numpy()), index=a["date"]).diff() - pd.Series(np.log(b["close"].to_numpy()), index=b["date"]).diff()


def test_d1_pik_na_nule_i_mutaciya_index_d_plyus_1_blokiruetsya():
    t, i = _bars(days=14, seed=5, sigma=0.001), _bars(days=14, seed=6, sigma=0.004, p0=500.0)
    res = R.proverka_d1(R.td_rel(t, i, "T"), _ref(t, i))
    assert max(res, key=res.get) == 0 and res[0] == pytest.approx(1.0, abs=1e-9) and set(res) == {-2, -1, 0, 1}
    m = i.copy()
    m[["open", "high", "low", "close"]] = i[["open", "high", "low", "close"]].shift(-BPD).ffill().to_numpy()
    with pytest.raises(ValueError, match="lag 0"):
        R.proverka_d1(R.td_rel(t, m, "T"), _ref(t, i))


def test_d1_mala_vyborka_ne_prohodit_molcha():
    t, i = _bars(days=2, seed=5), _bars(days=2, seed=6)
    with pytest.raises(ValueError):
        R.proverka_d1(R.td_rel(t, i, "T"), _ref(t, i))
