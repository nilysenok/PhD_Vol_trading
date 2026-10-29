"""V6-12a: оценщик точки куба v6 (fastbt/eval_v6.py). Синтетика (без данных) + реальные train4 (skip без ряда 18:50).
Гейты: v6 со всеми новыми осями «выкл.» = v5 побайтно; каждая подключённая ось меняет сигнал; мутация «ось не доходит до
оценщика» ловится; ключ кэша содержит все оси v6 (Д5); неподключённая ось — ValueError."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import canon_v5, canon_v6, eval_v5, eval_v6, kontekst_v6
from fastbt.eval_v6 import EvaluatorV6, IstochnikiV6, fkey, razdelit, rel_td_ind
from fastbt.indicators import compute_base
from fastbt.signals.interface import TickerData, get_family
from fastbt.signals import v5_s12, v5_s34  # noqa: F401  (регистрация семей v5)

SP = dict(ma_tip="sma", ma_window=20, dev_tip="std", z_entry=1.0, z_exit=0.5)


def _td(n=700, seed=7):
    r = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(r.normal(0, 0.015, n)))
    o = c * (1 + r.normal(0, 0.003, n))
    h, l = np.maximum(o, c) * 1.004, np.minimum(o, c) * 0.996
    v = r.uniform(1e6, 2e6, n)
    d = pd.bdate_range("2018-01-02", periods=n).values
    lr = np.r_[np.log(c[1:] / c[:-1]), 0.0]
    return TickerData("T", d, o, h, l, c, v, lr, {}), compute_base(c, h, l, o, v)


def fake(v6=None, ist=None, filtr=None, sid="S1", sp=SP, vhod="stop_zayavka"):
    e = object.__new__(EvaluatorV6)
    e.sid, e.sk, e.v6, e.ist, e._rel, e._dan = sid, SimpleNamespace(_td={"T": _td()}), v6 or {}, ist or IstochnikiV6(), {}, {}
    e.r5 = SimpleNamespace(filtr=filtr or {}, nov={"vhod_tip": vhod}, sp=sp)
    return e


def sig(e, sp=SP):
    return np.asarray(e._signals5(sp)["T"].sigs[0])


def _baza():
    td, ind = _td()
    return get_family("S1", setka="v5")(td, ind, [SP]).sig[0]


def test_bez_osej_v6_ravno_v5():
    assert np.array_equal(sig(fake()), _baza()) and (sig(fake()) != 0).sum() > 20


def test_dlit_zhestche():
    s1, s3 = sig(fake({"dlit": 1})), sig(fake({"dlit": 3}))
    assert np.array_equal(s1, _baza()) and 0 < (s3 != 0).sum() < (s1 != 0).sum()
    assert np.all((s3 == 0) | (s3 == s1))


def test_asim_long_i_sdvig():
    b, lg = _baza(), sig(fake({"asim": "long"}))
    assert (lg < 0).sum() == 0 and np.array_equal(lg, np.where(b > 0, b, 0))
    assert np.array_equal(sig(fake({"asim": 0})), b) and not np.array_equal(sig(fake({"asim": 2})), b)


def test_norm_s252_menyaet_signal():
    assert not np.array_equal(sig(fake({"norm": "s252"})), _baza())


def _ist_vse():
    n = len(_td()[0].dates)
    d = kontekst_v6.Dannye("T", _td()[0].dates, rezhim={"ma200": np.arange(n) % 2 == 0})
    ix = pd.DatetimeIndex(_td()[0].dates)
    cbr = pd.DataFrame({"date": ix[[300, 400]], "known_from": ix[[0, 0]]})
    z = pd.Series(5.0, index=ix)
    return IstochnikiV6(dannye=lambda t, td: d, vn=lambda t, f: (z, z), cbr=cbr)


OSI = [{"rezhim": "imoex_ma200"}, {"rezhim": "rang_sigma_rynka_08"}, {"sob_tip": "cbr"}, {"vn_uslovie": "spokojno_1"}]


def _dohodit(v6, **kw) -> bool:
    """Ось дошла до оценщика: сигнал изменился против v5."""
    ist = kw.pop("ist", None) or _ist_vse()
    if v6.get("rezhim") == "rang_sigma_rynka_08":
        ist.dannye = lambda t, td: kontekst_v6.Dannye("T", td.dates, rezhim={"rank08": np.arange(td.n) % 3 == 0})
    return not np.array_equal(sig(fake(v6, ist, **kw)), _baza())


@pytest.mark.parametrize("v6", OSI, ids=lambda x: next(iter(x)) + "=" + str(next(iter(x.values()))))
def test_os_kontekst_sobytiya_vn_doshla(v6):
    assert _dohodit(v6)


def test_vn_spokojno_blokiruet_vse_vhody_i_mutaciya_lovitsya(monkeypatch):
    assert (sig(fake({"vn_uslovie": "spokojno_1"}, _ist_vse())) != 0).sum() == 0
    monkeypatch.setattr(EvaluatorV6, "_ok6", lambda self, t, td: (np.ones(td.n, bool), np.ones(td.n, bool)))
    assert not any(_dohodit(v) for v in OSI)                      # мутация «ось не доходит» — тест выше упал бы


def test_sobytie_cbr_blokiruet_den_pered_zasedaniem():
    ist = _ist_vse()
    s = sig(fake({"sob_tip": "cbr"}, ist))
    b = _baza()
    assert s[299] == 0 and s[399] == 0 and (s != b).sum() > 0 and np.all((s == 0) | (s == b))


def test_os_bez_istochnika_valueerror():
    for v6 in OSI + [{"rel_signal": "rynok"}]:
        with pytest.raises(ValueError, match="IstochnikiV6"):
            sig(fake(v6))
    with pytest.raises(ValueError, match="дивиденды не сверены"):
        sig(fake({"sob_tip": "div"}, _ist_vse()))


def test_rel_signal_i_dyry():
    td, _ = _td()
    k = 1 + 0.03 * np.sin(np.arange(td.n) / 3)
    df = pd.DataFrame({"date": td.dates, "open": td.open * k, "high": td.high * k, "low": td.low * k, "close": td.close * k,
                       "volume": td.volume})
    ist = IstochnikiV6(rel=lambda t, r: df)
    assert not np.array_equal(sig(fake({"rel_signal": "rynok"}, ist)), _baza())
    with pytest.raises(ValueError, match="нет 1 дней"):
        rel_td_ind(df.iloc[1:], td)


def test_filtry_v6_i_otobrazhenie():
    assert (sig(fake(filtr={"vol_regime_max": 0.01})) != 0).sum() == 0
    f = {}
    EvaluatorV6._filtr6(f, "naklon", 0.1, "kontr"), EvaluatorV6._filtr6(f, "tihiy_othod", "da", "kontr")
    assert f == {"slope_max": 0.1, "tihiy_othod": 1}


def test_razdelit_nepodklyuchennaya_i_vykl():
    pt5 = {"ma_tip": "sma"}
    for ose in ({"time_decay": "da"}, {"tf": "10m"}, {"pauza_porog": 0.5}):
        with pytest.raises(ValueError, match="ещё не подключена"):
            razdelit("S1", {**pt5, **ose})
    with pytest.raises(ValueError, match="не подключено"):
        razdelit("S1", {**pt5, "tejk": "pol_puti"})
    p, v6 = razdelit("S1", {**pt5, "dlit": 1, "asim": 0, "norm": "okno", "naklon": 0.1})
    assert p == pt5 and v6 == {"naklon": 0.1}


def test_klyuch_soderzhit_vse_osi_v6():
    ya = {}
    assert fkey({}, ya) == ()
    assert fkey({"adx_th": 25}, {}) == (("adx_th", 25),)
    ks = {fkey({}, {k: v}) for k, v in (("dlit", 2), ("asim", 1), ("norm", "s252"), ("sob_do", 2), ("vn_faktor", "rvi"))}
    assert len(ks) == 5 and fkey({}, {"dlit": 2}) != fkey({}, {"dlit": 3})
    for sid in canon_v5.SIDS:
        assert {k for k in eval_v6.PODKLYUCHENO if k in canon_v6.novye(canon_v6._kub(sid))} <= set(canon_v6.novye(canon_v6._kub(sid)))


# ---------------- реальные данные train4 ----------------
real = pytest.mark.skipif(not eval_v5.__dict__ or not __import__("fastbt.daily_1850", fromlist=["x"]).out_path().exists(),
                          reason="нет дневного ряда 18:50")


def _para(sid, **kw):
    from fastbt import etap1
    inp, base = etap1.build_inputs(sid, period="train4"), etap1.base_point(sid)
    inp.sk.tickers = list(inp.sk.tickers)[:3]
    return eval_v5.EvaluatorV5(sid, inp, base), EvaluatorV6(sid, inp, base, **kw)


@real
@pytest.mark.parametrize("sid", canon_v5.SIDS)
def test_tochka_v6_bez_osej_ravna_v5_pobaytno(sid):
    e5, e6 = _para(sid)
    pts, _ = canon_v5.tochki_v5(canon_v5.load_kub(sid), 0, 60)
    for _, p in pts[:20]:
        r5, x5, t5 = e5.evaluate_v2(p, "train4")
        off = {n: v["setka"][0] for n, v in canon_v6.novye(canon_v6._kub(sid)).items() if n in ("dlit", "asim", "norm", "rel_signal")}
        r6, x6, t6 = e6.evaluate_v2({**p, **off}, "train4")
        r5.pop("t_sek", None), r6.pop("t_sek", None)
        b = lambda x: None if x is None else x.tobytes()      # noqa: E731  (ne_schitana — рядов нет)
        assert repr(r5) == repr(r6) and b(x5) == b(x6) and b(t5) == b(t6)


@real
def test_osi_bez_istochnikov_menyayut_ryad_s1():
    """Оси v6 без внешних рядов (dlit, asim, norm, фильтры листа V6-4) меняют excess портфеля S1."""
    from fastbt.eval_v5_osi import yakor
    _, e6 = _para("S1")
    sp = dict(ma_tip="sma", ma_window=20, dev_tip="std", z_entry=1.5, z_exit=0.5)
    base = e6.evaluate_v2({**yakor("S1"), **sp}, "train4")[1]
    for n, v in dict(dlit=2, asim=1, norm="s252", hurst_th=0.45, naklon=0.1, vol_regime_max=0.8, tihiy_othod="da").items():
        r, x, _ = e6.evaluate_v2({**yakor("S1"), **sp, n: v}, "train4")
        assert r["status"] == "ok" and not np.array_equal(x, base), n
