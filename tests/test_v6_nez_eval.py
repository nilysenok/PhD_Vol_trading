"""Независимая проверка PR #672 (V6-12a, fastbt/eval_v6.py), тестировщик (не автор).
(1) v5 ⊂ v6 побайтно на посчитанных точках (все новые оси куба — «выкл.»), счёт ok/ne_schitana по Sk;
(2) каждая подключённая ось доходит до результата — пары точек с известным эффектом на синтетике и источниках IstochnikiV6;
(3) fkey различает каждую ось v6 (Д5); (4) неподключённая ось — ValueError; (5) нет ряда в IstochnikiV6/Dannye — ValueError.
Мутации (в eval_v6.py по одной): ось не передаётся в сигнал; ось выпала из fkey; контекст на день d+1 — каждая роняет файл."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import canon_v5, canon_v6, eval_v5, eval_v6, kontekst_v6
from fastbt.eval_v6 import EvaluatorV6, IstochnikiV6, fkey, rel_td_ind
from fastbt.indicators import compute_base
from fastbt.signals import v5_s12, v5_s34  # noqa: F401  (регистрация семей v5)
from fastbt.signals.interface import TickerData, get_family
from fastbt.signals.v6_obertka import usloviya

SP = dict(ma_tip="sma", ma_window=20, dev_tip="std", z_entry=0.5, z_exit=0.5)
MODIF = {"ma_n", "porog_shiriny", "relf_k", "relf_okno", "relf_bench"}    # параметры другой оси, сами не включают контекст


# ---------------------------------------------------------------- синтетика
def _td(n=700, seed=11):
    r = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(r.normal(0, 0.015, n)))
    o = c * (1 + r.normal(0, 0.003, n))
    h, l = np.maximum(o, c) * 1.004, np.minimum(o, c) * 0.996
    v = r.uniform(1e6, 2e6, n)
    d = pd.bdate_range("2018-01-02", periods=n).values
    lr = np.r_[np.log(c[1:] / c[:-1]), 0.0]
    return TickerData("T", d, o, h, l, c, v, lr, {}), compute_base(c, h, l, o, v)


TD, IND = _td()
N = TD.n
IX = pd.DatetimeIndex(TD.dates)
FAM = get_family("S1", setka="v5")


def base(sp=SP):
    return np.asarray(FAM(TD, IND, [dict(sp)]).sig[0])


B = base()


def ev(v6=None, ist=None, filtr=None, sp=SP):
    e = object.__new__(EvaluatorV6)
    e.sid, e.sk, e.v6, e.ist, e._rel, e._dan = "S1", SimpleNamespace(_td={"T": (TD, IND)}), v6 or {}, ist or IstochnikiV6(), {}, {}
    e.r5 = SimpleNamespace(filtr=filtr or {}, nov={"vhod_tip": "stop_zayavka"}, sp=sp)
    return e


def sg(e, sp=SP):
    return np.asarray(e._signals5(sp)["T"].sigs[0])


def where_ok(ok):
    return np.where(ok, B, 0)


def test_baza_plotnaya():
    assert (B > 0).sum() > 40 and (B < 0).sum() > 40 and np.array_equal(sg(ev()), B)


# ---------------------------------------------------------------- (2) оси доходят до результата
def _flag(par=0):
    return np.arange(N) % 2 == par


@pytest.mark.parametrize("name,key", [("imoex_ma200", "ma200"), ("rang_sigma_rynka_08", "rank08")])
def test_rezhim_den_d_a_ne_d_plus_1(name, key):
    fl = _flag(0)
    ist = IstochnikiV6(dannye=lambda t, td: kontekst_v6.Dannye(t, td.dates, rezhim={key: fl}))
    s = sg(ev({"rezhim": name}, ist))
    assert np.array_equal(s, where_ok(fl)) and (s != B).sum() > 40 and (s != 0).sum() > 40
    assert np.all(s[~fl] == 0)                              # вход в день d разрешает флаг дня d, не d+1


def test_kontekst_mask_primenyaetsya_k_storonam_otdelno(monkeypatch):
    lo, sr = _flag(0), _flag(1)
    monkeypatch.setattr(eval_v6.kontekst_v6, "long_short", lambda kp, d: (lo, sr))
    s = sg(ev({"shirina": "protiv"}, IstochnikiV6(dannye=lambda t, td: None)))
    assert np.array_equal(s, np.where(((B > 0) & ~lo) | ((B < 0) & ~sr), 0, B)) and (s > 0).sum() and (s < 0).sum()


@pytest.mark.parametrize("k", eval_v6.KONTEKST)
def test_kazhdaya_os_konteksta_peredayotsya_v_modul(k, monkeypatch):
    nov = {}
    for sid in canon_v5.SIDS:
        nov.update({a: v["setka"] for a, v in canon_v6.novye(canon_v6._kub(sid)).items()})
    x = nov[k][1]
    seen = []
    monkeypatch.setattr(eval_v6.kontekst_v6, "long_short", lambda kp, d: (seen.append(dict(kp)), (_flag(0), _flag(0)))[1])
    s = sg(ev({k: x}, IstochnikiV6(dannye=lambda t, td: None)))
    assert len(seen) == 1 and seen[0][k] == (eval_v6.REZHIM_ALIAS.get(x, x) if k == "rezhim" else x)
    assert np.array_equal(s, where_ok(_flag(0)))


def _dan_real():
    r = np.random.default_rng(3)
    s = lambda a: a * np.exp(r.normal(0, .2, N))                  # noqa: E731
    return kontekst_v6.Dannye("T", TD.dates, close=TD.close, high=TD.high, low=TD.low, volume=TD.volume,
                              sig5=s(0.015 + 0 * TD.close), sig22=s(0.015 + 0 * TD.close), sig66=s(0.015 + 0 * TD.close),
                              Z=r.normal(0, 1, N))


@pytest.mark.parametrize("v6", [{"szhatie": "da"}, {"rv_perevorot": "da"}, {"obyom_anom": "da"}, {"post_shok": "2:0.5"},
                                {"szhatie": "da", "obyom_anom": "da"}])
def test_realnyj_kontekst_k21_k23_k20(v6):
    d = _dan_real()
    lo, sr = kontekst_v6.long_short(v6, d)                  # эталон: тот же модуль, прямой вызов
    exp = np.where(((B > 0) & ~lo) | ((B < 0) & ~sr), 0, B)
    s = sg(ev(v6, IstochnikiV6(dannye=lambda t, td: d)))
    assert np.array_equal(s, exp) and 0 < (s != B).sum()


def _cbr(idx, kf=0):
    return pd.DataFrame({"date": IX[idx], "known_from": IX[[kf] * len(idx)]})


def _jbl():
    j = next(i for i in range(10, N - 10) if (B[i - 3:i + 3] != 0).all())
    return j


@pytest.mark.parametrize("v6,okno", [({"sob_tip": "cbr"}, (-1, -1)), ({"sob_tip": "cbr", "sob_do": 3}, (-3, -1)),
                                     ({"sob_tip": "cbr", "sob_posle": 2}, (-1, 1))])
def test_sobytie_cbr_okno(v6, okno):
    j = _jbl()
    s = sg(ev(v6, IstochnikiV6(cbr=_cbr([j]))))
    bl = np.zeros(N, bool)
    bl[j + okno[0]:j + okno[1] + 1] = True
    assert np.array_equal(s, np.where(bl, 0, B)) and (s != B).sum() == bl.sum() and bl.sum() > 0


def test_sobytie_cbr_point_in_time_i_div():
    j = _jbl()
    s = sg(ev({"sob_tip": "cbr"}, IstochnikiV6(cbr=_cbr([j], kf=j + 5))))      # заседание объявлено позже d — не блокирует
    assert np.array_equal(s, B)
    div = pd.DataFrame({"ticker": ["T", "X"], "last_cum_date": [IX[j], IX[j + 1]], "announce_date": [IX[0], IX[0]]})
    s = sg(ev({"sob_tip": "div"}, IstochnikiV6(div=div, div_verified=True)))
    bl = np.zeros(N, bool)
    bl[j] = True
    assert np.array_equal(s, np.where(bl, 0, B)) and (s != B).sum() == 1
    j2 = next(i for i in range(j + 20, N - 10) if B[i - 1] != 0)
    both = IstochnikiV6(cbr=_cbr([j2]), div=div, div_verified=True)
    s = sg(ev({"sob_tip": "cbr_div"}, both))
    bl[j2 - 1] = True
    assert np.array_equal(s, np.where(bl, 0, B)) and (s != B).sum() == 2


def _vn(zv, bv=1.0, calls=None):
    def f(t, fak):
        if calls is not None:
            calls.append(fak)
        return (pd.Series(zv, index=IX) if np.isscalar(zv) else zv), pd.Series(bv, index=IX)
    return f


def test_vn_spokojno_1_i_2():
    k1, k2 = _jbl(), _jbl() + 8
    z = pd.Series(0.0, index=IX)
    z.iloc[k1], z.iloc[k2] = 1.5, 3.0
    s1 = sg(ev({"vn_uslovie": "spokojno_1"}, IstochnikiV6(vn=_vn(z))))
    s2 = sg(ev({"vn_uslovie": "spokojno_2"}, IstochnikiV6(vn=_vn(z))))
    b1, b2 = np.zeros(N, bool), np.zeros(N, bool)
    b1[[k1, k2]], b2[k2] = True, True
    assert np.array_equal(s1, np.where(b1, 0, B)) and np.array_equal(s2, np.where(b2, 0, B)) and B[k1] != 0 and B[k2] != 0


def test_vn_po_bete_i_protiv_bety_po_storonam():
    s = sg(ev({"vn_uslovie": "po_bete"}, IstochnikiV6(vn=_vn(1.0, 1.0))))
    assert np.array_equal(s, np.where(B < 0, 0, B)) and (s > 0).sum() > 40
    s = sg(ev({"vn_uslovie": "protiv_bety"}, IstochnikiV6(vn=_vn(1.0, 1.0))))
    assert np.array_equal(s, np.where(B > 0, 0, B)) and (s < 0).sum() > 40


def test_vn_faktor_dohodit_do_istochnika():
    c = []
    sg(ev({"vn_uslovie": "spokojno_1"}, IstochnikiV6(vn=_vn(0.0, calls=c))))
    sg(ev({"vn_uslovie": "spokojno_1", "vn_faktor": "rvi"}, IstochnikiV6(vn=_vn(0.0, calls=c))))
    assert c == ["brent", "rvi"]


def test_os_peresechenie_sobytiya_i_vn_po_i():
    j = _jbl()
    ist = IstochnikiV6(cbr=_cbr([j]), vn=_vn(1.0, 1.0))
    s = sg(ev({"sob_tip": "cbr", "vn_uslovie": "po_bete"}, ist))
    exp = np.where(B < 0, 0, B)
    exp[j - 1] = 0
    assert np.array_equal(s, exp)


def _kdf(k, sdv=0.0):
    f = k * (1 + sdv * np.sin(np.arange(N) / 5))
    return pd.DataFrame({"date": TD.dates, "open": TD.open * f, "high": TD.high * f, "low": TD.low * f, "close": TD.close * f,
                         "volume": TD.volume})


def test_rel_signal_istochnik_i_rezhim():
    calls = []
    dfs = {"sektor": _kdf(1.0, 0.05), "rynok": _kdf(1.0, 0.12)}
    ist = IstochnikiV6(rel=lambda t, r: (calls.append((t, r)), dfs[r])[1])
    out = {}
    for r in ("sektor", "rynok"):
        out[r] = sg(ev({"rel_signal": r}, ist))
        ts, is_ = rel_td_ind(dfs[r], TD)
        assert np.array_equal(out[r], np.asarray(FAM(ts, is_, [dict(SP)]).sig[0]))
    assert not np.array_equal(out["sektor"], B) and not np.array_equal(out["sektor"], out["rynok"])
    assert calls == [("T", "sektor"), ("T", "rynok")]
    calls.clear()
    assert np.array_equal(sg(ev({"rel_signal": "syroy"}, ist)), B) and not calls


def test_rel_signal_dyra_v_td_rel_valueerror():
    df = _kdf(1.0)
    df = df[df["date"] != df["date"].iloc[300]]
    with pytest.raises(ValueError):
        sg(ev({"rel_signal": "rynok"}, IstochnikiV6(rel=lambda t, r: df)))


def _run(c):
    r = np.zeros(len(c), int)
    for i in range(len(c)):
        r[i] = (r[i - 1] + 1 if i else 1) if c[i] else 0
    return r


@pytest.mark.parametrize("k", [1, 2, 3])
def test_dlit_k_dnej_podryad(k):
    cl, cs = usloviya("S1", TD, IND, SP)
    exp = np.where(((B > 0) & (_run(cl) >= k)) | ((B < 0) & (_run(cs) >= k)), B, 0)
    s = sg(ev({"dlit": k}))
    assert np.array_equal(s, exp) and (k == 1 or (s != B).sum() > 5)


def test_asim_napravlenie_poroga_shorta():
    lg = lambda x: x > 0                                                    # noqa: E731
    s0, sp1, sm1 = (sg(ev({"asim": a})) for a in (0, 1, -1))
    assert np.array_equal(s0, B)
    for s in (sp1, sm1):
        assert np.array_equal(lg(s), lg(B))                                 # лонги не тронуты
    assert (sp1 < 0).sum() < (B < 0).sum() and np.all(B[sp1 < 0] < 0)       # шорт строже (порог z выше)
    assert (sm1 < 0).sum() > (B < 0).sum() and np.all(B[B < 0] < 0)
    assert np.array_equal(sg(ev({"asim": "long"})), np.where(B > 0, B, 0))


def test_norm_s252_ravno_semye_s_norm():
    ps = {**SP, "norm": "s252"}
    exp = np.asarray(FAM(TD, IND, [ps]).sig[0])
    s = sg(ev({"norm": "s252"}))
    assert np.array_equal(s, exp) and not np.array_equal(s, B)


def test_filtry_v5_i_v6_primenyayutsya_k_signalu():
    assert (sg(ev(filtr={"vol_regime_max": 0.01})) != 0).sum() == 0
    exp = np.where(eval_v6.maska(TD, "kontr", {"hurst_th": 0.5}, B), B, 0)
    s = sg(ev(filtr={"hurst_th": 0.5}))
    assert np.array_equal(s, exp)


def test_kombinaciya_asim_dlit_kontekst_porjadok():
    fl = _flag(0)
    ist = IstochnikiV6(dannye=lambda t, td: kontekst_v6.Dannye(t, td.dates, rezhim={"ma200": fl}))
    s = sg(ev({"asim": "long", "dlit": 2, "rezhim": "imoex_ma200"}, ist))
    cl, _ = usloviya("S1", TD, IND, SP)
    exp = np.where((B > 0) & (_run(cl) >= 2) & fl, B, 0)
    assert np.array_equal(s, exp) and (s != 0).sum() > 5


# ---------------------------------------------------------------- состояние через evaluate_v2 (заглушка базового)
def _pt5(sid):
    return canon_v5.tochki_v5(canon_v5.load_kub(sid), 0, 5)[0][0][1]


def _ev2(monkeypatch, sid, pt):
    cap = []
    monkeypatch.setattr(eval_v6.EvaluatorV2, "evaluate_v2", lambda self, p, period: cap.append(p) or ("stub", p))
    e = object.__new__(EvaluatorV6)
    e.sid, e.ist = sid, IstochnikiV6()
    e.evaluate_v2(pt, "train4")
    return e, cap[0]


def _podkl(sid):
    nv = canon_v6.novye(canon_v6._kub(sid))
    return [(k, nv[k]["setka"]) for k in nv if k in eval_v6.PODKLYUCHENO]


@pytest.mark.parametrize("sid", canon_v5.SIDS)
def test_fkey_razlichaet_kazhduyu_os_i_znachenie(sid, monkeypatch):
    p5 = _pt5(sid)
    e0, _ = _ev2(monkeypatch, sid, p5)
    k0 = e0._fkey
    assert k0 == tuple(sorted(e0.r5.filtr.items())) and e0.v6 == {}
    for k, setka in _podkl(sid):
        e, _ = _ev2(monkeypatch, sid, {**p5, k: setka[0]})
        assert e._fkey == k0 and e.v6 == {}, (sid, k, "значение «выкл.» обязано давать ключ v5")
        keys = {}
        for x in setka[1:]:
            e, _ = _ev2(monkeypatch, sid, {**p5, k: x})
            assert e._fkey != k0, (sid, k, x, "ось не влияет на ключ кэша")
            assert e._fkey not in keys.values(), (sid, k, x, "два значения оси — один ключ")
            keys[x] = e._fkey
            assert (e.v6.get(k) == x) or k in eval_v6.FILTR6, (sid, k, x, "ось не дошла в e.v6")


def test_fkey_pary_osej_i_filtra():
    ks = {fkey({}, v) for v in ({"dlit": 2, "asim": 1}, {"dlit": 2}, {"asim": 1}, {"dlit": 1, "asim": 2}, {"asim": 2, "dlit": 1})}
    assert len(ks) == 4 and fkey({}, {"asim": 2, "dlit": 1}) == fkey({}, {"dlit": 1, "asim": 2})
    assert fkey({"adx_th": 25}, {"dlit": 2}) != fkey({"adx_th": 25}, {}) != fkey({}, {"dlit": 2})
    assert fkey({}, {"sob_do": 2}) != fkey({}, {"sob_posle": 2}) != fkey({}, {"sob_do": 2, "sob_posle": 0})


@pytest.mark.parametrize("sid", ["S3", "S4"])
def test_vhod_tip_dohodit(sid, monkeypatch):
    p5 = _pt5(sid)
    e, cap = _ev2(monkeypatch, sid, {**p5, "vhod_tip": "podtverzhdenie"})
    assert e.v6 == {"vhod_tip": "podtverzhdenie"} and e.r5.nov["vhod_tip"] == "podtverzhdenie"
    _, off = _ev2(monkeypatch, sid, p5)
    assert off.get("vhod_tip") == "stop_zayavka" and cap.get("vhod_tip", "podtverzhdenie") == "podtverzhdenie"   # умолчание v2


def test_filtry_v6_dohodyat_v_r5_filtr(monkeypatch):
    p5 = _pt5("S1")
    e, _ = _ev2(monkeypatch, "S1", {**p5, "hurst_th": 0.45, "naklon": 0.1, "vol_regime_max": 0.9, "tihiy_othod": "da"})
    assert e.r5.filtr["hurst_th"] == 0.45 and e.r5.filtr["vol_regime_max"] == 0.9 and e.r5.filtr["tihiy_othod"] == 1
    assert {"slope_min", "slope_max"} & set(e.r5.filtr)


# ---------------------------------------------------------------- (4) неподключённая ось
@pytest.mark.parametrize("sid", canon_v5.SIDS)
def test_nepodklyuchennaya_os_valueerror(sid, monkeypatch):
    p5 = _pt5(sid)
    nv = canon_v6.novye(canon_v6._kub(sid))
    n = 0
    for k in nv:
        if k in eval_v6.PODKLYUCHENO:
            continue
        for x in nv[k]["setka"][1:]:
            with pytest.raises(ValueError, match="ещё не подключена"):
                _ev2(monkeypatch, sid, {**p5, k: x})
            n += 1
    assert n >= 20, sid
    for name, r in canon_v6.rasshireno(canon_v6._kub(sid)).items():
        for x in r["v6_dobavleno"]:
            with pytest.raises(ValueError, match="не подключено"):
                _ev2(monkeypatch, sid, {**p5, name: x})


def test_vyklyuchennye_neprilyuchennye_osi_ne_meshayut(monkeypatch):
    for sid in canon_v5.SIDS:
        nv = canon_v6.novye(canon_v6._kub(sid))
        _ev2(monkeypatch, sid, {**_pt5(sid), **{k: v["setka"][0] for k, v in nv.items()}})


# ---------------------------------------------------------------- (5) нет ряда
OSI_ISTOCHNIK = [{"rezhim": "imoex_ma200"}, {"sektor_trend": "protiv"}, {"post_shok": "3:1"}, {"rel_signal": "sektor"},
                 {"rel_signal": "rynok"}, {"sob_tip": "cbr"}, {"sob_tip": "cbr_div"}, {"vn_uslovie": "po_bete"}]


@pytest.mark.parametrize("v6", OSI_ISTOCHNIK, ids=lambda v: "-".join(f"{k}={x}" for k, x in v.items()))
def test_net_istochnika_valueerror(v6):
    with pytest.raises(ValueError, match="IstochnikiV6"):
        sg(ev(v6))
    if "sob_tip" in v6:
        with pytest.raises(ValueError, match="IstochnikiV6"):
            sg(ev(v6, IstochnikiV6(cbr=pd.DataFrame({"date": [], "known_from": []}))))


def test_net_div_i_ne_sverennye_dividendy_valueerror():
    div = pd.DataFrame({"ticker": ["T"], "last_cum_date": [IX[100]], "announce_date": [IX[0]]})
    with pytest.raises(ValueError, match="не сверены"):
        sg(ev({"sob_tip": "div"}, IstochnikiV6(div=div)))
    with pytest.raises(ValueError, match="IstochnikiV6"):
        sg(ev({"sob_tip": "div"}, IstochnikiV6(div_verified=True)))


@pytest.mark.parametrize("v6", [{"rezhim": "imoex_ma200"}, {"rezhim": "rang_sigma_rynka_08"}, {"szhatie": "da"},
                                {"rv_perevorot": "da"}, {"obyom_anom": "da"}, {"post_shok": "3:1"}, {"sektor_trend": "protiv"},
                                {"shirina": "po_trendu"}, {"relf_napr": "protiv"}], ids=lambda v: "-".join(v))
def test_net_ryada_v_dannye_valueerror(v6):
    ist = IstochnikiV6(dannye=lambda t, td: kontekst_v6.Dannye(t, td.dates, rezhim={"ma200": _flag(0)}))
    if v6.get("rezhim") == "imoex_ma200":
        sg(ev(v6, ist))                                        # ряд есть — считается
        return
    with pytest.raises(ValueError):
        sg(ev(v6, ist))


# ---------------------------------------------------------------- (1) вложенность v5 ⊂ v6 на реальных train4
real = pytest.mark.skipif(not __import__("fastbt.daily_1850", fromlist=["x"]).out_path().exists(), reason="нет ряда 18:50")
CHET = {}


@real
@pytest.mark.parametrize("sid", canon_v5.SIDS)
def test_v5_vlozhen_v_v6_pobaytno_vse_novye_osi_vykl(sid):
    from fastbt import etap1
    inp, bp = etap1.build_inputs(sid, period="train4"), etap1.base_point(sid)
    inp.sk.tickers = list(inp.sk.tickers)[:3]
    e5, e6 = eval_v5.EvaluatorV5(sid, inp, bp), EvaluatorV6(sid, inp, bp)
    pts, _ = canon_v5.tochki_v5(canon_v5.load_kub(sid), 0, 60)
    off = {k: v["setka"][0] for k, v in canon_v6.novye(canon_v6._kub(sid)).items()}
    ok = ne = 0
    for _, p in pts[:20]:
        r5, x5, t5 = e5.evaluate_v2(p, "train4")
        r6, x6, t6 = e6.evaluate_v2({**p, **off}, "train4")
        r6b, x6b, t6b = e6.evaluate_v2(dict(p), "train4")        # точка v5 без осей v6 вообще
        for r in (r5, r6, r6b):
            r.pop("t_sek", None)
        assert r5["status"] == r6["status"] == r6b["status"], (sid, p)
        b = lambda x: None if x is None else x.tobytes()         # noqa: E731
        assert repr(r5) == repr(r6) == repr(r6b) and b(x5) == b(x6) == b(x6b) and b(t5) == b(t6) == b(t6b), (sid, p)
        ok += r5["status"] == "ok"
        ne += r5["status"] != "ok"
    CHET[sid] = (ok, ne)
    print(f"\n[вложенность] {sid}: посчитано ok={ok}, ne_schitana={ne} из {ok + ne}")
    assert ok >= 5, f"{sid}: слишком мало реально посчитанных точек ({ok}) — вложенность не доказана"


@real
def test_realnyj_konvejer_kontekst_zapreshchaet_vse_vhody():
    """Сквозной (evaluate_v2, S1, train4): контекст «всё запрещено» → сделок нет; «всё разрешено» → ряд = v5 побайтно."""
    from fastbt import etap1
    from fastbt.eval_v5_osi import yakor
    inp, bp = etap1.build_inputs("S1", period="train4"), etap1.base_point("S1")
    inp.sk.tickers = list(inp.sk.tickers)[:3]
    sp = dict(ma_tip="sma", ma_window=20, dev_tip="std", z_entry=1.5, z_exit=0.5)
    pt = {**yakor("S1"), **sp}
    r5, x5, _ = eval_v5.EvaluatorV5("S1", inp, bp).evaluate_v2(pt, "train4")
    mk = lambda v: IstochnikiV6(dannye=lambda t, td: kontekst_v6.Dannye(t, td.dates, rezhim={"ma200": np.full(td.n, v)}))   # noqa: E731
    r0, x0, _ = EvaluatorV6("S1", inp, bp, ist=mk(False)).evaluate_v2({**pt, "rezhim": "imoex_ma200"}, "train4")
    r1, x1, _ = EvaluatorV6("S1", inp, bp, ist=mk(True)).evaluate_v2({**pt, "rezhim": "imoex_ma200"}, "train4")
    assert r5["status"] == r0["status"] == r1["status"] == "ok"
    assert x5.tobytes() == x1.tobytes() and np.any(x5 != 0)
    assert not np.any(x0 != 0), "контекст запретил все входы, а excess ≠ 0"
