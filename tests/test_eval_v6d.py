"""V6-12d, синтетика (без данных): пивот S5 / VWAP S6 и pol_puti (eval_v6_piv), оси тренда на стоп-заявке (eval_v6_trend), свёртки canon_v6
(time_decay → «выкл.», reentry тренда), GOTOVY по осям, ключ кэша. Реальные данные — test_eval_v6d_real.py (FASTBT_V6_REAL=1)."""
from types import SimpleNamespace

import numpy as np
import pytest

from fastbt import canon_v5, canon_v6, eval_v6, eval_v6_piv as piv6, eval_v6_trend as trend6
from fastbt.etap1_eval_v2 import NeSchitana
from fastbt.eval_v6 import fkey, razdelit
from fastbt.exec_e_piv import PivUrovni

SIDS = canon_v5.SIDS


def _r(**kw):
    d = dict(ext={}, pt4={"k_tr": "net", "R": "net", "k_s": 2.0}, razmer="ravnyj", sp={"ma_tip": "sma"}, nov={"vhod_tip": "stop_zayavka"})
    d.update(kw)
    return SimpleNamespace(**d)


# ---------------- razobrat / proverit / primenit ----------------
def test_razobrat_piv_vwap_pol_puti():
    pt, e = piv6.razobrat({"stop_nach": "piv:half", "tejk": "piv:R"})
    assert e == {"piv_sl": "half", "piv_tp": "R"} and pt == {"stop_nach": piv6.ZAGLUSHKA, "tejk": "net"}
    assert piv6.razobrat({"stop_nach": "vwap:2.5", "tejk": "net"})[1] == {"piv_sl": "vwap", "vwap_m": 2.5}
    assert piv6.razobrat({"stop_nach": "sigma:2.0", "tejk": "pol_puti"})[1] == {"tejk": "pol_puti"}
    pt5 = {"stop_nach": "sigma:2.0", "tejk": "atr:1.0"}
    assert piv6.razobrat(pt5) == (pt5, {})                                      # без пивота — точка как есть


def test_proverit_otkazyvaet_gde_dvizhok_otkazal_by():
    ext = {"piv_sl": "next"}
    piv6.proverit(_r(), ext, "S5")
    for kw in (dict(ext={"atr_m": 1.0}), dict(ext={"b0": 1.0}), dict(ext={"pir": 0.5}), dict(razmer="risk_sigma"),
               dict(pt4={"k_tr": 1.0, "R": "net", "k_s": 2.0}), dict(pt4={"k_tr": "net", "R": 1.0, "k_s": 2.0})):
        with pytest.raises(NeSchitana):
            piv6.proverit(_r(**kw), ext, "S5")
    with pytest.raises(NeSchitana):
        piv6.proverit(_r(), {"piv_sl": "next", "tejk": "pol_puti"}, "S5")        # канал tejk_M один
    with pytest.raises(NeSchitana):
        piv6.proverit(_r(sp={"ma_tip": "ema"}), {"tejk": "pol_puti"}, "S1")      # M = SMA сигнала
    piv6.proverit(_r(sp={"ma_tip": "ema"}), {"tejk": "pol_puti"}, "S6")


def test_primenit_stop_bez_b_i_bufer_tolko_next_half():
    r = _r()
    piv6.primenit(r, {"piv_sl": "next"}, {"piv_buf": 0.2})
    assert r.pt4["k_s"] == "net" and r.ext["piv_buf"] == 0.2
    r = _r()
    piv6.primenit(r, {"piv_sl": "vwap", "vwap_m": 2.0}, {"piv_buf": 0.2})
    assert r.ext["piv_buf"] == 0.0 and r.ext["vwap_m"] == 2.0
    r = _r()
    piv6.primenit(r, {"piv_tp": "R"}, {"piv_buf": 0.2})
    assert r.pt4["k_s"] == 2.0 and "piv_buf" not in r.ext                       # тейк piv:R буфер не задаёт (Н4 #677)


# ---------------- уровни: строка d_sig, не d+1 ----------------
def _evr(sid="S5", n=12):
    ex = {k: np.arange(n, dtype=float) * (10 + i) for i, k in enumerate(("P", "S1", "S2", "R1", "R2", "vwap", "dev"))}
    e = SimpleNamespace(sid=sid, v6={}, r5=SimpleNamespace(sp={}), sk=SimpleNamespace(_td={}))
    e._piv_kesh = {("T", ()): ex}
    return e, ex


def _tm_ok(tm, sl=(3, 5)):
    e, ex = _evr()
    lv = tm(e, {"piv_sl": "next"}, "T", *sl)
    return all(np.array_equal(getattr(lv, k), ex[k][sl[0]:sl[0] + sl[1] + 1]) for k in ("P", "S1", "S2", "R1", "R2", "vwap", "dev"))


def test_urovni_berutsya_strokoy_d_a_ne_d_plyus_1():
    assert isinstance(piv6.tm(_evr()[0], {"piv_sl": "next"}, "T", 3, 5), PivUrovni)
    assert _tm_ok(piv6.tm)
    M = piv6.tm(_evr()[0], {"tejk": "pol_puti"}, "T", 3, 5)
    assert np.array_equal(M, _evr()[1]["P"][3:9])
    assert piv6.tm(_evr()[0], {}, "T", 3, 5) is None


def test_mutatsiya_pivot_s_d_plyus_1_lovitsya(monkeypatch):
    def sdvig(evr, ext, t, s0, b):
        x = piv6._extra(evr, t)
        return PivUrovni(**{k: x[k][s0 + 1:s0 + b + 2] for k in ("P", "S1", "S2", "R1", "R2", "vwap", "dev")})
    assert not _tm_ok(sdvig)


# ---------------- тренд ----------------
def test_trend_vernut_i_reentry():
    v6 = {"bezubytok": "atr:1.0", "pir": 0.5, "reentry": "da", "gap": "propusk"}
    assert trend6.vernut(True, {"a": 1}, v6) == {"a": 1, "bezubytok": "atr:1.0", "pir": 0.5}
    assert trend6.vernut(False, {"a": 1}, v6) == {"a": 1}
    r = _r()
    trend6.primenit(True, r, v6)
    assert r.ext == {"reentry_den": True}
    with pytest.raises(NeSchitana):
        trend6.primenit(True, _r(nov={"vhod_tip": "podtverzhdenie"}), v6)
    r = _r()
    trend6.primenit(False, r, v6), trend6.primenit(True, r, {})
    assert r.ext == {}


# ---------------- свёртки canon_v6 ----------------
def _q(sid, **kw):
    kub = canon_v6.load_kub(sid)
    base = {n: g[0] for n, g in canon_v5.osi(kub["v5"])}
    return kub, {**base, **{n: v["setka"][0] for n, v in canon_v6.novye(kub).items()}, **kw}


@pytest.mark.parametrize("sid", ("S3", "S4"))
def test_reentry_tolko_na_stop_zayavke(sid):
    kub, q = _q(sid, reentry="da")
    assert canon_v6.canon_v6(sid, {**q, "vhod_tip": "stop_zayavka"}, kub)["reentry"] == "da"
    assert canon_v6.canon_v6(sid, q, kub)["reentry"] == "da"                      # vhod_tip в точке нет — стоп-заявка (якорь)
    assert "reentry" not in canon_v6.canon_v6(sid, {**q, "vhod_tip": "podtverzhdenie"}, kub)   # подтверждение: заявки по уровню нет


@pytest.mark.parametrize("sid", SIDS)
def test_sobol_vtoroj_chastiy_otkryvaet_novye_osi_i_hranit_tochki(sid):
    kub = canon_v6.load_kub(sid)
    pts, vyb = canon_v6.tochki_v6(kub, 0, 2000)
    assert len(pts) / 2000 > 0.45, (sid, len(pts))                                # time_decay больше не выбрасывает точки (было ≈ 29 %)
    nov, cs = canon_v6.novye(kub), [c for _, c in pts]
    for n in ("piv_buf", "bezubytok", "stop_v_plyus", "pir", "reentry"):
        if n in nov and not (n == "piv_buf" and sid != "S5"):
            assert any(n in c for c in cs), f"{sid}.{n}: ось ни разу не выпала"
    assert not any(k in c for c in cs for k in ("tf",) + (("T", "smena") if sid in ("S3", "S4") else ()))   # tf, T и smena тренда — не подключены
    for n, r in canon_v6.rasshireno(kub).items():
        assert any(str(c.get(n)) in map(str, r["v6_dobavleno"]) for c in cs), f"{sid}.{n}: расширенные значения ни разу не выпали"
    for c in cs[:300]:
        razdelit(sid, c)                                                          # «ещё не подключена» не бывает


# ---------------- ряды отчёта: нога пирамиды открытой позиции ----------------
def test_otchet_noga_pir_otkrytoj_pozicii_ne_padaet():
    """Находка V6-12d: нога pir несёт reason="pir" с рождения, а exit_day = None, пока позиция открыта на конце окна → yr[None] — массив, TypeError."""
    from fastbt import tejk_signal
    from fastbt.exec_e_types import Trade
    dates = np.array(["2018-01-02", "2018-01-03", "2018-01-04"], "datetime64[ns]")
    zakr = Trade("T", 0, 1, 0, 10.0, exit_day=2, exit_bar=0, exit_px=11.0, reason="take")
    noga = Trade("T", 0, 2, 0, 10.5, reason="pir", parent=0)                    # позиция открыта на конце: выхода нет
    o = tejk_signal.otchet([([zakr, noga], dates)], (2018,))
    assert o["dolya_tejk_y"]["2018"] == 1.0                                       # посчитана только закрытая сделка


# ---------------- ключ кэша ----------------
def _kluchi_razlichayutsya(f) -> bool:
    ks = {f({}, {k: v}) for k, v in (("piv_buf", 0.1), ("piv_buf", 0.2), ("bezubytok", "atr:1.0"), ("stop_v_plyus", "2:1"), ("pir", 0.5),
                                      ("reentry", "da"))}
    return len(ks) == 6 and f({}, {}) == ()


def test_fkey_novye_osi_i_mutatsiya_os_vypala_lovitsya(monkeypatch):
    assert _kluchi_razlichayutsya(fkey)
    assert not _kluchi_razlichayutsya(lambda f, v: tuple(sorted(f.items())))      # ось выпала из ключа — ловится


def test_podklyucheno_ravno_v12():
    assert set(canon_v6.V12) == set(eval_v6.PODKLYUCHENO) - {"vhod_tip"}
    assert not {"T", "tf", "smena"} & set(eval_v6.PODKLYUCHENO)
