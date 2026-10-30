"""Несовместимые сочетания осей v6 → канон None (Ж-137 п. 7б; tolkovaniya.md [30.10 04:10]). Сеть и данные не нужны.
Для каждого из 7 сочетаний: плохая точка → None, соседняя совместимая → не None; мутации: правило снято, правило расширено."""
from unittest import mock

import pytest

from fastbt import canon_v6 as c6, canon_v6_nesovm as nsv
from fastbt.eval_v5_osi import yakor


def _g(sid, os, pref):
    return next(v for v in dict((n, g) for n, g, _ in c6.osi(c6._kub(sid)))[os] if str(v).startswith(pref))


def baza(sid, **kw):
    """Точка «всё на якоре / первом значении»: сигнальные оси — первые значения, v5 — якоря куба, новые оси — «выкл.»."""
    ox = c6.osi(c6._kub(sid))
    pt = {n: g[0] for n, g, _ in ox}
    pt.update({k: v for k, v in yakor(sid).items() if k in pt})
    return {**pt, **kw}


def nesovm(sid, **kw):
    return c6.canon_v6(sid, baza(sid, **kw)) is None


def test_baza_sovmestima():
    assert all(not nesovm(s) for s in ("S1", "S2", "S3", "S4", "S5", "S6"))


@pytest.mark.parametrize("sid", ["S1", "S2", "S5", "S6"])
def test_1_vyhod_lim_trebuet_close(sid):
    ok = dict(k_e=0.5, vyhod_lim="da", t_vyhod="close")
    assert not nesovm(sid, **ok)
    assert nesovm(sid, **{**ok, "t_vyhod": "1500"}) and nesovm(sid, **{**ok, "t_vyhod": "1700"})
    assert nesovm(sid, **{**ok, "t_vyhod": "open2"})                      # open2 — якорь куба, не close
    assert not nesovm(sid, **{**ok, "vyhod_lim": "net", "t_vyhod": "1500"})


@pytest.mark.parametrize("sid", ["S3", "S4"])
def test_2_parab_odin(sid):
    pb, a = _g(sid, "trail", "parab"), dict(stop_nach="atr:2.0")
    assert not nesovm(sid, trail=pb, **a)
    assert nesovm(sid, trail=pb, bezubytok="atr:1.0", **a) and nesovm(sid, trail=pb, stop_v_plyus="2:1", **a)
    assert nesovm(sid, trail=pb, pir=0.5, **a)
    assert not nesovm(sid, trail="atr:1.0", bezubytok="atr:1.0", **a)    # другие трейлы без parab — совместимы


@pytest.mark.parametrize("sid", ["S3", "S4"])
def test_3_trend_tejk_sily_na_stop_zayavke(sid):
    sila = _g(sid, "tejk", "sila")
    assert nesovm(sid, tejk=sila)
    assert not nesovm(sid, tejk=sila, vhod_tip="podtverzhdenie")         # отказ оценщика — только на RunStop
    assert not nesovm(sid, tejk="rr:1.0")


@pytest.mark.parametrize("sid", ["S1", "S2"])
def test_4_pol_puti_trebuet_sma(sid):
    pp = _g("S1", "tejk", "pol_puti")
    assert not nesovm(sid, tejk=pp, ma_tip="sma")
    assert nesovm(sid, tejk=pp, ma_tip="ema")
    assert not nesovm(sid, tejk="net", ma_tip="ema")


def test_5_piv_edinstvennyj_stop():
    piv, ok = _g("S5", "stop_nach", "piv:next"), {}
    assert not nesovm("S5", stop_nach=piv, **ok)
    assert nesovm("S5", stop_nach=piv, tejk="rr:1.0")
    assert not nesovm("S5", stop_nach=piv, tejk=_g("S5", "tejk", "piv:R"))   # второй пивотный канал — тейк, совместим
    assert nesovm("S6", stop_nach=_g("S6", "stop_nach", "vwap"), tejk="rr:1.0")


def test_6_piv_stop_i_pol_puti():
    assert nesovm("S5", stop_nach=_g("S5", "stop_nach", "piv:next"), tejk=_g("S5", "tejk", "pol_puti"))
    assert not nesovm("S5", tejk=_g("S5", "tejk", "pol_puti"))           # pol_puti без пивот-стопа


def test_7_rodnoy_i_norm_s252():
    ok = dict(tejk="rodnoy", ma_tip="sma", dev_tip="std", z_exit=0.5)
    assert not nesovm("S1", **ok)
    assert nesovm("S1", **ok, norm="s252") and nesovm("S1", **{**ok, "ma_tip": "ema"}) and nesovm("S1", **{**ok, "z_exit": "net"})
    assert nesovm("S5", tejk="rodnoy") and not nesovm("S6", tejk="rodnoy") and nesovm("S6", tejk="rodnoy", norm="s252")


def test_8_rodnoy_tau_vyhod_lim():
    ok = dict(tejk="rodnoy", ma_tip="sma", k_e=0.5, vyhod_lim="da", t_vyhod="close")
    assert not nesovm("S2", **ok, tau="net")                              # без τ rodnoy идёт по close
    assert nesovm("S2", **ok, tau=5)
    assert not nesovm("S2", **{**ok, "vyhod_lim": "net"}, tau=5)
    with mock.patch.object(nsv, "PRAVILA", nsv.PRAVILA[:-1]):
        assert not nesovm("S2", **ok, tau=5)                               # мутант без правила 8 пропускает точку, на которой движок падает


def test_rodnoy_ok_kopiya_sovpadaet_s_eval_v5():
    from fastbt.eval_v5 import rodnoy_ok
    for sid in ("S1", "S2", "S5", "S6", "S3"):
        for ma in ("sma", "ema", None):
            for dv in ("std", "atr", None):
                for ze in ("net", 0.5, None):
                    q = dict(ma_tip=ma, dev_tip=dv, z_exit=ze)
                    assert nsv.rodnoy_ok(sid, q) == rodnoy_ok(sid, q)


def _bez(i):
    return tuple(p for k, p in enumerate(nsv.PRAVILA) if k != i)


@pytest.mark.parametrize("i,sid,kw", [
    (0, "S1", dict(k_e=0.5, vyhod_lim="da", t_vyhod="1500")), (1, "S3", dict(trail="parab:0.02:0.1", bezubytok="atr:1.0", stop_nach="atr:2.0")),
    (2, "S3", dict(tejk="sila:0.5:1")), (3, "S1", dict(tejk="pol_puti", ma_tip="ema")),
    (4, "S5", dict(stop_nach="piv:next", tejk="rr:1.0")), (5, "S5", dict(stop_nach="piv:next", tejk="pol_puti")),
    (6, "S1", dict(tejk="rodnoy", norm="s252", ma_tip="sma", dev_tip="std", z_exit=0.5))])
def test_mutaciya_pravilo_snyato_lovitsya(i, sid, kw):
    assert nesovm(sid, **kw)
    with mock.patch.object(nsv, "PRAVILA", _bez(i)):
        assert not nesovm(sid, **kw)        # без правила i точка проходит → тесты выше (nesovm == True) на такой мутации падают


def test_mutaciya_vyhod_lim_vybrasyvaetsya_i_pri_close_lovitsya():
    ok = dict(k_e=0.5, vyhod_lim="da", t_vyhod="close")
    mut = lambda sid, q: q.get("vyhod_lim", "net") == "da"                # noqa: E731  (расширено: без условия t_vyhod ≠ close)
    with mock.patch.object(nsv, "PRAVILA", (mut,) + nsv.PRAVILA[1:]):
        assert nesovm("S1", **ok)           # мутант выбросил совместимую точку; test_1 («not nesovm») на нём падает
    assert not nesovm("S1", **ok)


def test_tochki_v6_vybrasyvaet_bol_she_i_bez_oshibok():
    kub = c6.load_kub("S3")
    pts, sch = c6.tochki_v6(kub, 0, 4096)
    with mock.patch.object(nsv, "PRAVILA", ()):
        pts0, sch0 = c6.tochki_v6(kub, 0, 4096)
    assert sch["vybrosheno"] > sch0["vybrosheno"] and len(pts) + sch["vybrosheno"] == 4096
    assert all(i in dict(pts0) for i, _ in pts)                          # правила только выбрасывают, точек не добавляют и не меняют
    assert all(dict(pts0)[i] == c for i, c in pts)
