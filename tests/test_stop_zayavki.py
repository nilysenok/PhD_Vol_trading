"""Л-22: ось vhod_tip = stop_zayavka (Ж-091 пп. 1.2–1.3) — fastbt/stop_zayavki.py, exec_e_stop.RunStop, exec_e_rules.stop_vhod,
ledger_v2.canon_v2. Синтетика; мутации: уровень по d+1, гэп по уровню, активна в баре выставления, уровень не обновлён вечером,
вместе с лимитом k_e, проход на тик вместо касания.
python3 -m pytest tests/test_stop_zayavki.py -q --tb=short"""
from types import SimpleNamespace

import numpy as np
import pytest

from fastbt import exec_e_rules, exec_e_stop, fast_etap1, ledger_v2, stop_zayavki
from fastbt.exec_e import Params, run
from fastbt.exec_e_types import Day

NAN = float("nan")


def D(*rows):
    a = np.array(rows, float).reshape(-1, 4)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


def P(**kw):
    return Params(vhod_tip="stop_zayavka", exit_signal=False, check="bary", stop_touch=True, comm=0.0, **kw)


def U(n, up=None, dn=None, xl=None, xs=None):
    f = lambda d: np.array([NAN] * n if d is None else d, float)             # noqa: E731
    return dict(up=f(up), dn=f(dn), xl=f(xl), xs=f(xs))


def go(days, u, p=None, f=run):
    n = len(days)
    return f("T", days, np.zeros(n), np.full(n, 0.02), p or P(), urovni=u)


FLAT = D((100, 100, 100, 100))
LONG = [FLAT, D((100, 105, 99, 102), (102, 111, 101, 109)), FLAT]            # buy-stop 110: касание на баре 1 дня 1


def test_vhod_po_kasaniyu_tsena_uroven():
    r = go(LONG, U(3, up=[110, NAN, NAN]))
    t = r.trades[0]
    assert (t.side, t.entry_day, t.entry_bar, t.entry_px, t.d_sig) == (1, 1, 1, 110.0, 0) and r.n_orders == 1 and r.n_fills == 1


def test_kasanie_h_ravno_urovnyu_ispolnyaet():                               # мутация «проход на тик вместо касания»
    days = [FLAT, D((100, 110, 99, 102)), FLAT]
    assert go(days, U(3, up=[110, NAN, NAN])).trades[0].entry_px == 110.0
    assert not go([FLAT, D((100, 109.99, 99, 102)), FLAT], U(3, up=[110, NAN, NAN])).trades


def test_gap_cherez_uroven_po_open():
    t = go([FLAT, D((115, 116, 114, 115)), FLAT], U(3, up=[110, NAN, NAN])).trades[0]
    assert t.entry_px == 115.0
    s = go([FLAT, D((90, 91, 89, 90)), FLAT], U(3, dn=[95, NAN, NAN])).trades[0]
    assert (s.side, s.entry_px) == (-1, 90.0)                                # шорт: sell-stop, open ниже уровня


def test_zayavka_ne_aktivna_v_bare_vystavleniya():
    days = [D((100, 120, 99, 100), (100, 100, 100, 100)), D((100, 105, 99, 102)), FLAT]
    r = go(days, U(3, up=[110, NAN, NAN]))                                    # high дня 0 = 120 ≥ 110, но заявка — на день 1
    assert not r.trades and r.n_orders == 1 and r.n_fills == 0


def test_dva_urovnya_v_odnom_bare_blizhayshiy_k_open():
    days = [FLAT, D((101, 112, 94, 100)), FLAT]
    t = go(days, U(3, up=[110, NAN, NAN], dn=[95, NAN, NAN])).trades[0]
    assert (t.side, t.entry_px) == (-1, 95.0)                                # |101−95| = 6 < |110−101| = 9
    assert go(days, U(3, up=[110, NAN, NAN], dn=[NAN] * 3)).trades[0].side == 1


def _vyhod_days():
    return [FLAT, D((100, 111, 99, 109)), D((110, 112, 109, 111)), D((111, 112, 110, 111)), FLAT]


def _vyhod_u(xl):
    return U(5, up=[110, NAN, NAN, NAN, NAN], xl=xl)


def test_vyhod_stop_zayavkoy_obnovlyaetsya_vecherom():
    r = go(_vyhod_days(), _vyhod_u([95, 108, 111.5, 50, 50]))                 # день 3: уровень вечера дня 2 (111,5), не 108 и не 50
    t = r.trades[0]
    assert (t.exit_day, t.exit_px, t.reason) == (3, 111.0, "kanal")           # гэп: min(open 111, уровень 111,5) = 111


@pytest.mark.parametrize("xl", [[95, 108, 108, 108, 108],                      # уровень не обновлён вечером
                                [95, 108, 111.5, 50, 50][1:] + [NAN],           # сдвиг на d+1 (значение дня i берётся от дня i+1)
                                [95, 108, 111.5, 50, 50]])
def test_mutatsii_urovnya_vyhoda_lovyatsya(xl):
    r = go(_vyhod_days(), _vyhod_u(xl)).trades[0]
    ok = (r.exit_day, r.exit_px) == (3, 111.0)
    assert ok is (xl == [95, 108, 111.5, 50, 50])


def test_vyhod_po_kasaniyu_ne_po_prokhodu():
    days = [FLAT, D((100, 111, 99, 109)), D((109, 110, 108, 109)), FLAT]
    r = go(days, U(4, up=[110, NAN, NAN, NAN], xl=[95, 108, NAN, NAN]))
    assert r.trades[0].exit_day == 2 and r.trades[0].exit_px == 108.0           # low = 108 касается стопа 108


def test_ne_primenyaetsya_s_limitom_ke():
    with pytest.raises(ValueError, match="stop_zayavka"):
        go(LONG, U(3, up=[110, NAN, NAN]), P(k=0.5))
    with pytest.raises(ValueError, match="urovni"):
        run("T", LONG, np.zeros(3), np.full(3, 0.02), P())
    with pytest.raises(ValueError, match="urovni"):
        run("T", LONG, np.zeros(3), np.full(3, 0.02), Params(), urovni=U(3))


def test_marshrut_bystryy_sloy_medlennyy_put():
    u = U(3, up=[110, NAN, NAN])
    a, b = go(LONG, u), go(LONG, u, f=fast_etap1.run)
    assert a.trades == b.trades and a.n_orders == b.n_orders


def _mut(monkeypatch, name, fn):
    monkeypatch.setattr(exec_e_stop, name, fn)


def test_mutatsiya_gap_po_urovnyu_lovitsya(monkeypatch):
    def bad(o, h, l, up, dn):
        r = exec_e_rules.stop_vhod(o, h, l, up, dn)
        return None if r is None else (r[0], r[1], up if r[0] > 0 else dn)
    _mut(monkeypatch, "stop_vhod", bad)
    assert go([FLAT, D((115, 116, 114, 115)), FLAT], U(3, up=[110, NAN, NAN])).trades[0].entry_px == 110.0   # ≠ 115 (тест выше)


def test_mutatsiya_prokhod_na_tik_lovitsya(monkeypatch):
    def bad(o, h, l, up, dn):
        return exec_e_rules.stop_vhod(o, h - 1e-9, l, up, dn)                   # строго «выше уровня» вместо h ≥ уровня
    _mut(monkeypatch, "stop_vhod", bad)
    assert not go([FLAT, D((100, 110, 99, 102)), FLAT], U(3, up=[110, NAN, NAN])).trades                  # касание 110 не исполнило


def test_mutatsiya_aktivna_v_bare_vystavleniya_lovitsya(monkeypatch):
    base = exec_e_stop.RunStop._decide

    def bad(s, i, j, close, nb):
        base(s, i, j, close, nb)
        if s.order is not None:
            s._try_order(i, j, s.days[i])                                     # заявка исполняется по барам дня выставления
    monkeypatch.setattr(exec_e_stop.RunStop, "_decide", bad)
    days = [D((100, 120, 99, 100), (100, 100, 100, 100)), D((100, 105, 99, 102)), FLAT]
    assert go(days, U(3, up=[110, NAN, NAN])).trades                         # в норме сделок нет (тест выше)


def _sig_ns(n=300, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    h, l = c + rng.uniform(0.2, 1.5, n), c - rng.uniform(0.2, 1.5, n)
    td = SimpleNamespace(close=c, high=h, low=l, volume=np.full(n, 2.0), n=n)
    ind = dict(adx14=np.full(n, 30.0), vol_sma20=np.full(n, 1.0), vf_pass=np.ones(n, bool))
    return td, ind


def test_urovni_s3_kanal_i_net_budushchego():
    td, ind = _sig_ns()
    u = stop_zayavki.urovni("S3", {"dc_window": 20}, td, ind)
    d = 250
    assert u["up"][d] == td.high[d - 19:d + 1].max() and u["dn"][d] == td.low[d - 19:d + 1].min()
    assert u["xl"][d] == u["dn"][d] and u["xs"][d] == u["up"][d] and np.isnan(u["up"][:200]).all()
    td2 = SimpleNamespace(**{**td.__dict__, "high": td.high.copy(), "low": td.low.copy()})
    td2.high[d + 1:] += 50.0
    td2.low[d + 1:] -= 50.0
    v = stop_zayavki.urovni("S3", {"dc_window": 20}, td2, ind)
    assert all(np.array_equal(u[k][:d + 1], v[k][:d + 1], equal_nan=True) for k in u)         # уровни дня d — по барам ≤ d


def test_urovni_s4_linia_st_storona_i_fitr():
    td, ind = _sig_ns()
    sp = {"atr_period": 10, "multiplier": 3.0}
    u = stop_zayavki.urovni("S4", sp, td, ind)
    ok = np.isfinite(u["xl"]) & (np.arange(td.n) >= 200)
    assert (u["up"][ok & (u["xl"] > td.close)] == u["xl"][ok & (u["xl"] > td.close)]).all()    # ST выше цены — buy-stop
    assert np.isnan(u["up"][ok & (u["xl"] < td.close)]).all() and np.isnan(u["dn"][ok & (u["xl"] > td.close)]).all()
    assert np.isfinite(u["up"][ok]).sum() + np.isfinite(u["dn"][ok]).sum() == ok.sum()
    bad = dict(ind, adx14=np.full(td.n, 10.0))
    w = stop_zayavki.urovni("S4", sp, td, bad)
    assert np.isnan(w["up"]).all() and np.isnan(w["dn"]).all() and np.isfinite(w["xl"]).any()  # фильтр гасит вход, не выход
    assert np.isnan(stop_zayavki.urovni("S4", sp, td, ind, "long")["dn"]).all()
    with pytest.raises(ValueError):
        stop_zayavki.urovni("S1", sp, td, ind)


def test_okno_allow():
    td, ind = _sig_ns()
    u = stop_zayavki.urovni("S3", {"dc_window": 20}, td, ind)
    al = np.ones(11, bool)
    al[3] = False
    w = stop_zayavki.okno(u, 240, 10, al)
    assert np.isnan(w["up"][3]) and w["up"][4] == u["up"][244] and len(w["xl"]) == 11
    with pytest.raises(ValueError):
        stop_zayavki.okno(u, 240, 10, al[:5])


PT = dict(vhod=0.5, ed_lim="atr14", zhizn=12, lestnica=0.5, gap="propusk", q_sily=0.8, proverka="zakrytie", storona="short_shag1")


def test_canon_v2_stop_zayavka_skhlopyvanie():
    c = ledger_v2.canon_v2("S3", {**PT, "vhod_tip": "stop_zayavka"})
    assert (c["vhod"], c["ed_lim"], c["zhizn"], c["lestnica"], c["gap"]) == ("rynok", "sigma", "den", "net", "vhod")
    assert (c["q_sily"], c["proverka"], c["storona"]) == (0.0, "bars5m", "obe")
    k = lambda pt: ledger_v2.klyuch_sha(ledger_v2.klyuch("S4", pt, "train4", 0.0004, None), "K", "V")    # noqa: E731
    assert k({**PT, "vhod_tip": "stop_zayavka"}) == k({**PT, "vhod_tip": "stop_zayavka", "vhod": 1.0, "zhizn": 36})
    assert k({**PT, "vhod_tip": "stop_zayavka"}) != k({**PT, "vhod_tip": "podtverzhdenie"})
    p = ledger_v2.canon_v2("S3", {**PT, "vhod_tip": "podtverzhdenie"})
    assert (p["vhod"], p["zhizn"], p["q_sily"], p["proverka"]) == (0.5, 12, 0.8, "zakrytie")  # podtverzhdenie — как раньше
    with pytest.raises(ValueError):
        ledger_v2.canon_v2("S1", {**PT, "vhod_tip": "stop_zayavka"})
