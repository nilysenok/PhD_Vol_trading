"""V6-a (Ж-111 пп. 1–3): жизнь заявки входа T, снятие при уходе/смене стороны сигнала, повторный вход в тот же день.
Синтетика с известным ответом. Мутации: № 2–4 (mutations.md) по теме; заявка не снята при смене сигнала (Ж-106 п. 20)."""
import numpy as np
import pytest

from fastbt import exec_e, exec_e_run, exec_e_zayavki, fast_etap1
from fastbt.exec_e import Params, run
from fastbt.exec_e_types import Day

SG = 0.01
DIP = (100.0, 100.0, 98.5, 99.5)       # low касается лимита 99
FLAT = (100.0, 100.0, 100.0, 100.0)


def D(*rows):
    a = np.array(rows, float).reshape(-1, 4)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


def flat(n=3):
    return D(*[FLAT] * n)


def go(days, sig, **kw):
    p = Params(k=1.0, comm=0.0, **kw)
    return run("X", days, np.array(sig), np.full(len(days), SG), p)


def entries(r):
    return [(t.entry_day, t.d_sig, t.side, round(t.entry_px, 6)) for t in r.trades]


# --- T ---
def _days_dip(dips):
    return [D(*[DIP] * 3) if i in dips else flat() for i in range(6)]


def test_T3_zayavka_zhivet_do_dnya_3_s_peresvetom():
    r = go(_days_dip({3}), [1] * 6, T=3)
    assert entries(r) == [(3, 2, 1, 99.0)]       # заявка от вечера 2 (age 3): пересчёт каждый вечер


def test_T2_taym_aut_ne_ispolnyaetsya():
    assert go(_days_dip({3}), [1] * 6, T=2).trades == []
    assert go(_days_dip({3}), [1] * 6, T=1).trades == []


def test_T1_poberaytno_prezhniy_put():
    sig = [1, 1, 0, 1, 1, -1, 0, 1]
    days = [D(*[DIP] * 3) if i in (2, 4, 6) else flat() for i in range(8)]
    a = go(days, sig)
    b = go(days, sig, T=1, reentry_den=False)
    for x in ("gross", "cost", "pos", "free"):
        assert getattr(a, x).tobytes() == getattr(b, x).tobytes()
    assert entries(a) == entries(b) and (a.n_orders, a.n_fills) == (b.n_orders, b.n_fills)


def test_signal_ushyol_zayavka_snyata():
    sig = [1, 1, 0, 1, 1, 1]
    r = go(_days_dip({3, 4}), sig, T=5)
    assert entries(r) == [(4, 3, 1, 98.505)]     # старая заявка (вечер 1) на день 3 снята; новая от close дня 3 (99,5) — на день 4


def test_smena_storony_snimaet_i_stavit_protivopolozhnuyu():
    day2 = D(DIP, (100.0, 101.5, 100.0, 100.0), FLAT)
    r = go([flat(), flat(), day2, flat()], [1, -1, -1, -1], T=5)
    assert entries(r) == [(2, 1, -1, 101.0)]     # лонг (low 98.5 дня 2) не исполнен: заявка снята в вечер 1


def test_mutatsiya_zayavka_ne_snyata_pri_smene_signala_lovitsya(monkeypatch):
    def zhivaya(s, o, i, j):                      # подложено: сторона сигнала не проверяется
        if o["retry"] or o.get("age", 1) >= s.p.T or j + 1 >= len(s.ix):
            return False
        s._place(s.close_of(i), o["sd"], s.sigma[i], j, i, retry=False, age=o.get("age", 1) + 1)
        return True
    monkeypatch.setattr(exec_e_run, "prodlit", zhivaya)
    r = go(_days_dip({3, 4}), [1, 1, 0, 1, 1, 1], T=5)
    assert entries(r) != [(4, 3, 1, 98.505)]
    day2 = D(DIP, (100.0, 101.5, 100.0, 100.0), FLAT)
    r = go([flat(), flat(), day2, flat()], [1, -1, -1, -1], T=5)
    assert entries(r) != [(2, 1, -1, 101.0)]


def test_mutatsiya_T_ignoriruetsya_lovitsya(monkeypatch):
    monkeypatch.setattr(exec_e_run, "prodlit", lambda s, o, i, j: exec_e_zayavki.prodlit(s, dict(o, age=0), i, j))
    assert go(_days_dip({3}), [1] * 6, T=2).trades != []


# --- повторный вход в тот же день ---
def _day_reentry(extra=()):
    rows = [DIP, (99.5, 100.5, 99.5, 100.0), (100.0, 100.0, 98.8, 99.0)] + list(extra)
    return D(*rows)


def test_reentry_den_vtoroy_vhod_v_tot_zhe_den():
    days = [flat(), _day_reentry(), flat(), flat()]
    kw = dict(a=1.0, b=1.0)
    a = go(days, [1] * 4, **kw)
    r = go(days, [1] * 4, reentry_den=True, **kw)
    assert len(a.trades) == 1 and a.trades[0].reason == "take"
    assert len(r.trades) == 2
    t0, t1 = r.trades
    assert (t0.reason, t0.exit_bar) == ("take", 1)
    assert (t1.entry_day, t1.entry_bar, t1.d_sig, round(t1.entry_px, 6)) == (1, 2, 0, 99.0)   # уровень вечера 0, бар после выхода


def test_reentry_den_ne_bolee_odnogo_za_den():
    take = (99.5, 100.5, 99.5, 100.0)
    days = [flat(), _day_reentry([take, (100.0, 100.0, 98.8, 99.0)]), flat()]
    r = go(days, [1] * 3, a=1.0, b=1.0, reentry_den=True)
    assert [(t.entry_day, t.entry_bar) for t in r.trades] == [(1, 0), (1, 2)]


def test_reentry_den_ne_posle_vyhoda_po_signalu():
    days = [flat(), D(DIP, FLAT, FLAT), D(DIP, DIP, DIP)]
    r = go(days, [1, 1, 0], reentry_den=True)
    assert [(t.entry_day, t.reason) for t in r.trades] == [(1, "signal")]


def test_mutatsiya_reentry_vyklyuchen_lovitsya(monkeypatch):
    monkeypatch.setattr(exec_e_zayavki, "INTRA", ())
    days = [flat(), _day_reentry(), flat()]
    assert len(go(days, [1] * 3, a=1.0, b=1.0, reentry_den=True).trades) == 1


# --- проверки параметров и маршрут ---
@pytest.mark.parametrize("kw", [dict(T=0), dict(T=2, k=None), dict(reentry_den=True, vhod_open=True, k=None),
                                dict(reentry_den=True, check="zakrytie")])
def test_nevernye_parametry(kw):
    args = dict(k=1.0, comm=0.0)
    args.update(kw)
    with pytest.raises(ValueError):
        run("X", [flat()], np.array([1]), np.array([SG]), Params(**args))


def test_marshrut_novye_rezhimy_na_medlennyy_put():
    assert not fast_etap1._slow_only(Params(k=1.0), None)
    assert fast_etap1._slow_only(Params(k=1.0, T=3), None)
    assert fast_etap1._slow_only(Params(k=1.0, reentry_den=True), None)
    days = _days_dip({3})
    p = Params(k=1.0, comm=0.0, T=3)
    sig, sg = np.ones(6), np.full(6, SG)
    a, b = exec_e.run("X", days, sig, sg, p), fast_etap1.run("X", days, sig, sg, p)
    assert entries(a) == entries(b) == [(3, 2, 1, 99.0)]
