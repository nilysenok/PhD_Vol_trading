"""Л6.3 куба v7 (Ж-146 п. 6, Ж-153 п. 1): вход в T того же дня в слое исполнения (fastbt/exec_e_t.py, Params.vhod_T).
Синтетика с известным ответом, без сети. 5-минутки 09:50..18:55; 10-минутки строятся из них (open10 = o5(m), close10 = c5(m+5)).
G1 выкл. = v6 побайтно; G2 вход в T по cena_T; G3 мутации «цена/бар после T»; G4 мутация «вход в баре сигнала вне T» (№ 55, Д2);
G5 T = 18:40 — close бара 18:40; G6 нет бара T — нет сделки (без ffill, Д4); + быстрый слой отдаёт такие точки эталону.
python3 -m pytest tests/test_exec_e_t.py -q"""
import numpy as np
import pandas as pd
import pytest

from fastbt import exec_e, exec_e_run, exec_e_t, fast_etap1
from fastbt import fast_exec as fx
from fastbt import t_reshenie_v7 as R
from fastbt.exec_e_types import Day, Params

DAYS = [pd.Timestamp("2019-03-04"), pd.Timestamp("2019-03-05"), pd.Timestamp("2019-03-06")]
MODS = list(range(590, 1140, 5))                  # начала 5-мин баров 09:50..18:55
COMM = 0.0004


def _o(k, m):
    return 100.0 * (k + 1) + (m - 590) / 100.0


def _day(k, drop=()):
    m = [x for x in MODS if x not in drop]
    o = np.array([_o(k, x) for x in m])
    return Day(o, o + 0.5, o - 0.5, o + 0.25, False, np.array(m))


def _bars10(drop=None):
    rows = []
    for k, d in enumerate(DAYS):
        for m in range(590, 1140, 10):
            if drop and m in drop.get(k, ()):
                continue
            rows.append((d + pd.Timedelta(minutes=m), _o(k, m), _o(k, m) + 1.0, _o(k, m) - 0.5, _o(k, m + 5) + 0.25, 10))
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _cena(T, drop=None):
    c = R.cena_T(_bars10(drop), T)
    return pd.Series(c["cena"].to_numpy(), index=c["date"]).reindex(DAYS).to_numpy()


def _run(sig, T="12:00", cena=None, days=None, p=None, **kw):
    days = days or [_day(k) for k in range(3)]
    p = p or Params(vhod_T=T, **kw)
    cena = _cena(T) if cena is None else cena
    return exec_e.run("T", days, np.array(sig), np.full(3, 0.01), p, cena_T=cena)


def _bar(m):
    return MODS.index(m)


def _ok(res, m, px, day=1):
    t = res.trades[0]
    return t.entry_day == day and t.entry_bar == _bar(m) and t.entry_px == px


def test_g1_vyklyuchen_raven_v6_pobajtno():
    days, sig, sg = [_day(k) for k in range(3)], np.array([0, 1, 1]), np.full(3, 0.01)
    p = Params(k=None)
    a = exec_e.run("T", days, sig, sg, p)
    b = exec_e_run.Run("T", days, sig, sg, p)
    for s in range(3):
        b.step(s, s)
    assert a.trades[0].entry_px == days[1].c[-1] and a.trades[0].entry_bar == len(days[1].c) - 1
    assert [t.entry_px for t in a.trades] == [t.entry_px for t in b.trades]
    assert np.array_equal(a.cost[:2], b.cost[:2]) and a.n_orders == 0
    with pytest.raises(ValueError):                           # cena_T без оси — отказ, а не молчаливое игнорирование
        exec_e.run("T", days, sig, sg, p, cena_T=np.ones(3))


def test_g2_vhod_v_T_po_cene_T_izvestnyj_otvet():
    r = _run([0, 1, 1])
    px = _o(1, 720)
    assert _ok(r, 720, px) and px == _cena("12:00")[1]
    cl = _day(1).c[-1]
    assert r.gross[0] == pytest.approx(np.log(cl / px)) and r.cost[0] == COMM       # день входа: от цены T до close, комиссия входа
    assert r.gross[1] == pytest.approx(np.log(_day(2).c[-1] / cl))
    assert r.cost[2] == COMM and r.n_orders == 0 and r.trades[0].fee_in == COMM
    assert r.pos[1] == 1.0


def test_g2_shorty_i_stop_v_bare_vhoda_isp_po_pravilam_dvizhka():
    r = _run([0, -1, -1], b=1.0)                              # шорт: цена T; стоп активен с бара после входа
    assert r.trades[0].side == -1 and _ok(r, 720, _o(1, 720))
    assert r.trades[0].reason == "stop" and r.trades[0].exit_bar > _bar(720)


@pytest.mark.parametrize("sdvig", ["12:10", "11:50"])
def test_g3_mutaciya_cena_ne_v_T_lovitsya(sdvig):
    r = _run([0, 1, 1], cena=_cena(sdvig))
    assert not _ok(r, 720, _o(1, 720))


def test_g3_mutaciya_bar_posle_T_lovitsya(monkeypatch):
    orig = exec_e_t.bar_vhoda
    monkeypatch.setattr(exec_e_t, "bar_vhoda", lambda d, T: orig(d, T + 10))
    r = _run([0, 1, 1])
    assert not _ok(r, 720, _o(1, 720))


def test_g4_mutaciya_vhod_v_bare_signala_vne_T_lovitsya(monkeypatch):
    monkeypatch.setattr(exec_e_t.RunT, "_decide", exec_e_run.Run._decide)      # вход по close d (бар сигнала), № 55
    r = _run([0, 1, 1])
    assert not _ok(r, 720, _o(1, 720)) and r.trades[0].entry_px == _day(1).c[-1]


def test_g5_18_40_close_bara_1840():
    r = _run([0, 1, 1], T="18:40")
    px = _o(1, 1125) + 0.25                                   # close 10-мин бара 18:40 = close 5-мин бара 18:45
    assert _ok(r, 1125, px) and px != _o(1, 1120)
    assert r.trades[0].entry_px == _cena("18:40")[1]
    assert not _ok(_run([0, 1, 1], T="18:40", cena=_cena("18:30")), 1125, px)    # мутация: open вместо close / другой бар


def test_g6_net_bara_T_net_sdelki_bez_podstanovki():
    cena = _cena("12:00", drop={1: (720,)})
    assert np.isnan(cena[1]) and np.isfinite(cena[[0, 2]]).all()
    r = _run([0, 1, 1], cena=cena)
    assert r.trades[0].entry_day == 2 and len(r.trades) == 1                   # день 1 пропущен, ffill нет
    with pytest.raises(ValueError):                           # мутация ffill: «чужая» цена дня 0 вне бара T дня 1 — инвариант _v_bare её ловит
        _run([0, 1, 1], cena=pd.Series(cena).ffill().to_numpy())
    d5 = [_day(0), _day(1, drop=(720,)), _day(2)]             # цена есть, а 5-мин бара T в данных нет — тоже без сделки
    assert _run([0, 1, 1], days=d5).trades[0].entry_day == 2


def test_net_povtornogo_vhoda_v_den_vyhoda_po_signalu():
    r = _run([1, -1, -1])
    assert r.trades[0].exit_day == 1 and r.trades[0].reason == "signal"
    assert r.trades[1].entry_day == 2 and r.trades[1].side == -1               # вход 12:00 дня 1 раньше выхода 18:55 — запрещён


def test_parametry_otkaz():
    for kw in (dict(vhod_T="12:60"), dict(vhod_T="09:50"), dict(vhod_T="12:00", k=1.0), dict(vhod_T="12:00", vhod_open=True),
               dict(vhod_T="12:00", T=2), dict(vhod_T="12:00", check="zakrytie")):
        with pytest.raises(ValueError):
            _run([0, 1, 1], p=Params(**kw))
    with pytest.raises(ValueError):
        _run([0, 1, 1], cena=np.ones(2))
    with pytest.raises(ValueError):                           # без Day.t бар T не найти
        o = np.array([100.0, 101.0])
        d = Day(o, o + 1, o - 1, o, False)
        _run([0, 1, 1], days=[d, d, d])


def test_bystryj_sloj_otdayot_tochku_etalonu():
    p = Params(vhod_T="12:00")
    assert fast_etap1._slow_only(p, None) and "vhod_T" in fx.ne_znaet(p)
    assert not fast_etap1._slow_only(Params(), None) and fx.ne_znaet(Params()) == []
    with pytest.raises(ValueError):                           # fast_etap1.run не несёт cena_T: на эталон — и громкий отказ, не молчаливый путь ядра
        fast_etap1.run("T", [_day(k) for k in range(3)], np.array([0, 1, 1]), np.full(3, 0.01), p)
