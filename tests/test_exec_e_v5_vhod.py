"""Л6 сетки v5 (Ж-127): сдвиг стоп-заявки тренда k_e·ATR, тейк от силы a0 + a1·ранг, частичный выход (доля, r). Синтетика, без сети.
Известный ответ: плоские дни 0..14 (close 100, ATR(14) = 2, σ̂ = 0,01), сигнал с дня 14, один бар-день 15.
Мутации: «ранг дня входа вместо дня сигнала», «добавка против позиции» (знак сдвига k_e / знак добавки за силу).
python3 -m pytest tests/test_exec_e_v5_vhod.py -q"""
import dataclasses

import numpy as np
import pytest

from fastbt import exec_e, exec_e_stop, exec_e_v6, fast_etap1
from fastbt.exec_e_types import Day, Params

N0 = 15


def _day(o, h, l, c):
    return Day(np.array(o, float), np.array(h, float), np.array(l, float), np.array(c, float), False)


def _flat():
    return _day([100, 100], [101, 101], [99, 99], [100, 100])


def _days(*bars):
    return [_flat() for _ in range(N0)] + [_day(*zip(*b)) for b in bars]


def _run(bars, p, rang=None):
    days = _days(*bars)
    sig = np.r_[np.zeros(N0 - 1), np.ones(len(days) - N0 + 1)]
    return exec_e.run("T", days, sig, np.full(len(days), 0.01), p, rang=rang)


def _stop(bars, k_e, up=102.0, dn=98.0):
    days = _days(*bars)
    n = len(days)
    nan = np.full(n, np.nan)
    u = dict(up=nan.copy(), dn=nan.copy(), xl=nan.copy(), xs=nan.copy())
    u["up"][N0 - 1], u["dn"][N0 - 1] = up, dn
    p = Params(vhod_tip="stop_zayavka", stop_touch=True, exit_signal=False, b=5.0, k_e=k_e)
    return exec_e.run("T", days, np.zeros(n), np.full(n, 0.01), p, urovni=u)


LONG_BAR = [[(100, 103.2, 100, 103)]]
SHORT_BAR = [[(100, 100, 97.5, 98)]]


# --- k_e ---
@pytest.mark.parametrize("k_e,px", [(0.0, 102.0), (0.5, 103.0), (-0.5, 101.0)])
def test_ke_sdvigaet_buy_stop_vverkh(k_e, px):
    t = _stop(LONG_BAR, k_e).trades[0]
    assert (t.side, round(t.entry_px, 9)) == (1, px)


def test_ke_sdvigaet_sell_stop_vniz():
    assert _stop(SHORT_BAR, 0.0).trades[0].side == -1
    assert _stop(SHORT_BAR, 0.5).trades == []                       # уровень 97 не достигнут (low 97,5)
    t = _stop([[(100, 100, 96.5, 97)]], 0.5).trades[0]
    assert (t.side, round(t.entry_px, 9)) == (-1, 97.0)


def test_ke_ne_vidit_atr_budushchego():
    """ATR на close d: изменение баров дня d+1 не меняет уровень входа."""
    a = _stop([[(100, 103.2, 100, 103)]], 0.5).trades[0].entry_px
    b = _stop([[(100, 150, 100, 103)]], 0.5).trades[0].entry_px
    assert a == 103.0 and b == 103.0


def test_mutaciya_ke_protiv_pozitsii_lovitsya(monkeypatch):
    ok = len(_stop(SHORT_BAR, 0.5).trades)
    monkeypatch.setattr(exec_e_stop, "ke_levels", lambda up, dn, k, a: (up - k * a, dn + k * a))
    assert len(_stop(SHORT_BAR, 0.5).trades) != ok                  # знак сдвига перевёрнут — шорт вошёл бы по 98


def test_ke_tolko_pri_stop_zayavke():
    with pytest.raises(ValueError):
        _run([[(100, 101, 99, 100)]], Params(k_e=0.5))


# --- тейк от силы ---
RANG = np.zeros(N0 + 1)
RANG[N0 - 1], RANG[N0] = 0.5, 0.0                                   # ранг дня сигнала 0,5; дня входа 0


def test_tejk_ot_sily_long_i_short():
    p = Params(b=5.0, a_rang=(1.0, 2.0))                            # a = 1 + 2·0,5 = 2 → T = 102 / 98
    t = _run([[(100, 102.5, 100, 102)]], p, RANG).trades[0]
    assert (t.reason, round(t.exit_px, 9)) == ("take", 102.0)
    assert _run([[(100, 101.5, 100, 101)]], p, RANG).trades[0].reason is None
    s = _run([[(100, 100, 97.5, 98)]], dataclasses.replace(p, side=-1), RANG).trades[0]
    assert (s.reason, round(s.exit_px, 9)) == ("take", 98.0)


def test_tejk_ot_sily_a1_nol_ravno_obychnomu():
    for seed in range(20):
        from test_fast_etap1_smoke import _days as rnd
        rng = np.random.default_rng(seed)
        days = rnd(rng, 60)
        sig = np.r_[np.zeros(20), (rng.random(40) < 0.5).astype(int)]
        sg, rg = np.full(60, 0.01), rng.random(60)
        p = Params(b=1.5, a=2.0, tau=4)
        a = exec_e.run("T", days, sig, sg, p)
        b = exec_e.run("T", days, sig, sg, dataclasses.replace(p, a=None, a_rang=(2.0, 0.0)), rang=rg)
        for f in ("gross", "cost", "pos"):
            assert getattr(a, f).tobytes() == getattr(b, f).tobytes(), (seed, f)


def test_mutaciya_rang_dnya_vhoda_vmesto_signala_lovitsya(monkeypatch):
    p = Params(b=5.0, a_rang=(1.0, 2.0))
    bars = [[(100, 101.5, 100, 101)]]
    ok = _run(bars, p, RANG).trades[0].reason
    orig = exec_e_v6.V6.v6_enter
    monkeypatch.setattr(exec_e_v6.V6, "v6_enter", lambda s, d: orig(s, min(d + 1, len(s.rang) - 1)))
    assert _run(bars, p, RANG).trades[0].reason != ok               # ранг 0 → T = 101 → тейк


def test_mutaciya_dobavka_protiv_pozitsii_lovitsya(monkeypatch):
    p = Params(b=5.0, a_rang=(1.0, 2.0))
    bars = [[(100, 101.5, 100, 101)]]
    ok = _run(bars, p, RANG).trades[0].reason
    monkeypatch.setattr(exec_e_v6.dataclasses, "replace", lambda q, a: q.__class__(**{**q.__dict__, "a": q.a_rang[0] - q.a_rang[1] * 0.5}))
    assert _run(bars, p, RANG).trades[0].reason != ok               # a = 0 → T = 100 → тейк сразу


def test_tejk_ot_sily_proverka_parametrov():
    bars = [[(100, 101, 99, 100)]]
    for bad in (dict(a_rang=(1.0, -1.0)), dict(a_rang=(0.0, 1.0)), dict(a_rang=(1.0, 1.0), a=1.0), dict(a_rang=(1.0, 1.0), R=1.0)):
        with pytest.raises(ValueError):
            _run(bars, Params(b=5.0, **bad), RANG)
    with pytest.raises(ValueError):
        _run(bars, Params(b=5.0, a_rang=(1.0, 1.0)))                # без ранга


def test_nechislovoj_rang_tejka_net():
    r = RANG.copy()
    r[N0 - 1] = np.nan
    assert _run([[(100, 150, 100, 140)]], Params(b=5.0, a_rang=(1.0, 2.0)), r).trades[0].reason is None


# --- частичный выход ---
PBARS = [[(100, 103.5, 100, 103), (103, 103, 103, 103)]]


def test_chast_doliya_i_uroven():
    p = Params(b=2.0, chast=(0.4, 1.5))                             # R = 2, уровень 100 + 1,5·2 = 103
    tr = _run(PBARS, p).trades
    part = [t for t in tr if t.reason == "part"][0]
    assert (round(part.exit_px, 9), part.frac, tr[0].frac) == (103.0, 0.4, 0.6)
    assert round(part.fee_out, 12) == round(0.0004 * 0.4, 12)
    assert not [t for t in _run(PBARS, Params(b=2.0, chast=(0.4, 2.0))).trades if t.reason == "part"]   # уровень 104


def test_chast_poloviny_na_1r_ravno_chast_1r_pobajtno():
    from test_fast_etap1_smoke import _days as rnd
    for seed in range(30):
        rng = np.random.default_rng(seed)
        days = rnd(rng, 60)
        sig = np.r_[np.zeros(20), (rng.random(40) < 0.5).astype(int)]
        sg = np.full(60, 0.01)
        p = Params(b=1.5, a=3.0, tau=5)
        a = exec_e.run("T", days, sig, sg, dataclasses.replace(p, chast_1r=True))
        b = exec_e.run("T", days, sig, sg, dataclasses.replace(p, chast=(0.5, 1.0)))
        for f in ("gross", "cost", "pos"):
            assert getattr(a, f).tobytes() == getattr(b, f).tobytes(), (seed, f)
        assert [(t.reason, t.exit_px, t.frac) for t in a.trades] == [(t.reason, t.exit_px, t.frac) for t in b.trades]


def test_chast_ne_v_bare_vhoda():
    """Уровень достижим лишь на баре входа → части нет (Ж-092: строго после бара входа)."""
    p = Params(b=2.0, chast=(0.5, 1.0), k=0.0)
    days = _days([(100, 100, 100, 100)])
    assert not [t for t in exec_e.run("T", days, np.r_[np.zeros(N0 - 1), np.ones(2)], np.full(16, 0.01), p).trades if t.reason == "part"]


def test_chast_proverka_parametrov():
    bars = [[(100, 101, 99, 100)]]
    for bad in (dict(chast=(0.0, 1.0)), dict(chast=(1.0, 1.0)), dict(chast=(0.5, 0.0)), dict(chast=(0.5, 1.0), chast_1r=True),
                dict(chast=(0.5, 1.0), check="zakrytie"), dict(chast=(0.5,))):
        with pytest.raises(ValueError):
            _run(bars, Params(b=2.0, **bad))
    with pytest.raises(ValueError):
        _run(bars, Params(chast=(0.5, 1.0)))                        # без b


def test_novye_osi_na_medlennom_puti():
    for kw in (dict(chast=(0.5, 1.0)), dict(a_rang=(1.0, 1.0)), dict(k_e=0.25)):
        assert fast_etap1._slow_only(Params(b=1.0, **kw), None), kw
    assert not fast_etap1._slow_only(Params(b=1.0), None)
