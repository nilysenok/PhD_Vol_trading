"""Л5 сетки v5 (Ж-127): единицы уровней ed_s / ed_t ∈ {sigma, atr, pct} и Chandelier (Params.chand). Синтетика, без сети.
Известный ответ: плоские дни с ATR(14) = 2 при close 100 (единица atr = 0,02), вход по close дня 14.
Мутации (mutations.md № 2, 3, 4 и «ATR дня d»): подмена правила в модуле → сценарий обязан упасть.
python3 -m pytest tests/test_exec_e_v5_ed.py -q"""
import dataclasses

import numpy as np
import pytest

from fastbt import exec_e, exec_e_run, fast_etap1
from fastbt.exec_e_rules import atr_series
from fastbt.exec_e_types import Day, Params

N0 = 15                                              # дни 0..14 плоские, сигнал с дня 14
U = 0.02                                             # ATR 2 / close 100


def _day(o, h, l, c):
    return Day(np.array(o, float), np.array(h, float), np.array(l, float), np.array(c, float), False)


def _flat():
    return _day([100, 100], [101, 101], [99, 99], [100, 100])


def _run(bars, p):
    days = [_flat() for _ in range(N0)] + [_day(*zip(*b)) for b in bars]
    sig = np.r_[np.zeros(N0 - 1), np.ones(len(days) - N0 + 1)]
    return exec_e.run("T", days, sig, np.full(len(days), 0.01), p, close=None)


def _first(bars, p):
    return _run(bars, p).trades[0]


BARS_STOP = [[(100, 100, 94, 95)]]                   # бар: o, h, l, c


def test_stop_pct_ravno_dole_vhoda():
    t = _first(BARS_STOP, Params(b=0.05, ed_s="pct"))
    assert (t.reason, round(t.exit_px, 9)) == ("stop", 95.0)


def test_stop_atr_i_take_atr():
    p = Params(b=2.0, a=3.0, ed_s="atr", ed_t="atr")        # S = 100(1 − 2·0,02) = 96, T = 100(1 + 3·0,02) = 106
    s = _first([[(100, 100, 95, 95)]], p)
    assert (s.reason, round(s.exit_px, 9)) == ("stop", 96.0)
    t = _first([[(100, 107, 100, 107)]], p)
    assert (t.reason, round(t.exit_px, 9)) == ("take", 106.0)


def test_edinitsy_stopa_i_tejka_nezavisimy():
    p = Params(b=0.05, a=3.0, ed_s="pct", ed_t="atr")       # стоп 95 (pct), тейк 106 (atr)
    assert round(_first([[(100, 106.5, 100, 106)]], p).exit_px, 9) == 106.0
    assert round(_first(BARS_STOP, p).exit_px, 9) == 95.0


def test_stop_nizhe_vhoda_dlya_longa_i_vyshe_dlya_shorta():
    for ed, b in (("sigma", 2.0), ("atr", 2.0), ("pct", 0.03)):
        lo = _first([[(100, 100, 50, 60)]], Params(b=b, ed_s=ed))
        assert lo.reason == "stop" and lo.exit_px < lo.entry_px, ed
        sh = _first([[(100, 150, 100, 140)]], Params(b=b, ed_s=ed, side=-1))
        assert sh.reason == "stop" and sh.exit_px > sh.entry_px, ed


def test_ed_sigma_ravno_podmene_ryada_edinitsy():
    """atr ≡ путь sigma с σ̂ := ATR/close (известный ответ: единица — только ряд уровня)."""
    from test_fast_etap1_smoke import _days
    for seed in range(30):
        rng = np.random.default_rng(seed)
        days = _days(rng, 60)
        sig = np.r_[np.zeros(20), (rng.random(40) < 0.5).astype(int)]
        sg = np.full(60, 0.01)
        u = atr_series(days) / np.array([d.c[-1] if not d.pusto else np.nan for d in days])
        u = np.array(pd_ffill(u))
        p = Params(b=1.5, a=2.0, tau=4, comm=0.0004)
        a = exec_e.run("T", days, sig, sg, dataclasses.replace(p, ed_s="atr", ed_t="atr"))
        b = exec_e.run("T", days, sig, u, p)
        for f in ("gross", "cost", "pos"):
            assert getattr(a, f).tobytes() == getattr(b, f).tobytes(), (seed, f)
        assert [t.exit_px for t in a.trades] == [t.exit_px for t in b.trades]


def pd_ffill(x):
    out, last = [], np.nan
    for v in x:
        last = v if np.isfinite(v) else last
        out.append(last)
    return out


def test_sigma_idet_prezhnim_putyom_i_bystryj_sloj_ravnyj_etalonu():
    from test_fast_etap1_smoke import _case, _same
    for seed in range(40):
        days, sig, sigma, p, close, div = _case(seed)
        try:
            ref = exec_e.run("T", days, sig, sigma, p, close=close, div=div)
        except ValueError:
            continue
        _same(ref, exec_e.run("T", days, sig, sigma, dataclasses.replace(p, ed_s="sigma", ed_t="sigma"), close=close, div=div))
        _same(ref, fast_etap1.run("T", days, sig, sigma, p, close=close, div=div))


def test_bystryj_sloj_umeet_perenesennye_osi():
    """П4 (план v5): ed_s / ed_t / chand перенесены в быстрый слой (сверка — tests/test_fast_v5_ekviv.py)."""
    assert not fast_etap1._slow_only(Params(b=1.0, ed_s="atr"), None)
    assert not fast_etap1._slow_only(Params(b=1.0, ed_t="pct"), None)
    assert not fast_etap1._slow_only(Params(b=1.0, chand=2.0), None)
    assert not fast_etap1._slow_only(Params(b=1.0), None)


NE_PERENESENY = [dict(k_e=0.5), dict(a_rang=(1.0, 1.0)), dict(chast=(0.5, 1.0)), dict(parab=(0.02, 0.2)),
                 dict(zhizn=3), dict(reentry_den=True)]


@pytest.mark.parametrize("kw", NE_PERENESENY)
def test_slow_only_dlya_neperenesennykh_osej(kw):
    """Защита #488: ось, которой быстрый слой не умеет, не должна молча игнорироваться — только эталон."""
    assert fast_etap1._slow_only(Params(b=1.0, **kw), None)
    assert fast_etap1._slow_only(Params(b=1.0, ed_s="atr", chand=2.0, **kw), None)


def test_mutatsiya_slow_only_vsegda_false_lovitsya(monkeypatch):
    monkeypatch.setattr(fast_etap1, "_slow_only", lambda p, ed, *extra: False)
    with pytest.raises(AssertionError):
        test_slow_only_dlya_neperenesennykh_osej(NE_PERENESENY[0])


def test_proverka_parametrov():
    for bad in (dict(ed_s="x"), dict(ed_t="x"), dict(chand=2.0), dict(b=1.0, chand=0.0), dict(b=1.0, chand=2.0, check="zakrytie")):
        with pytest.raises(ValueError):
            _run(BARS_STOP, Params(**bad))


# --- ATR причинен: единица дня i не видит дней > i ---
def _u(days):
    return exec_e.unit_series(days, None, "atr", None, atr_series(days))


def _prefix_ok(unit_fn):
    days = [_flat() for _ in range(20)] + [_day([100], [130], [70], [100]) for _ in range(3)]
    return np.array_equal(unit_fn(days)[:20], unit_fn(days[:20]), equal_nan=True)


def test_atr_prefiks_prichinnost():
    assert _prefix_ok(_u)
    u = _u([_flat() for _ in range(15)])
    assert np.isnan(u[:13]).all() and u[13] == pytest.approx(U)             # до 14 дней — nan


def test_mutatsiya_atr_dnya_d_lovitsya():
    def mut(days):
        a = _u(days)
        return np.r_[a[1:], np.nan]                     # ATR следующего дня
    assert not _prefix_ok(mut)


# --- мутации № 2, 3, 4 при единицах atr ---
P_ATR = Params(b=2.0, a=3.0, ed_s="atr", ed_t="atr")     # S = 96, T = 106


def _scen_poryadok():                                  # № 3: стоп и тейк в одном баре — стоп
    assert _first([[(100, 110, 90, 100)]], P_ATR).reason == "stop"


def _scen_kasanie():                                   # № 2: касание стопа / тейка — не исполнение
    S, T = 100 * (1 - 2 * U), 100 * (1 + 3 * U)
    assert _first([[(100, T, S, 100)]], P_ATR).reason != "stop"
    assert _first([[(100, T, S, 100)]], P_ATR).reason != "take"


def _scen_gep():                                       # № 4: гэп через стоп — по open
    t = _first([[(90, 91, 89, 90)]], P_ATR)
    assert (t.reason, t.exit_px) == ("stop", 90.0)


@pytest.mark.parametrize("scen", [_scen_poryadok, _scen_kasanie, _scen_gep])
def test_scenarii_na_pravke_prokhodyat(scen):
    scen()


def _mut_stop_hit_touch(sd, o, h, l, S, touch):
    return (l <= S if sd > 0 else h >= S), min(o, S) if sd > 0 else max(o, S)


def _mut_stop_hit_level(sd, o, h, l, S, touch):
    return (l < S if sd > 0 else h > S), S


def _mut_stop_hit_never(sd, o, h, l, S, touch):
    return False, S


@pytest.mark.parametrize("scen,mut", [(_scen_kasanie, _mut_stop_hit_touch), (_scen_gep, _mut_stop_hit_level),
                                      (_scen_poryadok, _mut_stop_hit_never)])
def test_mutatsii_2_3_4_lovyatsya(monkeypatch, scen, mut):
    monkeypatch.setattr(exec_e_run, "stop_hit", mut)
    with pytest.raises((AssertionError, ValueError)) as ei:           # #617: исполнение вне бара — ValueError
        scen()
    if ei.type is ValueError:
        assert "вне [low; high]" in str(ei.value), str(ei.value)


# --- Chandelier ---
def _chand_days():
    rally = [(100, 110, 100, 110)]
    drop = [(108, 108, 100, 100)]
    return [rally, drop]


def test_chandelier_ukhod_po_urovnyu():
    bars = _chand_days()
    p = Params(b=5.0, ed_s="atr", chand=2.0)             # начальный стоп 90; Chandelier поднимает его
    days = [_flat() for _ in range(N0)] + [_day(*zip(*b)) for b in bars]
    a = atr_series(days)
    expect = 110.0 - 2.0 * a[N0]                          # 11 дней до дня 16 включают день 15 (high 110), ATR на конец дня 15
    t = _first(bars, p)
    assert (t.reason, t.exit_day) == ("trail", N0 + 1)
    assert t.exit_px == pytest.approx(expect, abs=1e-9)
    assert _first(bars, dataclasses.replace(p, chand=None)).reason != "trail"       # без Chandelier выхода нет


def test_chandelier_okno_11_dnej_bez_tekushchego_dnya():
    """Мутация: окно с текущим днём (high дня j) — уровень выше, выход раньше/по другой цене; тест различает."""
    bars = [[(100, 110, 100, 110)], [(108, 140, 100, 100)]]       # на дне 16 high 140 не должен попасть в уровень дня 16
    t = _first(bars, Params(b=5.0, ed_s="atr", chand=2.0))
    days = [_flat() for _ in range(N0)] + [_day(*zip(*b)) for b in bars]
    assert t.exit_px == pytest.approx(110.0 - 2.0 * atr_series(days)[N0], abs=1e-9)


def test_chandelier_shortovyj_zerkalo():
    bars = [[(100, 100, 90, 90)], [(92, 100, 92, 100)]]
    p = Params(b=5.0, ed_s="atr", chand=2.0, side=-1)
    days = [_flat() for _ in range(N0)] + [_day(*zip(*b)) for b in bars]
    expect = 90.0 + 2.0 * atr_series(days)[N0]
    t = _first(bars, p)                                  # sig = +1 при Params.side = −1 — шорт
    assert t.side == -1 and t.reason == "trail" and t.exit_px == pytest.approx(expect, abs=1e-9)
