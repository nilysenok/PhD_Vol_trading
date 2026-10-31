"""Лист s8_s12_boj (31.10): боевой путь S8–S12 для пилота-2. Синтетика, без сети; реальный прогон драйвера — в конце (пропуск без данных).
Б1 cena_T — из реальной ленты (не скорректированной на дивиденды); Б2 цена T = close последнего бара, закрывающегося ≤ T (17:00, 18:00, 18:40);
Б3 граница ленты S10–S12 = period_v2.chtenie_konec(polnaya). Б4 и G6 (драйвер, панель) — в test_s8_s12_boj_2.py.
Мутация = подмена функции/параметра, на которой проверка `_ok*` обязана упасть (AssertionError).
python3 -m pytest tests/test_s8_s12_boj.py -q"""
import numpy as np
import pandas as pd
import pytest

from fastbt import exec_e, exec_e_t, impuls_v7, period_v2
from fastbt import eval_v7_impuls as SI
from fastbt import eval_v7_ist as E
from fastbt.eval_v7 import EvaluatorV7, Ist7
from fastbt.exec_e_types import Day, Params

TS = {"17:00": 1020, "18:00": 1080, "18:40": 1120}
MODS = np.arange(590, 1120, 5)                       # 5-мин бары начинаются 09:50..18:35 (баров 18:40 и позже нет, как у AFLT)
DATES = pd.bdate_range("2019-03-04", periods=3)


def _px(k, m):
    """Уровень цены дня k в минуту m; open бара, начинающегося в m, = _px + 0.3 (скачок), close бара, закрывающегося в m, = _px."""
    return 100.0 * (k + 1) + (m - 590) / 100.0


def _day(k):
    o = np.array([_px(k, m) + 0.3 for m in MODS])
    c = np.array([_px(k, m + 5) for m in MODS])
    return Day(o, np.maximum(o, c) + 0.05, np.minimum(o, c) - 0.05, c, False, MODS)


def _b10():
    """10-минутки: слот s = бар [10:00 + 10 s; +10), open = _px(m) + 0.3, close = _px(m + 10); слотов 0..51 (бара 18:40 нет, как у AFLT)."""
    rows = [(d, s, _px(k, 600 + 10 * s) + 0.3, _px(k, 610 + 10 * s)) for k, d in enumerate(DATES) for s in range(52)]
    return pd.DataFrame(rows, columns=["date", "slot", "open", "close"])


def _ozhid_cena(k, T):
    """Л6.3/Д2: open первого бара ≥ T; 18:40 без бара — close дня (аукцион 18:50), т.е. close последнего бара."""
    return _px(k, 1120) if T == "18:40" else _px(k, TS[T]) + 0.3


# ---------------- Б2: цена и бар входа T (open первого бара ≥ T; 18:40 — запасной путь close дня) ----------------
def _ok_bar(bar_vhoda):
    for T, t in TS.items():
        ozhid = len(MODS) - 1 if T == "18:40" else int(np.where(MODS == t)[0][0])
        assert bar_vhoda(_day(0), t) == ozhid, T
    assert bar_vhoda(Day(*[a[:5] for a in (_day(0).o, _day(0).h, _day(0).l, _day(0).c)], False, MODS[:5]), 1020) is None   # нет бара ≥ T (17:00) — None (Д4)


def test_b2_bar_vhoda_pervyj_bar_ne_ranshe_T_i_zapasnoj_1840():
    _ok_bar(exec_e_t.bar_vhoda)


def test_b2_mutatsiya_bar_zakryvayushchijsya_v_T_lovitsya():
    def mut(day, t):                                                   # бар [T−5; T): цена сигнала, Д2 нарушен
        k = int(np.searchsorted(day.t, t - 5, side="left"))
        return k if k < len(day.t) and day.t[k] == t - 5 else None
    with pytest.raises(AssertionError):
        _ok_bar(mut)


def _ok_cena(cena_t):
    for T in TS:
        assert np.allclose(cena_t(_b10(), DATES, T), [_ozhid_cena(k, T) for k in range(3)], atol=1e-12, rtol=0), T


def test_b2_cena_t_open_pervogo_bara_ne_ranshe_T_i_close_dnya_dlya_1840():
    _ok_cena(SI.cena_t)


def test_b2_mutatsiya_close_bara_zakryvayushchegosya_v_T_lovitsya():
    def mut(b10, dates, T):                                            # close 10-мин бара [T−10; T): исполнение по цене сигнала
        b = b10[b10["slot"] == (TS[T] - 10 - 600) // 10]
        return b.set_index("date")["close"].reindex(dates).to_numpy(float)
    with pytest.raises(AssertionError):
        _ok_cena(mut)


@pytest.mark.parametrize("T", TS)
def test_b2_vhod_v_T_bez_isklyucheniya_vne_low_high(T):
    """Вход RunT на барах без 18:40: цена внутри [low; high] бара входа (иначе «исполнение вне»), бар входа — [T−5; T)."""
    days = [_day(k) for k in range(3)]
    cena = SI.cena_t(_b10(), DATES, T)
    r = exec_e.run("T", days, np.array([0, 1, 1]), np.full(3, 0.01), Params(vhod_T=T, tau=2), cena_T=cena)
    ozhid_bar = len(MODS) - 1 if T == "18:40" else int(np.where(MODS == TS[T])[0][0])
    assert r.trades and r.trades[0].entry_bar == ozhid_bar and r.trades[0].entry_px == cena[1] == _ozhid_cena(1, T)


# ---------------- Б1: cena_T из реальных цен ----------------
def _ist(real_is_adj=False):
    import test_eval_v7_ist as tv
    raw = tv._raw()
    s = tv.E.sobrat(raw, tv._im(), tv.KARTA, tv.TK, "2018-01-01", "2019-01-01", korp=tv._korp0,
                    mnozh=lambda t, dts, c: np.where(np.asarray(dts) < np.datetime64("2018-02-20"), 0.5, 1.0))
    return s, raw


def test_b1_istochniki_otdayut_realnuyu_lentu_i_skorrektirovannuyu():
    s, raw = _ist()
    t = "AAA"
    d = pd.Timestamp("2018-02-01")                                    # до отсечки: сигнальная лента ×0.5, исполнительная — реальная
    a, r = s.b10[t], s.b10_real[t]
    assert np.allclose(a[a["date"] == d]["close"].to_numpy() * 2, r[r["date"] == d]["close"].to_numpy())
    assert not np.allclose(a[a["date"] == d]["close"], r[r["date"] == d]["close"])     # лента действительно разная
    assert s.ist7().b10_real is s.b10_real and s.ist7().b10 is s.b10


def _ok_cena_real(s, ist7):
    e = object.__new__(EvaluatorV7)
    e.fdr, e.ist7, e._cena_T = None, ist7, "17:00"
    dates = pd.DatetimeIndex([pd.Timestamp("2018-02-01"), pd.Timestamp("2018-03-01")])
    got = e.fdr_kw()["cena_T"]("AAA", dates)
    r = s.b10_real["AAA"]
    ozhid = [float(r[(r["date"] == d) & (r["slot"] == (1020 - 600) // 10)]["open"].iloc[0]) for d in dates]      # open бара 17:00 реальной ленты
    assert np.allclose(got, ozhid, atol=1e-12, rtol=0)


def test_b1_cena_T_iz_realnoj_lenty_na_otsechke():
    s, _ = _ist()
    _ok_cena_real(s, s.ist7())


def test_b1_mutatsiya_cena_T_iz_skorrektirovannoj_lovitsya():
    s, _ = _ist()
    i7 = s.ist7()
    mut = Ist7(b10=i7.b10, panel=i7.panel, otn=i7.otn, b10_real=i7.b10)           # исполнительная лента = скорректированная
    with pytest.raises(AssertionError):
        _ok_cena_real(s, mut)


def test_b1_sha_ischet_realnuyu_lentu():
    s, raw = _ist()
    b = {t: v.copy() for t, v in s.b10_real.items()}
    b["AAA"].loc[b["AAA"].index[0], "close"] += 1.0
    assert E.sha_istochnikov(s.b10, s.panel, s.fon, b) != s.sha
    assert Ist7(b10=s.b10, b10_real=b).sha != Ist7(b10=s.b10, b10_real=s.b10_real).sha


# ---------------- Б3: граница ленты S10–S12 ----------------
def _lenta(start, period=None):
    import test_impuls_v7 as ti
    b = ti.b10_iz(lambda i, s: 100.0, 3, start=start)
    if period:
        b.attrs["period"] = period                                               # так ленту помечает загрузчик (eval_v7_ist.zagruzit)
    return b


def _ok_granitsa():
    assert period_v2.chtenie_konec("polnaya") == pd.Timestamp("2025-01-01") and period_v2.chtenie_konec("train4") == pd.Timestamp("2020-01-01")
    with pytest.raises(ValueError, match="2020-01-01"):                          # без метки периода — train4, как было
        impuls_v7.signal_dnya(_lenta("2022-01-10"), "S11", "18:30")
    impuls_v7.signal_dnya(_lenta("2022-01-10", "polnaya"), "S11", "18:30")       # 2022 — внутри зоны polnaya: не отвергается
    impuls_v7.signal_dnya(_lenta("2024-12-26", "polnaya"), "S10", "18:30")       # последний день 2024-12-30 < 2025
    with pytest.raises(ValueError, match="2025-01-01"):
        impuls_v7.signal_dnya(_lenta("2025-01-06", "polnaya"), "S10", "18:30")


def test_b3_granitsa_iz_period_v2_train4_po_umolchaniyu_polnaya_yavno():
    _ok_granitsa()


def test_b3_mutatsiya_granitsa_snyata_lovitsya(monkeypatch):
    monkeypatch.setattr(period_v2, "chtenie_konec", lambda p: pd.Timestamp("2100-01-01"))
    with pytest.raises(BaseException):                                           # граница не из period_v2 / снята — проверка падает
        _ok_granitsa()


