"""Д-4: остановка торгов 28.02–23.03.2022 (справка И-1), аудит на синтетике, без данных >= 2022.
Факты И-1: 3 недели без торгов, 24.03 — сокращённая сессия на 33 акциях, до 31.03 нет бара 18:40–18:50.
Строки ряда = торговые дни (разрыв — нет строк или pusto-строки). xfail(strict) = дефект (правка — отдельный лист).
Мутации: № 4 (гэп по open, а не по уровню), окно по календарю вместо строк, 18:50 в дне."""
import math

import numpy as np
import pandas as pd
import pytest

from fastbt import daily_1850, dividendy, msp, sigma_real, sobytiya_e1
from fastbt.etap1_data import align_f
from fastbt.exec_e import Params, atr_series, run
from test_exec_e import D, F

GAP = [D([], True)] * 5       # строки без баров внутри разрыва (вариант «календарь с pusto»)


def go(days, sig, sigma=0.02, **kw):
    kw.setdefault("comm", 0.0)
    return run("T", days, sig, np.full(len(days), sigma), Params(**kw))


def ts(day, hm):
    return pd.Timestamp(f"{day} {hm}")


def bars10(day, hms, px):
    return pd.DataFrame({"timestamp": [ts(day, h) for h in hms], "open": px, "high": px, "low": px, "close": px, "volume": 10})


# --- (1) окна σ̂/RV: по строкам, не по календарю ---
def _rv():
    d = list(pd.bdate_range("2022-02-14", "2022-02-25")) + list(pd.bdate_range("2022-03-24", "2022-04-05"))
    return pd.Series(np.linspace(1e-4, 4e-4, len(d)), index=pd.DatetimeIndex(d, name="date"))


def check_okno_po_strokam(f):
    rv, w = _rv(), 5
    i = list(rv.index).index(pd.Timestamp("2022-03-24"))
    return f(rv, w).iloc[i] == pytest.approx(math.sqrt(rv.iloc[i - w + 1:i + 1].mean()))


def test_sigma_okno_po_strokam_cherez_razryv():
    assert check_okno_po_strokam(sigma_real.sigma_rv)


def test_mut_okno_po_kalendaryu_lovitsya():
    def kalendar(rv, w):                       # подложено: окно w календарных строк (разрыв заполнен NaN)
        full = rv.reindex(pd.date_range(rv.index[0], rv.index[-1], freq="D"))
        return np.sqrt(full.rolling(w, min_periods=1).mean()).reindex(rv.index)
    assert not check_okno_po_strokam(kalendar)


def test_ewma_cherez_razryv_bez_dyr():
    assert sigma_real.sigma_ewma(pd.Series(np.full(19, 1e-4), index=_rv().index)).isna().all()      # EWMA_MIN = 22 > 19
    s = sigma_real.sigma_ewma(pd.Series(np.full(30, 1e-4), index=pd.bdate_range("2022-02-14", periods=30)))
    assert np.allclose(s.dropna(), 1e-2) and s.notna().sum() == 9


def test_align_f_den_bez_sigma_nan_bez_podstanovki():     # тикер без бара 24.03 → NaN, не перенос (Д4)
    s = _rv().drop(pd.Timestamp("2022-03-24"))
    v = align_f(s, pd.DatetimeIndex(_rv().index))
    assert np.isnan(v[list(_rv().index).index(pd.Timestamp("2022-03-24"))]) and np.isfinite(v).sum() == len(v) - 1


@pytest.mark.xfail(strict=True, reason="ДЕФЕКТ Д4-1: RV сокращённой сессии 24.03 (часть баров) входит в окно как полный день и занижает "
                   "σ̂. Правка: день с числом баров < доли (напр. 0,5) от медианы полного — NaN в окне rv (отдельный лист)")
def test_sokrashchennaya_sessiya_ne_zanizhaet_sigma():
    full = [f"{h:02d}:{m:02d}" for h in range(10, 18) for m in (0, 10, 20, 30, 40, 50)]
    rng = np.random.default_rng(1)
    bars = []
    for day in pd.bdate_range("2022-03-14", "2022-03-23").append(pd.DatetimeIndex(["2022-03-25"])):
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, len(full))))
        bars.append(bars10(day.date().isoformat(), full, px))
    short = bars10("2022-03-24", full[:4], [100, 100.2, 100.1, 100.3])        # 4 бара из 48
    rv = sigma_real.rv_1850(pd.concat(bars + [short]))
    base = sigma_real.rv_1850(pd.concat(bars))
    ref = math.sqrt(base.loc[:"2022-03-23"].mean())                          # σ̂ по полным дням до сокращённой сессии
    assert sigma_real.sigma_rv(rv, 5).loc["2022-03-24"] >= 0.97 * ref


# --- (2) сборщик дневки: нет бара 18:40 до 31.03, сокращённая сессия ---
def test_aggregate_bez_bara_1840_close_posledniy_bar():
    b = pd.concat([bars10("2022-03-28", ["09:50", "12:00", "18:30"], 100.0),
                   bars10("2022-03-28", ["18:50"], 999.0)])
    r = daily_1850.aggregate(b, "T").iloc[0]
    assert (r["close"], r["high"], r["volume"]) == (100.0, 100.0, 30)           # 18:50 отброшен, день не потерян


def test_mut_aggregate_vklyuchaet_bar_1850_lovitsya():
    b = pd.concat([bars10("2022-03-28", ["09:50", "18:30"], 100.0), bars10("2022-03-28", ["18:50"], 999.0)])
    assert daily_1850.aggregate(b, "T", cut_min=18 * 60 + 51).iloc[0]["close"] == 999.0   # мутация видна, штатная — 100
    assert daily_1850.aggregate(b, "T").iloc[0]["close"] == 100.0


def test_aggregate_tiker_bez_barov_net_stroki_a_s_odnim_baro_est():
    b = pd.concat([bars10("2022-03-24", ["10:00"], 50.0), bars10("2022-03-25", ["10:00", "18:40"], 51.0)])
    r = daily_1850.aggregate(b, "T")
    assert list(r["date"]) == [pd.Timestamp("2022-03-24"), pd.Timestamp("2022-03-25")]
    assert pd.isna(sigma_real.rv_1850(b.iloc[:1]).get(pd.Timestamp("2022-03-24")))      # один бар: RV нет → σ̂ NaN, вход не ставится


def test_vhod_pri_sigma_nan_ne_ustanavlivaetsya():
    days = [D(F(100.0)), D(F(100.0)), D(F(100.0))]
    assert go(days, [1, 1, 0], sigma=np.nan).trades == []


# --- (3) позиция через разрыв: стоп/тейк по open первого дня (мутация № 4), срок в строках ---
@pytest.mark.parametrize("gap", [[], GAP], ids=["строк_нет", "pusto"])
def test_stop_cherez_razryv_po_open(gap):
    days = [D(F(100.0))] + gap + [D([(90, 91, 89, 90)]), D(F(90.0))]
    r = go(days, [1] * len(days), b=1.0)                 # стоп 98; open 90 < 98
    t = r.trades[0]
    assert (t.reason, t.exit_px, t.exit_day) == ("stop", 90.0, 1 + len(gap))
    assert t.exit_px != pytest.approx(98.0)                            # мутация № 4: по уровню
    assert r.gross.sum() == pytest.approx(math.log(90 / 100))


@pytest.mark.parametrize("gap", [[], GAP], ids=["строк_нет", "pusto"])
def test_tejk_i_shortovyj_stop_cherez_razryv_po_open(gap):
    n = len(gap)
    t = go([D(F(100.0))] + gap + [D([(110, 111, 109, 110)])], [1] * (n + 2), a=1.0).trades[0]    # тейк 102, open 110
    assert (t.reason, t.exit_px) == ("take", 110.0)
    s = go([D(F(100.0))] + gap + [D([(110, 111, 109, 110)])], [1] * (n + 2), side=-1, b=1.0).trades[0]   # шорт, стоп 102
    assert (s.reason, s.exit_px) == ("stop", 110.0)


def test_pozitsiya_derzhitsya_v_razryve_bez_pnl_i_izderzhek():
    days = [D(F(100.0))] + GAP + [D(F(100.0))]
    r = go(days, [1] * 7, comm=0.0004)
    assert list(r.pos[:7]) == [1.0] * 7 and (r.gross[1:6] == 0).all() and (r.cost[1:6] == 0).all()


def test_tau_v_torgovyh_dnyah_a_ne_v_kalendarnyh_strokah():
    sig = [1] * 9
    a = go([D(F(100.0))] + GAP + [D(F(100.0))] * 3, sig, tau=2).trades[0]       # торговые строки: 0, 6, 7, 8
    b = go([D(F(100.0))] * 4, [1] * 4, tau=2).trades[0]                          # те же 4 торговые строки без разрыва
    assert (a.reason, b.reason) == ("vremya", "vremya")
    assert a.exit_day == 7 and b.exit_day == 2                                   # 2 торговых дня: 6 → 7, а не 0 + 2


def test_atr_cherez_razryv_po_torgovym_dnyam_i_gep_v_tr():
    days = [D([(100, 102, 98, 100)]), D([(100, 102, 98, 100)])] + GAP + [D([(90, 92, 88, 90)])]
    a = np.array(atr_series(days, n=2))
    assert math.isnan(a[0]) and a[1] == pytest.approx(4.0)
    assert (a[2:7] == a[1]).all()                                                # pusto переносит прошлое
    assert a[7] == pytest.approx((4.0 + 12.0) / 2)                              # TR = max(4, |92−100|, |88−100|) = 12, окно — 2 торговых дня


def test_zayavka_perezhivaet_razryv__reshenie_vladeltsa():
    """ХАРАКТЕРИЗАЦИЯ, не дефект: лимит d стоит до первой торговой строки после разрыва (3 недели) и исполняется по open гэпа.
    Политика (снимать ли заявки и сигнал d на остановке) — решение владельца."""
    r = go([D(F(100.0))] + GAP + [D([(90, 91, 89, 90)])], [1] * 7, k=0.5)
    t = r.trades[0]
    assert (t.entry_day, t.entry_px, t.d_sig) == (6, 90.0, 0)


# --- (4) ставка, дивиденды, события ---
def test_stavka_nachislyaetsya_za_vse_kalendarnye_dni_razryva():
    cal = msp.cal_days(pd.DatetimeIndex(["2022-02-25", "2022-03-24", "2022-03-25"]))
    assert list(cal) == [27.0, 1.0, 1.0]


def test_dividend_v_pusto_perehodit_na_sleduyushchiy_torgovyj_den():
    days = [D(F(100.0))] + GAP + [D(F(100.0))]
    div = np.zeros(7)
    div[3] = 1.0                                                                 # отсечка в разрыве
    r = run("T", days, np.ones(7), np.full(7, 0.02), Params(comm=0.0), div=div, nalog=1.0)
    assert r.gross.sum() == pytest.approx(math.log(101 / 100))                   # D получена один раз, на первой торговой строке


def test_dividend_gap_divs_den_gepa_pervyj_bar_posle_last_cum():
    dates = np.array(["2022-02-24", "2022-02-25", "2022-03-24", "2022-03-25"], dtype="datetime64[ns]")
    ev = pd.DataFrame({"last_cum_date": pd.to_datetime(["2022-02-25"]), "dps_rub": [5.0]})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dividendy, "events", lambda t: ev)
        assert list(dividendy.gap_divs("X", dates)) == [0, 0, 5.0, 0]          # ex-день — 24.03, первый бар после last_cum


def test_last_cum_reestr_v_razryve_t_plus_2_po_torgovym_dnyam():
    cal = pd.to_datetime(["2022-02-21", "2022-02-22", "2022-02-24", "2022-02-25", "2022-03-24", "2022-03-25"]).values
    mp = pytest.MonkeyPatch()
    mp.setattr(dividendy, "_calendar", lambda: cal)
    try:   # реестр 10.03 в разрыве: покупка 24.02 рассчитывается на 24.03 (> реестра), поэтому последний cum — 22.02
        r = dividendy.last_cum(pd.Series(pd.to_datetime(["2022-03-10"])))
        assert r.iloc[0] == pd.Timestamp("2022-02-22")
    finally:
        mp.undo()


def test_cbr_zasedanie_v_razryve_blok_na_poslednem_torgovom_dne_do_razryva():
    """Аналог на датах 2019 (события < 2022 по зоне). Заседание внутри разрыва и известное заранее блокирует вход в день
    перед разрывом (j−1): безвредно, но семантика «d+1» условна — в отчёте как замечание, не дефект."""
    d = pd.DatetimeIndex(["2019-02-22", "2019-03-25", "2019-03-26"])
    cbr = pd.DataFrame({"date": pd.to_datetime(["2019-03-11"]), "known_from": pd.to_datetime(["2019-01-01"])})
    assert list(sobytiya_e1.cbr_block(d, cbr)) == [True, False, False]
    cbr["known_from"] = pd.to_datetime(["2019-03-11"])                          # объявлено внутри разрыва → PIT: не блокируем
    assert not sobytiya_e1.cbr_block(d, cbr).any()
