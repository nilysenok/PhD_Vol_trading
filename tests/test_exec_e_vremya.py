"""Л-3: время 5-минутных баров — bar_do, vyhod_plan, Day.t в build_days, t в ключе кэша (Д5).

Мутации (в отдельной копии, см. описание PR): bar_do берёт бар, кончающийся после hhmm (`t <= hhmm`);
сигнал в 15:00 дня x (vyhod_plan по signal → 0); t не в _flat; build_days не заполняет t.
"""
import numpy as np
import pandas as pd
import pytest

from fastbt import cache
from fastbt import etap1 as E
from fastbt import etap1_data as D
from fastbt.exec_e_types import Day
from fastbt.exec_e_vremya import bar_do, vyhod_plan


def tday(p=100.0, n=108, start=590, skip=()):
    """День с n барами от start (минуты от полуночи; 590 = 09:50); пропуски — skip. Цены — для ключа и проверок."""
    idx = [i for i in range(n) if i not in skip]
    t = np.array([start + 5 * i for i in idx], dtype=np.int64)
    c = np.array([p + 0.01 * i for i in idx])
    return Day(c - 0.005, c, c - 0.005, c, False, t=t)


# --- bar_do: известный ответ ---
def test_bar_do_izvestnyj_otvet_108_barov_s_0950():
    d = tday(n=108, start=590)
    assert (bar_do(d, 900), bar_do(d, 1020)) == (61, 85)      # 900 = 15:00, 1020 = 17:00 (минуты)


def test_bar_do_granica_15_00_vklyuchaet_bar_konchayushchijsya_rovno():
    d = tday()
    assert d.t[61] + 5 == 900 and d.t[62] + 5 > 900          # 61: 14:55–15:00; 62: 15:00–15:05
    assert (bar_do(d, 899), bar_do(d, 900), bar_do(d, 904), bar_do(d, 905)) == (60, 61, 61, 62)


def test_bar_do_korotkij_den_do_14_00_none():
    assert bar_do(tday(n=40), 900) is None and bar_do(tday(n=40), 1020) is None   # последний бар 13:05


def test_bar_do_den_do_14_55_dayet_indeks():
    assert bar_do(tday(n=61), 900) == 60                       # последний бар 14:50–14:55, 14:55 нет


def test_bar_do_propushchennyj_bar_14_55_sdvigaet_indeks():
    assert bar_do(tday(skip=(61,)), 900) == 60                 # 14:55 нет: последний — 14:50 (индекс 60)
    assert bar_do(tday(skip=(10,)), 900) == 60                 # дыра раньше — индекс сдвинут на 1


def test_bar_do_bez_vremeni_none():
    assert bar_do(Day(np.ones(1), np.ones(1), np.ones(1), np.ones(1)), 900) is None
    assert bar_do(Day(np.empty(0), np.empty(0), np.empty(0), np.empty(0), True), 900) is None


# --- vyhod_plan: таблица контракта §1.1 ---
@pytest.mark.parametrize("tv,pr,ozhid", [
    ("close", "signal", (0, None)), ("close", "vremya", (0, None)),
    ("1500", "vremya", (0, 900)), ("1500", "signal", (1, 900)),
    ("1700", "vremya", (0, 1020)), ("1700", "signal", (1, 1020)),
    ("open2", "vremya", (1, None)), ("open2", "signal", (1, None)),
])
def test_vyhod_plan_tablica(tv, pr, ozhid):
    assert vyhod_plan(tv, pr) == ozhid


def test_vyhod_plan_signal_15_00_ne_v_den_x():
    # мутация «сигнал в 15:00 дня x»: сдвиг по сигналу обязан быть 1 (Д2)
    assert vyhod_plan("1500", "signal")[0] == 1


@pytest.mark.parametrize("tv,pr", [("1500", "stop"), ("abc", "signal"), ("1530", "vremya")])
def test_vyhod_plan_nedopustimye_znacheniya(tv, pr):
    with pytest.raises(ValueError):
        vyhod_plan(tv, pr)


# --- Д5: t входит в ключ кэша, если задан ---
def test_d5_t_v_chasti_klyucha_menyaet_klyuch():
    a = tday()
    b = Day(a.o, a.h, a.l, a.c, False, t=a.t + 1)           # те же цены, другое время баров
    assert np.array_equal(E._flat(a)[0], E._flat(b)[0]) and E._flat(a)[1] == E._flat(b)[1]   # цены и pusto одинаковы
    assert not np.array_equal(E._flat(a)[-1], E._flat(b)[-1])                                # различается только t
    assert cache.key("etap1_inputs", *E._flat(a)) != cache.key("etap1_inputs", *E._flat(b))


def test_d5_day_bez_t_daet_prezhnie_chasti():
    a = Day(np.ones(2), np.ones(2), np.ones(2), np.ones(2), False)
    assert len(E._flat(a)) == 2                             # v1.0-путь: t = None не добавляет частей


# --- build_days: Day.t из времени баров источника ---
def _bars(day, n, start=590, pusto=False):
    ts = pd.Timestamp(day) + pd.to_timedelta(start + 5 * np.arange(n), unit="min")
    c = 100.0 + 0.01 * np.arange(n)
    return pd.DataFrame(dict(ts=ts, date=pd.DatetimeIndex(ts).normalize(), open=c - 0.005, high=c, low=c - 0.005,
                             close=c, pusto=np.full(n, pusto)))


def test_build_days_zapolnyaet_t_i_pusto_none(monkeypatch):
    bars = _bars("2019-06-03", 108)
    monkeypatch.setattr(D, "load_bars5m", lambda *a, **k: bars)
    dates = np.array(["2019-06-03", "2019-06-04", "2019-06-05"], dtype="datetime64[D]")
    days, _ = D.build_days("T", dates, np.array([1.0, 1.0, 1.0]), 0, 2, pd.Timestamp("2019-12-31"))
    assert days[0].t[:3].tolist() == [590, 595, 600] and len(days[0].t) == 108
    assert (bar_do(days[0], 900), bar_do(days[0], 1020)) == (61, 85)
    assert days[1].pusto and days[1].t is None               # день без баров
    assert days[2].t is None                                 # строка i1 — одна свеча close Н6


def test_build_days_pustoj_bar_ne_vhodit_v_t(monkeypatch):
    bars = _bars("2019-06-03", 108)
    bars.loc[61, "pusto"] = True                             # бар 14:55 пустой: в день не входит
    monkeypatch.setattr(D, "load_bars5m", lambda *a, **k: bars)
    dates = np.array(["2019-06-03", "2019-06-04"], dtype="datetime64[D]")
    days, _ = D.build_days("T", dates, np.array([1.0, 1.0]), 0, 1, pd.Timestamp("2019-12-31"))
    assert len(days[0].t) == 107 and bar_do(days[0], 900) == 60
