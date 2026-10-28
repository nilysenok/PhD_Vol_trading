"""П-2 (проверяющий): время баров и ось t_vyhod (Л-3 + Л-2), контракт progon_etap1 §1.1.
Код (fastbt/exec_e_vremya.py, Day.t, Result.x_t, Run2) пишется параллельно: до него тесты красные.
Мутации (в отдельной копии): bar_do берёт бар, кончающийся после hhmm; выход по сигналу в 15:00 дня x вместо x+1 (Д2);
выход по τ — не в день x; комиссия не в строке начала интервала (№5); стоп после лимита выхода не первый (№3);
open2 берёт close/не open первого бара; нет бара — выход не по close Н6; x_t считает выходы по close."""
import numpy as np
import pytest

from fastbt.exec_e import Day, Params, run

C = 0.0004


def tday(p, n=108, start=590, skip=()):
    """День с n барами от start (мин. от полуночи); close бара i = p + 0.01·i, open = close − 0.005, low = open, high = close."""
    idx = [i for i in range(n) if i not in skip]
    t = np.array([start + 5 * i for i in idx])
    c = np.array([p + 0.01 * i for i in idx])
    return Day(c - 0.005, c, c - 0.005, c, False, t=t)


# --- bar_do: индекс последнего бара с t + 5 <= hhmm ---
def test_bar_do_izvestnyj_otvet():
    from fastbt.exec_e_vremya import bar_do
    d = tday(100.0)
    assert (bar_do(d, 900), bar_do(d, 1020)) == (61, 85)


def test_bar_do_granitsa_vklyuchaet_konchayushchijsya_rovno():
    from fastbt.exec_e_vremya import bar_do
    d = tday(100.0)
    assert d.t[61] + 5 == 900 and d.t[62] + 5 > 900          # бар 61 кончается в 15:00, бар 62 — позже
    assert bar_do(d, 899) == 60 and bar_do(d, 900) == 61 and bar_do(d, 904) == 61 and bar_do(d, 905) == 62


def test_bar_do_korotkij_den_none():
    from fastbt.exec_e_vremya import bar_do
    assert bar_do(tday(100.0, n=40), 900) is None and bar_do(tday(100.0, n=40), 1020) is None


def test_bar_do_korotkij_den_do_14_55():
    from fastbt.exec_e_vremya import bar_do
    d = tday(100.0, n=61)                                      # последний бар 14:50–14:55 → индекс 60 при hhmm=900
    assert bar_do(d, 900) == 60


def test_bar_do_propushchennyj_bar_sdvigaet_indeks():
    from fastbt.exec_e_vremya import bar_do
    assert bar_do(tday(100.0, skip=(61,)), 900) == 60          # бара 14:55 нет — последний кончается в 14:55 (бар 60)
    assert bar_do(tday(100.0, skip=(10,)), 900) == 60          # дыра раньше — индекс сдвигается на 1


def test_bar_do_start_10_00():
    from fastbt.exec_e_vremya import bar_do
    assert bar_do(tday(100.0, n=100, start=600), 900) == 59


def test_day_t_po_umolchaniyu_none():
    assert Day(np.ones(1), np.ones(1), np.ones(1), np.ones(1)).t is None


# --- vyhod_plan ---
@pytest.mark.parametrize("tv,pr,ожид", [
    ("close", "signal", (0, None)), ("close", "vremya", (0, None)),
    ("1500", "vremya", (0, 900)), ("1500", "signal", (1, 900)),
    ("1700", "vremya", (0, 1020)), ("1700", "signal", (1, 1020)),
    ("open2", "vremya", (1, None)), ("open2", "signal", (1, None)),
])
def test_vyhod_plan(tv, pr, ожид):
    from fastbt.exec_e_vremya import vyhod_plan
    assert vyhod_plan(tv, pr) == ожид


# --- исполнение выхода: ось t_vyhod в движке ---
def lenta(n_after=1, **dk):
    return [tday(100.0 + d, **dk) for d in range(4 + n_after)]


def go(days, sig, **kw):
    kw = dict(dict(comm=C, tau=None), **kw)
    return run("T", days, np.asarray(sig), np.full(len(days), 0.02), Params(**kw))


SIG_X = [1, 1, 0, 0, 0]               # решение о выходе по сигналу — на close дня x = 2


@pytest.mark.parametrize("tv,bar", [("1500", 61), ("1700", 85)])
def test_signal_vyhod_na_x_plyus_1(tv, bar):
    days = lenta()
    r = go(days, SIG_X, t_vyhod=tv)
    t = r.trades[0]
    assert (t.exit_day, t.exit_bar, t.reason) == (3, bar, "signal")
    assert t.exit_px == days[3].c[bar]
    assert r.x_t == 1


def test_signal_open2():
    days = lenta()
    t = go(days, SIG_X, t_vyhod="open2").trades[0]
    assert (t.exit_day, t.exit_bar, t.reason, t.exit_px) == (3, 0, "signal", days[3].o[0])


def test_close_prezhnee_povedenie():
    days = lenta()
    r = go(days, SIG_X, t_vyhod="close")
    t = r.trades[0]
    assert (t.exit_day, t.exit_px, r.x_t) == (2, days[2].c[-1], 0)


def test_tau_vyhod_v_den_x_i_komissiya_v_nachale_intervala():
    days = lenta()
    r = go(days, [1] * 5, tau=2, reentry=False, t_vyhod="1500")
    t = r.trades[0]
    assert (t.exit_day, t.exit_bar, t.reason, t.exit_px) == (2, 61, "vremya", days[2].c[61])   # τ известен заранее: день x
    assert r.cost[0] == C and r.cost[1] == C and r.cost[2] == 0.0      # вход по close 0; выход внутри дня 2 — в строке 1
    assert r.gross[1] == pytest.approx(np.log(days[2].c[61] / days[1].c[-1]))


def test_tau_open2_na_sleduyushchij_den():
    days = lenta()
    t = go(days, [1] * 5, tau=2, reentry=False, t_vyhod="open2").trades[0]
    assert (t.exit_day, t.exit_bar, t.reason, t.exit_px) == (3, 0, "vremya", days[3].o[0])


def test_signal_komissiya_v_stroke_nachala_intervala_i_dokhod():
    days = lenta()
    r = go(days, SIG_X, t_vyhod="1500")
    assert r.cost[0] == C and r.cost[2] == C and r.cost[3] == 0.0         # выход дня 3 — в строке 2 (начало интервала)
    assert r.gross[2] == pytest.approx(np.log(days[3].c[61] / days[2].c[-1]))
    assert r.gross[3] == 0.0 and r.pos[2] == 1.0 and r.pos[3] == 0.0


def test_shortovyj_signal_1500():
    days = lenta()
    t = go(days, [-1, -1, 0, 0, 0], t_vyhod="1500").trades[0]
    assert (t.side, t.exit_day, t.exit_bar, t.exit_px) == (-1, 3, 61, days[3].c[61])


def test_stop_do_bar_do_idyot_pervym():
    days = lenta()
    c = days[3]
    o, h, lo, cl = (np.array(x) for x in (c.o, c.h, c.l, c.c))
    lo[10] = 90.0                                              # S = 101.07·0.98 ≈ 99.05: стоп на баре 10 дня x+1
    days[3] = Day(o, h, lo, cl, False, t=c.t)
    r = go(days, SIG_X, b=1.0, t_vyhod="1500")
    t = r.trades[0]
    assert (t.reason, t.exit_day, t.exit_bar) == ("stop", 3, 10)
    assert t.exit_px == pytest.approx(days[0].c[-1] * 0.98)       # вход по close дня 0, стоп b·σ̂ = 2 %
    assert r.x_t == 0                                          # стоп — не «не по close»


def test_korotkij_den_vykhod_po_close_N6():
    days = lenta()
    days[3] = tday(103.0, n=40)
    r = go(days, SIG_X, t_vyhod="1500")
    t = r.trades[0]
    assert (t.exit_day, t.reason, t.exit_px) == (3, "signal", days[3].c[-1])
    assert r.x_t == 0
    r2 = run("T", days, np.asarray(SIG_X), np.full(5, 0.02), Params(comm=C, tau=None, t_vyhod="1500"), close=[101, 102, 103, 104.5, 105])
    assert r2.trades[0].exit_px == 104.5                       # close Н6 дня исполнения, не последний 5-мин бар


@pytest.mark.parametrize("tv", ["close", "1500", "1700", "open2"])
def test_net_vyhoda_po_signalu_ne_menyaet_nichego(tv):
    days = lenta()
    r = go(days, [1, 1, 1, 1, 1], t_vyhod=tv)                  # ни τ, ни сигнала: позиция открыта на конце окна (Н5)
    assert r.trades[0].reason is None and r.x_t == 0
    assert r.cost[-1] == C                                     # комиссия выхода на последней строке


# --- mutations.md № 2–5 на пути Run2 (новые поля включены — движок Run2, не прежний Run) ---
def r2(days, sig, **kw):
    kw = dict(dict(comm=C, tick=0.01), **kw)
    return run("T", days, np.asarray(sig), np.full(len(days), 0.02), Params(**kw))


def flat(x, n=3):
    return Day(*(np.full(n, x),) * 4, False)


def bars(o, h, lo, c):
    return Day(*(np.array(v, float) for v in (o, h, lo, c)), False)


def test_no2_kasanie_ne_ispolnenie_na_run2():
    for low in (99.01, 99.0):                                  # L = 99: на тик выше и касание — не исполнение
        d1 = bars([100] * 3, [100] * 3, [100, low, 100], [100] * 3)
        assert r2([flat(100.0), d1, flat(100.0)], [1, 0, 0], k=0.5, zhizn=12).trades == []


def test_no3_stop_i_tejk_v_odnom_bare_stop_pervyj():
    d1 = bars([100, 100], [100, 103], [100, 97], [100, 100])
    t = r2([flat(100.0), d1, flat(100.0)], [1, 1, 1], a=1.0, b=1.0, stop_tik=1).trades[0]
    assert (t.reason, t.exit_px) == ("stop", pytest.approx(97.99))


def test_no4_gep_cherez_stop_po_open():
    d1 = bars([95, 95], [96, 96], [94, 94], [95, 95])
    t = r2([flat(100.0), d1, flat(100.0)], [1, 1, 1], b=1.0, stop_tik=1).trades[0]
    assert (t.reason, t.exit_px) == ("stop", 95.0)


def test_no5_komissiya_vykhoda_na_konce_okna():
    r = r2([flat(100.0)] * 3, [1, 1, 1], stop_tik=1, b=1.0)
    assert r.trades[0].reason is None and r.cost[0] == C and r.cost[-1] == C
