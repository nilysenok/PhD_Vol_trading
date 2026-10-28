"""П-11 (проверяющий): правила торговли v2, Ж-111 (журнал; п. 2–4, 7–8, 10, 12, 15) и Ж-114 п. 5. ДО кода V6-a/b/c: красные до кода V6. Skip/xfail нет.
Мутации владельца (Ж-111 п. 20): M1 уровни по d+1 (test_uroven_ne_po_d_plus_1, test_bezubytok_po_sigma_d); M2 пирамидинг против позиции
 (test_piramida_protiv_pozitsii); M3 заявка не снята при смене стороны (test_zayavka_snyata_pri_smene_storony). Свои: M4 заявка дольше T
 (test_zayavka_zhivet_T_dney); M5 reentry при «нет» (test_reentry); M6 третья добавка (test_piramida_ne_bolshe_dvuh); M7 риск > 2ρ
 (test_piramida_risk_2rho); M8 ступенчатый трейлинг ослабил стоп (test_trejling_stupen); M9 безубыток по d+1 (test_bezubytok_po_sigma_d);
 M10–M12 (ФДР t0 занятому, ранг по всему периоду, m(score) вне [0,25; 2]) — в test_p11_v2_c.py.
ПОДГОНКА П-11б (тестировщик, Ж-081) под код #499/#501/#504; требования те же. Причины: (1) Params: reentry→reentry_den, trail_step→step_s, plus→sp (имена кода);
 (2) go() не передавал T в test_zayavka_zhivet_T_dney / _smene_storony / _peresmotr_vecherom — ошибка теста (без T=1 заявка не живёт), добавлено T=…; (3) причины выхода в коде: безубыток "bu", трейлинг и «стоп в плюс» "trail",
 «выхода нет» = закрытие по сигналу ("signal"), а не exit_px None; (4) контроль σ d+1 в безубытке инвертирован в тесте: при σ_d (верно) выход 99,0, при σ d+1 порога 108,9 нет — теперь assert == 99,0;
 (5) риск 2ρ в коде — по расстоянию до стопа (Σ размер·(цена − стоп) ≤ 2·R), а не Σ frac ≤ 2: добавка 1,0 урезается до 0,8, пара 0,5 → 0,5 + 0,25; тест считает риск независимо. Толкование «риск ставки» — на подтверждение (класс A/B);
 (6) бары AFLT без сделок (NaN, 48 из 4558) — не бары, отбрасываются.
ДОПУЩЕНИЯ (были до кода; сигнатур V6 в спецификации §13 нет; п. А1–А2 после подгонки верны, А3 — устарело, см. test_p11_v2_c.py):
 А1. V6-a/b — `fastbt.exec_e.run(tk, days, sig, sigma, Params(...))`; новые поля Params: T (дней жизни заявки с первого вечера сигнала; после T заявка снята и
     не ставится заново, пока сигнал не исчез/сменил сторону), pir (0|0.5|1, добавка), trail_step (s, в σ), b0 (σ; безубыток: стоп = цена входа),
     plus=(p, q) (после +p·σ стоп на +q·σ). Уровни считаются по σ_d и close d вечером d; заявка на день j берёт уровень вечера j−1. reentry — прежнее поле.
 А2. Пирамидинг: k-я добавка (k = 1, 2) — стоп-заявка на P_вх·(1 ± k·σ_d) в сторону прибыли, цена max/min(open, уровень); добавка — Trade с
     parent = индекс родителя, frac = pir; риск ставки = Σ frac родителя и добавок ≤ 2. Ступень трейлинга: стоп += s·σ_d·P_вх за каждые +s·σ от входа, не ослабляется.
 А3. V6-c — модуль `fastbt.rang_sily`: rang_para(score, okno=252) -> ndarray (ранг score[d] в окне [d−okno, d−1], доля строго меньших), rang_obshchiy(S[тик, d], okno=252),
     m_score(rang, tip), razmer(rho, m, k_s, sigma, side=1) = side·ρ·m/(k_s·σ). ФДР: `fastbt.fdr.fdr_t0(noch, zan_den, rf_noch, rf_den)` по дням =
     (1 − noch)·rf_noch + (1 − zan_den)·rf_den; noch — занятость ночи, zan_den — средняя занятость по 5-мин барам дня.
Запуск: FASTBT_BARS1M=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/D25/bars1m python3 -m pytest tests/proverka_progon/test_p11_v2.py -q --tb=short"""
import numpy as np
import pytest

from fastbt import bars5m
from fastbt.exec_e import Day, Params, run

SG = 0.02
L = 99.0                      # лимит лонга: 100·(1 − 0,5·0,02)
ENT = (100.0, 100.0, 98.5, 100.0)   # бар входа по L


def D(rows):
    a = np.array(rows, float).reshape(-1, 4)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3], False)


def F(x, n=1):
    return [(x, x, x, x)] * n


def go(days, sig, sg=SG, **kw):
    s = np.full(len(days), sg) if np.isscalar(sg) else np.asarray(sg, float)
    kw.setdefault("k", 0.5)
    return run("T", days, sig, s, Params(**kw))


def touch(n=20, b=5, low=98.5):
    rows = F(100.0, n)
    rows[b] = (100.0, 100.0, low, 100.0)
    return D(rows)


def kids(r):
    return [t for t in r.trades if t.parent is not None]


# --- V6-a: заявка со сроком жизни ---
@pytest.mark.parametrize("T", [1, 2, 3, 5])
def test_zayavka_zhivet_T_dney(T):
    sig = [1] * (T + 1) + [0, 0]
    for j, n in ((T, 1), (T + 1, 0)):                      # на день T исполняется, на T+1 заявка снята (M4)
        days = [D(F(100.0, 20)) for _ in range(T + 3)]
        days[j] = touch()
        assert len(go(days, sig, T=T).trades) == n


def test_zayavka_snyata_pri_ischeznovenii():
    days = [D(F(100.0, 20)), D(F(100.0, 20)), touch(), D(F(100.0, 20))]
    assert go(days, [1, 0, 0, 0]).trades == []
    days[1], days[2] = touch(), D(F(100.0, 20))
    assert len(go(days, [1, 0, 0, 0]).trades) == 1       # контроль: на день 1 заявка ещё стоит


def test_zayavka_snyata_pri_smene_storony():              # M3
    days = [D(F(100.0, 20)), D(F(100.0, 20)), touch(), D(F(100.0, 20))]
    assert go(days, [1, -1, -1, -1], T=5).trades == []   # лонг-лимит снят вечером 1, шорт-лимит 101 не тронут
    days[1], days[2] = touch(), D(F(100.0, 20))
    t = go(days, [1, -1, -1, -1], T=5).trades
    assert len(t) == 1 and t[0].side == 1 and t[0].entry_day == 1


def test_uroven_peresmotr_vecherom():
    d1 = D([(100.0, 100.0, 100.0, 110.0)] + F(110.0, 19))
    d2 = D(F(110.0, 5) + [(110.0, 110.0, 108.5, 110.0)] + F(110.0, 14))
    t = go([D(F(100.0, 20)), d1, d2, D(F(110.0, 20))], [1, 1, 1, 0], T=3).trades
    assert len(t) == 1 and t[0].entry_day == 2 and t[0].entry_px == pytest.approx(108.9)


def test_uroven_ne_po_d_plus_1():                          # M1: σ и close дня d+1 в уровне
    d1 = D([ENT] + [(100.0, 100.0, 90.0, 90.0)] + F(90.0, 18))
    t = go([D(F(100.0, 20)), d1, D(F(90.0, 20))], [1, 0, 0], sg=[SG, 0.10, 0.10]).trades
    assert len(t) == 1 and t[0].entry_px == pytest.approx(L) and t[0].entry_bar == 0


def test_reentry():                                        # M5; Ж-111 п. 3
    rows = [(100.0, 100.0, 100.0, 100.0), ENT, (99.0, 99.0, 97.0, 98.0), (98.0, 100.0, 98.0, 100.0), ENT] + F(100.0, 15)
    days = [D(F(100.0, 20)), D(rows), D(F(100.0, 20))]
    on = go(days, [1, 1, 0], b=1.0, reentry_den=True).trades
    assert len(on) == 2 and on[0].reason == "stop" and on[1].entry_day == 1
    assert len(go(days, [1, 1, 0], b=1.0, reentry_den=False).trades) == 1


# --- V6-b: пирамидинг ---
def run_pir(path, sig=1, **kw):
    sig_arr = [sig, sig, 0]
    kw.setdefault("b", 4.0)
    return go([D(F(100.0, 20)), D(path + F(path[-1][3], 20 - len(path))), D(F(100.0, 20))], sig_arr, **kw)


UP = [ENT, (100.0, 101.5, 100.0, 101.5), (101.5, 103.5, 101.5, 103.5), (103.5, 105.5, 103.5, 105.5)]


def test_piramida_odna_dobavka_pri_plyus_1_sigma():
    r = run_pir(UP[:2], pir=1.0)
    k = kids(r)
    # риск 2ρ по расстоянию до стопа (код V6-b): R = 99·0,08 = 7,92; добавка на 100,98 при стопе 91,08: min(1; 7,92/9,9) = 0,8
    assert len(k) == 1 and k[0].entry_px == pytest.approx(99 * 1.02) and k[0].frac == pytest.approx(0.8) and k[0].side == 1
    assert r.trades[k[0].parent].parent is None
    assert kids(run_pir(UP[:2], pir=0.0)) == []


def test_piramida_ne_bolshe_dvuh():                         # M6
    k = kids(run_pir(UP, pir=0.5))
    assert [round(t.entry_px, 2) for t in k] == [100.98, 102.96]


def test_piramida_risk_2rho():                              # M7
    stop, r0 = 99 * 0.92, 99 * 0.08                          # стоп b = 4σ: 91,08; риск базовой ноги 7,92

    def risk(r):                                             # риск ставки = Σ размер·(цена входа − стоп)
        return sum(t.frac * (t.entry_px - stop) for t in r.trades)
    r = run_pir(UP, pir=1.0)
    assert risk(r) <= 2 * r0 + 1e-9 and len(kids(r)) == 1 and sum(t.frac for t in r.trades) == pytest.approx(1.8)
    r = run_pir(UP, pir=0.5)                                 # 0,5 на 100,98; вторая урезана до 0,25 (остаток риска 2,97 / 11,88)
    assert risk(r) <= 2 * r0 + 1e-9 and [t.frac for t in kids(r)] == pytest.approx([0.5, 0.25])
    assert risk(r) == pytest.approx(2 * r0)


def test_piramida_protiv_pozitsii():                        # M2
    dn = [ENT, (100.0, 100.0, 97.0, 97.0), (97.0, 97.0, 94.0, 94.0)]
    assert kids(run_pir(dn, pir=1.0)) == []
    sh_up = [(100.0, 101.5, 100.0, 100.0), (100.0, 104.5, 100.0, 104.5), (104.5, 108.0, 104.5, 108.0)]
    assert kids(run_pir(sh_up, sig=-1, pir=1.0)) == []                      # шорт: рост цены = против
    sh_dn = [(100.0, 101.5, 100.0, 100.0), (100.0, 100.0, 98.5, 98.5)]
    k = kids(run_pir(sh_dn, sig=-1, pir=1.0))
    assert len(k) == 1 and k[0].side == -1 and k[0].entry_px == pytest.approx(101 * 0.98)


def test_trejling_stupen():                                 # M8: s = 1, b = 2: S0 = 95,04, шаг 1,98
    path = [ENT, (100.0, 103.0, 100.0, 103.0), (103.0, 103.0, 100.0, 100.0), (100.0, 100.0, 98.5, 98.5)]
    t = run_pir(path, b=2.0, step_s=1.0).trades[0]
    assert t.reason == "trail" and t.exit_px == pytest.approx(99.0)          # 95,04 + 2·1,98, после отката не ослаблен
    assert run_pir(path, b=2.0).trades[0].reason == "signal"                 # контроль: без трейлинга стоп 95,04 не тронут, выход по сигналу


def test_bezubytok_po_sigma_d():                            # M9, M1: порог 99·(1 + 1σ_d) = 100,98
    path = [ENT, (100.0, 101.5, 100.0, 101.5), (101.5, 101.5, 98.5, 98.5)]
    t = run_pir(path, b=2.0, b0=1.0).trades[0]
    assert t.reason == "bu" and t.exit_px == pytest.approx(99.0)
    days = [D(F(100.0, 20)), D(path + F(98.5, 17)), D(F(100.0, 20))]
    assert go(days, [1, 1, 0], sg=[SG, 0.10, 0.10], b=2.0, b0=1.0).trades[0].exit_px == pytest.approx(99.0)   # σ_d, не σ d+1 (при d+1 порог 108,9, выхода нет)
    low = [ENT, (100.0, 100.5, 100.0, 100.5), (100.5, 100.5, 98.5, 98.5)]
    assert run_pir(low, b=2.0, b0=1.0).trades[0].reason == "signal"          # порог не достигнут: выход по сигналу, не bu


def test_stop_v_plyus():
    path = [ENT, (100.0, 102.0, 100.0, 102.0), (102.0, 102.0, 99.5, 99.5)]
    t = run_pir(path, b=2.0, sp=(1.5, 0.5)).trades[0]
    assert t.reason == "trail" and t.exit_px == pytest.approx(99 * 1.01)


# --- данные <= 2019 ---
def test_dannye_aflt_2018_invarianty(tmp_path):
    assert bars5m.SRC.exists(), "нужен FASTBT_BARS1M=…/data/regen/D25/bars1m"
    df = bars5m.load_bars5m("AFLT", "2018-03-01", "2018-04-30", cache=tmp_path)
    days = []
    for _, g in df.dropna(subset=["open", "high", "low", "close"]).groupby("date", sort=True):   # бары без сделок (NaN) — не бары
        days.append(Day(g.open.values, g.high.values, g.low.values, g.close.values, False))
    assert 30 < len(days) <= 45
    sig = np.ones(len(days))
    sig[-3:] = 0
    r = go(days, sig, sg=0.015, T=2, b=2.0, pir=0.5, step_s=1.0, b0=1.0, reentry_den=True)
    assert r.trades
    for t in r.trades:
        assert t.entry_day >= t.d_sig + 1
        d = days[t.entry_day]
        assert d.l[t.entry_bar] <= t.entry_px <= d.h[t.entry_bar] or t.parent is None and d.o[t.entry_bar] == t.entry_px
    assert all(r.trades[t.parent].parent is None and t.entry_px > r.trades[t.parent].entry_px for t in kids(r) if t.side == 1)
    fam = {}
    for i, t in enumerate(r.trades):
        key = i if t.parent is None else t.parent
        fam[key] = fam.get(key, 0.0) + t.frac
    assert all(v <= 2.0 + 1e-12 for v in fam.values())
