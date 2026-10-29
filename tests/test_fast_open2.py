"""Ф-open2 (Ж-137 п. 6): ядро fast_exec знает t_vyhod = open2 — выход по сигналу дня x по open бара 10:00 дня x+1 (стоп и тейк
строго до него), выход по сроку τ — по close дня истечения; эталон — exec_e.run (Run2, exec_e_vremya.vyhod_plan). Допуск 0.
1) известный ответ (лента из test_vyhod_tau_open2): ядро = эталон = ручные числа;
2) 1200 синтетических путей побайтно (Result, включая x_t): дни с барами от 09:50 / 10:00 / 10:05, короткие дни до 10:00, дни без времени,
   пустые дни, сигнал и τ в один день, сигнал на последнем дне окна, шорт, гэпы, дивиденды, close Н6;
3) мутации: ядро без open2 (pi[P_TV] = 0) расходится с эталоном; пустое покрытие по t_vyhod ловится;
4) pokryto: open2 принят, 1500 / 1700 — нет.
Реальные пути (≤ 2021): FASTBT_V5_N=200 pytest tests/test_fast_v5_pokryto.py (оси кубов S1…S6 с якорем open2)."""
import dataclasses

import numpy as np
import pytest

from fastbt import exec_e, fast_etap1 as fe, fast_exec as fx
from fastbt.exec_e_types import Day, Params
from test_vyhod_tau_open2 import lenta as lenta_ruchn, rn

TICK = 0.01
N_PUT = 1200


def _day(rng, px):
    kind = int(rng.integers(0, 10))
    if kind == 0:
        e = np.empty(0)
        return Day(e, e, e, e, True), px
    k = int(rng.integers(1, 14))
    start = [590, 600, 605, 540, 540, 590, 600, 590, 610, 600][kind]      # 540: день целиком до 10:00; 610: бара 10:00 нет, первый позже
    c = np.round(px * np.exp(np.cumsum(rng.normal(0, 0.012, k))) / TICK) * TICK
    o = np.r_[np.round(px * np.exp(rng.normal(0, 0.02)) / TICK) * TICK, c[:-1]]           # гэп открытия
    h = np.maximum(o, c) + TICK * rng.integers(0, 40, k)
    lo = np.minimum(o, c) - TICK * rng.integers(0, 40, k)
    t = None if kind == 8 and rng.random() < 0.7 else np.arange(start, start + 5 * k, 5)
    return Day(o, h, lo, c, False, t), float(c[-1])


def lenta(seed):
    rng = np.random.default_rng(seed)
    n, px, days = int(rng.integers(8, 70)), 100.0, []
    for _ in range(n):
        d, px = _day(rng, px)
        days.append(d)
    sig = np.repeat(rng.choice([-1, 0, 1], n // 2 + 1), rng.integers(1, 4, n // 2 + 1))[:n]
    sig = np.r_[sig, np.zeros(n - len(sig))].astype(float)
    sigma = rng.uniform(0.005, 0.03, n)
    close = np.array([d.c[-1] * (1 + rng.normal(0, 0.002)) if not d.pusto else 1.0 for d in days])
    div = np.where(rng.random(n) < 0.05, rng.uniform(0.5, 3.0, n), 0.0)
    return days, sig, sigma, (close if rng.random() < 0.6 else None), (div if rng.random() < 0.5 else None)


def tochka(seed) -> Params:
    rng = np.random.default_rng(777 + seed)
    ch = lambda *v: v[int(rng.integers(len(v)))]                       # noqa: E731
    b = ch(None, 1.0, 2.0, 3.0)
    return Params(k=ch(None, None, 0.25, 0.5), f=ch("propusk", "zakrytie_d1", "povtor"), a=ch(None, 1.0, 2.0), b=b,
                  tau=ch(None, 1, 2, 3, 5), s=ch("fiks", "ezhedn"), comm=0.0004, exit_signal=ch(True, True, True, False),
                  side=ch(1, -1), tick=ch(None, 0.0, TICK), stop_touch=ch(True, False), k_tr=ch(None, 1.5) if b else None,
                  be=ch(False, True) if b else False, reentry=ch(True, False), reentry_den=ch(False, False, True),
                  T=ch(1, 1, 2), t_vyhod="open2", check="bary")


def _res(f, days, sig, sigma, p, close, div):
    try:
        return f("T", days, sig, sigma, p, close=close, div=div)
    except ValueError as e:
        return str(e)


def _eq(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    for f in ("gross", "cost", "pos", "free", "fdr"):
        if getattr(a, f).tobytes() != getattr(b, f).tobytes():
            return False
    k = ("n_orders", "n_fills", "x_orders", "x_fills", "x_t")
    return ([getattr(a, x) for x in k] == [getattr(b, x) for x in k]
            and [dataclasses.astuple(t) for t in a.trades] == [dataclasses.astuple(t) for t in b.trades])


def _ob(seed):
    p = tochka(seed)
    return (p,) + lenta(50_000 + seed)


def test_sintetika_open2_yadro_ravno_etalonu_na_1200_putyah():
    plokho, cnt = [], dict(open_x1=0, tau=0, short=0, poslednij=0, ne_znaet=0)
    for s in range(N_PUT):
        p, days, sig, sigma, close, div = _ob(s)
        cnt["ne_znaet"] += bool(fx.ne_znaet(p))
        ref = _res(exec_e.run, days, sig, sigma, p, close, div)
        if not _eq(ref, _res(lambda *a, **k: fx.run_v5(*a, **k), days, sig, sigma, p, close, div)):
            plokho.append(s)
            continue
        if isinstance(ref, str):
            continue
        cnt["open_x1"] += ref.x_t > 0
        cnt["short"] += any(t.side < 0 and t.reason == "signal" for t in ref.trades)
        cnt["tau"] += any(t.reason == "vremya" for t in ref.trades)
        cnt["poslednij"] += any(t.reason == "signal" and t.exit_day != t.entry_day and t.exit_bar is not None
                                and t.exit_day == max(i for i, d in enumerate(days) if not d.pusto) for t in ref.trades)
    print("покрытие синтетики:", cnt)
    assert not plokho, plokho[:10]
    assert cnt["ne_znaet"] == 0 and min(cnt["open_x1"], cnt["tau"], cnt["short"]) >= 150, cnt        # оси реально работают


def test_izvestnyj_otvet_yadro_ravno_ruchnomu_chislu():
    kw = dict(comm=0.0, check="bary", t_vyhod="open2")
    for sig, tau, ozhid in (((1, 1, 1, 1, 1), 2, (2, 103.0, "vremya")),     # τ: close дня истечения
                            ((1, 0, 0, 0, 0), None, (2, 102.0, "signal")),    # сигнал x=1: open x+1
                            ((1, 1, 0, 0, 0), 2, (2, 103.0, "vremya")),       # сигнал и τ в один день x=2: выигрывает срок
                            ((1, 0, 0, 0, 0), 3, (2, 102.0, "signal")),       # τ ещё не наступил
                            ((-1, 0, 0, 0, 0), None, None)):
        days, p = lenta_ruchn(), Params(**kw, tau=tau, side=1 if sig[0] > 0 else -1)
        sg = np.full(len(days), 0.05)
        r = fx.run_v5("T", days, np.array(sig, float), sg, p)
        e = exec_e.run("T", days, np.array(sig, float), sg, p)
        assert _eq(e, r)
        t = r.trades[0]
        if ozhid:
            assert (t.exit_day, t.exit_px, t.reason) == ozhid
    # сигнал на последнем дне окна: выхода на следующий день нет — close последнего дня
    days = lenta_ruchn()
    r = fx.run_v5("T", days, np.array([1, 1, 1, 1, 0], float), np.full(5, 0.05), Params(comm=0.0, check="bary", t_vyhod="open2"))
    assert (r.trades[0].exit_day, r.trades[0].exit_px, r.trades[0].reason) == (4, 121.0, "signal")


def test_shorty_i_gep_cherez_stop_do_10_00():
    """Шорт, выход по сигналу — open 10:00 x+1; гэп вверх на open через стоп (b = 1, σ = 0,05) выходит по стопу до 10:00 не раньше."""
    for seed in range(40):
        p = dataclasses.replace(tochka(seed), side=-1, exit_signal=True, k=None)
        days, sig, sigma, close, div = lenta(900 + seed)
        assert _eq(_res(exec_e.run, days, sig, sigma, p, close, div), _res(fx.run_v5, days, sig, sigma, p, close, div))


def test_mutatsiya_yadro_bez_open2_lovitsya(monkeypatch):
    """Мутация «сигнал по close x в ядре»: параметр P_TV обнулён — ядро считает выход по close дня решения; сверка обязана упасть."""
    orig = fx.params_v5

    def mut(p, nal):
        pf, pi = orig(p, nal)
        pi[fx.P_TV] = 0
        return pf, pi

    monkeypatch.setattr(fx, "params_v5", mut)
    plokho = 0
    for s in range(300):
        p, days, sig, sigma, close, div = _ob(s)
        plokho += not _eq(_res(exec_e.run, days, sig, sigma, p, close, div), _res(fx.run_v5, days, sig, sigma, p, close, div))
    assert plokho >= 30, plokho


def test_mutatsiya_cut_ne_zadan_lovitsya(monkeypatch):
    """Мутация: cut_open2 отдаёт «времени нет» (−1) — ядро выходит по close Н6 дня x+1, а не по open 10:00; сверка обязана упасть."""
    monkeypatch.setattr(fx, "cut_open2", lambda days, p: np.full(len(days), -1, np.int64))
    plokho = 0
    for s in range(300):
        p, days, sig, sigma, close, div = _ob(s)
        plokho += not _eq(_res(exec_e.run, days, sig, sigma, p, close, div), _res(fx.run_v5, days, sig, sigma, p, close, div))
    assert plokho >= 20, plokho


def test_pokryto_open2_prinyat_1500_net():
    pt = dict(ed_lim="sigma", zhizn="den", gap="vhod", k_x="net", storona="obe", tejk_signal="net", rho="net", m_tip="1",
              rang_tip="tikernyj")
    assert fe.pokryto({**pt, "t_vyhod": "close"}) and fe.pokryto({**pt, "t_vyhod": "open2"})
    assert not fe.pokryto({**pt, "t_vyhod": "1500"}) and not fe.pokryto({**pt, "t_vyhod": "1700"})
