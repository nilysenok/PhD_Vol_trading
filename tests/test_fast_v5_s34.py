"""Ф-сл2 (лист 2 из П-0): ядро fast_exec для осей S3/S4 — stop_zayavka + urovni, atr_m, T, reentry_den, chast_1r / chast.
Эталон — медленный путь `exec_e.run`; `fast_exec.run_v5` обязан совпасть ПОБАЙТНО (допуск 0, proverka.md п. 3).
1) синтетика: 1200 путей (600 стоп-заявок S3/S4, 600 лимитных с T / reentry_den), дни pusto, дивиденды, close Н6, тик;
2) бары train4 (2016–2019, AFLT/SBER): 50 точек S3 и 50 точек S4 на тикер, уровни канала и Supertrend из дневных баров;
3) крайние ленты (пустая, 1 день, все pusto, все уровни nan, нулевая σ); 4) мутации эталона (monkeypatch) расходятся с ядром;
5) замер с/точку (малая проверка, один процесс).
FASTBT_BARS1M=<основная копия>/data/regen/D25/bars1m python3 -m pytest tests/test_fast_v5_s34.py -q"""
import dataclasses
import time

import numpy as np
import pytest

from fastbt import exec_e, exec_e_run, exec_e_zayavki, fast_exec as fx
from fastbt.exec_e_types import Day, Params
from fastbt.signals.s34 import rolling_ext, supertrend
from test_fast_etap1_smoke import TICK, _days, _same

N_SZ, N_LIM = 600, 600
NAN = float("nan")


def ch(rng, *v):
    return v[int(rng.integers(len(v)))]


def tochka(i: int) -> Params:
    """i < N_SZ — стоп-заявки S3/S4 (k_e, atr_m, tejk a, chast); иначе лимит с T > 1 и reentry_den (atr_m, chast, сигнальные выходы)."""
    rng = np.random.default_rng(20261029 + i)
    sz = i < N_SZ
    atr_m = ch(rng, None, 1.0, 2.0, 3.0)
    b = None if (atr_m and rng.random() < 0.2) else ch(rng, 1.0, 2.0, 3.0)
    cf = ch(rng, None, None, "1r", (0.5, 1.0), (1 / 3, 1.5), (0.5, 2.0)) if b is not None else None
    kw = dict(chast_1r=cf == "1r", chast=cf if isinstance(cf, tuple) else None)
    if sz:
        kw.update(vhod_tip="stop_zayavka", exit_signal=False, stop_touch=True, k_e=ch(rng, 0.0, 0.0, -0.5, 0.25))
    else:
        kw.update(k=ch(rng, 0.25, 0.5, 1.0), T=ch(rng, 1, 2, 3), reentry_den=bool(i % 2), f=ch(rng, "propusk", "zakrytie_d1", "povtor"),
                  exit_signal=ch(rng, True, False), stop_touch=ch(rng, True, False), vyhod_lim=ch(rng, False, False, True),
                  reentry=ch(rng, True, False))
    return Params(b=b, a=ch(rng, None, 1.0, 2.0), tau=ch(rng, None, 3, 5), s=ch(rng, "fiks", "ezhedn"), comm=0.0004,
                  side=ch(rng, 1, -1), tick=ch(rng, None, 0.0, TICK), k_tr=ch(rng, None, 1.5) if b else None,
                  be=ch(rng, False, True) if b else False, atr_m=atr_m, **kw)


def lenta(seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 100))
    days = _days(rng, n)
    sig = np.repeat(rng.choice([-1, 0, 1], n // 3 + 1), 3)[:n].astype(float)
    sigma = rng.uniform(0.005, 0.03, n)
    sigma[rng.random(n) < 0.05] = NAN
    close = np.array([d.c[-1] * (1 + rng.normal(0, 0.002)) if not d.pusto else 1.0 for d in days])
    div = np.where(rng.random(n) < 0.05, rng.uniform(0.5, 3.0, n), 0.0)
    c = np.array([NAN if d.pusto else d.c[-1] for d in days])
    for k in range(1, n):
        c[k] = c[k - 1] if np.isnan(c[k]) else c[k]
    c[0] = 100.0 if np.isnan(c[0]) else c[0]
    u = lambda s, w: np.where(rng.random(n) < 0.1, NAN, c * (1 + s * rng.uniform(0, w, n)))   # noqa: E731
    urov = dict(up=u(1, 0.02), dn=u(-1, 0.02), xl=u(-1, 0.03), xs=u(1, 0.03))
    return days, sig, sigma, (close if rng.random() < 0.7 else None), (div if rng.random() < 0.5 else None), urov


def sverit(days, sig, sigma, p, close, div, urov) -> bool:
    kw = dict(close=close, div=div, urovni=urov if p.vhod_tip == "stop_zayavka" else None)
    try:
        a = exec_e.run("T", days, sig, sigma, p, **kw)
    except ValueError as e:
        with pytest.raises(ValueError) as ei:
            fx.run_v5("T", days, sig, sigma, p, **kw)
        return str(ei.value) == str(e)
    try:
        _same(a, fx.run_v5("T", days, sig, sigma, p, **kw))
    except AssertionError:
        return False
    return True


def test_sintetika_yadro_ravno_etalonu_na_1200_putyah():
    plokho, cnt = [], dict(kanal=0, part=0, trail=0, povtor_den=0, T=0, fills=0)
    for i in range(N_SZ + N_LIM):
        p = tochka(i)
        assert not fx.ne_znaet(p), p
        days, sig, sigma, close, div, urov = lenta(10_000 + i)
        if not sverit(days, sig, sigma, p, close, div, urov):
            plokho.append(i)
            continue
        try:
            r = exec_e.run("T", days, sig, sigma, p, close=close, div=div, urovni=urov if i < N_SZ else None)
        except ValueError:
            continue
        rs = [t.reason for t in r.trades]
        cnt["kanal"] += "kanal" in rs
        cnt["part"] += "part" in rs
        cnt["trail"] += p.atr_m is not None and "trail" in rs
        cnt["povtor_den"] += any(a.exit_day == b.entry_day and a.exit_bar is not None and b.entry_bar > a.exit_bar
                                 for a, b in zip(r.trades, r.trades[1:]))
        cnt["fills"] += r.n_fills > 0
    assert not plokho, plokho[:10]
    assert min(cnt["kanal"], cnt["part"], cnt["trail"], cnt["povtor_den"]) >= 20 and cnt["fills"] >= 800, cnt   # оси реально работают


def test_osi_menyayut_lentu_i_T_zhivet():
    """Ось не «ничего не делает»: T > 1 даёт заявок больше, чем T = 1, хотя бы на части лент."""
    bol = 0
    for i in range(N_SZ, N_SZ + 120):
        p = dataclasses.replace(tochka(i), T=3, k=0.5, f="propusk", vyhod_lim=False)
        days, sig, sigma, close, div, _ = lenta(10_000 + i)
        try:
            a = exec_e.run("T", days, sig, sigma, p, close=close, div=div)
            b = exec_e.run("T", days, sig, sigma, dataclasses.replace(p, T=1), close=close, div=div)
        except ValueError:
            continue
        bol += a.n_orders > b.n_orders
    assert bol >= 10, bol


def test_krajnie_lenty():
    e, one = np.empty(0), Day(np.array([100.0]), np.array([101.0]), np.array([99.0]), np.array([100.0]), False)
    pust = Day(e, e, e, e, True)
    u = lambda n, v: dict(up=np.full(n, v), dn=np.full(n, v), xl=np.full(n, v), xs=np.full(n, v))     # noqa: E731
    for days in ([], [one], [one, one], [pust, pust, pust], [one, pust, one, one]):
        n = len(days)
        for sigma in (np.full(n, 0.02), np.zeros(n), np.full(n, NAN)):
            for v in (NAN, 100.5, 99.5):
                for i in (0, N_SZ + 1):
                    p = tochka(i)
                    assert sverit(days, np.ones(n), sigma, p, None, None, u(n, v)), (n, v, i)


# --- бары train4 (≤ 2019): 50 точек S3 и 50 точек S4 на тикер ---
def urovni_s3(days, w):
    h, l = np.array([max(d.h) for d in days]), np.array([min(d.l) for d in days])
    hc, lc = rolling_ext(h, w, True), rolling_ext(l, w, False)
    return dict(up=hc, dn=lc, xl=lc, xs=hc)


def urovni_s4(days, per, mult):
    h, l, c = (np.array([f(d) for d in days]) for f in (lambda d: max(d.h), lambda d: min(d.l), lambda d: d.c[-1]))
    st = supertrend(h, l, c, per, mult)
    return dict(up=np.where(st > c, st, NAN), dn=np.where(st < c, st, NAN), xl=st, xs=st)


def tochka_real(i: int, s4: bool) -> tuple:
    rng = np.random.default_rng(777 + i + 1000 * s4)
    p = dataclasses.replace(tochka(i % N_SZ), tau=ch(rng, None, 10, 20), s="fiks", tick=ch(rng, None, 0.01))
    return p, (ch(rng, 10, 15, 20, 25), ) if not s4 else (ch(rng, 10, 14, 20), ch(rng, 2.0, 3.0, 3.5))


def prover_real(args) -> list:
    t, days, sigma, i, s4 = args
    p, par = tochka_real(i, s4)
    urov = urovni_s4(days, *par) if s4 else urovni_s3(days, *par)
    rng = np.random.default_rng(i)
    div = np.where(rng.random(len(days)) < 0.004, rng.uniform(0.5, 3.0, len(days)), 0.0)
    ok = sverit(days, np.ones(len(days)), sigma, p, None, div, urov)
    nt = len(fx.run_v5(t, days, np.ones(len(days)), sigma, p, div=div, urovni=urov).trades)
    return [(t, i, s4, ok, nt)]


@pytest.fixture(scope="module")
def realnye():
    from test_fast_v5_ekviv import _realnye
    return {t: _realnye(t) for t in ("AFLT", "SBER")}


def test_realnye_S3_S4_50_tochek_pobajtno(realnye):
    import multiprocessing as mp
    zad = [(t, d, s, i, s4) for t, (d, s) in realnye.items() for s4 in (False, True) for i in range(50)]
    with mp.get_context("fork").Pool(2) as pool:
        rows = [r for part in pool.map(prover_real, zad, chunksize=5) for r in part]
    assert len(rows) == 200
    assert not [r for r in rows if not r[3]], [r for r in rows if not r[3]][:10]
    for s4 in (False, True):                                                    # точки реально торгуют (не пустые ленты)
        assert sum(r[4] > 20 for r in rows if r[2] == s4) >= 60, (s4, sorted(r[4] for r in rows if r[2] == s4))


def test_zamer_s_na_tochku(realnye, capsys):
    """G4 (малая проверка, 1 процесс): с/точку эталон / ядро на 20 точках S3 и S4 (AFLT, train4); потолок — одна машина."""
    days, sigma = realnye["AFLT"]
    kw = dict(close=None, div=None)
    for s4 in (False, True):
        pts = [(tochka_real(i, s4)[0], (urovni_s4 if s4 else urovni_s3)(days, *tochka_real(i, s4)[1])) for i in range(20)]
        fx.run_v5("T", days, np.ones(len(days)), sigma, pts[0][0], urovni=pts[0][1], **kw)     # компиляция вне замера
        t = []
        for f in (exec_e.run, fx.run_v5):
            t0 = time.perf_counter()
            for p, u in pts:
                f("T", days, np.ones(len(days)), sigma, p, urovni=u if p.vhod_tip == "stop_zayavka" else None, **kw)
            t.append((time.perf_counter() - t0) / len(pts))
        with capsys.disabled():
            print(f"\nзамер {'S4' if s4 else 'S3'}: эталон {t[0]:.3f} с/точку, ядро {t[1]:.3f} с/точку, ×{t[0] / t[1]:.1f}")
        assert t[1] < t[0]


# --- мутации эталона: ядро честное, эталон с подложенной ошибкой — сверка обязана разойтись (и мутант действует) ---
def _kasanie(mp):                          # (а) лимит исполняется по касанию (l ≤ L / h ≥ L), а не проходом на тик
    f = lambda sd, h, l, L, tick: (l <= L) if sd > 0 else (h >= L)         # noqa: E731
    mp.setattr(exec_e_run, "limit_mask", f)
    mp.setattr(exec_e_zayavki, "limit_mask", f)


def _stop_po_S(mp):                        # (б) «уже пробитый» стоп в баре входа исполняется по уровню S, а не по цене входа
    def f(s, i, j, d, b, px, S):
        if S is None:
            return False
        hit, xp = exec_e_run.stop_hit(s.tr.side, px, d.h[b], d.l[b], S, s.p.stop_touch)
        if hit:
            s._exit(i, j, b, S, s.sr)
        return hit
    mp.setattr(exec_e_run.Run, "_stop_vhoda", f)


def _reentry_v_bare(mp):                   # (в) повторный вход в тот же бар, что и выход (b0 = бар выхода, а не следующий)
    orig = exec_e_zayavki.reenter

    def f(s, i, j, d):
        x = s.last_x
        if x is not None:
            x.exit_bar -= 1
        try:
            orig(s, i, j, d)
        finally:
            if x is not None:
                x.exit_bar += 1
    mp.setattr(exec_e_run, "reenter", f)


LIM = range(N_SZ, N_SZ + N_LIM)
MUT = {"kasanie": (_kasanie, lambda p: dataclasses.replace(p, tick=TICK), LIM),
       "stop_po_S": (_stop_po_S, lambda p: dataclasses.replace(p, chand=2.0, b=p.b or 1.0), LIM),   # «пробитый» стоп на входе — у Chandelier
       "reentry_v_bare": (_reentry_v_bare, lambda p: dataclasses.replace(p, reentry_den=True, exit_signal=True, stop_touch=True), LIM)}


@pytest.mark.parametrize("name", list(MUT))
def test_mutatsiya_etalona_lovitsya(name, monkeypatch):
    mut, osi, rng = MUT[name]
    chislo = 0
    for i in rng:
        p = osi(tochka(i))
        days, sig, sigma, close, div, urov = lenta(20_000 + i)
        kw = dict(close=close, div=div, urovni=urov if p.vhod_tip == "stop_zayavka" else None)
        try:
            ok = fx.run_v5("T", days, sig, sigma, p, **kw)
        except (ValueError, NotImplementedError):
            continue
        _same(exec_e.run("T", days, sig, sigma, p, **kw), ok)                           # без мутации — равны
        with monkeypatch.context() as m:
            mut(m)
            try:
                bad = exec_e.run("T", days, sig, sigma, p, **kw)
            except ValueError:
                chislo += 1
                continue
        try:
            _same(bad, ok)
        except AssertionError:
            chislo += 1
    assert chislo >= 3, f"мутация {name} не поймана: расхождений {chislo}"
