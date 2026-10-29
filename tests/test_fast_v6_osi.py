"""Ф-v6-osi: оси исполнения куба v6 в ядре fast_exec (fast_exec_isp): gap_propusk, zhizn, вход ed (ось ed_lim), t_vyhod 1500 / 1700.
Эталон — exec_e.run (Run2: gap_over, zhizn_hits, _place, _leave, _scan, exec_e_vremya); допуск 0 (побайтно, включая x_t).
1) по оси: >= 1000 синтетических путей ядро = эталон; ось реально работает (результат без оси отличается в >= 100 путях);
2) мутации по оси (параметр оси сброшен, off-by-one по жизни заявки / бару hhmm, ed отброшен или без nan) — сверка обязана упасть;
3) pokryto: оси приняты под флагом YADRO_ISP_GOTOV, лесенка — тоже, снят флаг — эталон.
FASTBT_V6_OSI_N — число путей на ось (по умолчанию 1000)."""
import dataclasses
import os

import numpy as np
import pytest

from fastbt import exec_e, fast_etap1 as fe, fast_exec as fx
from fastbt.exec_e_types import Day, Params
from test_fast_open2 import _eq, _res

TICK = 0.01
N_PUT = int(os.environ.get("FASTBT_V6_OSI_N", "1000"))
N_MUT = 300


def _day(rng, px):
    kind = int(rng.integers(0, 12))
    if kind == 0:
        e = np.empty(0)
        return Day(e, e, e, e, True), px
    k = int(rng.choice([2, 5, 13, 30, 48, 55, 62, 70, 90, 108]))      # 48–55: день кончается до 14:00–14:30 (короткий для 15:00)
    start = [590, 600, 600, 590, 600, 605, 600, 590, 600, 610, 600, 600][kind]
    c = np.round(px * np.exp(np.cumsum(rng.normal(0, 0.004, k))) / TICK) * TICK
    o = np.r_[np.round(px * np.exp(rng.normal(0, 0.02)) / TICK) * TICK, c[:-1]]
    h = np.maximum(o, c) + TICK * rng.integers(0, 25, k)
    lo = np.minimum(o, c) - TICK * rng.integers(0, 25, k)
    t = None if kind == 8 and rng.random() < 0.7 else np.arange(start, start + 5 * k, 5)
    return Day(o, h, lo, c, False, t), float(c[-1])


def lenta(seed):
    rng = np.random.default_rng(seed)
    n, px, days = int(rng.integers(8, 40)), 100.0, []
    for _ in range(n):
        d, px = _day(rng, px)
        days.append(d)
    sig = np.repeat(rng.choice([-1, 0, 1], n // 2 + 1), rng.integers(1, 4, n // 2 + 1))[:n]
    sig = np.r_[sig, np.zeros(n - len(sig))].astype(float)
    sigma = rng.uniform(0.005, 0.03, n)
    close = np.array([d.c[-1] * (1 + rng.normal(0, 0.002)) if not d.pusto else 1.0 for d in days])
    div = np.where(rng.random(n) < 0.05, rng.uniform(0.5, 3.0, n), 0.0)
    ed = rng.uniform(0.003, 0.05, n)
    ed[rng.random(n) < 0.12] = np.nan                                 # ATR ещё не набран / нет бара
    return days, sig, sigma, (close if rng.random() < 0.6 else None), (div if rng.random() < 0.5 else None), ed


def tochka(seed, **kw) -> Params:
    rng = np.random.default_rng(4242 + seed)
    ch = lambda *v: v[int(rng.integers(len(v)))]                       # noqa: E731
    b = ch(None, 1.0, 2.0, 3.0)
    base = dict(k=ch(0.25, 0.5, 1.0, 0.25), f=ch("propusk", "zakrytie_d1", "povtor"), a=ch(None, 1.0, 2.0), b=b,
                tau=ch(None, 1, 2, 3, 5), s=ch("fiks", "ezhedn"), comm=0.0004, exit_signal=ch(True, True, False),
                side=ch(1, -1), tick=ch(None, 0.0, TICK), stop_touch=ch(True, False), k_tr=ch(None, 1.5) if b else None,
                be=ch(False, True) if b else False, reentry=ch(True, False), reentry_den=ch(False, False, True),
                T=ch(1, 1, 2), check="bary")
    return Params(**{**base, **kw})


def _run(f, p, seed, ed_on=True, **kw):
    days, sig, sigma, close, div, ed = lenta(70_000 + seed)
    return _res(lambda *a, **k: f(*a, **k, ed=ed if ed_on else None, **kw), days, sig, sigma, p, close, div)


def _sverka(osi, n=N_PUT, ed_on=False, f=None):
    """(плохие номера путей, число путей, где ось меняет результат против точки без неё). f — подменённый run_v5."""
    plokho, deistvuet = [], 0
    for s in range(n):
        p = tochka(s, **osi(s))
        ref = _run(exec_e.run, p, s, ed_on)
        if not _eq(ref, _run(f or fx.run_v5, p, s, ed_on)):
            plokho.append(s)
        else:
            deistvuet += not _eq(ref, _run(exec_e.run, tochka(s), s, False))
    return plokho, deistvuet


OSI = dict(gap=lambda s: dict(gap_propusk=True), zhizn=lambda s: dict(zhizn=(12, 36, 60, 12)[s % 4]),
           t_hh=lambda s: dict(t_vyhod=("1500", "1700")[s % 2]), ed=lambda s: {},
           vse=lambda s: dict(gap_propusk=bool(s % 2), zhizn=(None, 12, 36, 60)[s % 4], t_vyhod=("close", "open2", "1500", "1700")[s % 4]))


@pytest.mark.parametrize("os_", ["gap", "zhizn", "t_hh", "ed", "vse"])
def test_osi_yadro_ravno_etalonu_i_os_deistvuet(os_):
    plokho, deist = _sverka(OSI[os_], ed_on=os_ in ("ed", "vse"))
    assert not plokho, plokho[:10]
    assert deist >= (0 if os_ == "vse" else 100), deist


def test_t_vyhod_vyhody_na_bare_hhmm_idut_po_sroku_i_signalu():
    tau = sig = 0
    for s in range(N_PUT):
        ref = _run(exec_e.run, tochka(s, **OSI["t_hh"](s)), s)
        if not isinstance(ref, str) and ref.x_t > 0:
            tau += any(t.reason == "vremya" for t in ref.trades)
            sig += any(t.reason == "signal" for t in ref.trades)
    assert min(tau, sig) >= 80, (tau, sig)


def test_dlina_ed_kak_u_etalona():
    days, sig, sigma, close, div, ed = lenta(70_001)
    r, q = (_res(lambda *x, **k: f(*x, **k, ed=ed[:-1]), days, sig, sigma, tochka(1), close, div) for f in (exec_e.run, fx.run_v5))
    assert isinstance(r, str) and r == q


# --- мутации: сверка обязана упасть на подложенной ошибке ---
def _mut_params(monkeypatch, **idx):
    orig = fx.params_v5

    def mut(p, nal):
        pf, pi = orig(p, nal)
        for name, f in idx.items():
            pi[getattr(fx, name)] = f(pi[getattr(fx, name)])
        return pf, pi
    monkeypatch.setattr(fx, "params_v5", mut)


def _cut(monkeypatch, f):
    orig = fx.isp.cut_hhmm
    monkeypatch.setattr(fx.isp, "cut_hhmm", lambda days, p: f(orig(days, p)))


MUT = {"gap_otklyuchen": ("gap", 5, lambda m: _mut_params(m, P_GAP=lambda v: 0)),
       "zhizn_off_by_one": ("zhizn", 5, lambda m: _mut_params(m, P_ZH=lambda v: v + 1)),
       "zhizn_ne_zadana": ("zhizn", 20, lambda m: _mut_params(m, P_ZH=lambda v: 0)),
       "t_vyhod_kak_close": ("t_hh", 30, lambda m: _mut_params(m, P_TH=lambda v: 0)),
       "bar_hhmm_so_sdvigom": ("t_hh", 30, lambda m: _cut(m, lambda c: np.where(c >= 0, c + 1, -1))),
       "bar_hhmm_ne_zadan": ("t_hh", 30, lambda m: _cut(m, lambda c: np.full(len(c), -1, np.int64)))}


@pytest.mark.parametrize("nazv", list(MUT))
def test_mutatsiya_lovitsya(nazv, monkeypatch):
    os_, nuzhno, podlozhit = MUT[nazv]
    podlozhit(monkeypatch)
    assert len(_sverka(OSI[os_], N_MUT)[0]) >= nuzhno


@pytest.mark.parametrize("nazv, ed_mut, nuzhno", [("ed_otbroshen", lambda ed: None, 20), ("ed_nan_zamenen", lambda ed: np.nan_to_num(ed, nan=0.02), 10)])
def test_mutatsiya_ed_lovitsya(nazv, ed_mut, nuzhno):
    """Нечисловая ed[i] — заявки нет (Run2._place); ядро без ed или с nan, замененным числом, расходится с эталоном."""
    orig = fx.run_v5
    mut = lambda *a, **k: orig(*a, **{**k, "ed": ed_mut(k["ed"])})       # noqa: E731
    assert len(_sverka(OSI["ed"], N_MUT, True, mut)[0]) >= nuzhno


# --- покрытие ---
def _pt(**kw):
    return dict(ed_lim="sigma", zhizn="den", lestnica="net", gap="vhod", k_x="net", storona="obe", tejk_signal="net", rho="net",
                m_tip="1", rang_tip="tikernyj", **kw)


def test_pokryto_osi_isp_pod_flagom(monkeypatch):
    osi = [dict(gap="propusk"), dict(zhizn="12"), dict(zhizn="60"), dict(ed_lim="pct"), dict(ed_lim="diap"), dict(ed_lim="atr14"),
           dict(lestnica="0.5"), dict(lestnica="1.0"), dict(t_vyhod="1500"), dict(t_vyhod="1700")]
    assert all(fe.pokryto({**_pt(), **o}) for o in osi)
    assert fe.pokryto({**_pt(), "gap": "propusk", "zhizn": "36", "ed_lim": "atr14", "lestnica": "1.0", "t_vyhod": "1700"})
    assert not any(fe.pokryto({**_pt(), **o}) for o in (dict(gap="zakr"), dict(zhizn="7"), dict(ed_lim="mediana"), dict(lestnica="dva")))
    monkeypatch.setattr(fx, "YADRO_ISP_GOTOV", False)
    assert not any(fe.pokryto({**_pt(), **o}) for o in osi)


def test_run_otdayot_yadru_pri_flage_i_etalonu_bez(monkeypatch):
    """fast_etap1.run берёт вызов с ed / zhizn / gap ядром при флаге и эталоном — без; результат тот же."""
    p = dataclasses.replace(tochka(3), k=0.5, zhizn=12, gap_propusk=True)
    days, sig, sigma, close, div, ed = lenta(70_003)
    ref, calls, orig = _res(lambda *a, **k: exec_e.run(*a, **k, ed=ed), days, sig, sigma, p, close, div), [], fx.run_v5
    monkeypatch.setattr(fx, "run_v5", lambda *a, **k: calls.append(1) or orig(*a, **k))
    go = lambda: _res(lambda *a, **k: fe.run(*a, **k, ed=ed), days, sig, sigma, p, close, div)   # noqa: E731
    assert _eq(ref, go()) and calls
    calls.clear()
    monkeypatch.setattr(fx, "YADRO_ISP_GOTOV", False)
    assert _eq(ref, go()) and not calls
