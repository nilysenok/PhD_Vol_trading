"""Ф-сл3 (лист 3 из П-0): оси ядра b0, b0_c, sp, step_s, a_rang и вход rang в fast_exec.kernel_v5 (fastbt/fast_exec_v6.py).
Эталон — `exec_e.run` (Run + V6); `fast_exec.run_v5` обязан совпасть ПОБАЙТНО (допуск 0, proverka.md п. 3). pir — вне ядра (3c).
1) синтетика: 1200 путей (ранг с nan, дни pusto, дивиденды, close Н6, тик); 2) 50 точек на реальных барах train4 (AFLT, ≤ 2019);
3) крайние ленты; 4) мутации эталона (ранг дня входа, «стоп в плюс» в том же баре) и мутации ядра — тест обязан падать.
FASTBT_BARS1M=<основная копия>/data/regen/D25/bars1m python3 -m pytest tests/test_fast_v5_osi3.py -q"""
import dataclasses

import numpy as np
import pytest

from fastbt import exec_e, exec_e_v6, fast_exec as fx, fast_exec_v6 as v6
from fastbt.exec_e_types import Day, Params
from test_fast_etap1_smoke import TICK, _same
from test_fast_v5_s34 import ch, lenta

N = 1200
NAN = float("nan")


def tochka(i: int) -> Params:
    """Лимитный вход (podtverzhdenie) с b и bary: ось (b0 | b0_c | be | sp | step_s | a_rang) в разных сочетаниях."""
    rng = np.random.default_rng(20261030 + i)
    b = ch(rng, 1.0, 2.0, 3.0)
    kw = {}
    k_osi = i % 6
    if k_osi == 0:
        kw["b0"] = ch(rng, 0.5, 1.0, 1.5)
    elif k_osi == 1:
        kw["b0_c"] = ch(rng, 1.0, 2.0)
    elif k_osi == 2:
        kw["sp"] = ch(rng, (1.0, 0.25), (1.5, 0.5), (2.0, 0.0))
    elif k_osi == 3:
        kw["step_s"] = ch(rng, 0.5, 1.0)
    elif k_osi == 4:
        kw["a_rang"] = ch(rng, (1.0, 1.0), (0.5, 2.0), (2.0, 0.0))
    else:
        kw.update(b0=0.75, sp=(1.0, 0.25), step_s=ch(rng, 0.5, 1.0), a_rang=(1.5, 1.0))
    if k_osi < 5 and rng.random() < 0.4:                     # вторая ось поверх (b0/b0_c/be взаимоисключающие)
        kw.setdefault("sp", ch(rng, (1.0, 0.25), (2.0, 0.5)))
        kw.setdefault("a_rang", (1.0, 1.0)) if "b0" not in kw else None
    if "b0" not in kw and "b0_c" not in kw and rng.random() < 0.3:
        kw["be"] = True
    return Params(b=b, k=ch(rng, 0.25, 0.5, 1.0), f=ch(rng, "propusk", "zakrytie_d1", "povtor"), tau=ch(rng, None, 3, 5),
                  s=ch(rng, "fiks", "ezhedn"), comm=0.0004, side=ch(rng, 1, -1), tick=ch(rng, None, 0.0, TICK),
                  k_tr=ch(rng, None, 1.5), exit_signal=ch(rng, True, False), stop_touch=ch(rng, True, False),
                  reentry=ch(rng, True, False), **kw)


def rang_of(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.uniform(0, 1, n)
    r[rng.random(n) < 0.1] = NAN
    return r


def sverit(days, sig, sigma, p, close, div, rang) -> bool:
    kw = dict(close=close, div=div, rang=rang)
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
    plokho, cnt = [], dict(trail=0, bu=0, take=0, rano=0)
    for i in range(N):
        p = tochka(i)
        assert not fx.ne_znaet(p), p
        days, sig, sigma, close, div, _ = lenta(30_000 + i)
        rang = rang_of(i, len(days))
        if not sverit(days, sig, sigma, p, close, div, rang):
            plokho.append(i)
            continue
        r = exec_e.run("T", days, sig, sigma, p, close=close, div=div, rang=rang)
        rs = [t.reason for t in r.trades]
        cnt["trail"] += (p.sp is not None or p.step_s is not None) and "trail" in rs
        cnt["bu"] += (p.b0 is not None or p.b0_c is not None) and "bu" in rs
        cnt["take"] += p.a_rang is not None and "take" in rs
    assert not plokho, plokho[:10]
    assert min(cnt["trail"], cnt["bu"], cnt["take"]) >= 20, cnt          # оси реально работают, не «ничего не делают»


def test_krajnie_lenty():
    e, one = np.empty(0), Day(np.array([100.0]), np.array([101.0]), np.array([99.0]), np.array([100.0]), False)
    pust = Day(e, e, e, e, True)
    for days in ([], [one], [one, one], [pust, pust, pust], [one, pust, one, one]):
        n = len(days)
        for sigma in (np.full(n, 0.02), np.zeros(n), np.full(n, NAN)):
            for rang in (np.full(n, NAN), np.full(n, 0.5), np.zeros(n), np.ones(n)):
                for i in (0, 1, 2, 3, 4, 5):
                    assert sverit(days, np.ones(n), sigma, tochka(i), None, None, rang), (n, i)


def test_otkazy_te_zhe():
    days, sig, sigma, close, div, _ = lenta(31_000)
    for p, rang in ((dataclasses.replace(tochka(1), b0_c=1.0), None),                 # b0_c без ранга
                    (dataclasses.replace(tochka(4), a_rang=(1.0, 1.0)), rang_of(1, len(days) - 1)),   # ранг не той длины
                    (dataclasses.replace(tochka(0), b0=1.0, be=True), None)):          # be и b0 вместе
        assert sverit(days, sig, sigma, p, close, div, rang), p


def test_pir_teper_v_yadre():
    assert fx.ne_znaet(dataclasses.replace(tochka(0), pir=0.5)) == []      # 3c: пирамидинг в ядре (tests/test_fast_v5_pir.py)


@pytest.fixture(scope="module")
def aflt():
    from test_fast_v5_ekviv import _realnye
    return _realnye("AFLT")


def test_realnye_50_tochek_pobajtno(aflt):
    days, sigma = aflt
    n, ok, torg = len(days), [], 0
    for i in range(50):
        rng = np.random.default_rng(i)
        p = dataclasses.replace(tochka(i), tau=ch(rng, None, 10, 20), s="fiks")
        sig = np.repeat(rng.choice([-1, 0, 1], n // 4 + 1), 4)[:n].astype(float)
        rang = rang_of(i, n)
        div = np.where(rng.random(n) < 0.004, rng.uniform(0.5, 3.0, n), 0.0)
        ok.append(sverit(days, sig, sigma, p, None, div, rang))
        torg += len(fx.run_v5("T", days, sig, sigma, p, div=div, rang=rang).trades) > 20
    assert all(ok), [i for i, x in enumerate(ok) if not x]
    assert torg >= 30, torg


# --- мутации: тест обязан падать ---
def _rang_dnya_vhoda(mp):                 # эталон: b0_c / a_rang берут ранг дня входа, а не дня сигнала (mutations: сдвиг входа)
    orig = exec_e_v6.V6.v6_enter

    def f(s, d_sig):
        orig(s, s.trades[-1].entry_day)
    mp.setattr(exec_e_v6.V6, "v6_enter", f)


def _plus_v_tom_zhe_bare(mp):             # эталон: sp включается по high того же бара, где проход (pq_on сразу)
    mp.setattr(exec_e_v6, "plus_level", lambda sd, px, sg, q: px * (1 + sd * (q + 0.5) * sg))


@pytest.mark.parametrize("name,osi", [("rang_dnya_vhoda", (1, 4)), ("plus_level", (2,))])
def test_mutatsiya_etalona_lovitsya(name, osi, monkeypatch):
    mut = {"rang_dnya_vhoda": _rang_dnya_vhoda, "plus_level": _plus_v_tom_zhe_bare}[name]
    chislo = 0
    for i in range(N):
        if i % 6 not in osi:
            continue
        p = tochka(i)
        days, sig, sigma, close, div, _ = lenta(32_000 + i)
        rang = rang_of(i, len(days))
        kw = dict(close=close, div=div, rang=rang)
        try:
            ok = fx.run_v5("T", days, sig, sigma, p, **kw)
        except ValueError:
            continue
        with monkeypatch.context() as m:
            mut(m)
            try:
                bad = exec_e.run("T", days, sig, sigma, p, **kw)
            except (ValueError, IndexError):
                chislo += 1
                continue
        try:
            _same(bad, ok)
        except AssertionError:
            chislo += 1
    assert chislo >= 5, f"мутация {name} не поймана: расхождений {chislo}"


def test_mutatsiya_yadra_lovitsya(monkeypatch):
    """Ядро с подложенной ошибкой (ранг берётся с обратным знаком a1) расходится с эталоном — сверка не «слепая»."""
    orig = v6.rows

    def bad(rang, n, pf, pi, be):
        pf = pf.copy()
        pf[v6.F_AA1] = -pf[v6.F_AA1]
        return orig.py_func(rang, n, pf, pi, be)
    monkeypatch.setattr(fx.v6, "rows", bad)
    monkeypatch.setattr(fx, "kernel_v5", fx.kernel_v5.py_func)
    chislo = 0
    for i in range(4, N, 6):
        p = dataclasses.replace(tochka(i), a_rang=(1.0, 1.0))
        days, sig, sigma, close, div, _ = lenta(30_000 + i)
        rang = rang_of(i, len(days))
        chislo += not sverit(days, sig, sigma, p, close, div, rang)
        if chislo >= 3:
            break
    assert chislo >= 3
