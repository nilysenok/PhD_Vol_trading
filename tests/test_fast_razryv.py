"""Ф-razryv: правило Д5-2 (Params.razryv, #695) в ядре быстрого слоя: заявка (лимит / стоп-заявка OCO) снимается на шаге после разрыва
торгов тикера > razryv календарных дней, открытая позиция держится, повторный вход в тот день не ставится. Прежде `fast_etap1.run` отдавал
в эталон ВЕСЬ ряд с хоть одним таким разрывом (быстрый слой на wf и train4 с прогревом 2014–2015 был фактически выключен).
1) 1200 синтетических путей S3/S4 с разрывами 4, 5, 6, 7, 30 дней (лимит / стоп-заявка / T / reentry_den / шорт / позиция через разрыв):
   `fx.run_v5` и `fast_etap1.run` = `exec_e.run` побайтно; 2) 400 путей прежнего ядра (`fast_etap1._case`); правило реально работает (счётчики);
3) мутации: «заявка не снимается в ядре» и «разрыв ≥ вместо >» расходятся с эталоном; 4) ядро реально вызвано (не эталон).
FASTBT_MAIN=<копия> python3 -m pytest tests/test_fast_razryv.py -q"""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from fastbt import exec_e, fast_etap1, fast_exec as fx
from test_fast_etap1_smoke import _case, _same
from test_fast_v5_s34 import lenta, tochka

N = 1200
GAPS = (4, 5, 6, 7, 30)


def daty(seed: int, n: int) -> pd.DatetimeIndex:
    rng = np.random.default_rng(seed)
    step = np.where(rng.random(n) < 0.12, rng.choice(GAPS, n), 1)
    return pd.DatetimeIndex(pd.Timestamp("2019-03-01") + pd.to_timedelta(np.cumsum(step), unit="D"))


def sverit(days, sig, sigma, p, close, div, urov, dates, f) -> bool:
    kw = dict(close=close, div=div, urovni=urov if p.vhod_tip == "stop_zayavka" else None, dates=dates)
    try:
        a = exec_e.run("T", days, sig, sigma, p, **kw)
    except ValueError as e:
        with pytest.raises(ValueError) as ei:
            f("T", days, sig, sigma, p, **kw)
        return str(ei.value) == str(e)
    try:
        _same(a, f("T", days, sig, sigma, p, **kw))
    except AssertionError:
        return False
    return True


def progon():
    plokho, cnt = [], dict(valid=0, snyata=0, shorty=0, cherez=0)
    for i in range(N):
        p, (days, sig, sigma, close, div, urov) = tochka(i), lenta(50_000 + i)
        assert fx.ne_znaet(p) == [], p
        dates = daty(i, len(days))
        for f in (fx.run_v5, fast_etap1.run):
            if not sverit(days, sig, sigma, p, close, div, urov, dates, f):
                plokho.append(i)
                break
        else:
            try:
                kw = dict(close=close, div=div, urovni=urov if p.vhod_tip == "stop_zayavka" else None)
                a, b = exec_e.run("T", days, sig, sigma, p, dates=dates, **kw), exec_e.run("T", days, sig, sigma, p, **kw)
            except ValueError:
                continue
            cnt["valid"] += 1
            cnt["snyata"] += a.n_orders - a.n_fills < b.n_orders - b.n_fills or a.n_fills != b.n_fills
            cnt["shorty"] += any(t.side < 0 for t in a.trades)
            ix = [k for k, d in enumerate(days) if not d.pusto]
            st = exec_e.stale_steps(dates, ix, p.razryv)
            cnt["cherez"] += any(t.exit_day is not None and any(ix[j] > t.entry_day for j in st if ix[j] <= t.exit_day) for t in a.trades)
    return plokho, cnt


def lovitsya() -> bool:
    """Мутация поймана: расхождение с эталоном или падение ядра (оно повторяет и отказы эталона, например σ̂ входа nan при тике)."""
    try:
        return bool(progon()[0])
    except (ValueError, pytest.fail.Exception):         # ядро упало / не повторило отказ эталона — тоже расхождение
        return True


def test_sintetika_1200_putej_s_razryvami_yadro_ravno_etalonu():
    plokho, cnt = progon()
    assert plokho == [], plokho[:20]
    assert cnt["valid"] > 900 and cnt["snyata"] > 100 and cnt["shorty"] > 100 and cnt["cherez"] > 20, cnt     # правило реально бьёт, позиция живёт через разрыв


@pytest.mark.parametrize("seed", range(400))
def test_staroe_yadro_s_razryvami_ravno_etalonu(seed):
    days, sig, sigma, p, close, div = _case(seed)
    p = dataclasses.replace(p, razryv=5)
    dates = daty(seed, len(days))
    try:
        ref = exec_e.run("T", days, sig, sigma, p, close=close, div=div, dates=dates)
    except ValueError:
        return
    _same(ref, fast_etap1.run("T", days, sig, sigma, p, close=close, div=div, dates=dates))


def test_yadro_vyzvano_a_ne_etalon(monkeypatch):
    """Ряд с разрывом 30 дней идёт в ядро: эталон, подменённый на бомбу, не вызывается."""
    def bomba(*a, **k):
        raise AssertionError("ушло в эталон")
    monkeypatch.setattr(fast_etap1, "_SLOW_RUN", bomba)
    days, sig, sigma, close, div, _ = lenta(7)
    r = fast_etap1.run("T", days, sig, sigma, tochka(700), close=close, div=div, dates=daty(7, len(days)))     # лимит, T, reentry_den
    assert len(r.gross) == len(days)


def test_stale_mask_ravno_stale_steps():
    for s in range(200):
        days = lenta(s)[0]
        dates = daty(s, len(days))
        pusto = np.array([d.pusto for d in days])
        ix = [k for k, d in enumerate(days) if not d.pusto]
        for rz in (3, 5, 6):
            assert set(np.flatnonzero(fx.stale_mask(dates, pusto, rz))) == set(exec_e.stale_steps(dates, ix, rz))
    assert fx.stale_mask(dates, pusto, None).size == 0 and fx.stale_mask(None, pusto, 5).size == 0


def test_razryv_pyat_dnej_ne_snimaet_shest_snimaet():
    from test_sessiya_razryv import go
    from test_fast_etap1_smoke import TICK  # noqa: F401
    assert go(5, 5).trades and not go(6, 5).trades          # граница эталона: строго «>»


def test_mut_zayavka_ne_snimaetsya_v_yadre(monkeypatch):
    monkeypatch.setattr(fx, "stale_mask", lambda dates, pusto, razryv: np.empty(0, np.bool_))
    assert lovitsya()


def test_mut_razryv_ge_vmesto_gt(monkeypatch):
    def ge(dates, pusto, razryv):
        ix = np.flatnonzero(~pusto)
        if razryv is None or dates is None or len(ix) < 2:
            return np.empty(0, np.bool_)
        d = pd.DatetimeIndex(dates)[ix]
        return np.r_[False, np.asarray((d[1:] - d[:-1]).days >= razryv)].astype(np.bool_)
    monkeypatch.setattr(fx, "stale_mask", ge)
    assert lovitsya()
