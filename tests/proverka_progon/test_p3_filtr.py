"""П-3 (проверяющий), лист Л-4: ось `filtr` по контракту etap1 §2.1 и заданию Л-4. Пишется до кода.
Допущения о коде (контракт сказал лишь сигнатуру `razresheno(imya, t, inp, sd, dates)`):
 - ряды тикера читаются из `inp.sk._td[t][0]` (dates, close, high, low, volume, log_ret), σ — `inp.sigma["rv5|rv22|rv66"][t]`,
   события — `inp.sobytiya[имя][t]` («вход запрещён»), IMOEX — `fastbt.rezhim_e1.imoex_daily()` (date, close, sig_mkt).
Данные синтетические; ни 2022+, ни метрик. python3 -m pytest tests/proverka_progon/test_p3_filtr.py -q --tb=short"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import rezhim_e1

N, T = 300, "SBER"
DATES = pd.bdate_range("2016-01-04", periods=N).to_numpy("datetime64[D]")
SIG = 0.01
NAMES = ("net", "cbr", "div", "post_shok", "szhatie", "rv_perevorot", "obyom", "rel_sila_imoex", "sektor_trend")


def make(r=None, vol=None, w=None, imoex_r=None, rv5=None, ticker=T):
    """Фейковые входы: цена из доходностей r, объём vol, полуширина бара w, IMOEX из imoex_r, σ = SIG."""
    r = np.zeros(N) if r is None else np.asarray(r, float)
    c = 100 * np.exp(np.cumsum(r))
    w = np.full(N, 0.01) if w is None else np.asarray(w, float)
    td = SimpleNamespace(dates=DATES, close=c, high=c * (1 + w), low=c * (1 - w), open=c,
                         volume=np.ones(N) if vol is None else np.asarray(vol, float), log_ret=np.r_[r[1:], 0.0])
    # log_ret в TickerData — доход d→d+1; r_d (по close d) = r[d]: оба ряда согласованы тестом ниже
    ir = np.zeros(N) if imoex_r is None else np.asarray(imoex_r, float)
    imoex = pd.DataFrame({"date": pd.DatetimeIndex(DATES), "close": 3000 * np.exp(np.cumsum(ir)),
                          "sig_mkt": SIG, "nbars": 50})
    s5 = np.full(N, SIG) if rv5 is None else np.asarray(rv5, float)
    inp = SimpleNamespace(sk=SimpleNamespace(_td={ticker: (td, None)}, tickers=[ticker]),
                          sigma={"rv5": {ticker: s5}, "rv22": {ticker: np.full(N, SIG)}, "rv66": {ticker: np.full(N, SIG)}},
                          sobytiya={k: {ticker: np.zeros(N, bool)} for k in ("net", "cbr", "div")},
                          meta={ticker: {"dates": DATES, "s0": 0, "i1": N - 1}}, tick={ticker: 0.01})
    return inp, imoex


def run(name, inp, imoex, sd, ticker=T, mp=None):
    mp.setattr(rezhim_e1, "imoex_daily", lambda path=None: imoex)
    from fastbt import filtr_e1_rynok as f
    return f.razresheno(name, ticker, inp, np.full(N, sd, float) if np.isscalar(sd) else sd, DATES)


@pytest.fixture
def mp(monkeypatch):
    return monkeypatch


def test_znacheniya_dev_et_devyat():
    from fastbt import filtr_e1_rynok as f
    assert tuple(f.ZNACHENIYA) == NAMES


def test_net_vse_true_i_cbr_pobajtno(mp):
    inp, im = make()
    inp.sobytiya["cbr"][T] = np.arange(N) % 7 == 0
    assert run("net", inp, im, 1, mp=mp).tolist() == [True] * N
    assert run("cbr", inp, im, 1, mp=mp).tobytes() == (~inp.sobytiya["cbr"][T]).tobytes()


def test_obyom_znaet_otvet(mp):
    vol = np.ones(N)
    vol[100] = 4.0                                   # 4 / mean(prev 20) = 4 > 3, r = 0 -> |z| < 0.5
    inp, im = make(vol=vol)
    for sd in (1, -1):                               # фильтр не зависит от стороны
        a = run("obyom", inp, im, sd, mp=mp)
        assert a[100] and a.sum() == 1 and not a[:21].any()
    r = np.zeros(N)
    r[100] = 0.01                                    # |z| = 1 >= 0.5: запрещено
    inp2, im2 = make(r=r, vol=vol)
    assert not run("obyom", inp2, im2, 1, mp=mp)[100]


def test_post_shok_obe_storony(mp):
    r = np.zeros(N)
    r[150] = -0.035                                  # z = -3.5; лонг: -sd*z = 3.5 >= 3
    inp, im = make(r=r)
    L, S = run("post_shok", inp, im, 1, mp=mp), run("post_shok", inp, im, -1, mp=mp)
    assert L[150:155].all() and not L[149] and not L[155]
    assert not S.any()                               # перепутанная сторона даёт True на шорте — ловится здесь
    inp, im = make(r=-r)
    assert run("post_shok", inp, im, -1, mp=mp)[150:155].all() and not run("post_shok", inp, im, 1, mp=mp).any()
    ir = np.zeros(N)
    ir[150] = -0.03                                  # |Z| = 3 > 1: шок рынка, не идиосинкратический
    inp, im = make(r=np.where(np.arange(N) == 150, -0.035, 0.0), imoex_r=ir)
    assert not run("post_shok", inp, im, 1, mp=mp)[150:155].any()


def test_szhatie_obe_storony_nezavisimo(mp):
    d = np.arange(N)
    inp, im = make(w=0.05 - 0.00015 * d)             # диапазон падает каждый день: каждый день — минимум
    for sd in (1, -1):
        a = run("szhatie", inp, im, sd, mp=mp)
        assert a[140:].all() and not a[:20].any()    # нет истории — запрещено
    inp, im = make(w=0.01 + 0.00015 * d)             # растёт: сжатия нет
    assert not run("szhatie", inp, im, 1, mp=mp)[140:].any()


def test_rv_perevorot(mp):
    rv5 = np.full(N, SIG)
    rv5[100], rv5[101] = 0.02, 0.0105               # (0.02/0.01)^2 = 4 > 3 в день 100, затем спад
    inp, im = make(rv5=rv5)
    for sd in (1, -1):
        a = run("rv_perevorot", inp, im, sd, mp=mp)
        assert not a[100] and a[101] and a[102] and not a[130]


def test_rel_sila_imoex_storona(mp):
    r = np.zeros(N)
    r[40:45] = 0.01                                  # бумага +5 дней, рынок 0: 0.05/(0.01·√5) = 2.24 >= 1
    inp, im = make(r=r)
    assert run("rel_sila_imoex", inp, im, 1, mp=mp)[44]
    assert not run("rel_sila_imoex", inp, im, -1, mp=mp)[44]
    assert not run("rel_sila_imoex", inp, im, 1, mp=mp)[20]


def test_sektor_trend_aflt_i_lsrg_ne_dejstvuet(mp):
    for tk in ("AFLT", "LSRG"):
        inp, im = make(ticker=tk)
        assert run("sektor_trend", inp, im, 1, ticker=tk, mp=mp).all()


def _causal(name, mp, sd, shift=0, cuts=range(150, 290, 7)):
    """Результат на днях <= cut не должен зависеть от рядов после cut — для каждого cut из набора.
    shift=-1 — подложенная утечка d+1. Данные с толстыми хвостами и убывающим диапазоном: каждый фильтр
    срабатывает на части дней, поэтому подмена хвоста меняет ответ утёкшего фильтра (а не False в обоих вариантах)."""
    for cut in cuts:
        rng = np.random.default_rng(cut)
        r = rng.normal(0, SIG, N) * np.where(rng.random(N) < 0.08, 4, 1)
        vol = np.exp(rng.normal(0, 0.5, N)) * np.where(rng.random(N) < 0.05, 6, 1)
        w = 0.05 - 0.00015 * np.arange(N) + 0.002 * rng.random(N)
        out = []
        for tail in (0, 1):
            rr, vv, ww = r.copy(), vol.copy(), w.copy()
            if tail:
                rr[cut + 1:] = rng.normal(0, 0.05, N - cut - 1)
                vv[cut + 1:] *= 9
                ww[cut + 1:] *= 4
            inp, im = make(r=rr, vol=vv, w=ww, imoex_r=rr * 0.5, rv5=np.abs(rr) + 0.005)
            if shift:                                # мутация: фильтр читает ряды сдвинутыми на -1
                td = inp.sk._td[T][0]
                for k in ("close", "high", "low", "volume"):
                    setattr(td, k, np.roll(getattr(td, k), shift))
            out.append(np.asarray(run(name, inp, im, sd, mp=mp))[:cut + 1])
        if out[0].tobytes() != out[1].tobytes():
            return False
    return True


@pytest.mark.parametrize("name", NAMES[3:8])
@pytest.mark.parametrize("sd", [1, -1])
def test_net_utechki_budushchego(name, sd, mp):
    assert _causal(name, mp, sd)


@pytest.mark.parametrize("name", ["post_shok", "szhatie", "obyom", "rel_sila_imoex"])
def test_mutaciya_dannye_d_plus_1_lovitsya(name, mp):
    assert not _causal(name, mp, 1, shift=-1)    # сдвиг -1 тянет день cut+1 в cut: утечка видна хотя бы на одном cut


def test_ne_schitana_div_vozvrashchaet_none(mp):
    inp, im = make()
    r = run("div", inp, im, 1, mp=mp)
    assert r is None or np.asarray(r).tobytes() == (~inp.sobytiya["div"][T]).tobytes()
    # если даты 2016 не сверены, обязан быть None, а не молчаливый net: проверка на реальных входах — в test_p3_eval_v2
