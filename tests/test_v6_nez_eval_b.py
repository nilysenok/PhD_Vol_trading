"""V6-12b, независимая проверка (тестировщик, не автор): оси движка и тень паузы EvaluatorV6 дают известный эффект.
Движок (синтетика, без данных): time_decay — половина тейка после tau//2; zhizn — заявка живёт N баров.
Оценщик (train4 ≤ 2021, S1, 3 тикера, ≈ 20–30 с; нет дневного ряда 18:50 — skip): ось доходит до Params/σ̂/ed движка;
пауза: нет новых входов в дни паузы, открытые позиции не закрываются, пауза дня d не видит excess ≥ d (подмена строк ≥ d не меняет маску ≤ d);
GOTOVY: открыты ровно подключённые оси (osi + Соболь); fkey различает каждую новую ось. Тяжёлое (все Sk) — V6_NEZ_TYAZHELOE=1.
Мутации (в коде, по одной; откат git checkout -- fastbt/): e1_pauza.ROW_LAG = -1; fkey без оси; V12 + "tf"."""
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fastbt import canon_v5, canon_v6, daily_1850, eval_v5, eval_v6, eval_v6_osi as osi6
from fastbt.eval_v5_osi import yakor
from fastbt.eval_v6 import EvaluatorV6, fkey
from fastbt.exec_e import Params, run
from fastbt.exec_e_types import Day

OSI12B = osi6.DVIG + osi6.PAUZA
real = pytest.mark.skipif(not daily_1850.out_path().exists(), reason="нет дневного ряда 18:50")
TYAZH = pytest.mark.skipif(not os.environ.get("V6_NEZ_TYAZHELOE"), reason="тяжёлое: V6_NEZ_TYAZHELOE=1")

# ---------------- движок: известный эффект на синтетике ----------------
SG, FLAT, UP = 0.01, (100.0, 100.0, 100.0, 100.0), (100.0, 102.5, 100.0, 100.0)    # UP: +2,5 % — между половиной (2 %) и полным (4 %) тейка


def _day(*rows):
    a = np.array(rows, float).reshape(-1, 4)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


def _td(rows, **kw):
    days = [_day(r) for r in rows]
    return run("X", days, np.ones(len(days)), np.full(len(days), SG), Params(**dict(a=4.0, tau=4, comm=0.0) | kw)).trades


def test_time_decay_polovina_tejka_posle_tau_popolam():
    rows = [FLAT, FLAT, UP, UP, FLAT, FLAT]
    t = _td(rows, time_decay=True)[0]
    assert (t.exit_day, round(t.exit_px, 6), t.reason) == (3, 102.0, "take")       # день 2 (j−ej = tau//2) — ещё полный тейк
    assert _td([FLAT, FLAT, UP, FLAT, FLAT, FLAT], time_decay=True)[0].reason == "vremya"          # UP только на tau//2-й день — не берёт
    assert _td(rows)[0].reason != "take"                                           # ось выключена — полный тейк 4 % не достигнут


def _zh(b_low, zhizn):
    n = 20
    def f(l=100.0):
        return Day(np.full(n, 100.0), np.full(n, 100.0), np.where(np.arange(n) == b_low, l, 100.0), np.full(n, 100.0))
    days = [_day(FLAT), f(98.5), _day(FLAT)]
    return run("X", days, np.array([1.0, 0, 0]), np.full(3, SG), Params(k=0.5, zhizn=zhizn, comm=0.0)).trades


def test_zhizn_zayavka_zhivyot_N_barov():
    assert len(_zh(11, 12)) == 1 and _zh(11, 12)[0].entry_bar == 11                  # бар b < N — исполнена
    assert _zh(12, 12) == [] and _zh(13, 12) == []                                  # b ≥ N — снята
    assert len(_zh(12, 36)) == 1 and len(_zh(15, None)) == 1                        # длиннее жизнь / весь день


# ---------------- оценщик: ось доходит до движка ----------------
_K = {}
SP1 = dict(ma_tip="sma", ma_window=20, dev_tip="std", z_entry=1.0, z_exit=0.5)       # мягкий сигнал: на 3 тикерах есть сделки
TD = {"tau": 10, "tejk": "pct:0.05"}
PZ = {"pauza_porog": 0.5, "pauza_okno": 42}


def _e6(sid="S1"):
    if sid not in _K:
        from fastbt import etap1
        inp = etap1.build_inputs(sid, period="train4")
        inp.sk.tickers = list(inp.sk.tickers)[:3]
        _K[sid] = EvaluatorV6(sid, inp, etap1.base_point(sid))
    return _K[sid]


@pytest.fixture
def spy(monkeypatch):
    """Перехват eval_v5.run_rows: на каждый тикер — (Params, σ̂, ed)."""
    got, orig = [], eval_v5.run_rows

    def f(t, days, sig, sg, p, *a, **kw):
        got.append((t, p, np.asarray(sg), kw.get("ed")))
        return orig(t, days, sig, sg, p, *a, **kw)
    monkeypatch.setattr(eval_v5, "run_rows", f)
    return got


def _pt(**kw):
    return {**yakor("S1"), **SP1, "k_e": 1.0, **kw}


def _run(spy, **kw):
    spy.clear()
    e = _e6()
    r, x, _ = e.evaluate_v2(_pt(**kw), "train4")
    assert x is not None, r.get("prichina")
    return e, list(spy)


@real
def test_time_decay_i_zhizn_v_params(spy):
    _, g = _run(spy)
    assert all(not p.time_decay and p.zhizn is None for _, p, _, _ in g)
    _, g = _run(spy, time_decay="da", **TD)
    assert len(g) == 3 and all(p.time_decay is True and p.tau == 10 for _, p, _, _ in g)
    _, g = _run(spy, zhizn=12)
    assert all(p.zhizn == 12 and not p.time_decay for _, p, _, _ in g)


@real
def test_sigma_ist_podstavlyaet_ryad(spy):
    e, g0 = _run(spy)
    _, g = _run(spy, sigma_ist="rv5")
    for (t, _, s0, _), (_, _, s5, _) in zip(g0, g):
        eq = lambda a, b: np.array_equal(a, b, equal_nan=True)                       # noqa: E731
        assert eq(s0, e.inp.sigma["rv22"][t][:len(s0)]) and eq(s5, e.inp.sigma["rv5"][t][:len(s5)]) and not eq(s0, s5)


@real
def test_ed_lim_edinica_otstupa(spy):
    e, g0 = _run(spy)
    assert all(ed is None for *_, ed in g0)                                          # sigma — по σ̂ движка
    for lim in ("atr14", "diap"):
        _, g = _run(spy, ed_lim=lim)
        for t, _, _, ed in g:
            td, ind = e.sk._td[t]
            s0, b = e.inp.meta[t]["s0"], len(ed) - 1
            sl = slice(s0, s0 + b + 1)
            want = ind["atr14"][sl] / td.close[sl] if lim == "atr14" else (td.high[sl] - td.low[sl]) / td.close[sl]
            assert np.array_equal(np.asarray(ed), want)


# ---------------- тень паузы ----------------
@real
def test_pauza_net_novyh_vhodov_i_pozitsii_ne_zakryvayutsya():
    e = _e6()
    r0, x0, _ = e.evaluate_v2(_pt(), "train4")
    base = [[(t.d_sig, t.entry_day, t.exit_day, t.reason) for t in r.trades] for r in e._last[4]]
    r1, x1, _ = e.evaluate_v2(_pt(**PZ), "train4")
    assert not np.array_equal(x0, x1)
    nblok = nspan = 0
    for ti, t in enumerate(e.sk.tickers):
        dts = e.inp.meta[t]["dates"][:e._b(t, "train4") + 1]
        al = osi6.razreshit(e.v6, t, dts, e._ten)
        nblok += int((~al).sum())
        pz = {(t_.d_sig, t_.entry_day, t_.exit_day, t_.reason) for t_ in e._last[4][ti].trades}
        assert all(al[d] for d, *_ in pz), t                                          # вход в день паузы — нет
        for tr in base[ti]:
            if al[tr[0]] and tr[2] is not None:                                      # вход разрешён → сделка прежняя, выход не тронут
                if (~al[tr[1]:tr[2] + 1]).any():
                    nspan += 1
                    assert tr in pz, (t, tr)                                         # сделка через день паузы: выход тот же, не закрыта
    assert nblok > 50
    assert nspan > 0, "нет сделки, пережившей день паузы — тест ничего не доказывает"


@real
def test_pauza_dnya_d_ne_vidit_excess_d_i_pozzhe():
    e = _e6()
    e.evaluate_v2(_pt(**PZ), "train4")
    t = e.sk.tickers[0]
    s = e._ten[t]
    dts = pd.DatetimeIndex(s.index)
    base = osi6.razreshit(e.v6, t, dts, e._ten)
    assert (~base).sum() > 0 and base.sum() > 0
    for d in (150, 400, 700):
        bad = s.copy()
        bad.iloc[d:] = np.where(np.arange(len(bad) - d) % 2, -9.0, 9.0)               # мусор в строках ≥ d
        m = osi6.razreshit(e.v6, t, dts, {t: bad})
        assert np.array_equal(m[:d + 1], base[:d + 1]), d                            # маска ≤ d зависит от строк ≤ d−1
    bad = s.copy()
    bad.iloc[399] = -9.0
    assert not np.array_equal(osi6.razreshit(e.v6, t, dts, {t: bad})[400:], base[400:])   # строка d−1 — видна (тест не вырожден)


# ---------------- GOTOVY ----------------
@pytest.mark.parametrize("sid", canon_v5.SIDS)
def test_gotovy_otkryvaet_rovno_podklyuchennye_osi(sid):
    kub = canon_v6.load_kub(sid)
    nov = set(canon_v6.novye(kub))
    otkr = {n for n, _, _ in canon_v6.osi(kub, canon_v6.GOTOVY)} & nov
    podkl = nov & set(eval_v6.PODKLYUCHENO) - {"vhod_tip"}        # vhod_tip подключён (12a), но токена V6-12:vhod_tip нет — в куб не открыт (наблюдение)
    assert otkr == podkl, (sid, sorted(otkr ^ podkl))
    assert not {"tf", "lestnica", "gap", "t_vyhod", "vyhod_lim", "f_neisp"} & otkr
    assert {n for n in OSI12B if n in nov} <= otkr


@pytest.mark.parametrize("sid", canon_v5.SIDS)
def test_sobol_v6_ne_variruet_zakrytye(sid):
    kub = canon_v6.load_kub(sid)
    nov = set(canon_v6.novye(kub))
    zakr = nov - set(eval_v6.PODKLYUCHENO)
    pts = [c for _, c in canon_v6.tochki_v6(kub, 0, 1200)[0]]
    vid = set().union(*[set(c) for c in pts])
    assert not vid & zakr, vid & zakr
    assert {n for n in OSI12B if n in nov} <= vid
    for c in pts[:300]:
        eval_v6.razdelit(sid, c)                                                     # «ещё не подключена» не бросается


# ---------------- fkey ----------------
def test_fkey_razlichaet_kazhduyu_os_i_znachenie():
    vals = [(k, v) for sid in canon_v5.SIDS for k, s in canon_v6.novye(canon_v6.load_kub(sid)).items() if k in OSI12B for v in s["setka"][1:]]
    assert {k for k, _ in vals} == set(OSI12B)
    keys = {fkey({}, {k: v}) for k, v in set(vals)}
    assert len(keys) == len(set(vals)) and fkey({}, {}) not in keys
    for k in OSI12B:                                                                  # ось к уже заданной точке меняет ключ
        v = next(v for kk, v in vals if kk == k)
        assert fkey({"a": 1}, {"pauza_porog": 0.5, k: v}) != fkey({"a": 1}, {"pauza_porog": 0.5}) or k == "pauza_porog"


@real
def test_evaluate_v6_fkey_soderzhit_os(monkeypatch):
    monkeypatch.setattr(eval_v6.EvaluatorV2, "evaluate_v2", lambda self, pt, period, *a: None)       # считать не нужно: ключ ставится до
    e, ks = _e6(), {}
    for k, kw in (("0", {}), ("td", {"time_decay": "da", **TD}), ("sg", {"sigma_ist": "rv5"}), ("ed", {"ed_lim": "atr14"}),
                  ("zh", {"zhizn": 12}), ("zh2", {"zhizn": 36}), ("pp", {"pauza_porog": 0.5}), ("po", {"pauza_porog": 0.5, "pauza_okno": 42})):
        e.evaluate_v2(_pt(**kw), "train4")
        ks[k] = e._fkey
    assert len(set(ks.values())) == len(ks), ks


@real
@TYAZH
@pytest.mark.parametrize("sid", canon_v5.SIDS)
def test_vykl_12b_ravno_v5_pobaytno_tyazheloe(sid):
    from fastbt import etap1
    inp = etap1.build_inputs(sid, period="train4")
    inp.sk.tickers = list(inp.sk.tickers)[:3]
    base = etap1.base_point(sid)
    e5, e6 = eval_v5.EvaluatorV5(sid, inp, base), EvaluatorV6(sid, inp, base)
    nov = canon_v6.novye(canon_v6._kub(sid))
    off = {n: nov[n]["setka"][0] for n in OSI12B if n in nov}
    for _, p in canon_v5.tochki_v5(canon_v5.load_kub(sid), 0, 40)[0][:10]:
        a, b = e5.evaluate_v2(p, "train4"), e6.evaluate_v2({**p, **off}, "train4")
        a[0].pop("t_sek", None), b[0].pop("t_sek", None)
        assert repr(a[0]) == repr(b[0]) and (a[1] is None) == (b[1] is None) and (a[1] is None or a[1].tobytes() == b[1].tobytes())
