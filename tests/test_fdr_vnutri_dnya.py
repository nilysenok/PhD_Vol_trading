"""К-Ф2 (Ж-065 п. 3): ФДР внутри дня — port_rows(noch), run_e1(fdr), Evaluator(fdr), имя ledger по режиму.
Контракт docs/otchety/2026-10-28_kontrakt_fdr_iter16_20.md §1.3–1.5: Г1, Г3, М-Ф1…М-Ф6 + добавочные мутации noch_iz_pos.
Синтетика, данные не читаются. Образец мутации: подмена функции, проверка обязана падать AssertionError.
python3 -m pytest tests/test_fdr_vnutri_dnya.py -q"""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from fastbt import etap1_run, fdr as F_, zh051_plato
from fastbt.s9 import Engine
from test_zh047_pomoshch import D, F, P, SG

C = 0.0004


def engine(n, npair=1, seed=0):
    rng = np.random.default_rng(seed)
    e = Engine.__new__(Engine)
    e.r = {"date": np.tile(np.arange(n), npair)}
    e.first = np.tile(np.r_[True, np.zeros(n - 1, bool)], npair)
    e.lr_row = rng.normal(0, 0.02, n * npair)
    e.dates = pd.bdate_range("2019-01-02", periods=n)
    e.cnt = np.full(n, float(npair))
    e.kr = np.full(n, 0.07)
    e.rf, e.cal = e.kr / 365.0, np.array([1.0, 3.0] * n)[:n]
    return e


def run1(days, sig, p, fdr):
    n = len(days)
    return etap1_run.run_e1("T", days, sig, np.full(n, SG), p, win=(0, n - 1), fdr=fdr)


def stavka(r):
    """(total − r, rf·cal) по строкам окна одного тикера: ставка на свободный капитал."""
    e = engine(len(r.g))
    df = e.port_rows(r.g, r.sides, r.expo, C, noch=r.noch)
    return (df["total"] - df["r"]).to_numpy(), e.rf * e.cal


def intraday():
    """Лимит в баре 3 дня d+1, стоп в баре 10 того же дня (k = 1: L = 93,75; стоп 87,89)."""
    b = [(100.0,) * 4] * 3 + [(100, 100, 93.5, 94)] + [(94.0,) * 4] * 6 + [(94, 94, 87.5, 88)] + [(88.0,) * 4]
    return [D(F(100.0)), D(b), D(F(88.0))], (1, 1, 0), P(k=1.0, R=None, reentry=False)


def at_close(sig, n=5):
    return [D(F(100.0)) for _ in range(n)], sig, P(R=None, b=None)


def chk_mf1():
    r = run1(*intraday(), "t0")
    assert [t.entry_day for t in r.trades] == [1] and [t.exit_day for t in r.trades] == [1]
    assert np.array_equal(r.noch, [0.0, 0.0])
    col, rfcal = stavka(r)
    assert np.allclose(col, rfcal, rtol=1e-9, atol=0), "обе ночи свободны: fdr[d] = fdr[d+1] = rf·cal"


def chk_t0_t1():
    days, sig, p = at_close((1, 1, 0, 0, 0))
    t0, t1 = run1(days, sig, p, "t0").noch, run1(days, sig, p, "t1").noch
    assert t0[0] == 1.0, "ночь входа по close занята (М-Ф4)"
    assert t0[2] == 0.0, "T+0: строка выхода x свободна (М-Ф2)"
    assert t1[2] == 1.0 and t1[3] == 0.0, "T+1: строка x занята, x+1 свободна (М-Ф3)"


def chk_flip():
    days, sig, p = [D(F(100.0)) for _ in range(6)], (1, 1, -1, -1, 0, 0), P(R=None, b=None)
    assert run1(days, sig, p, "t1").noch[2] == 1.0, "переворот: min(1, ·), а не 2 (М-Ф5)"


def test_mf1_vnutridnevnaya_sdelka_noch_svobodna():
    chk_mf1()
    r = run1(*intraday(), None)
    assert r.noch is None and r.expo[0] > 0, "старое правило занимало ночь долей баров"


def test_mf2_mf3_mf4_t0_t1():
    chk_t0_t1()


def test_mf5_perevorot_min_edinitsy():
    chk_flip()


def test_t1_ne_menshe_zanyatosti_t0_i_stavka_ne_bolshe():
    days, sig, p = at_close((1, 1, 0, 0, 0))
    a, b = run1(days, sig, p, "t0"), run1(days, sig, p, "t1")
    assert (b.noch >= a.noch).all() and (stavka(b)[0] <= stavka(a)[0] + 1e-18).all() and b.noch.sum() > a.noch.sum()


def test_noch_menyaet_tolko_stavku_exposure_i_plecho_po_starym():
    r = run1(*intraday(), "t0")
    e = engine(2)
    a, b = e.port_rows(r.g, r.sides, r.expo, C), e.port_rows(r.g, r.sides, r.expo, C, noch=r.noch)
    for c in ("r", "gross", "exposure", "turnover"):
        assert np.array_equal(a[c], b[c]), c
    assert not np.array_equal(a["total"], b["total"])
    z = np.zeros(2)
    h = e.port_rows(z, z, np.full(2, 1.5), C, noch=z)
    mg = 0.5 * (e.kr + 0.02) / 365.0 * e.cal
    assert np.allclose(h["total"] - h["r"], e.rf * e.cal - mg, rtol=1e-12, atol=0), "mg по exposure, ставка по noch"


# ---- Г1, Г3 ----
def old_finish(e, r, gross, exp_, turn, cap):
    used = cap * exp_
    fdr = np.maximum(0.0, 1.0 - used) * e.rf * e.cal
    mg = np.maximum(0.0, used - 1.0) * (e.kr + 0.02) / 365.0 * e.cal
    total = r + fdr - mg
    return pd.DataFrame(dict(r=r, gross=gross, exposure=exp_, turnover=turn, total=total, excess=total - e.rf * e.cal,
                             excess_gross=gross + fdr - mg - e.rf * e.cal), index=e.dates)


def rows_of(e, pos):
    dp = np.diff(pos, prepend=0.0)
    dp[e.first] = pos[e.first]
    last = np.r_[e.first[1:], True]
    return pos * e.lr_row, np.abs(dp) + np.where(last, np.abs(pos), 0.0), np.abs(pos)


def test_g1_noch_none_ravno_staromu_port_rows():
    e = engine(30, 5, seed=1)
    g, s, x = rows_of(e, np.random.default_rng(2).normal(0, 0.8, 150))
    di, bc = e.r["date"], lambda v: np.bincount(di, v, 30) / e.cnt  # noqa: E731
    old = old_finish(e, bc(g - s * C), bc(g), bc(x), bc(s), 1.0)
    pd.testing.assert_frame_equal(e.port_rows(g, s, x, C), old, check_exact=True)
    pd.testing.assert_frame_equal(e.port_rows(g, s, x, C, noch=None), old, check_exact=True)


def chk_g3():
    e = engine(30, 5, seed=3)
    pos = np.random.default_rng(4).normal(0, 0.8, 150) * np.random.default_rng(5).integers(0, 2, 150)
    g, s, x = rows_of(e, pos)
    pd.testing.assert_frame_equal(e.port_rows(g, s, x, C, noch=F_.noch_iz_pos(pos, e.first, "t0")), e.port(pos, C),
                                  check_exact=True)


def test_g3_A_pri_t0_pobaytovo_kak_port():
    chk_g3()


# ---- noch_iz_pos: добавочные мутации (замечание проверяющего к #377) ----
def chk_noch_pos():
    pos = np.array([0.5, 0.2, -0.3, 0.9, 0.4, 0.4])
    first = np.array([True, False, False, False, True, False])
    assert np.array_equal(F_.noch_iz_pos(pos, first, "t0"), np.abs(pos))
    # t1: стр. 1 сократила 0,5→0,2 (+0,3); стр. 2 переворот (+0,2); стр. 3 переворот (0,9+0,3 → min 1);
    # стр. 4 — первая пары: хвост прошлой пары (0,9) не считается; стр. 5 без изменения
    assert np.allclose(F_.noch_iz_pos(pos, first, "t1"), [0.5, 0.5, 0.5, 1.0, 0.4, 0.4])


def test_noch_iz_pos_reper():
    chk_noch_pos()


def mut_pos(kind, orig):
    def bad(pos, first, rezhim):
        pos = np.asarray(pos, float)
        ap, prev = np.abs(pos), np.abs(np.r_[0.0, pos[:-1]])
        if kind == "t0_sdvig" and rezhim == "t0":
            return prev
        if kind in ("t1_bez_first", "t1_bez_min") and rezhim == "t1":
            flip = np.sign(pos) != np.sign(np.r_[0.0, pos[:-1]])
            zak = np.where(flip, prev, np.maximum(0.0, prev - ap))
            if kind == "t1_bez_min":
                zak[np.asarray(first, bool)] = 0.0
                return ap + zak
            return np.minimum(1.0, ap + zak)
        return orig(pos, first, rezhim)
    return bad


@pytest.mark.parametrize("kind", ["t0_sdvig", "t1_bez_first", "t1_bez_min"])
def test_mutatsiya_noch_iz_pos_lovitsya(monkeypatch, kind):
    monkeypatch.setattr(F_, "noch_iz_pos", mut_pos(kind, F_.noch_iz_pos))
    with pytest.raises(AssertionError):
        chk_noch_pos()


# ---- мутации М-Ф1…М-Ф5 на noch_iz_sdelok и _finish ----
def mut_sdelok(kind, orig):
    def bad(trades, w_tr, n, rezhim):
        ts = list(trades)
        if kind == "exit_day_vklyuchen":
            ts = [dataclasses.replace(t, exit_day=None if t.exit_day is None else t.exit_day + 1) for t in ts]
        if kind == "noch_vhoda_svobodna":
            ts = [dataclasses.replace(t, entry_day=t.entry_day + 1) for t in ts]
        if kind == "t1_vmesto_t0" and rezhim == "t0":
            rezhim = "t1"
        if kind == "t0_vmesto_t1" and rezhim == "t1":
            rezhim = "t0"
        if kind == "bez_min" and rezhim == "t1":
            out = orig(ts, w_tr, n, "t0")
            for t in ts:
                if t.exit_day is not None and 0 <= t.exit_day < n:
                    out[t.exit_day] += w_tr[t.d_sig]
            return out
        return orig(ts, w_tr, n, rezhim)
    return bad


@pytest.mark.parametrize("kind,chk", [("exit_day_vklyuchen", chk_mf1), ("t1_vmesto_t0", chk_t0_t1),
                                      ("t0_vmesto_t1", chk_t0_t1), ("noch_vhoda_svobodna", chk_t0_t1),
                                      ("bez_min", chk_flip)])
def test_mutatsii_mf1_mf5(monkeypatch, kind, chk):
    monkeypatch.setattr(F_, "noch_iz_sdelok", mut_sdelok(kind, F_.noch_iz_sdelok))
    with pytest.raises(AssertionError):
        chk()


def test_mutatsiya_finish_ignoriruet_noch_lovitsya(monkeypatch):
    orig = Engine._finish
    monkeypatch.setattr(Engine, "_finish", lambda self, r, g, x, t, cap, noch_=None: orig(self, r, g, x, t, cap))
    with pytest.raises(AssertionError):
        chk_mf1()


# ---- М-Ф6: режим входит в имя ledger ----
def chk_mf6():
    assert len({zh051_plato.ledger_name(1, m) for m in (None, "t0", "t1")}) == 3
    assert zh051_plato.ledger_name(1, "t0") == "ledger_it1_fdr-t0.pkl"
    assert zh051_plato.ledger_name(3) == "ledger_it3.pkl", "умолчание — прежнее имя"

    class Ev:
        fdr = "t1"

        def evaluate(self, pt, period):
            return {}
    assert zh051_plato.PlusLedger(Ev().evaluate).name(2) == "ledger_it2_fdr-t1.pkl"


def test_mf6_ledger_po_rezhimu_otdelnyy_fayl():
    chk_mf6()
    with pytest.raises(ValueError):
        zh051_plato.ledger_name(1, "t2")


def test_mutatsiya_mf6_rezhim_ne_v_imeni(monkeypatch):
    monkeypatch.setattr(zh051_plato, "ledger_name", lambda it, fdr=None: f"ledger_it{it}.pkl")
    with pytest.raises(AssertionError):
        chk_mf6()


def _net_dannyh():
    from fastbt import bars5m
    return not bars5m.SRC.exists()


@pytest.mark.skipif(_net_dannyh(), reason="нет 1-минуток D25 (FASTBT_BARS1M)")
def test_g2_g3_yakor_S1_fdr_none_v10_t0_pobaytovo():
    """Реальные данные 2018–2019, якорь S1: fdr=None — строка v1.0; t0 у A (все сделки по close) = база побайтно."""
    from fastbt import etap1
    from fastbt.etap1_eval import Evaluator
    inp, base = etap1.build_inputs("S1"), etap1.base_point("S1")
    ra, rb = Evaluator("S1", inp, base).evaluate({}, "train"), Evaluator("S1", inp, base, fdr="t0").evaluate({}, "train")
    assert ra["v"] == "1.0" and "fdr" not in ra and rb["v"] == "1.1" and rb["fdr"] == "t0"
    for y in ra["excess_d"]:
        assert np.array_equal(np.array(ra["excess_d"][y]), np.array(rb["excess_d"][y])), y


def test_nevernyy_rezhim_oshibka():
    from types import SimpleNamespace

    from fastbt.etap1_eval import Evaluator
    inp = SimpleNamespace(sk=None)
    with pytest.raises(ValueError):
        run1(*at_close((1, 1, 0, 0, 0)), "t2")
    with pytest.raises(ValueError):
        Evaluator("S1", inp, {}, fdr="t2")
    assert Evaluator("S1", inp, {}).fdr is None
