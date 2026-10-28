"""Л-5: оценщик v2 (fastbt/etap1_eval_v2.py), контракт docs/otchety/2026-10-28_progon_etap1.md §1, IG-1…IG-3.
Данные ≤ 2019: FASTBT_MAIN=<копия>, FASTBT_BARS1M=…/data/regen/D25/bars1m. Значений метрик в выводе нет (сравнение — sha и ==).
python3 -m pytest tests/test_etap1_eval_v2.py -q --tb=no"""
import hashlib
import sys
import types
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fastbt import etap1, etap1_eval_v2 as v2, exec_e
from fastbt.etap1_eval import YAKOR, Evaluator
from fastbt.etap1_poisk import load_setka
from fastbt.etap1_run import run_e1

PER, COMM, SIDS = "train4", 0.0004, ("S1", "S2", "S3", "S4", "S5", "S6")
_INP = {}


def inp_of(sid):
    if sid not in _INP:
        _INP[sid] = etap1.build_inputs(sid, period=PER)
    return _INP[sid]


def make(sid, cls=v2.EvaluatorV2):
    return cls(sid, inp_of(sid), etap1.base_point(sid), COMM)


def rand_points(sid, n, seed):
    rng = np.random.default_rng(seed)
    ax = load_setka(sid)["axes"]
    return [{k: v[int(rng.integers(len(v)))] for k, v in ax.items()} for _ in range(n)]


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()


# ---- IG-2: все новые оси = умолчания → побайтно как Evaluator ----
@pytest.mark.parametrize("sid", SIDS)
def test_umolchaniya(sid):
    e1, e2 = make(sid, Evaluator), make(sid)
    for pt in rand_points(sid, 50, 20261028):
        old = e1.evaluate(pt, PER)
        row, x, tik = e2.evaluate_v2(pt, PER)
        if "ne_schitana" in old:
            assert row["status"] == "ne_schitana" and x is None
            continue
        ref = np.concatenate([np.asarray(old["excess_d"][y], float) for y in sorted(old["excess_d"])])
        assert sha(x) == sha(ref) and x.dtype == np.float64 and tik.shape == (17, len(x))
        for k in ("sharpe", "msr", "sortino", "kalmar", "mdd", "T", "skew", "kurt", "prichiny", "dolya_ispolneniya_limita",
                  "kod", "vhody"):
            assert repr(row[k]) == repr(old[k]), k
        for k in ("by_year", "msr_by_year", "sdelok_y", "storony_y"):
            assert repr({int(a): b for a, b in row[k].items()}) == repr(old[k]), k


# ---- IG-3: тождество рядов ----
def tozh(ev, pt, vozm=None):
    row, x, tik = ev.evaluate_v2(pt, PER, vozm)
    if tik is None:
        return 0.0, 0.0, row
    m = np.nanmean(tik, axis=0)
    return float(np.max(np.abs(m - x))), float(np.max(np.abs(m.astype(np.float32).astype(np.float64)
                                                           - np.nanmean(tik.astype(np.float32), 0, dtype=np.float64)))), row


@pytest.mark.parametrize("sid", SIDS)
def test_tozhdestvo(sid):
    ev = make(sid)
    for pt in rand_points(sid, 4, 7):
        d64, d32, row = tozh(ev, pt)
        if row["status"] == "ok":
            assert d64 <= 1e-12 and d32 <= 1e-8


def test_tozhdestvo_mutatsiya_bez_rf(monkeypatch):
    ev, pt = make("S1"), {"vhod": 0.5, "k_s": "net", "tau": 3}
    assert tozh(ev, pt)[0] <= 1e-12
    monkeypatch.setattr(v2, "_carry", lambda eng, inv: np.zeros(len(inv)))
    assert tozh(ev, pt)[0] > 1e-8


# ---- IG-1: якорь через EvaluatorV2 = боевой s9 ----
def test_yakor_v2_equals_s9():
    from test_yakor_v1 import POINTS, TOL, deltas
    for sid, sigp in POINTS:
        d = deltas(make(sid), sid, sigp)
        assert len(d) == 4 and max(d.values()) <= TOL


# ---- новые оси ----
PT = {"vhod": 0.5, "k_s": "net", "R": "net", "tau": 3}


def rows_of(ev, pt, nov, vz=None):
    nov = {**v2.NOVYE, **nov}
    ev._ctx = dict(nov=nov, vozm=vz or {}, default=False)
    full = ev.full(pt)
    return [r for _, r, _, _ in ev._rows2(full, PER, False, ev._ctx)]


def ladder_half_ok(ev):
    net, lad = rows_of(ev, PT, {}), rows_of(ev, PT, {"lestnica": 1e3})
    assert sum(abs(r.g).sum() for r in net) > 0
    return all(np.array_equal(a.g * 0.5, b.g) and np.array_equal(a.sides * 0.5, b.sides) and np.array_equal(a.expo * 0.5, b.expo)
               for a, b in zip(net, lad))


def test_lestnica_x_beskonechnost_polovina_vesa():
    assert ladder_half_ok(make("S1"))


def test_lestnica_mutatsiya_ves_ne_polovina(monkeypatch):
    orig = v2.EvaluatorV2._ladder

    def full_weight(t, days, sig, sg, p, w, lest, kw):                              # мутация: вес половины w, а не w/2
        return orig(t, days, sig, sg, p, w * 2, lest, kw) if lest != "net" else orig(t, days, sig, sg, p, w, lest, kw)
    monkeypatch.setattr(v2.EvaluatorV2, "_ladder", staticmethod(full_weight))
    assert not ladder_half_ok(make("S1"))


def test_lestnica_dve_polovinki_sovpadayut_pri_x0():
    ev = make("S1")
    net, lad = rows_of(ev, PT, {}), rows_of(ev, PT, {"lestnica": 0.0})
    assert all(np.allclose(a.g, b.g, atol=1e-15) and np.allclose(a.sides, b.sides, atol=1e-15) for a, b in zip(net, lad))


def storona_long_ok(ev):
    return all(tr.side > 0 for r in rows_of(ev, PT, {"storona": "long"}) for tr in r.trades)


def test_storona_long_net_shortov(monkeypatch):
    ev = make("S1")
    assert any(tr.side < 0 for r in rows_of(ev, PT, {}) for tr in r.trades)         # контроль: при obe шорты есть
    assert storona_long_ok(ev)
    monkeypatch.setattr(v2.np, "maximum", lambda a, b: a)                              # мутация: long пропускает шорт
    assert not storona_long_ok(ev)


def test_storona_short_shag1_pokazyvaet_kray_i_menyaet_shorty():
    ev = make("S1")
    ev._kray = False
    r_obe, r_sh = rows_of(ev, PT, {}), rows_of(ev, PT, {"storona": "short_shag1"})
    L = lambda rs, s: sum(tr.side == s for r in rs for tr in r.trades)             # noqa: E731
    assert L(r_sh, 1) == L(r_obe, 1) and L(r_sh, -1) < L(r_obe, -1)                  # строже порог → шортов меньше
    assert not ev._kray
    ev.base["z_entry"] = 99                                                          # центр вне сетки сверху → край
    rows_of(ev, {**PT, "z_entry": max(load_setka("S1")["axes"]["z_entry"])}, {"storona": "short_shag1"})
    assert ev._kray


def test_zaem_tolko_na_shortakh(monkeypatch):
    ev = make("S1")
    row, x, _ = ev.evaluate_v2(PT, PER, None)
    assert any(row["by_quarter_zaem"]["10"][k] < row["by_quarter"][k] for k in row["by_quarter"])
    lg, _, _ = ev.evaluate_v2({**PT, "storona": "long"}, PER)
    assert lg["by_quarter_zaem"]["5"] == lg["by_quarter"] == lg["by_quarter_zaem"]["10"]
    orig = v2.run_e1
    def lonely(*a, **k):
        r = orig(*a, **k)
        return replace(r, short=r.expo)                                                # мутация: заём на лонгах
    monkeypatch.setattr(v2, "run_e1", lonely)
    mt, _, _ = ev.evaluate_v2({**PT, "storona": "long"}, PER)
    assert mt["by_quarter_zaem"]["10"] != mt["by_quarter"]


def test_filtr_none_ne_schitana(monkeypatch):
    stub = types.ModuleType("fastbt.filtr_e1_rynok")
    stub.razresheno = lambda name, inp, t, sig: None
    monkeypatch.setitem(sys.modules, "fastbt.filtr_e1_rynok", stub)
    ev = make("S1")
    row, x, tik = ev.evaluate_v2({**PT, "filtr": "obyom"}, PER)
    assert row["status"] == "ne_schitana" and x is None and tik is None and np.isnan(row["sharpe"])
    stub.razresheno = lambda name, inp, t, sig: np.ones(len(sig), bool)
    assert ev.evaluate_v2({**PT, "filtr": "obyom"}, PER)[0]["status"] == "ok"
    stub.razresheno = lambda name, inp, t, sig: None                                   # мутация: None как net
    monkeypatch.setattr(v2.EvaluatorV2, "_allow", lambda self, pt, t, d, s, f: np.ones(len(s), bool))
    assert ev.evaluate_v2({**PT, "filtr": "obyom"}, PER)[0]["status"] != "ne_schitana"


def test_k_x_tolko_bez_stopa(monkeypatch):
    ev, t = make("S1"), "SBER"
    p = lambda f, pt: f(ev.full(pt), t, 5.0, False, {**v2.NOVYE, "k_x": 1.0}, {})                # noqa: E731
    assert p(ev._params, {"vhod": 0.5, "k_s": "net", "R": "net"}).a == 1.0
    assert p(ev._params, {"vhod": 0.5, "k_s": 1.5, "R": "net"}).a is None
    orig = ev._params
    mut = lambda pt, t_, h, an, nov, vz: replace(orig(pt, t_, h, an, nov, vz), a=1.0)           # noqa: E731
    assert p(mut, {"vhod": 0.5, "k_s": 1.5, "R": "net"}).a == 1.0


def test_ed_probros_i_massivy(monkeypatch):
    seen = []
    monkeypatch.setattr(exec_e, "run", lambda *a, **k: seen.append(k) or (_ for _ in ()).throw(RuntimeError))
    ev, t = make("S1"), "SBER"
    m = ev.inp.meta[t]
    sg = np.full(50, 0.02)
    for ed in (None, np.ones(3)):
        with pytest.raises(RuntimeError):
            run_e1(t, [], np.zeros(0), np.zeros(0), exec_e.Params(), ed=ed)
    assert "ed" not in seen[0] and seen[1]["ed"].shape == (3,)
    s0, b = m["s0"], ev._b(t, PER)
    td, ind = ev.sk._td[t]
    dates = m["dates"][:b + 1]
    assert np.array_equal(ev._ed(t, "pct", sg, dates, s0, b), np.full(b + 1, 0.01))
    assert np.array_equal(ev._ed(t, "diap", sg, dates, s0, b), (td.high - td.low)[s0:s0 + b + 1] / td.close[s0:s0 + b + 1])
    assert np.array_equal(ev._ed(t, "atr14", sg, dates, s0, b), ind["atr14"][s0:s0 + b + 1] / td.close[s0:s0 + b + 1])
    assert ev._ed(t, "sigma", sg, dates, s0, b) is None


def test_vozmushcheniya():
    ev = make("S1")
    base = ev.evaluate_v2(PT, PER)
    assert ev.evaluate_v2(PT, PER, {"sigma_mult": 1.0})[1].tobytes() == base[1].tobytes()
    assert ev.evaluate_v2(PT, PER, {"sigma_mult": 1.1})[1].tobytes() != base[1].tobytes()
    hi = ev.evaluate_v2(PT, PER, {"komissiya": 0.0005})
    assert hi[0]["komissiya"] == 0.0005 and hi[1].sum() < base[1].sum() and ev.comm == COMM
    nul = ev.evaluate_v2(PT, PER, {"sig_fn": lambda t, s, d: np.zeros(len(s))})
    assert nul[0]["dolya_ispolneniya_limita"] != nul[0]["dolya_ispolneniya_limita"] or sum(nul[0]["sdelok_y"].values()) == 0
    m = v2.mediana_goda(np.array([1.0, 3.0, 9.0, np.nan, 5.0]), pd.to_datetime(["2016-01-04"] * 3 + ["2017-01-03"] * 2))
    assert m[0] == m[1] == m[2] == 3.0 and m[3] == m[4] == 5.0                          # медиана своего года, не всех лет


# ---- кварталы и целевая: известный ответ ----
def test_kvartaly_granitsy_i_tselevaya():
    d = pd.to_datetime(["2016-03-31", "2016-04-01", "2016-04-04", "2016-04-05"])
    x = np.array([0.1, 0.1, 0.3, 0.2])
    q = v2.kvartaly(d, x, (2016,))
    assert len(q) == 4 and np.isnan(q["2016Q3"]) and q["2016Q1"] == 0.0
    expect = np.mean(x[1:]) / np.std(x[1:], ddof=1) * np.sqrt(252)
    assert abs(q["2016Q2"] - expect) < 1e-12
    med, dolya = v2.tselevaya({**q, "2016Q1": 1.0})
    assert np.isnan(med) and dolya == 0.5                                           # Q1 и Q2 положительны; Q3, Q4 — nan
    assert v2.tselevaya({"a": 1.0, "b": -1.0, "c": 3.0, "d": 0.0}) == (0.5, 0.5)
