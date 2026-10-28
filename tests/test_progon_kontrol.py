"""Л-9: плацебо, контроли, возмущения (fastbt/progon_kontrol.py). Синтетика без данных, подставной evaluate_v2.
python3 -m pytest tests/test_progon_kontrol.py -q"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import ledger_v2, progon_kontrol as pk
from fastbt.etap1_eval_v2 import NOVYE, mediana_goda

DATES = pd.bdate_range("2015-01-01", "2019-12-31").to_numpy()
YEARS = np.asarray(pd.DatetimeIndex(DATES).year)
TICKS = [f"T{i:02d}" for i in range(17)]
BASE_PT = {"sigma": "rv22", "k_s": 1.5, "vhod": 0.5, **NOVYE}


def base_sig(i):
    rng, v, d = np.random.default_rng(100 + i), np.zeros(len(DATES)), 5
    while d < len(DATES) - 40:
        n = int(rng.integers(3, 15))
        v[d:d + n] = rng.choice([-1.0, 1.0])
        d += n + int(rng.integers(2, 20))
    return v


SIGS = {t: base_sig(i) for i, t in enumerate(TICKS)}


class Fake:
    """Подставной оценщик: значение = Σ позиций·доходность, ×(σ-множитель) − комиссия; возмущения должны дойти до счёта."""
    sid, comm, fdr = "S1", 0.0004, None

    def __init__(self):
        sg = {t: np.where(YEARS == 2015, 0.02 + 0.001 * i, 0.03) for i, t in enumerate(TICKS)}
        self.inp = SimpleNamespace(sigma={"rv22": sg}, meta={t: {"dates": DATES} for t in TICKS}, digest=lambda: "vhody-test")
        self.calls, self.ret = [], np.random.default_rng(7).normal(size=len(DATES))

    def evaluate_v2(self, pt, period, vozm=None):
        vz = dict(vozm or {})
        sigs = {t: np.asarray(vz["sig_fn"](t, SIGS[t].copy(), DATES) if "sig_fn" in vz else SIGS[t], float) for t in TICKS}
        sg0 = self.inp.sigma["rv22"][TICKS[0]][0]
        val = sum(float((s * self.ret).sum()) for s in sigs.values()) * vz.get("sigma_mult", 1.0) - 100 * vz.get("komissiya", self.comm)
        val += 0.001 * sg0 + 0.01 * (pt["vhod"] != 0.5) + (0.5 if vz.get("sigma_lim_mediana") else 0.0)
        self.calls.append((pt, vz, sigs))
        runs = sum(len(pk.runs_of(s)) for s in sigs.values())
        bq = {f"{y}Q{q}": val for y in (2016, 2017, 2018, 2019) for q in range(1, 5)}
        row = {"v": "2.0", "format": "etap1-ledger", "period": period, "komissiya": vz.get("komissiya", self.comm), "fdr": None,
               "status": "ok", "prichina": None, "vozm": vz or None, "by_year": {str(y): val for y in (2016, 2017, 2018, 2019)},
               "by_quarter": bq, "mediana_kv": val, "dolya_pol_kv": 1.0, "sharpe": val, "sdelok_y": {"2016": runs}, "flagi": {}}
        return row, np.full(20, val), np.full((17, 20), val)


def per_year(sig):
    rs = pk.runs_of(sig)
    return {y: sorted((ln, v) for st, ln, v in rs if YEARS[st] == y) for y in np.unique(YEARS)}


def test_plasebo_sohranyaet_chislo_storony_dlitelnosti_v_tikero_gode():
    for seed in range(3):
        fn = pk.placebo_sig_fn(seed)
        for t in TICKS[:6]:
            out = fn(t, SIGS[t], DATES)
            assert per_year(out) == per_year(SIGS[t]), (seed, t)
            assert out[-1] == 0.0 and not np.array_equal(out, SIGS[t])


def test_plasebo_menyaet_dni_vosproizvodimo_i_raznoe_po_seed():
    a, a2, b = (pk.placebo_sig_fn(s)(TICKS[0], SIGS[TICKS[0]], DATES) for s in (0, 0, 1))
    assert np.array_equal(a, a2) and not np.array_equal(a, b)
    st = lambda s: {x[0] for x in pk.runs_of(s)}                                        # noqa: E731
    assert st(a) != st(SIGS[TICKS[0]])


def test_k2_lag0_ravno_ishodnomu_i_sdvig_krugovoy():
    t = TICKS[1]
    assert np.array_equal(pk.k2_sig_fn(0)(t, SIGS[t], DATES), SIGS[t])                   # mutations.md № 9
    s = pk.k2_sig_fn(100)(t, SIGS[t], DATES)
    assert not np.array_equal(s, SIGS[t]) and s[-1] == 0.0
    assert sorted((x[1], x[2]) for x in pk.runs_of(s)) == sorted((x[1], x[2]) for x in pk.runs_of(np.roll(SIGS[t], 100)))
    assert all(MIN <= l <= 1000 for l in pk.k2_lags() for MIN in (63,)) and len(pk.k2_lags()) == 20


def test_k1_sigma_konstanta_sredneye_progreva_i_vozvrat():
    ev = Fake()
    old = {t: a.copy() for t, a in ev.inp.sigma["rv22"].items()}
    with pk.sigma_const(ev, "rv22"):
        for i, t in enumerate(TICKS):
            a = ev.inp.sigma["rv22"][t]
            assert np.all(a == a[0]) and a[0] == pytest.approx(0.02 + 0.001 * i)
    assert all(np.array_equal(ev.inp.sigma["rv22"][t], old[t]) for t in TICKS)


def test_sigma_mediana_goda_po_godam_a_ne_po_vsem():
    sg = np.where(YEARS == 2015, 1.0, np.where(YEARS == 2016, 3.0, 5.0))
    m = mediana_goda(sg, DATES)
    assert set(m[YEARS == 2015]) == {1.0} and set(m[YEARS == 2016]) == {3.0} and set(m[YEARS == 2017]) == {5.0}


def test_p_perm_izvestnyy_otvet_i_minimum():
    assert pk.p_perm(5.0, [1, 2, 3]) == pytest.approx(1 / 4)
    assert pk.p_perm(0.0, [1, 2, 3]) == pytest.approx(1.0)
    assert pk.p_perm(100.0, range(10)) == pytest.approx(1 / 11)
    assert pk.p_perm(1.0, [float("nan")]) == pytest.approx(1 / 2) and np.isnan(pk.p_perm(float("nan"), [1]))


def test_plan_sostav_k3_i_vozmushcheniya():
    z = pk.plan({**BASE_PT, "storona": "long", "filtr": "cbr", "k_x": 1.0})
    k = {x.kontrol: x for x in z if x.kontrol}
    assert sum(x.startswith("placebo:") for x in k) == 10 and sum(x.startswith("k2:") for x in k) == 20
    assert {kk: k["k3"].pt[kk] for kk in NOVYE} == NOVYE and k["k3"].pt["vhod"] == 0.5
    assert k["sigma_mediana"].zaglyad and k["sigma_mediana"].ev_vozm == {"sigma_lim_mediana": True}
    assert [x.vozm for x in z if x.istochnik == "vozm"] == list(pk.VOZM)
    assert all(x.istochnik in ("kontrol", "vozm") for x in z)


def run(tmp_path):
    ev, (root, csv) = Fake(), (tmp_path / "kontrol", tmp_path / "kontrol_S1.csv")
    return ev, root, csv, pk.run_strategiya(ev, {"c0": BASE_PT}, root, csv, T=20)["c0"]


def test_vozmushcheniya_dohodyat_do_scheta_i_dayut_effekt(tmp_path):
    ev, _, _, res = run(tmp_path)
    r = {(z.vozm and tuple(z.vozm.items())): row for z, row in res if z.istochnik == "vozm"}
    base = res[0][1]["mediana_kv"]
    assert r[(("sigma_mult", 1.1),)]["mediana_kv"] != base and r[(("sigma_mult", 0.9),)]["mediana_kv"] != r[(("sigma_mult", 1.1),)]["mediana_kv"]
    assert r[(("komissiya", 0.0005),)]["komissiya"] == 0.0005 and r[(("komissiya", 0.0005),)]["mediana_kv"] < r[(("komissiya", 0.0004),)]["mediana_kv"]
    assert any(c[1].get("bar_plus") == 1 for c in ev.calls) and any(c[1].get("stop_tik") == 1 for c in ev.calls)


def test_stroki_kontrolya_v_kontrol_i_ne_v_n_osnovnogo_ledger(tmp_path):
    ev, root, csv, res = run(tmp_path)
    main = ledger_v2.Pisatel(tmp_path / "main", "S1", 0, dict(T=20, kod=res[1][1]["kod"], vhody="vhody-test"))
    row = {**res[1][1], "klyuch_sha": "a" * 64, "istochnik": "sobol1", "kontrol": None}
    main.add(row, np.zeros(20), np.zeros((17, 20)))
    main.close()
    assert ledger_v2.Chitatel(tmp_path / "main", "S1").n_unique() == 1
    k = ledger_v2.Chitatel(root, "S1")
    assert k.n_unique() == len(res) - 1 == 39 and {r["istochnik"] for r in k.rows()} == {"kontrol", "vozm"}
    assert not (tmp_path / "main" / "S1" / "p01").exists()
    with pytest.raises(ValueError):
        pk.kontrol_pisatel(tmp_path / "main", "S1", 1, dict(T=20, kod="x", vhody="y"))
    assert "потолок" in csv.read_text(encoding="utf-8").splitlines()[0]
    t = pd.read_csv(csv, comment="#")
    assert len(t) == 40 and t.loc[t.kontrol == "sigma_mediana", "zaglyadyvanie"].all() and t[t.kontrol.str.startswith("placebo", na=False)].p.nunique() == 1


def test_k2_lag0_stroka_ravna_baze(tmp_path):
    ev = Fake()
    z = [x for x in pk.plan(BASE_PT, lags=[0]) if x.kontrol == "k2:lag0"][0]
    assert ev.evaluate_v2(z.pt, "train4", z.ev_vozm)[0]["mediana_kv"] == ev.evaluate_v2(BASE_PT, "train4")[0]["mediana_kv"]
