"""Д-8 (Ж-145): таблица (6) — топ-10 против к2 и плацебо. Гейты G8.1–G8.4. Тесты на реальном ledger — только при забое и FASTBT_MAIN (иначе skip)."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import diag_v6_perescet as dp, diag_v6_t6 as t6, progon_kontrol as pk, progon_sobol as ps, progon_v6 as v6

L = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/server_2021/train4_v6_31-10/ledger")
DANNYE = (L / "train4_v6_s1" / "S4" / "manifest.json").exists() and os.environ.get("FASTBT_MAIN") not in (None, "")


def _p_bez_edinicy(fakt, vals):
    """Мутация G8.2: p без «+1» — минимум 0, не бывает p = 1/(n+1)."""
    v = np.asarray(list(vals), float)
    return float((v >= fakt).sum()) / len(v)


def _rec(fakt, plac, k2):
    """Синтетическая запись точки: fakt/plac/k2 — dict метрик или числа (все метрики равны числу)."""
    m = lambda x: x if isinstance(x, dict) else dict(sharpe=x, msr_ned=x, norm=x, sdelok=x, cena=0.0, n=40)    # noqa: E731
    ks = [dict(vid="plac", par=i, **m(v)) for i, v in enumerate(plac)] + [dict(vid="k2", par=i, **m(v)) for i, v in enumerate(k2)]
    return dict(sid="S1", prohod="s1", sobol_i=1, fakt=m(fakt), kontroli=ks)


def test_g81_p_izvestnyj_otvet():
    assert pk.p_perm(1.0, [0.0, 0.0, 2.0, 3.0]) == pytest.approx(3 / 5)          # (1 + 2) / (4 + 1)
    assert pk.p_perm(9.0, [0.0] * 10) == pytest.approx(1 / 11)                    # минимум плацебо при n = 10
    assert np.isnan(pk.p_perm(float("nan"), [1.0]))
    r = t6.stroka(_rec(5.0, [0.0] * 10, [0.0] * 20), 1)
    assert r["p_sharpe_plac"] == pytest.approx(1 / 11) and r["p_sharpe_k2"] == pytest.approx(1 / 21)
    assert r["p_msr_plac"] == pytest.approx(1 / 11)
    assert t6.stroka(_rec(0.0, [1.0] * 10, [1.0] * 20), 1)["p_sharpe_plac"] == 1.0   # все контроли лучше факта


def test_g82_mutaciya_p_bez_edinicy_lovitsya():
    ok = t6.stroka(_rec(5.0, [0.0] * 10, [0.0] * 20), 1)
    bad = t6.stroka(_rec(5.0, [0.0] * 10, [0.0] * 20), 1, p_fn=_p_bez_edinicy)
    assert bad["p_sharpe_plac"] == 0.0 and ok["p_sharpe_plac"] > 0.0               # ловится: p = 0 невозможен
    assert bad["p_sharpe_k2"] != pytest.approx(ok["p_sharpe_k2"])


def test_g83_k2_lag0_ravno_ishodnomu_pobajtno():
    sig = np.random.default_rng(1).choice([-1.0, 0.0, 1.0], 400)
    assert pk.k2_sig_fn(0)("T", sig, None).tobytes() == sig.tobytes()
    assert pk.k2_sig_fn(100)("T", sig, pd.date_range("2016-01-01", periods=400)).tobytes() != sig.tobytes()   # ненулевой сдвиг меняет ряд


def _sig_po_godam():
    d = pd.date_range("2016-01-01", "2019-12-31", freq="B")
    sig = np.zeros(len(d))
    yr = d.year.to_numpy()
    for y in range(2016, 2020):
        lo = int(np.flatnonzero(yr == y)[0])
        for s, ln, v in ((10, 5, 1.0), (40, 3, -1.0), (80, 12, 1.0), (150, 1, -1.0)):
            sig[lo + s:lo + s + ln] = v
    return d, sig


@pytest.mark.parametrize("seed", range(5))
def test_g84_placebo_sohranyaet_chislo_i_dliny_otrezkov(seed):
    d, sig = _sig_po_godam()
    out = pk.placebo_sig_fn(seed)("T", sig, d)
    f = lambda s: sorted((ln, v) for _, ln, v in pk.runs_of(s))             # noqa: E731
    assert f(out) == f(sig) and len(pk.runs_of(out)) == 16
    for y in range(2016, 2020):                                                 # число и длины — в каждом году отдельно
        m = d.year.to_numpy() == y
        assert f(out[m]) == f(sig[m])
    assert not np.array_equal(out, sig)                                         # и ряд действительно переставлен


class _Ev:
    """Минимальный оценщик: σ с NaN в последний день, прогрев s0 = 3."""
    class inp:
        sigma = {"s": {"T": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan])}}
        meta = {"T": {"s0": 3}}

    class r5:
        pt4 = {"sigma": "s"}


def test_bez_kraya_obnulyaet_proguev_i_posledniy_den_i_schitaet():
    sig, schet = np.ones(8), [0]
    out = t6.bez_kraya(_Ev, lambda t, s, d: s.copy(), schet)("T", sig, None)
    assert out.tolist() == [0, 0, 0, 1, 1, 1, 0, 0]          # прогрев (0–2), день 6 (σ дня 7 = NaN), день 7
    assert schet[0] == 5
    sig0, s0 = np.zeros(8), [0]
    assert t6.bez_kraya(_Ev, lambda t, s, d: s.copy(), s0)("T", sig0, None).tolist() == [0] * 8 and s0[0] == 0   # нулевые не считаются
    k2 = t6.bez_kraya(_Ev, pk.k2_sig_fn(0), [0])("T", np.r_[np.zeros(3), 1.0, 0, 0, 0, 0], None)   # к2 лаг 0 — исходный (не затронут)
    assert k2.tolist() == [0, 0, 0, 1, 0, 0, 0, 0]


def test_net_signala_zh148_p4_i_svod():
    sil = dict(sharpe=1.0, msr_ned=1.0, norm=50.0, sdelok=3.0, cena=49.0, n=40)
    # плацебо с такими же норм. и сделочным, как у точки, → сигнала нет (даже если Sharpe дн. выше)
    sl = t6.stroka(_rec(sil, [dict(sil, sharpe=0.0)] * 10, [0.0] * 20), 1)
    assert sl["net_signala"] is True
    ok = t6.stroka(_rec(sil, [dict(sharpe=0.0, msr_ned=0.0, norm=1.0, sdelok=0.1, cena=0.0, n=40)] * 10, [0.0] * 20), 1)
    assert ok["net_signala"] is False
    df = pd.DataFrame([ok, dict(ok, sid="S2", fakt_sharpe=-1.0)])               # S2: факт хуже p95 контролей → слабое место
    sv, vyvod = t6.svod(df)
    assert "Sk: S2" in vyvod and "S1" not in vyvod.split("Ж-148")[0]
    assert sv.set_index("sid").loc["S1", "dolya_luchshe_plac"] == 1.0


def test_tablica_iz_kesha(tmp_path):
    top = tmp_path / "top10.csv"
    top.write_text("sid,rang,sobol_i,prohod\nS1,1,1,s1\n", encoding="utf-8")
    (tmp_path / "t6_S1_s1_000001.json").write_text(json.dumps(_rec(5.0, [0.0] * 10, [0.0] * 20)), encoding="utf-8")
    vyvod = t6.tablica(top, tmp_path, tmp_path / "t6.csv")
    assert vyvod.startswith("(6): не слабое место")
    txt = (tmp_path / "t6.csv").read_text(encoding="utf-8")
    assert txt.startswith("# статус: разработка train4") and "потолок" in txt
    assert len(pd.read_csv(tmp_path / "t6.csv", comment="#")) == 2             # точка + svod


@pytest.mark.skipif(not DANNYE, reason="нужны забор ledger и FASTBT_MAIN=<теневой корень>")
def test_g83_real_k2_lag0_ravno_tochke_pobajtno():
    sid, i = "S1", 8002          # топ-1 S1 (40 сделок); точка с минимальным sobol_i — без сделок, к2 на ней не отличим от факта
    zam = v6.zamorozka(L / "train4_morris", [sid], lambda s: ps.kub_put(s, ps.Rezhim("v6", "train4", "seed", False, None)))[0][2]
    pt, rz, _ = dp.proverit_klyuch(sid, "seed", i, L / "train4_v6_s1", zam)
    ps.fabrika, sr = None, ps.sreda(sid, rz)
    _, x0, _ = sr.evr.evaluate_v2(pt, "train4", None)
    _, x1, _ = sr.evr.evaluate_v2(pt, "train4", {"sig_fn": pk.k2_sig_fn(0)})
    assert np.asarray(x0).tobytes() == np.asarray(x1).tobytes()
    _, x2, _ = sr.evr.evaluate_v2(pt, "train4", {"sig_fn": pk.k2_sig_fn(300)})
    assert np.asarray(x2).tobytes() != np.asarray(x0).tobytes()                # мутация: сдвиг действительно меняет счёт
