"""П-5 / Л-11: независимые тесты `fastbt/progon_urovni.py` (PR #439) по контракту (§1.4, §3 IG-4, §6) и заданию Л-11. Синтетика.
Требования не ослаблены. Изменения против первой версии (9b9b354) — под фактический API:
  UMOLCHANIYA -> setka()["umolchaniya"];  vse_konfigi -> konfiguracii();  a_shtrih(komponenty, konf, daty, pozicii, proshli);
  vesa_strategij -> _vesa_strategij(S, daty, ves, porog, act) (приватная, единственный доступ к весам);
  tormoz: по коду множитель x0,5 (не обнуление) — тест проверяет момент реакции (со дня d+1), требование «до d−1» то же;
  tablica_1296 требует proshli и pozicii; колонка роли — status;  IG-4 — через pary_iz_dvizhka + Engine.port.
python3 -m pytest tests/proverka_progon/test_p5_urovni.py -q"""
import importlib

import numpy as np
import pandas as pd
import pytest

TOL = 1e-12
UMOL = dict(ves="ravnyj", lam="net", rho="net", sostav="vse6", tsel_vol="net", okno_vol=63, tormoz_mdd="net")


def pu():
    return importlib.import_module("fastbt.progon_urovni")


def _daty(T):
    return pd.bdate_range("2017-01-02", periods=T)


def _komp(T=300, seed=1):
    rng = np.random.default_rng(seed)
    k = {s: rng.normal(0, .01, (17, T)) for s in ("S1", "S2", "S3")}
    k["S1"][2:, :] = np.nan                      # у S1 два тикера, у S2 пять, у S3 все: веса по парам различаются
    k["S1"][1, 100:120] = np.nan
    k["S2"][5:, :] = np.nan
    return k


def test_umolchaniya_i_chislo_konfiguracij():
    sk = pu().setka()
    assert dict(sk["umolchaniya"]) == UMOL and sk["v"] == 1
    u2, u3 = sk["uroven2"], sk["uroven3"]
    assert len(u2["ves"]) * len(u2["lam"]) * len(u2["rho"]) * len(u2["sostav"]) == 54
    assert len(u3["tsel_vol"]) * len(u3["okno_vol"]) * len(u3["tormoz_mdd"]) == 24
    vk = pu().konfiguracii()
    assert len(vk) == 1296 and len({tuple(sorted((k, str(v)) for k, v in c.items())) for c in vk}) == 1296
    osn = [c for c in vk if c["osnovnaya"]]
    assert len(osn) == 1 and {k: osn[0][k] for k in UMOL} == UMOL
    assert sum(all(c[k] == v for k, v in UMOL.items()) for c in vk) == 1


def test_a_shtrih_umolchaniya_sredneye_par():
    k, T = _komp(), 300
    pairs = np.concatenate([k[s] for s in ("S1", "S2", "S3")], axis=0)
    got = pu().a_shtrih(k, UMOL, _daty(T))
    assert got.shape == (T,) and np.max(np.abs(got - np.nanmean(pairs, axis=0))) <= TOL
    po_strat = np.nanmean([np.nanmean(k[s], axis=0) for s in k], axis=0)
    assert np.max(np.abs(got - po_strat)) > 1e-4                               # не среднее средних по стратегиям


def test_a_shtrih_den_bez_par_nan_i_vhod_ne_menyaetsya():
    k = {"S1": np.full((17, 30), np.nan)}
    k["S1"][0, 1] = 0.5
    c = k["S1"].copy()
    out = pu().a_shtrih(k, UMOL, _daty(30))
    assert np.isnan(out[0]) and out[1] == 0.5
    assert np.array_equal(k["S1"], c, equal_nan=True)


def test_umolchaniya_ne_obr_korr():
    k, T = _komp(), 300
    a = pu().a_shtrih(k, UMOL, _daty(T))
    b = pu().a_shtrih(k, {**UMOL, "ves": "obr_korr_126"}, _daty(T))
    assert np.nanmax(np.abs(a - b)) > 1e-9


def _kor(seed=3, T=400):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, .01, T)
    s = {"S1": z + rng.normal(0, .003, T), "S2": z + rng.normal(0, .003, T), "S3": rng.normal(0, .01, T)}
    return np.stack([s["S1"], s["S2"], s["S3"]])


def _w(S, ves="ravnyj", rho=None):
    return pu()._vesa_strategij(S, _daty(S.shape[1]), ves, rho, [0, 1, 2])


def test_obr_korr_dayot_bolshe_nezavisimoj():
    w = _w(_kor(), "obr_korr_126")[:, 300]
    assert abs(w.sum() - 1) <= TOL and w[2] > w[0] and w[2] > w[1]


@pytest.mark.parametrize("ves", ["ravnyj", "obr_korr_126", "obr_korr_252"])
@pytest.mark.parametrize("rho", [None, 0.5, 0.7])
def test_ves_i_rho_tolko_po_dannym_do_d_minus_1(ves, rho):
    S, d = _kor(), 300
    w0 = _w(S, ves, rho)
    S2 = S.copy()
    S2[:, d:] = np.random.default_rng(9).normal(0, 1.0, S2[:, d:].shape)     # день d и всё после — переписаны
    w1 = _w(S2, ves, rho)
    assert np.max(np.abs(w0[:, :d + 1] - w1[:, :d + 1])) <= TOL, (ves, rho)  # веса дня d не видят d и позже; не весь период


def test_dubl_rho_070_odin_ves_na_dvoih():
    r = np.random.default_rng(4)
    x = r.normal(0, .01, 400)
    S = np.stack([x, x.copy(), r.normal(0, .01, 400)])
    w = _w(S, "ravnyj", 0.7)[:, 300]
    assert abs(w[0] - w[1]) <= TOL and abs((w[0] + w[1]) - w[2]) <= TOL
    assert np.max(np.abs(_w(S, "ravnyj", None)[:, 300] - 1 / 3)) <= TOL        # rho = net: дублей нет


def test_nedubl_pri_korrelyacii_nizhe_poroga_ves_ne_delitsya():
    r = np.random.default_rng(6)
    a = r.normal(0, .01, 600)
    S = np.stack([a, 0.3 * a + 0.95 * r.normal(0, .01, 600), r.normal(0, .01, 600)])
    assert np.max(np.abs(_w(S, "ravnyj", 0.7)[:, 400] - 1 / 3)) <= TOL


def test_a_shtrih_bez_buduschego_na_vyborke_konfiguracij():
    k, T, d0 = _komp(), 300, 200
    vk = [c for c in pu().konfiguracii() if c["sostav"] == "vse6" and c["lam"] == "net"][::7]
    assert len(vk) > 30
    for konf in vk:
        a0 = pu().a_shtrih(k, konf, _daty(T))
        k2 = {s: v.copy() for s, v in k.items()}
        for v in k2.values():
            v[:, d0:] = np.random.default_rng(1).normal(0, .5, v[:, d0:].shape)
        a1 = pu().a_shtrih(k2, konf, _daty(T))
        assert np.allclose(a0[:d0], a1[:d0], atol=TOL, equal_nan=True), konf


def test_tormoz_mdd_po_dannym_do_d_minus_1():
    x = np.full((1, 80), 0.01)
    x[0, 50] = -0.2
    base = pu().a_shtrih({"S1": x}, UMOL, _daty(80))
    br = pu().a_shtrih({"S1": x}, {**UMOL, "tormoz_mdd": 10}, _daty(80))
    assert br[50] == base[50] == -0.2                    # просадка, начавшаяся в день d, в день d ещё не тормозит
    assert br[51] < base[51] and base[51] == 0.01        # со следующего дня — тормоз
    assert np.allclose(br[:50], base[:50], atol=TOL)


def test_tsel_vol_okno_do_d_minus_1_i_bez_plecha():
    r = np.random.default_rng(8)
    x = r.normal(0, .02, (1, 300))
    base = pu().a_shtrih({"S1": x}, UMOL, _daty(300))
    sc = pu().a_shtrih({"S1": x}, {**UMOL, "tsel_vol": 10}, _daty(300))
    m = np.abs(base) > 0
    assert np.all(np.abs(sc[m]) <= np.abs(base[m]) + TOL)                       # масштаб ≤ 1: плеча нет
    x2 = x.copy()
    x2[0, 200:] *= 50.0
    sc2 = pu().a_shtrih({"S1": x2}, {**UMOL, "tsel_vol": 10}, _daty(300))
    assert np.array_equal(sc[:200], sc2[:200])                                  # будущая волатильность не видна


def test_lam_lomit_sovpadenie_signalov_i_ne_trogaet_raznoimennye():
    T = 80
    x = np.full((1, T), 0.01)
    k = {"S1": x.copy(), "S2": x.copy()}
    same = {"S1": np.ones((1, T)), "S2": np.ones((1, T))}
    opp = {"S1": np.ones((1, T)), "S2": -np.ones((1, T))}
    f = lambda lam, poz: pu().a_shtrih(k, {**UMOL, "lam": lam}, _daty(T), pozicii=poz)   # noqa: E731
    assert np.allclose(f(1, same), 0.005, atol=TOL)                             # |Σ a·pos| = 1 > λ·max a = 0,5 -> множитель 0,5
    assert np.allclose(f(2, same), 0.01, atol=TOL)                              # λ = 2: cap = 1 -> без урезания
    assert np.allclose(f(1, opp), 0.01, atol=TOL)                               # противоположные: сумма 0, лимит не действует
    with pytest.raises(ValueError):
        f(1, None)                                                              # λ без позиций — отказ, не молчаливое «net»


def test_sostav_proshedshie_bez_spiska_otkaz():
    with pytest.raises(ValueError):
        pu().a_shtrih(_komp(60), {**UMOL, "sostav": "proshedshie"}, _daty(60))
    a = pu().a_shtrih(_komp(60), {**UMOL, "sostav": "proshedshie"}, _daty(60), proshli=["S3"])
    assert np.allclose(a, np.nanmean(_komp(60)["S3"], axis=0), atol=TOL, equal_nan=True)


def test_tablica_1296_odna_osnovnaya_bez_ranzhirovaniya(tmp_path):
    T = 150
    k = _komp(T)
    poz = {s: np.ones((17, T)) for s in k}
    t = pu().tablica_1296(k, _daty(T), pozicii=poz, proshli=["S1", "S2"], out=tmp_path / "u.csv")
    assert len(t) == 1296 and int(t["osnovnaya"].sum()) == 1
    assert not any(c.lower() in ("rang", "rank", "luchshaya", "vybrana") for c in t.columns)
    assert dict(t[t["osnovnaya"]].iloc[0][list(UMOL)]) == UMOL
    assert set(t[~t["osnovnaya"]]["status"]) == {"chuvstvitelnost"} and list(t.index) == list(range(1296))
    assert not hasattr(pu(), "vybrat_konfig") and not hasattr(pu(), "luchshaya_konfig")


def test_a_shtrih_ravno_a():
    """IG-4 (свой): A' при умолчаниях = боевая A через Engine.port, ≤ 1e-12, 2017–2019 (запас Ж-082 п. 4; 2016 — IG-0)."""
    from fastbt import ekviv_s9
    cache = ekviv_s9.a_cache(2019)
    eng = ekviv_s9.s9_engine(cache, ekviv_s9.windows("train4")[2019], "train4")
    ref = eng.port(eng.A_row, comm=0.0004)["excess"]
    komp, _, daty = pu().pary_iz_dvizhka(eng, 0.0004)
    assert all(v.dtype == np.float64 for v in komp.values())
    mine = pd.Series(pu().a_shtrih(komp, UMOL, daty), index=pd.to_datetime(daty))
    for y in (2017, 2018, 2019):
        r = ref[ref.index.year == y]
        assert len(r) > 200 and np.abs(r.to_numpy()).max() > 0
        m = mine.reindex(r.index)
        assert not m.isna().any(), y
        assert float(np.abs(m.to_numpy() - r.to_numpy()).max()) <= TOL, y
