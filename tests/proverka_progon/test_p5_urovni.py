"""П-5 / Л-11: независимые тесты `fastbt/progon_urovni.py` (контракт §1.4, §3 IG-4, §6; задание Л-11). Кода нет -> красные
(ImportError), без skip/xfail (Ж-081). Ожидаемый API (фиксирует проверяющий):
  UMOLCHANIYA: dict ves/lam/rho/sostav/tsel_vol/okno_vol/tormoz_mdd;  vse_konfigi() -> list[dict] (1296)
  a_shtrih(komponenty: {sid: float64[17,T]}, konf) -> ndarray[T];  vesa_strategij(komponenty, konf, d) -> {sid: w}
  tablica_1296(komponenty, daty) -> DataFrame;  komponenty_yakor(gody) -> (komponenty, daty[datetime64])
IG-4 (`-k a_shtrih_ravno_a`) — свой, независимый: эталон = Engine.port(A_row) на кэше боевой A (ekviv_s9.a_cache), не код Л-11.
python3 -m pytest tests/proverka_progon/test_p5_urovni.py -q"""
import importlib

import numpy as np
import pytest

TOL = 1e-12
UMOL = dict(ves="ravnyj", lam="net", rho="net", sostav="vse6", tsel_vol="net", okno_vol=63, tormoz_mdd="net")


def pu():
    return importlib.import_module("fastbt.progon_urovni")


def _komp(T=300, seed=1):
    rng = np.random.default_rng(seed)
    k = {"S1": rng.normal(0, .01, (17, T)), "S2": rng.normal(0, .01, (17, T)), "S3": rng.normal(0, .01, (17, T))}
    k["S1"][2:, :] = np.nan                      # у S1 один-два тикера, у S2 пять, у S3 все -> веса по парам различаются
    k["S1"][1, 100:120] = np.nan
    k["S2"][5:, :] = np.nan
    return k


def test_umolchaniya_i_chislo_konfiguracij():
    assert dict(pu().UMOLCHANIYA) == UMOL
    vk = pu().vse_konfigi()
    assert len(vk) == 54 * 24 == 1296
    assert len({tuple(sorted(k.items())) for k in vk}) == 1296
    assert sum(k == UMOL for k in vk) == 1


def test_a_shtrih_umolchaniya_sredneye_par():
    k = _komp()
    T = 300
    pairs = np.concatenate([k[s] for s in ("S1", "S2", "S3")], axis=0)        # (51, T)
    ozh = np.nanmean(pairs, axis=0)
    got = pu().a_shtrih(k, UMOL)
    assert got.shape == (T,) and np.max(np.abs(got - ozh)) <= TOL
    ravnye_po_strat = np.nanmean([np.nanmean(k[s], axis=0) for s in k], axis=0)
    assert np.max(np.abs(got - ravnye_po_strat)) > 1e-4                        # это не среднее средних


def test_a_shtrih_den_bez_par_nan_i_vhod_ne_menyaetsya():
    k = {"S1": np.full((17, 4), np.nan)}
    k["S1"][0, 1] = 0.5
    c = {s: v.copy() for s, v in k.items()}
    out = pu().a_shtrih(k, UMOL)
    assert np.isnan(out[0]) and out[1] == 0.5
    assert np.array_equal(k["S1"], c["S1"], equal_nan=True)


def test_mutaciya_umolchaniya_s_vesom_obr_korr_otlichaetsya():
    k = _komp()
    a = pu().a_shtrih(k, UMOL)
    b = pu().a_shtrih(k, {**UMOL, "ves": "obr_korr_126"})
    assert np.nanmax(np.abs(a - b)) > 1e-9           # если умолчания тихо = obr_korr, первый тест выше упадёт, этот — защита от константы


def _kor(seed=3, T=400):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, .01, T)
    s1 = z + rng.normal(0, .003, T)
    s2 = z + rng.normal(0, .003, T)
    s3 = rng.normal(0, .01, T)
    return {s: np.tile(v, (17, 1)) for s, v in (("S1", s1), ("S2", s2), ("S3", s3))}


def test_obr_korr_dayot_bolshe_nezavisimoj():
    w = pu().vesa_strategij(_kor(), {**UMOL, "ves": "obr_korr_126"}, 300)
    assert abs(sum(w.values()) - 1) <= TOL and w["S3"] > w["S1"] and w["S3"] > w["S2"]


@pytest.mark.parametrize("ves", ["ravnyj", "obr_korr_126", "obr_korr_252"])
@pytest.mark.parametrize("rho", ["net", 0.5, 0.7])
def test_ves_i_rho_tolko_po_dannym_do_d_minus_1(ves, rho):
    k, d = _kor(), 300
    konf = {**UMOL, "ves": ves, "rho": rho}
    w0 = pu().vesa_strategij(k, konf, d)
    k2 = {s: v.copy() for s, v in k.items()}
    for v in k2.values():
        v[:, d:] = np.random.default_rng(9).normal(0, 1.0, v[:, d:].shape)    # будущее, включая сам день d, переписано
    w1 = pu().vesa_strategij(k2, konf, d)
    assert all(abs(w0[s] - w1[s]) <= TOL for s in w0), (ves, rho)             # окно корреляции без дня d и без всего периода


def test_dubl_rho_070_odin_ves_na_dvoih():
    x = np.random.default_rng(4).normal(0, .01, (17, 400))
    k = {"S1": x, "S2": x.copy(), "S3": np.random.default_rng(5).normal(0, .01, (17, 400))}
    w = pu().vesa_strategij(k, {**UMOL, "rho": 0.7}, 300)
    assert abs(w["S1"] - w["S2"]) <= TOL and abs((w["S1"] + w["S2"]) - w["S3"]) <= TOL   # пара-дубль делит один вес
    w0 = pu().vesa_strategij(k, UMOL, 300)
    assert all(abs(v - 1 / 3) <= TOL for v in w0.values())                     # rho = net: дублей нет


def test_nedubl_pri_korrelyacii_nizhe_poroga_ves_ne_delitsya():
    r = np.random.default_rng(6)
    a = r.normal(0, .01, 600)
    k = {"S1": np.tile(a, (17, 1)), "S2": np.tile(0.3 * a + 0.95 * r.normal(0, .01, 600), (17, 1)),
         "S3": np.tile(r.normal(0, .01, 600), (17, 1))}
    w = pu().vesa_strategij(k, {**UMOL, "rho": 0.7}, 400)
    assert all(abs(v - 1 / 3) <= TOL for v in w.values())


def test_a_shtrih_bez_buduschego_na_vyborke_konfiguracij():
    k, d0 = _komp(), 200
    vk = pu().vse_konfigi()[::37]
    assert len(vk) > 30
    for konf in vk:
        a0 = pu().a_shtrih(k, konf)
        k2 = {s: v.copy() for s, v in k.items()}
        for v in k2.values():
            v[:, d0:] = np.random.default_rng(1).normal(0, .5, v[:, d0:].shape)
        a1 = pu().a_shtrih(k2, konf)
        assert np.allclose(a0[:d0], a1[:d0], atol=TOL, equal_nan=True), konf


def test_tormoz_mdd_po_dannym_do_d_minus_1():
    x = np.full((1, 80), 0.01)
    x[0, 50] = -0.2
    k = {"S1": x}
    base = pu().a_shtrih(k, UMOL)
    br = pu().a_shtrih(k, {**UMOL, "tormoz_mdd": 0.10})
    assert br[50] == base[50] == -0.2                  # просадка, начавшаяся в день d, в день d ещё не тормозит
    assert br[51] == 0.0 and base[51] == 0.01          # со следующего дня — тормоз
    assert np.allclose(br[:50], base[:50], atol=TOL)


def test_tablica_1296_odna_osnovnaya_bez_ranzhirovaniya():
    k = {s: v[:, :200] for s, v in _komp().items()}
    daty = np.arange(np.datetime64("2018-01-01"), np.datetime64("2018-01-01") + 200)
    t = pu().tablica_1296(k, daty)
    assert len(t) == 1296 and int(t["osnovnaya"].sum()) == 1
    assert not any(c.lower() in ("rang", "rank", "luchshaya", "vybrana") for c in t.columns)
    assert dict(t[t["osnovnaya"]].iloc[0][list(UMOL)]) == UMOL
    assert set(t[~t["osnovnaya"]]["rol"]) == {"chuvstvitelnost"}
    assert not hasattr(pu(), "vybrat_konfig") and not hasattr(pu(), "luchshaya_konfig")


def test_yaml_urovni_54_na_24():
    import yaml
    from pathlib import Path
    d = yaml.safe_load(Path("fastbt/designs/progon/urovni_2_3.yaml").read_text())
    assert d["v"] == 1
    assert d["umolchaniya"] == UMOL


def test_a_shtrih_ravno_a():
    """IG-4 (свой): A' при умолчаниях = боевая A через Engine.port, ≤ 1e-12, годы 2017–2019 (запас Ж-082 п. 4; 2016 — IG-0)."""
    import pandas as pd
    from fastbt import ekviv_s9
    cache = ekviv_s9.a_cache(2019)
    eng = ekviv_s9.s9_engine(cache, ekviv_s9.windows("train4")[2019], "train4")
    ref = eng.port(eng.A_row, comm=0.0004)["excess"]
    komp, daty = pu().komponenty_yakor((2017, 2018, 2019))
    assert all(v.dtype == np.float64 for v in komp.values())
    mine = pd.Series(pu().a_shtrih(komp, UMOL), index=pd.to_datetime(daty))
    for y in (2017, 2018, 2019):
        r = ref[ref.index.year == y]
        assert len(r) > 200 and np.abs(r.to_numpy()).max() > 0
        m = mine.reindex(r.index)
        assert not m.isna().any(), y
        assert float(np.abs(m.to_numpy() - r.to_numpy()).max()) <= TOL, y
