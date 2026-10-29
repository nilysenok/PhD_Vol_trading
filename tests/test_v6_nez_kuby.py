"""Независимый тест (тестировщик, не автор) кубов v6 S1-S6 (#655, #659): вложенность, формат осей, объём, свёртки, частоты, GOTOVY, хэш.
Синтетика и yaml; данных и сети нет. Объём и списки осей считаются заново из сырых yaml, без fastbt.canon_v6.obem/osi."""
import hashlib
import importlib.util
import math
from pathlib import Path

import pytest
import yaml

from fastbt import canon_v5 as C5
from fastbt import canon_v6 as C6

R = Path(C6.ROOT)
SIDS = [f"S{k}" for k in range(1, 7)]
SEKC = ("signal", "filtr", "vhod", "vyhod")
NOV = ("kontekst", "ispolnenie")
OTKL = {"S3": 0.03, "S4": 0.03}                  # тейк тренда 35, а не 36 значений (шапка kub_S3.yaml, принято)
TABLICA = {"S1": 1.87e38, "S2": 1.17e38, "S3": 1.76e31, "S4": 7.75e31, "S5": 2.07e37, "S6": 2.81e37}


def raw(kat, sid):
    return yaml.safe_load((R / f"fastbt/designs/{kat}/kub_{sid}.yaml").read_text(encoding="utf-8"))


def novye_raw(k6):
    return {n: v for s in NOV for n, v in k6.get(s, {}).items()}


def v5_obem(sid):
    return math.prod(len(s["setka"]) for sec in SEKC for s in raw("v5", sid).get(sec, {}).values())


def obem_nezav(sid, zam=()):
    """V6(F) = V5 x prod(L_i, i in A\\F, дети замороженных родителей тоже) x prod(L6/L5, j in E\\F) — §3.2, прямо из yaml."""
    k5, k6 = raw("v5", sid), raw("v6", sid)
    nov = novye_raw(k6)
    z = set(zam)
    while True:
        add = {n for n, v in nov.items() if v.get("roditel") and set(v["roditel"]) <= z} - z
        if not add:
            break
        z |= add
    v = v5_obem(sid)
    for n, s in nov.items():
        if n not in z:
            v *= len(s["setka"])
    for sec in SEKC:
        for n, s in k6.get(sec, {}).items():
            if "v6_dobavleno" in s and n not in z:
                L5 = len(k5[sec][n]["setka"])
                v *= (L5 + len(s["v6_dobavleno"])) / L5
    return v


def vykl(k6):
    return {n: v["setka"][0] for n, v in novye_raw(k6).items()}


def tochki_v5(sid, n):
    return C5.tochki_v5(C5.load_kub(sid), 0, n)[0]


# --- v5 ⊂ v6 ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sid", SIDS)
def test_v5_vlozhena_v_v6_na_tochkakh_v5(sid):
    k6 = raw("v6", sid)
    pts = tochki_v5(sid, 600)
    assert len(pts) > 100
    for i, c in pts:
        assert C6.canon_v6(sid, {**c, **vykl(k6)}) == c, i


@pytest.mark.parametrize("sid", SIDS)
def test_vklyuchennaya_novaya_os_popadaet_v_kanon(sid):
    """Обратное: включённая (не свёрнутая) новая ось остаётся в каноне — иначе v5 ⊂ v6 выполнялась бы тривиально."""
    k6 = raw("v6", sid)
    c = tochki_v5(sid, 1)[0][1]
    nprov = 0
    for n, v in novye_raw(k6).items():
        if v.get("roditel") or n in ("dlit", "piv_buf", "lestnica", "ed_lim", "vyhod_lim"):
            continue
        q = C6.canon_v6(sid, {**c, **vykl(k6), n: v["setka"][1]})
        assert q is not None and q.get(n) == v["setka"][1], n
        nprov += 1
    assert nprov >= 15


# --- формат осей -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sid", SIDS)
def test_kazhdaya_novaya_os_imeet_vykl_smysl_otkuda(sid):
    k6 = raw("v6", sid)
    nov = novye_raw(k6)
    assert len(nov) >= 20
    for n, v in nov.items():
        assert v.get("vykl") == v["setka"][0], n
        assert isinstance(v.get("smysl"), str) and len(v["smysl"].strip()) > 5, n
        assert isinstance(v.get("otkuda"), str) and v["otkuda"].strip(), n
        assert v.get("zhdet"), n
    assert set(nov) & {n for sec in SEKC for n in raw("v5", sid).get(sec, {})} == set()
    for sec in SEKC:
        for n, s in k6.get(sec, {}).items():
            if "v6_dobavleno" in s:
                assert s.get("zhdet"), n
    assert k6["baza"]["kub_sha"] == hashlib.sha256((R / f"fastbt/designs/v5/kub_{sid}.yaml").read_bytes()).hexdigest()


# --- объём -------------------------------------------------------------------------------------------------------------
def _obem_modul():
    spec = importlib.util.spec_from_file_location("obem_v6_t", R / "fastbt/designs/v6/obem_v6.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("sid", SIDS)
def test_obem_nezavisimo_ravno_obem_v6(sid):
    k6 = raw("v6", sid)
    nom = obem_nezav(sid)
    kub = C6.load_kub(sid)
    assert C6.obem(kub) == pytest.approx(nom, rel=1e-9)
    tf = len(k6["kontekst"]["tf"]["setka"])
    assert nom / tf == pytest.approx(TABLICA[sid], rel=OTKL.get(sid, 0.005) + 0.005)
    assert C6.obem(kub, ["tf"]) == pytest.approx(nom / tf, rel=1e-9)
    assert obem_nezav(sid, ["tf"]) == pytest.approx(nom / tf, rel=1e-9)


def test_obem_v6_py_tablica_sovpadaet_s_nezavisimym():
    m = _obem_modul()
    F = ["relf_napr", "shirina", "tf", "tejk", "sob_tip"]
    for sid, nosi, v5, v6, bez, zam in m.tablica(F):
        k6 = raw("v6", sid)
        F_ = [f for f in F if f in novye_raw(k6) or f in k6.get("vyhod", {})]
        assert v5 == pytest.approx(v5_obem(sid))
        assert v6 == pytest.approx(obem_nezav(sid), rel=1e-9)
        assert zam == pytest.approx(obem_nezav(sid, F_), rel=1e-9), sid
        assert v5 <= zam < v6


@pytest.mark.parametrize("sid", SIDS)
def test_zamorozka_vsekh_novykh_i_rasshirennykh_daet_v5(sid):
    k6 = raw("v6", sid)
    F = list(novye_raw(k6)) + [n for sec in SEKC for n, s in k6.get(sec, {}).items() if "v6_dobavleno" in s]
    assert obem_nezav(sid, F) == pytest.approx(v5_obem(sid))


# --- свёртки -----------------------------------------------------------------------------------------------------------
def test_s4_period_pri_norm_s252_dayot_odinakovyj_kanon():
    sid = "S4"
    k6, k5 = raw("v6", sid), raw("v5", sid)
    per = list(k5["signal"]["period"]["setka"])
    assert len(per) >= 3
    for i, c in tochki_v5(sid, 40):
        base = {**c, **vykl(k6), "norm": "s252"}
        kan = [C6.canon_v6(sid, {**base, "period": p}) for p in per]
        assert all(k is not None and k == kan[0] and k["norm"] == "s252" for k in kan), i
    c = tochki_v5(sid, 1)[0][1]                       # при norm = okno (выкл.) period действует
    k = [C6.canon_v6(sid, {**c, **vykl(k6), "period": p}) for p in per]
    assert len({str(x) for x in k}) == len(per)


@pytest.mark.parametrize("sid", ["S3", "S4"])
def test_dlit_trenda_bez_podtverzhdeniya_svernut(sid):
    k6 = raw("v6", sid)
    dl = list(k6["kontekst"]["dlit"]["setka"])
    assert len(dl) >= 3
    for i, c in tochki_v5(sid, 30):
        c = {**c, **vykl(k6)}
        c.pop("vhod_tip", None)
        for vt in ("stop_zayavka", None):
            q = dict(c) if vt is None else {**c, "vhod_tip": vt}
            kan = [C6.canon_v6(sid, {**q, "dlit": d}) for d in dl]
            assert all(k == kan[0] and "dlit" not in k for k in kan), (i, vt)
        kan = [C6.canon_v6(sid, {**c, "vhod_tip": "podtverzhdenie", "dlit": d}) for d in dl]
        assert [k.get("dlit", dl[0]) for k in kan] == dl and "dlit" not in kan[0], i   # «выкл.» (1) в канон не пишется


def test_dlit_kontrtrenda_ne_svorachivaetsya():
    k6 = raw("v6", "S1")
    c = tochki_v5("S1", 1)[0][1]
    d = k6["kontekst"]["dlit"]["setka"][-1]
    assert C6.canon_v6("S1", {**c, **vykl(k6), "dlit": d})["dlit"] == d


# --- частоты Соболя ----------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sobol20k():
    return {s: [c for _, c in C6.tochki_v6(C6.load_kub(s), 0, 20000, gotovy=None)[0]] for s in SIDS}


@pytest.mark.parametrize("sid", SIDS)
def test_chastota_vykl_zapreshchayushchikh_filtrov_09(sid, sobol20k):
    k6 = raw("v6", sid)
    nov = novye_raw(k6)
    flt = [n for n, v in nov.items() if v.get("filtr")]
    assert len(flt) == (12 if sid in ("S3", "S4") else 14)
    pts = sobol20k[sid]
    assert len(pts) > 10000                          # часть точек несовместима по canon_v5 (~40 %) и выбрасывается
    for f in flt:
        assert nov[f]["p_vykl"] == 0.9
        dolya = sum(f not in c for c in pts) / len(pts)
        assert abs(dolya - 0.9) < 0.012, (f, dolya)
    vkl = sum(sum(f in c for f in flt) for c in pts) / len(pts)
    assert abs(vkl - 0.1 * len(flt)) < 0.15


@pytest.mark.parametrize("sid", SIDS)
def test_rel_signal_p05_i_deti_tolko_pri_roditele(sid, sobol20k):
    nov = novye_raw(raw("v6", sid))
    pts = sobol20k[sid]
    assert abs(sum("rel_signal" not in c for c in pts) / len(pts) - 0.5) < 0.015
    for n, v in nov.items():
        if v.get("roditel"):
            assert not any(n in c and not any(r in c for r in v["roditel"]) for c in pts), n


# --- GOTOVY и zhdet ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sid", SIDS)
def test_pustoj_gotovy_ne_variruet_zhdushchie_osi(sid):
    assert C6.GOTOVY == frozenset()
    k6, k5 = raw("v6", sid), raw("v5", sid)
    kub = C6.load_kub(sid)
    ox = {n: g for n, g, _ in C6.osi(kub)}
    assert set(ox) == {n for sec in SEKC for n in k5.get(sec, {})}
    for sec in SEKC:
        for n, s in k5.get(sec, {}).items():
            assert ox[n] == list(s["setka"]), n
    pts, _ = C6.tochki_v6(kub, 0, 1500)
    assert not any(set(c) & set(novye_raw(k6)) for _, c in pts)


@pytest.mark.parametrize("sid", SIDS)
def test_gotovy_vklyuchaet_tolko_osi_s_zakrytymi_listami(sid):
    k6 = raw("v6", sid)
    nov = novye_raw(k6)
    kub = C6.load_kub(sid)
    for g in ({"V6-12"}, {"V6-12", "V6-2"}, None):
        ozh = {n for n, v in nov.items() if g is None or set(v["zhdet"]) <= set(g)}
        got = {n for n, _, _ in C6.osi(kub, g)} & set(nov)
        assert got == ozh, (g, got ^ ozh)
        pts, _ = C6.tochki_v6(kub, 0, 800, gotovy=g)
        assert set().union(*[set(c) for _, c in pts]) & set(nov) <= ozh


# --- детерминизм хэша --------------------------------------------------------------------------------------------------
def h(pts):
    return hashlib.sha256(repr([(i, sorted((k, str(v)) for k, v in c.items())) for i, c in pts]).encode()).hexdigest()


@pytest.mark.parametrize("sid", SIDS)
def test_khesh_tochek_determinirovan_i_zavisit_ot_seed(sid):
    kub = C6.load_kub(sid)
    a = C6.tochki_v6(kub, 0, 700, gotovy=None)[0]
    assert h(a) == h(C6.tochki_v6(C6.load_kub(sid), 0, 700, gotovy=None)[0])
    c1 = C6.tochki_v6(kub, 0, 300, gotovy=None)[0] + C6.tochki_v6(kub, 300, 700, gotovy=None)[0]
    assert h(c1) == h(a)                                             # продолжение последовательности Соболя
    other = dict(kub, sobol=dict(kub["sobol"], seed=kub["sobol"]["seed"] + 1))
    assert h(C6.tochki_v6(other, 0, 700, gotovy=None)[0]) != h(a)
    assert h(C6.tochki_v6(kub, 0, 700, seed="seed2", gotovy=None)[0]) != h(a)
