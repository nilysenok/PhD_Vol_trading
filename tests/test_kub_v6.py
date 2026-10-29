"""V6-13: кубы v6 (fastbt/designs/v6/kub_S{k}.yaml) и канон fastbt/canon_v6.py. Синтетика и yaml, данных нет, сети нет."""
import copy

import pytest

from fastbt import canon_v5 as C5
from fastbt import canon_v6 as C6

SIDS = [s for s in C6.SIDS if (C6.KUB6 / f"kub_{s}.yaml").exists()]
KUBY = {s: C6.load_kub(s) for s in SIDS}
NFILTR = {"kontr": 14, "trend": 12}                         # запрещающие фильтры с p = 0,9 (инвентаризация §3.5)
TABLICA = {"S1": 1.87e38, "S2": 1.17e38, "S3": 1.76e31, "S4": 7.75e31, "S5": 2.07e37, "S6": 2.81e37}   # §3.1, без оси tf
OTKL = {"S3": 0.029, "S4": 0.029}                           # ↺ тейк тренда 35, а не 36 (шапка kub_S3.yaml)


def vykl(kub):
    return {n: v["setka"][0] for n, v in C6.novye(kub).items()}


@pytest.mark.parametrize("sid", SIDS)
def test_proverka_kuba(sid):
    C6.proverit(KUBY[sid])


@pytest.mark.parametrize("sid", SIDS)
def test_vykl_pervoe_i_filtry_p09(sid):
    kub = KUBY[sid]
    p09 = [n for n, v in C6.novye(kub).items() if v.get("p_vykl") == C6.P_FILTR]
    assert len(p09) == NFILTR[C5.tip(sid)]
    assert all(v.get("p_vykl") in (0.5, 0.9, None) for v in C6.novye(kub).values())
    assert all(v["zhdet"] and v["smysl"] and v["otkuda"] and v["vykl"] == v["setka"][0] for v in C6.novye(kub).values())


@pytest.mark.parametrize("sid", SIDS)
def test_snyatye_yakorya_sovpadayut_s_vykl(sid):
    kub, o = KUBY[sid], vykl(KUBY[sid])
    for sec, d in kub["snyato"].items():
        for staroe, nov in d.items():
            if staroe != "storona":                          # obe ↔ asim 0: другое имя значения
                assert str(kub["v5"][sec][staroe]) == str(o[nov]), (staroe, nov)


def vlozhena(sid, canon, n=400):
    """v5 ⊂ v6: канон v5 + все новые оси «выкл.» → канон v6 без новых осей и равен канону v5."""
    kub = KUBY[sid]
    pts, _ = C5.tochki_v5(kub["v5"], 0, n)
    return all(canon(sid, {**c, **vykl(kub)}, kub) == c for _, c in pts)


@pytest.mark.parametrize("sid", SIDS)
def test_v5_vlozhena_v_v6(sid):
    assert vlozhena(sid, C6.canon_v6)


def test_mutaciya_vykl_ne_vybrasyvaetsya_lovitsya():
    def plokhoj(sid, pt, kub):
        c = C6.canon_v6(sid, pt, kub)
        return None if c is None else {**c, **vykl(kub)}
    assert not vlozhena(SIDS[0], plokhoj)


@pytest.mark.parametrize("sid", SIDS)
def test_sobol_bez_gotovykh_osej_raven_v5(sid):
    kub = copy.deepcopy(KUBY[sid])
    kub["sobol"] = kub["v5"]["sobol"]
    assert C6.tochki_v6(kub, 0, 300, gotovy=frozenset())[0] == C5.tochki_v5(kub["v5"], 0, 300)[0]   # V6-12b: GOTOVY уже не пуст


def test_gotovy_vhodyat_v_sobol_po_listam():
    kub = KUBY["S1"]
    imena = lambda g: {n for n, _, _ in C6.osi(kub, g)}                       # noqa: E731
    assert not imena(frozenset()) & set(C6.novye(kub))
    assert "tf" in imena({"V6-12"}) and "rel_signal" not in imena({"V6-12"})
    assert "pol_puti" in dict((n, g) for n, g, _ in C6.osi(kub, {"V6-12"}))["tejk"]
    assert imena(None) >= set(C6.novye(kub))
    assert not any(set(c) & set(C6.novye(kub)) for _, c in C6.tochki_v6(kub, 0, 200, gotovy=frozenset())[0])


def test_p_vykl_09_filtr_i_chislo_vklyuchennykh():
    kub = KUBY["S1"]
    pts = [c for _, c in C6.tochki_v6(kub, 0, 4096, gotovy=None)[0]]
    filtry = [n for n, v in C6.novye(kub).items() if v.get("p_vykl") == C6.P_FILTR]
    dolya = sum(f not in c for c in pts for f in filtry) / (len(pts) * len(filtry))
    assert abs(dolya - 0.9) < 0.02
    vkl = sum(sum(f in c for f in filtry) for c in pts) / len(pts)
    assert abs(vkl - 0.1 * len(filtry)) < 0.25                               # 1,4 вместо 8–9 при равномерной выборке
    assert not any(c.get("ma_n") and not (c.get("sektor_trend") or c.get("shirina")) for c in pts)


def test_deti_shodyatsya_v_kanon_pri_vyklyuchennom_roditele():
    kub = KUBY["S1"]
    baza = {n: g[0] for n, g in C5.osi(kub["v5"])}
    a = C6.canon_v6("S1", {**baza, **vykl(kub), "relf_k": 0.5, "relf_okno": 20}, kub)
    assert a == C6.canon_v6("S1", {**baza, **vykl(kub)}, kub) and "relf_k" not in a
    b = C6.canon_v6("S1", {**baza, **vykl(kub), "relf_napr": "protiv", "relf_k": 0.5}, kub)
    assert b["relf_k"] == 0.5 and "relf_okno" not in b and b["relf_napr"] == "protiv"
    assert "lestnica" not in C6.canon_v6("S1", {**baza, **vykl(kub), "lestnica": 0.5, "k_e": 0}, kub)
    assert C6.canon_v6("S1", {**baza, **vykl(kub), "lestnica": 0.5, "k_e": 0.5}, kub)["lestnica"] == 0.5


def test_dlit_trenda_svorachivaetsya_bez_podtverzhdeniya():            # V6-2 (#647)
    kub = KUBY["S3"]
    baza = {n: g[0] for n, g in C5.osi(kub["v5"])}
    assert "dlit" not in C6.canon_v6("S3", {**baza, **vykl(kub), "dlit": 2}, kub)
    q = {**baza, **vykl(kub), "dlit": 2, "vhod_tip": "podtverzhdenie"}
    assert C6.canon_v6("S3", q, kub)["dlit"] == 2


@pytest.mark.parametrize("sid", SIDS)
def test_rasshireno_soderzhit_v5_i_obem_zamorozhen_raven_v5(sid):
    kub = KUBY[sid]
    base = dict(C5.osi(kub["v5"]))
    for n, g, _ in C6.osi(kub, None):
        if n in base:
            assert g[:len(base[n])] == base[n]
    zam = list(C6.novye(kub)) + list(C6.rasshireno(kub))
    assert C6.obem(kub, zam) == C5.obem(kub["v5"])


@pytest.mark.parametrize("sid", SIDS)
def test_obem_sovpadaet_s_tablicej_inventarizacii(sid):
    kub = KUBY[sid]
    v = C6.obem(kub, ["tf"])
    assert v == pytest.approx(TABLICA[sid], rel=0.05)
    assert abs(v / TABLICA[sid] - 1) <= OTKL.get(sid, 0.005) + 0.005
    assert C6.obem(kub) == pytest.approx(5 * v)


def test_period_s4_svorachivaetsya_pri_norm_s252():                          # V6-5 (#648)
    kub = KUBY["S4"]
    baza = {n: g[0] for n, g in C5.osi(kub["v5"])}
    a = C6.canon_v6("S4", {**baza, **vykl(kub), "norm": "s252", "period": 80}, kub)
    b = C6.canon_v6("S4", {**baza, **vykl(kub), "norm": "s252", "period": 5}, kub)
    assert a == b and a["norm"] == "s252"
    assert C6.canon_v6("S4", {**baza, **vykl(kub), "period": 80}, kub)["period"] == 80   # при norm = okno period действует


def test_piv_buf_tolko_pri_pivotnom_stope():                      # S5, И11; замечание Н4 #677: piv:R у тейка буфер не задаёт
    kub = KUBY["S5"]
    baza = {n: g[0] for n, g in C5.osi(kub["v5"])}
    q = {**baza, **vykl(kub), "piv_buf": 0.1}
    assert "piv_buf" not in C6.canon_v6("S5", q, kub)
    for t in ("piv:R", "piv:half"):
        assert "piv_buf" not in C6.canon_v6("S5", {**q, "tejk": t}, kub)
    for s in ("piv:next", "piv:half"):
        assert C6.canon_v6("S5", {**q, "stop_nach": s}, kub)["piv_buf"] == 0.1


def test_zamorozka_zamorazhivaet_detej():
    import importlib.util
    spec = importlib.util.spec_from_file_location("obem_v6", C6.KUB6 / "obem_v6.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    kub = KUBY["S1"]
    assert {"relf_k", "relf_okno", "relf_bench"} <= m.zamorozhennye(kub, ["relf_napr"])
    assert "ma_n" not in m.zamorozhennye(kub, ["shirina"])                   # второй родитель (sektor_trend) не заморожен
    for sid, _, v5, v6, bez, zam in m.tablica(list(C6.novye(kub)) + ["tejk"]):
        assert v5 <= zam <= v6 and bez == pytest.approx(v6 / 5)


def test_proverit_otvergaet_plokhoe():
    kub = copy.deepcopy(KUBY["S1"])
    plokhie = [
        ("S1", lambda k: k["kontekst"]["dlit"].update(roditel=["rel_signal"])),            # и p_vykl, и roditel
        ("S1", lambda k: k["vyhod"]["tejk"].update(v6_dobavleno=["rodnoy"])),              # значение v5 уже есть
        ("S3", lambda k: k["vyhod"]["tejk"].update(v6_dobavleno=["piv:R"])),               # у тренда нет пивотов
        ("S1", lambda k: k["kontekst"]["tf"].update(setka=["den", "den"])),
        ("S1", lambda k: k["kontekst"]["tf"].update(vykl="10m")),                          # «выкл.» не первое значение
        ("S1", lambda k: k["kontekst"]["tf"].pop("smysl")),                                # нет смысла — нет оси
        ("S1", lambda k: k["kontekst"]["tf"].update(filtr=True)),                          # filtr без p_vykl = 0,9
        ("S1", lambda k: k["baza"].update(kub_sha="0" * 64)),                              # куб v5 менялся после коммита v6
    ]
    for sid, mut in plokhie:
        kub = copy.deepcopy(KUBY[sid])
        mut(kub)
        with pytest.raises(ValueError):
            C6.proverit(kub)
