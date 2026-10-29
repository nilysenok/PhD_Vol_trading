"""Л8: кубы v5 (fastbt/designs/v5/kub_S{1..6}.yaml) и канон fastbt/canon_v5.py. Синтетика и yaml, данных нет, сети нет."""
import contextlib
import copy
import importlib.util
import io
import itertools
import math
from pathlib import Path

import pytest

from fastbt import canon_v5 as C
from fastbt.ma_tipy import TIPY
from fastbt.signals import v5_s12, v5_s34, v5_s56

ROOT = Path(__file__).resolve().parent.parent
KUBY = {s: C.load_kub(s) for s in C.SIDS}

# obem_v5.py (архитектор, §2.1) → ось куба; расхождения — с причиной (README «Сверка»: строка на каждое).
SIG_MAP = {"S1": dict(ma_tip="ma_tip", okno="ma_window", z_vyhod="z_exit"), "S2": dict(ma_tip="ma_tip", okno="bb_window",
           shirina="bb_std", podtv="podtv"), "S3": dict(okno="okno", podtv="podtv"), "S4": dict(period="period", mnozh="mnozh"),
           "S5": dict(tip="pivot_type", uroven="uroven", period="pivot_period", atr_filtr="atr_filtr", fs="filter_set"),
           "S6": dict(okno="vwap_window", porog="dev_mult", dir_bias="dir_bias", fs="filter_set")}
OBSH_MAP = dict(adx="adx_th", atr_lo="atr_lo", obyom="vol_th", rsi="rsi_th", mom="mom_n", dow="dow", szhatie="squeeze_pct",
                hurst=None, naklon_max=None, naklon_min=None, k_e="k_e", T="T", reentry="reentry", stop="stop_nach", tejk="tejk",
                trail="trail", bezub="bezubytok", stop_plyus="stop_v_plyus", tau="tau", chast="chast", pir="pir", smena="smena",
                razmer_tip="razmer_tip", m_sily="m_sily", rang_tip="rang_tip")
RASH = {  # (тип, ось obem): (уровней в кубе, причина)
    ("kontr", "hurst"): (0, "v5_filtr не считает"), ("kontr", "naklon_max"): (0, "v5_filtr не считает"),
    ("trend", "naklon_min"): (0, "v5_filtr не считает"),
    ("trend", "rsi"): (6, "номинал Р 6, канон 2 (test_rsi_trenda_svyortka)"),
    ("kontr", "stop"): (26, "pct 9 (zh126 + 0,003) + atr 11 (chand свёрнут в atr, Ж-129 (6)) + σ 6"),
    ("kontr", "bezub"): (8, "σ снят: безубыток от ATR входа, Ж-129 (6)"),
    ("trend", "stop"): (6, "стоп-заявка: только σ (exec_e.py:112)"), ("trend", "tejk"): (17, "стоп-заявка: без pct/atr/rodnoy"),
    ("trend", "trail"): (17, "линия — сама стоп-заявка; step — exec_e_v6.py:27"),
    ("trend", "bezub"): (0, "exec_e_v6.py:27"), ("trend", "stop_plyus"): (0, "exec_e_v6.py:27"), ("trend", "pir"): (0, "exec_e_v6.py:27"),
    ("trend", "T"): (0, "exec_e.py:104"), ("trend", "reentry"): (0, "exec_e.py:104"), ("trend", "smena"): (0, "выход — стоп-заявка"),
}


def obem_v5():
    sp = importlib.util.spec_from_file_location("obem_v5", ROOT / "fastbt/designs/v5/obem_v5.py")
    m = importlib.util.module_from_spec(sp)
    with contextlib.redirect_stdout(io.StringIO()):
        sp.loader.exec_module(m)
    return m


def urovni(kub):
    return {n: len(g) for n, g in C.osi(kub)}


@pytest.mark.parametrize("sid", C.SIDS)
def test_kub_format_i_obyavleno_do_scheta(sid):
    k = KUBY[sid]
    C.proverit(k)
    assert k["sobol"]["N1"] == 32768 and k["sobol"]["seed"] != k["sobol"]["seed2"] and k["sobol"]["scramble"] is True
    assert k["periody"] == ["train4", "v2"] and k["period"] == "train4" and k["komissiya"] == 0.0004 and k["fdr"] == "t0d"
    assert k["zakrepleno"]["tf"] == "den" and set(k["faza3"]) == {"consec", "min_hold", "cooldown", "min_sily"}
    assert "uroven_razmera" in k["msp"] and "uroven_razmera" not in urovni(k)       # Ж-129 (5): уровень — слой портфеля


def test_signal_osi_ravny_semyam_v5():
    for sid, g in (("S3", v5_s34.S3_GRID_V5), ("S4", v5_s34.S4_GRID_V5), ("S5", v5_s56.G5), ("S6", v5_s56.G6)):
        assert {n: v["setka"] for n, v in KUBY[sid]["signal"].items()} == g, sid
    for sid in ("S1", "S2"):
        assert set(KUBY[sid]["signal"]["ma_tip"]["setka"]) == set(TIPY)
    assert KUBY["S2"]["signal"]["podtv"]["setka"] == list(v5_s12.PODTV)
    assert KUBY["S1"]["signal"]["dev_tip"]["setka"] == list(v5_s12.DEV_TIPY)


@pytest.mark.parametrize("sid", C.SIDS)
def test_urovni_ravny_obem_v5_krome_zapisannyh(sid):
    m, k, t = obem_v5(), KUBY[sid], C.tip(sid)
    u = urovni(k)
    trend = sid in m.TREND
    obsh = {**(m.FILTR_T if trend else m.FILTR_K), **(m.VHOD_T if trend else m.VHOD_K), **(m.VYHOD_T if trend else m.VYHOD_K)}
    for a, n in itertools.chain(((a, (SIG_MAP[sid].get(a), n)) for a, n in m.SIG[sid].items()),
                                ((a, (OBSH_MAP[a], n)) for a, n in obsh.items())):
        os_, n_obem = n
        if sid == "S1" and a == "porog":
            assert u["dev_tip"] * u["z_entry"] == 26 >= n_obem          # zh126 2 × 13 шире §1 (23): действует zh126
            continue
        want = RASH.get((t, a), (n_obem,))[0]
        assert u.get(os_, 0) == want, (sid, a, os_, u.get(os_), want)


def test_rsi_trenda_svyortka():
    for v in (20, 25, 30, 35, 40):
        assert C.canon_v5("S3", {"rsi_th": v})["rsi_th"] == C.RSI_TREND
    assert C.canon_v5("S3", {"rsi_th": 0})["rsi_th"] == 0
    assert C.canon_v5("S1", {"rsi_th": 35})["rsi_th"] == 35                 # у контртренда порог действует


def test_rang_tip_svyortka():
    base = {"m_sily": "1", "tejk": "rr:1.0", "bezubytok": "net", "stop_nach": "sigma:2.0"}
    kan = {C.canon_v5("S1", {**base, "rang_tip": r})["rang_tip"] for r in ("tikernyj", "obshchiy", "srednee")}
    assert kan == {"tikernyj"}
    for ext in ({"m_sily": "lin"}, {"tejk": "sila:0.5:1"}, {"bezubytok": "rang:2"}):
        assert C.canon_v5("S1", {**base, **ext, "rang_tip": "srednee"})["rang_tip"] == "srednee", ext


def test_vybros_i_chand():
    assert C.canon_v5("S1", {"stop_nach": "chand:2.5"})["stop_nach"] == "atr:2.5"           # Ж-129 (6)
    assert C.canon_v5("S1", {"stop_nach": "sigma:2.0", "bezubytok": "atr:1.0"}) is None       # ed_s одна
    assert C.canon_v5("S1", {"stop_nach": "pct:0.01", "bezubytok": "rang:1"}) is None
    assert C.canon_v5("S1", {"stop_nach": "atr:2.0", "bezubytok": "atr:1.0"}) is not None
    assert C.canon_v5("S1", {"stop_nach": "chand:2.0", "bezubytok": "atr:1.0"}) is not None


def test_kanon_idempotent_i_ne_trogaet_drugie_osi():
    pts, _ = C.tochki_v5(KUBY["S1"], 0, 300)
    for _, p in pts:
        assert C.canon_v5("S1", p) == p
    p0 = {"ma_window": 20, "tau": 5, "chast": "1/2:1", "m_sily": "1", "rang_tip": "obshchiy"}
    assert {k: v for k, v in C.canon_v5("S1", p0).items() if k != "rang_tip"} == {k: v for k, v in p0.items() if k != "rang_tip"}


def test_obem_kanon_ravno_pereboru_na_malom_kube():
    k = copy.deepcopy(KUBY["S3"])
    for sec in C.SEKCII:
        for n, v in k[sec].items():
            v["setka"] = v["setka"][:1] if n not in C.PRAVILA else v["setka"][:2] if n == "stop_nach" else v["setka"]
    k["vyhod"]["tejk"]["setka"] = ["net", "sila:0.5:1"]
    ox = C.osi(k)
    kan = {tuple(sorted((a, str(b)) for a, b in C.canon_v5("S3", dict(zip([n for n, _ in ox], c))).items()))
           for c in itertools.product(*[g for _, g in ox])}
    assert C.obem_kanon(k) == (len(kan), 0) and len(kan) < C.obem(k)


def test_obem_kanon_vybros_schitaetsya_pereborom():
    k = copy.deepcopy(KUBY["S1"])
    k["vyhod"]["stop_nach"]["setka"] = ["pct:0.01", "atr:1.0", "sigma:2.0"]
    k["vyhod"]["bezubytok"]["setka"] = ["net", "atr:1.0", "rang:1"]
    for n in ("tejk", "m_sily", "rang_tip"):
        k["vyhod"][n]["setka"] = k["vyhod"][n]["setka"][:1]
    k["filtr"]["rsi_th"]["setka"] = [0]
    ost = C.obem(k) // 9
    assert C.obem_kanon(k) == (6 * ost, 3 * ost)        # выброшены (pct, atr), (pct, rang), (σ, atr)


def test_dymovoj_200_tochek_sobol():
    for sid in C.SIDS:
        k = KUBY[sid]
        pts, sch = C.tochki_v5(k, 0, 200)
        assert len(pts) + sch["vybrosheno"] == 200 and [i for i, _ in pts] == sorted({i for i, _ in pts})
        g = dict(C.osi(k))
        assert all(str(v) in map(str, g[n]) for _, p in pts for n, v in p.items())
        pts2, _ = C.tochki_v5(k, 0, 200, seed="seed2")
        assert pts != pts2 and C.tochki_v5(k, 0, 200)[0] == pts                      # seed детерминирует
        assert C.tochki_v5(k, 100, 200)[0] == [x for x in pts if x[0] >= 100]          # продолжение = префикс


def test_balans_sobola_po_krupnejshej_osi():
    pts, sch = C.tochki_v5(KUBY["S3"], 0, 4096)
    assert sch["vybrosheno"] == 0
    cnt = {}
    for _, p in pts:
        cnt[p["tejk"]] = cnt.get(p["tejk"], 0) + 1
    assert len(cnt) == 17 and max(cnt.values()) - min(cnt.values()) <= 2         # 2¹² точек: каждая 1/4096 — ровно одна


@pytest.mark.parametrize("sid", C.SIDS)
def test_k3_tolko_osi_v4(sid):
    k3 = C.k3_kub(KUBY[sid])
    u, polnyj = urovni(k3), urovni(KUBY[sid])
    assert set(u) == set(polnyj) and math.prod(u.values()) < 20000
    for n in u:
        if n not in KUBY[sid]["k3"]:
            assert u[n] == 1, n


def test_mutacii_proverit_lovit():
    k = copy.deepcopy(KUBY["S3"])
    k["vyhod"]["stop_nach"]["setka"].append("pct:0.01")              # стоп в % у стоп-заявки — движок не умеет
    with pytest.raises(ValueError, match="stop_nach"):
        C.proverit(k)
    k = copy.deepcopy(KUBY["S1"])
    k["vyhod"]["trail"]["setka"].append("parab:0.02:0.1")            # параболический — только тренд
    with pytest.raises(ValueError, match="trail"):
        C.proverit(k)
    k = copy.deepcopy(KUBY["S1"])
    k["filtr"]["hurst_th"] = {"setka": [0, 0.4]}                      # ось, которую оценщик молча проигнорировал бы
    with pytest.raises(ValueError, match="hurst_th"):
        C.proverit(k)
    k = copy.deepcopy(KUBY["S2"])
    k["vyhod"]["sigma"] = {"setka": ["rv22", "rv5"]}                  # ось zh047 в кубе и в якорях
    with pytest.raises(ValueError, match="якорях"):
        C.proverit(k)
