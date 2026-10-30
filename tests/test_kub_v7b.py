"""V7-1b: кубы v7 S8–S12 и канон (Ж-153 пп. 1–4, Ж-156 п. 3; план §1.1). S7 не входит. Данных не читает, Соболь не запускает."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from fastbt import canon_v6 as c6, canon_v7 as c7, canon_v7b as b

IMP, SECH = ("S10", "S11", "S12"), ("S8", "S9")
OSI_S3 = {"stop_nach": 26, "tejk": 35, "trail": 19, "chast": 7, "razmer_tip": 2, "m_sily": 3, "rang_tip": 3, "sigma_ist": 4,
          "bezubytok": 8, "stop_v_plyus": 4, "pir": 3}


@pytest.fixture(scope="module")
def kuby():
    return {s: b.load_kub(s) for s in IMP + SECH}


def _tochki(kub, n, seed, gotovy=None):
    ox, rng = b.osi(kub, gotovy or b.GOTOVY_B), np.random.default_rng(seed)
    return [c7.tochka(ox, rng.random(len(ox))) for _ in range(n)]


def test_sostav_bez_s7():
    assert b.SIDS_B == ("S8", "S9", "S10", "S11", "S12")
    assert not (c7.KUB7 / "kub_S7.yaml").exists()


@pytest.mark.parametrize("sid", IMP)
def test_impuls_osi_signal_i_vyhod_iz_s3(sid, kuby):
    k = kuby[sid]
    assert k["signal"]["tau"]["setka"] == [2, 3, 5] and k["signal"]["tau"]["spravka"] == 1       # τ = 1 — справочно, вне N
    assert k["signal"]["vhod"]["setka"] == ["d1", "T"]
    assert k["signal"]["T"]["setka"] == ["17:00", "18:00", "18:40"] and k["signal"]["T"]["roditel"] == {"vhod": "T"}
    ox = {n: g for n, g, _ in b.osi(k, c6.GOTOVY)}
    assert {n: len(ox[n]) for n in OSI_S3} == OSI_S3
    s3 = {n: g for n, g, _ in c6.osi(c6.load_kub("S3"), c6.GOTOVY)}
    assert all(ox[n] == s3[n] for n in OSI_S3)                        # значения те же, что в кубе S3 v6
    assert set(ox) == {"tau", "vhod", "T"} | set(OSI_S3)              # k_okup закрыт до V7-3 (gotovy)
    assert set(dict((n, g) for n, g, _ in b.osi(k, b.GOTOVY_B))) == set(ox) | {"k_okup"}
    assert not {"tau_v5", "k_e", "t_vyhod", "time_decay", "ed_lim"} & set(ox)          # не входят (план §1.1)


def test_n1_i_seed(kuby):
    for sid, n in {"S8": 50_000, "S9": 50_000, "S10": 200_000, "S11": 200_000, "S12": 200_000}.items():
        k = kuby[sid]
        assert k["sobol"]["N"] == n, sid
        assert (k["sobol"]["seed"], k["sobol"]["seed2"], k["sobol"]["morris_seed"]) == (20261105, 20261106, 20261104)
        assert k["v"] == 5 and k["setka"] == "v7"
        assert k["osi7"]["k_okup"]["setka"] == ["net", 3]
        b.proverit(k)


def test_T_svorachivaetsya_pri_d1_signalov_chetyre_na_tau(kuby):
    k, sig = kuby["S10"], set()
    base = next(p for p in _tochki(k, 200, 1) if b.canon_v7b("S10", p, k))
    for tau, vhod, T in itertools.product((2, 3, 5), ("d1", "T"), ("17:00", "18:00", "18:40")):
        c = b.canon_v7b("S10", dict(base, tau=tau, vhod=vhod, T=T), k)
        assert c is not None and ("T" in c) == (vhod == "T")
        sig.add((c["tau"], c["vhod"], c.get("T")))
    assert len(sig) == 3 * (1 + 3)                                     # плана §1.1: 12 канонических сочетаний сигнала


def test_pravila_trenda_s3_vyrezayut_dolyu(kuby):
    k = kuby["S10"]
    pts = _tochki(k, 3000, 2)
    share = sum(b.canon_v7b("S10", p, k) is None for p in pts) / len(pts)
    assert 0.35 < share < 0.55                                         # ≈ 44 % (план §1.2: канон S3)
    for p in pts[:300]:                                                # точка = canon_v6 S3 на общих осях
        core = {n: v for n, v in p.items() if n in OSI_S3 and n != "pir"} | {"pir": p["pir"]}
        a = c6.canon_v6("S3", core, k["s3"])
        assert (b.canon_v7b("S10", p, k) is None) == (a is None)


def test_mutatsiya_pravila_s3_snyaty_lovitsya(kuby, monkeypatch):
    k = kuby["S10"]
    monkeypatch.setattr(c6, "canon_v6", lambda sid, pt, kub=None: dict(pt))
    pts = _tochki(k, 500, 2)
    assert all(b.canon_v7b("S10", p, k) is not None for p in pts)      # без правил ничего не выброшено → тест доли выше упал бы


def test_k_okup_vykl_ubiraetsya(kuby):
    k = kuby["S11"]
    p = next(p for p in _tochki(k, 200, 3) if b.canon_v7b("S11", dict(p, k_okup="net"), k))
    assert "k_okup" not in b.canon_v7b("S11", dict(p, k_okup="net"), k)
    assert b.canon_v7b("S11", dict(p, k_okup=3), k)["k_okup"] == 3


@pytest.mark.parametrize("sid", SECH)
def test_s8_s9_reshetka_108(sid, kuby):
    k = kuby[sid]
    assert [n for n, _, _ in b.osi(k, b.GOTOVY_B)] == ["porog", "bufer", "cap", "rebal", "k_okup"]
    assert b.obem(k) == 3 * 3 * 3 * 2 * 2 == 108
    assert k["signal"]["porog"]["setka"] == [0.3, 0.5, 0.7] and k["signal"]["rebal"]["setka"] == ["den", "nedelya"]
    ox = b.osi(k, b.GOTOVY_B)
    uniq = {tuple(sorted(b.canon_v7b(sid, dict(zip((n for n, _, _ in ox), v)), k).items()))
            for v in itertools.product(*[g for _, g, _ in ox])}
    assert len(uniq) == 108                                            # N₁ = 50 000 индексов Соболя, различных точек 108


def test_khesh_s3_v6_i_tablica(kuby):
    for sid in IMP:
        assert kuby[sid]["baza"] == {"setka": "v6", "kub_sha": c7.sha_v6("S3"), "iz": "S3"}
    assert not c7.sha_nesovpadenia()
    names = {ln.split()[1] for ln in (c7.KUB7 / c7.SUMS).read_text(encoding="utf-8").splitlines() if ln.strip()}
    assert names == {f"kub_S{i}.yaml" for i in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)}
    with pytest.raises(ValueError, match="kub_sha"):
        b.proverit(dict(kuby["S10"], baza={"setka": "v6", "kub_sha": "0" * 64, "iz": "S3"}))
