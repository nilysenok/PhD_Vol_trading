"""V7-1a: кубы v7 S1–S6 и канон v7 (Ж-153, Ж-155 п. 2, Ж-156 п. 3, Ж-157 п. 2). Данных не читает, Соболь не запускает (точки — numpy rng)."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from fastbt import canon_v5 as c5, canon_v6 as c6, canon_v7 as c7, ledger_v2 as lm

SIDS = c7.SIDS
KONTR = ("S1", "S2", "S5", "S6")
N1 = {"S1": 100_000, "S2": 100_000, "S3": 100_000, "S4": 100_000, "S5": 100_000, "S6": 50_000}


@pytest.fixture(scope="module")
def kuby():
    return {s: c7.load_kub(s) for s in SIDS}


def _tochki(ox, n, seed):
    rng = np.random.default_rng(seed)
    return [c7.tochka(ox, rng.random(len(ox))) for _ in range(n)]


def _ozhid(g):
    i = len(g) // 2
    return list(g[max(0, i - 1): i + 2])


def test_suzhenie_centr_plyus_minus_odin():
    assert c7.suzhenie([1, 2, 3, 4]) == [2, 3, 4]          # центр setka[2] = 3
    assert c7.suzhenie(["a", "b"]) == ["a", "b"]           # две точки — остаются обе
    assert c7.suzhenie([1, 2, 3]) == [1, 2, 3]
    assert c7.suzhenie(list(range(19))) == [8, 9, 10]
    assert c7.suzhenie([7]) == [7]


@pytest.mark.parametrize("sid", SIDS)
def test_signal_osi_centr_pm1_ot_setki_v6(sid, kuby):
    k, k6 = kuby[sid], c6.load_kub(sid)
    base = dict((n, g) for n, g, _ in c6.osi(k6, c6.GOTOVY))
    assert tuple(k["signal"]) == c5.SIGNAL[sid]
    for n, spec in k["signal"].items():
        assert spec["setka"] == _ozhid(base[n]), n
        assert spec["centr"] == base[n][len(base[n]) // 2], n
    got = dict((n, g) for n, g, _ in c7.osi(k, c6.GOTOVY))
    for n in c5.SIGNAL[sid]:
        assert got[n] == _ozhid(base[n])


def test_s1_primer_iz_plana():
    s = c7.load_kub("S1")["signal"]
    assert s["z_exit"]["setka"] == [0.3, 0.5, 0.7]
    assert s["ma_window"]["setka"] == [25, 30, 35]


@pytest.mark.parametrize("sid", SIDS)
def test_obshchie_osi_v6_kak_est(sid, kuby):
    sig = set(c5.SIGNAL[sid])
    a = [(n, g) for n, g, _ in c6.osi(c6.load_kub(sid), c6.GOTOVY) if n not in sig]
    b = [(n, g) for n, g, _ in c7.osi(kuby[sid], c6.GOTOVY) if n not in sig and n not in c7.NOVYE7]
    assert a == b


@pytest.mark.parametrize("sid", SIDS)
def test_novye_osi_k_okup_i_otnositelno(sid, kuby):
    n7 = c7.novye(kuby[sid])
    assert n7["k_okup"]["setka"] == ["net", 3] and n7["k_okup"]["vykl"] == "net"        # Ж-156 п. 3: значение 2 — позже добором
    if sid in ("S1", "S2", "S3", "S4"):
        assert n7["otnositelno"]["setka"] == ["net", "sektor", "imoex"]
    else:
        assert "otnositelno" not in n7
    assert all(v["p_vykl"] == 0.5 for v in n7.values())
    assert (kuby[sid].get("min_uderzhanie") == {"dney": 2, "tau1": "spravka"}) == (sid in KONTR)


def test_n1_po_semejstvam_i_seed(kuby):
    for sid in SIDS:
        s = kuby[sid]["sobol"]
        assert s["N"] == N1[sid], sid
        assert (s["seed"], s["seed2"], s["morris_seed"]) == (20261105, 20261106, 20261104)
        assert kuby[sid]["v"] == 5 and kuby[sid]["setka"] == "v7"
        assert kuby[sid]["baza"] == {"setka": "v6", "kub_sha": c7.sha_v6(sid)}
        c7.proverit(kuby[sid])


@pytest.mark.parametrize("sid", SIDS)
def test_vykl_v7_ravno_v6_klyuch(sid, kuby):
    """v6 ⊂ v7: точка v6 с выкл. осями v7 даёт тот же канон (и тот же klyuch_n), годные точки v6 не выбрасываются."""
    k6, k7 = c6.load_kub(sid), kuby[sid]
    n_god = 0
    for pt in _tochki(c6.osi(k6, c6.GOTOVY), 400, 7):
        a = c6.canon_v6(sid, pt, k6)
        b = c7.canon_v7(sid, dict(pt, k_okup="net", **({"otnositelno": "net"} if "otnositelno" in c7.novye(k7) else {})), k7)
        assert a == b
        if a is not None:
            n_god += 1
            kl = lambda c: lm.klyuch_n(sid, lm.klyuch(sid, c, "train4", 0.0004, "t0d"))      # noqa: E731
            assert kl(a) == kl(b)
    assert n_god > 50


def test_k_okup_i_otnositelno_ostayutsya_v_kanone(kuby):
    k = kuby["S1"]
    pt = next(p for p in _tochki(c6.osi(c6.load_kub("S1"), c6.GOTOVY), 50, 1) if c6.canon_v6("S1", p, k["v6"]))
    base = c7.canon_v7("S1", dict(pt, k_okup="net", otnositelno="net"), k)
    assert "k_okup" not in base and "otnositelno" not in base
    assert c7.canon_v7("S1", dict(pt, k_okup=3, otnositelno="net"), k)["k_okup"] == 3
    assert c7.canon_v7("S1", dict(pt, k_okup="net", otnositelno="imoex"), k)["otnositelno"] == "imoex"


def test_mutatsiya_k_okup_vykl_ne_ubiraetsya_lovitsya(kuby):
    """Мутация: k_okup = выкл. остаётся в каноне — ключ точки v6 ≠ ключу v7 (ловит test_vykl_v7_ravno_v6_klyuch; здесь — прямо)."""
    k = kuby["S3"]
    for pt in _tochki(c6.osi(c6.load_kub("S3"), c6.GOTOVY), 100, 3):
        c = c7.canon_v7("S3", dict(pt, k_okup="net", otnositelno="net"), k)
        assert c is None or ("k_okup" not in c and "otnositelno" not in c)


def test_sha256sums_sovpadayut():
    assert c7.sha_nesovpadenia() == []


def test_mutatsiya_yaml_izmenyon_bez_khesha_lovitsya(tmp_path):
    for f in list(c7.KUB7.glob("kub_S*.yaml")) + [c7.KUB7 / "SHA256SUMS"]:
        shutil.copy(f, tmp_path / f.name)
    assert c7.sha_nesovpadenia(tmp_path) == []
    p = tmp_path / "kub_S2.yaml"
    p.write_text(p.read_text(encoding="utf-8") + "# правка\n", encoding="utf-8")
    assert c7.sha_nesovpadenia(tmp_path) == ["kub_S2.yaml"]
    (tmp_path / "SHA256SUMS").write_text("", encoding="utf-8")
    assert "kub_S1.yaml" in c7.sha_nesovpadenia(tmp_path)           # файл без строки в таблице — тоже расхождение


def test_mutatsiya_kub_sha_v6_ne_sovpadaet_lovitsya(kuby):
    k = dict(kuby["S1"], baza={"setka": "v6", "kub_sha": "0" * 64})
    with pytest.raises(ValueError, match="kub_sha"):
        c7.proverit(k)
    s = {n: dict(v) for n, v in kuby["S1"]["signal"].items()}
    s["ma_window"]["setka"] = [25, 30, 35, 40]
    with pytest.raises(ValueError, match="ma_window"):
        c7.proverit(dict(kuby["S1"], signal=s))
