"""Тестировщик #721: ловят мутации плана Морриса v6, не пойманные tests/test_morris_v6_plan.py.
Данных и сети нет: синтетический куб, подставной оценщик, шесть реальных кубов v6 — только канон.

Мутации: (а) None-шаг берёт уровень другой оси; (б) пропущенная ось — активный шаг; (в) seed не влияет на починку;
(г) счётчик теряет none_kanon; (д) мёртвая ось заморожена / помечена «мало»; (е) быстрый слой выключен по умолчанию или на непокрытых;
(ж) эффект считается не по шагу своей оси."""
from __future__ import annotations

import argparse
import contextlib
import json

import numpy as np
import pytest
import yaml

from fastbt import canon_v6, fast_etap1, morris_v6 as mv, morris_v6_zamorozka as mz, progon_morris_v6 as pm, progon_morris_v6_status as st, progon_sobol as ps

L = [2, 2, 3, 2, 2, 4, 2]
W = np.array([1.0, 2.0, 3.0, 5.0, 0.0, 7.0, 11.0])      # веса линейной модели; w[4] = 0 (дочерняя ось сворачивается)
DATY = np.arange(np.datetime64("2016-01-01"), np.datetime64("2019-01-01"))
SHUM = np.random.default_rng(7).normal(0, 0.01, DATY.size)
REZ = ps.Rezhim("v6", "train4", "seed", False, "t0d")


def klyuch(idx):
    if (idx[0] == 1 and idx[1] == 1) or (idx[2] == 2 and idx[5] == 3) or idx[5] == 1:
        return None
    return (*idx[:4], idx[4] if idx[3] else 0, idx[5], 0)


def plan(seed, r=16):
    return mv.traektorii_kanon(L, r, klyuch, seed, 4)


def _raznitsa_osi(I, por):
    """Свойство: соседние точки различаются не более чем осью шага (por[m, t-1]), и только ею."""
    for m, t in np.ndindex(por.shape):
        d = np.flatnonzero(I[m, t + 1] != I[m, t])
        assert list(d) in ([], [por[m, t]]), f"траектория {m}, шаг {t}: изменены оси {list(d)}, должна быть {por[m, t]}"


@pytest.mark.parametrize("seed", (1, 7, 20261101))
def test_sosednie_tochki_otlichayutsya_odnoj_os_yu(seed):
    _raznitsa_osi(*plan(seed))


def _keys(I):
    uniq, nom = {}, np.empty(I.shape[:2], dtype=np.int64)
    for m, t in np.ndindex(nom.shape):
        nom[m, t] = uniq.setdefault(klyuch(tuple(int(i) for i in I[m, t])), len(uniq))
    return nom, list(uniq)


def test_effekt_na_svoej_osi_i_propusk_ne_aktiven():
    """Y = Σ w_a·ключ_a: эффект оси a = w_a·Δключ_a на шаге этой оси; ось, не двинувшая ключ, и мёртвая ось 6 — NaN (мутации б, ж)."""
    I, por = plan(5)
    nom, kl = _keys(I)
    Yp = np.array([sum(w * k for w, k in zip(W, key)) for key in kl])
    E = mv.effekty(Yp[nom], nom, por)
    assert np.isnan(E[:, 6]).all(), "мёртвая ось получила эффект (пропуск засчитан активным шагом)"
    akt = 0
    for m, t in np.ndindex(por.shape):
        a, k0, k1 = por[m, t], kl[nom[m, t]], kl[nom[m, t + 1]]
        if k0 == k1:
            assert np.isnan(E[m, a]), "шаг без смены ключа посчитан активным"
        else:
            akt += 1
            assert E[m, a] == pytest.approx(W[a] * (k1[a] - k0[a])), f"эффект {m},{a} не по шагу своей оси"
    assert akt > 0


def test_v_seed_zavisit_pochinka(monkeypatch):
    """Мутация в: rng починки без seed. База траекторий фиксирована, меняется только seed → планы обязаны разойтись."""
    orig = mv.traektorii
    monkeypatch.setattr(mv, "traektorii", lambda L_, r, seed, mnozh: orig(L_, r, 5, mnozh))
    a, a2, b = (mv.traektorii_kanon(L, 16, klyuch, s, 4)[0] for s in (1, 1, 2))
    assert (a == a2).all() and not (a == b).all()


def test_raznye_seed_raznye_plany_ne_odin_plan():
    assert not (plan(1)[0] == plan(2)[0]).all()


# ── реальные кубы: только канон ────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sid", canon_v6.SIDS)
def test_real_kub_sosednie_tochki_odna_os_i_effekt_na_nej(sid, monkeypatch):
    kub, ox = pm.setka(sid)
    zahv, orig = {}, mv.traektorii_kanon
    monkeypatch.setattr(mv, "traektorii_kanon", lambda *a, **k: zahv.setdefault("I", orig(*a, **k)))
    nom, por, pts = pm.plan(sid, kub, ox, 3, 2, int(kub["sobol"]["morris_seed"]))
    I, por2 = zahv["I"]
    assert (por == por2).all() and all(pts[n] is not None for n in nom.ravel()), "в плане None-точка"
    _raznitsa_osi(I, por)
    Y = nom.astype(float) ** 2 + nom                 # любая функция точки
    E = mv.effekty(Y, nom, por)
    for m, t in np.ndindex(por.shape):
        a = por[m, t]
        want = Y[m, t + 1] - Y[m, t] if nom[m, t + 1] != nom[m, t] else np.nan
        assert (np.isnan(E[m, a]) and np.isnan(want)) or E[m, a] == want
    assert np.isfinite(E).sum() > 0


# ── счётчик статусов и быстрый слой ────────────────────────────────────────────────────────────────────────────────

class Fake:
    fdr = "t0d"

    def evaluate_v2(self, pt, period):
        if pt.get("rel_signal") == "sektor":
            return {"status": "ne_schitana"}, None, None
        return {"status": "ok"}, SHUM, None


@pytest.fixture
def sled(monkeypatch, tmp_path):
    f = tmp_path / "fast.log"
    monkeypatch.setattr(ps, "fabrika", lambda sid: ps.Sreda(Fake(), "kod", "vhody", DATY, ["T"]))
    monkeypatch.setattr(ps, "fast_ok", lambda pt: pt.get("pokryto", True))

    @contextlib.contextmanager
    def patched():
        f.open("a").write("x\n")
        yield

    monkeypatch.setattr(fast_etap1, "patched", patched)
    return f


def progon(tmp_path, **kw):
    return pm.strategiya("S1", argparse.Namespace(procs=2, prodolzhit=False, **kw), tmp_path, REZ, 12, mnozh=2)


def test_g_schetchik_sohranyaet_none_kanon(sled, tmp_path, monkeypatch):
    orig = pm.plan

    def s_none(*a):
        nom, por, pts = orig(*a)
        pts[0] = None
        return nom, por, pts

    monkeypatch.setattr(pm, "plan", s_none)
    i = progon(tmp_path)
    assert i["statusy"]["none_kanon"] == 1, i["statusy"]
    assert sum(i["statusy"].values()) == i["tochek"]
    y = yaml.safe_load((tmp_path / "S1" / "zamorozka.yaml").read_text("utf-8"))
    assert y["statusy"]["none_kanon"] == 1

def test_g_kusok_schitaet_none_kanon(sled, monkeypatch):
    monkeypatch.setattr(ps, "_W", dict(rz=REZ, pts={"S1": [None, {}, {"rel_signal": "sektor"}, None]}, fast=False))
    _, _, ys, _, s = pm._kusok(("S1", 0, 4))
    assert s == {"ok": 1, "ne_schitana": 1, "none_kanon": 2, "oshibka": 0}
    assert np.isnan(ys[0]).all() and np.isfinite(ys[1]).all()


def test_e_bystryj_sloj_ne_na_nepokrytyh(sled, monkeypatch):
    sr = ps.Sreda(Fake(), "k", "v", DATY, ["T"])
    monkeypatch.setattr(ps, "fast_ok", lambda pt: False)
    st.status_okna(sr, {}, "train4", True)
    assert not sled.exists(), "быстрый слой включён на непокрытой точке"
    monkeypatch.setattr(ps, "fast_ok", lambda pt: True)
    st.status_okna(sr, {}, "train4", True)
    assert sled.exists()


def test_e_main_po_umolchaniyu_fast_true_i_no_fast(monkeypatch, tmp_path):
    vid = []
    monkeypatch.setattr(pm, "strategiya", lambda sid, a, out, rz, r: vid.append(a.fast) or dict(
        tochek=0, statusy={}, oshibok=0, zamorozheno=[], osi_akt={}, prichiny={}))
    base = ["--kub", "v6", "--period", "train4", "--out", str(tmp_path), "--strategii", "S1"]
    assert pm.main(base) == 0 and pm.main(base + ["--no-fast"]) == 0
    assert vid == [True, False]


# ── мёртвые оси ────────────────────────────────────────────────────────────────────────────────────────────────────

def _stat(rows):
    """rows — по окнам: [(n_akt, U95) по осям]."""
    return {w: {"n_akt": np.array([r[0] for r in rows[i]]), "U95": np.array([r[1] for r in rows[i]]),
                "mu_star": np.zeros(len(rows[i])), "sigma": np.zeros(len(rows[i]))} for i, w in enumerate(mz.OKNA)}


OSI = [{"os": n, "gruppa": "g", "novaya": True} for n in ("mertv", "chast", "malo", "nul")]


def test_d_mertvaya_os_ne_zamorozhena_ne_znachima_i_chast_ne_mertvaya(tmp_path):
    nan = float("nan")
    stat = _stat([[(0, nan), (0, 0.0), (5, 0.0), (40, 0.0)], [(0, nan), (30, 0.0), (5, 0.0), (40, 0.0)],
                  [(0, nan), (30, 0.0), (5, 0.0), (40, 0.0)]])
    res = mz.reshenie(OSI, stat)
    assert res["mertv"][0] is False and mz.MERTVAYA in res["mertv"][1]
    assert res["chast"][0] is False and mz.MALO in res["chast"][1] and mz.MERTVAYA not in res["chast"][1]
    assert res["malo"][0] is False and res["nul"][0] is True
    pom = mz.pometki(OSI, stat)
    assert pom["mertv"]["pometka"] == mz.MERTVAYA and pom["chast"]["pometka"] == mz.MALO
    p = tmp_path / "z.yaml"
    zam = mz.zapisat_yaml(p, "2026-10-30", {}, res, {"ok": 1}, pom)
    y = yaml.safe_load(p.read_text("utf-8"))
    assert zam == ["nul"] == y["osi"] and y["mertvye_ne_ocenyeny"] == ["mertv"] and y["malo_shagov"] == ["chast", "malo"]
    assert "mertv" not in y["prichiny"] and y["statusy"] == {"ok": 1}
    assert json.dumps(y["osi_aktivnye"]["mertv"]["n_akt"]) == "[0, 0, 0]"
