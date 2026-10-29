"""План Морриса v6 (Ж-131 п. 2, Ж-137 п. 2): траектории только по каноническим точкам, счётчик статусов, мёртвые оси, быстрый слой.
Данных нет, сети нет: синтетический куб с запрещёнными сочетаниями и подставной оценщик.

Мутации: план снова допускает None-точки (mv.traektorii вместо traektorii_kanon); мёртвая ось помечена «заморожена»;
драйвер Морриса не включает быстрый слой (fast=False)."""
from __future__ import annotations

import argparse
import contextlib

import numpy as np
import pytest
import yaml

from fastbt import fast_etap1, morris_v6 as mv, morris_v6_zamorozka as mz, progon_morris_v6 as pm, progon_morris_v6_status as st, progon_sobol as ps

L = [2, 2, 3, 2, 2, 4, 2]          # ось 4 — дочерняя оси 3 (при 3 = 0 свёрнута), ось 6 — мёртвая (всегда свёрнута)
SEEDS = (1, 7, 20261101)


def klyuch(idx):
    """Запрещено: (0, 1) == (1, 1); ось 2 = 2 вместе с осью 5 = 3. Ось 4 свёрнута при оси 3 = 0, ось 6 — всегда."""
    if (idx[0] == 1 and idx[1] == 1) or (idx[2] == 2 and idx[5] == 3):
        return None
    return (*idx[:4], idx[4] if idx[3] else 0, idx[5], 0)


def _tochki(I):
    return [tuple(int(i) for i in p) for tr in I for p in tr]


def _aktivnye(I, por):
    """Число активных шагов по осям: ключ точки изменился на шаге, меняющем эту ось."""
    n = np.zeros(I.shape[2], int)
    for m in range(len(I)):
        for t in range(1, I.shape[1]):
            n[por[m, t - 1]] += klyuch(tuple(I[m, t])) != klyuch(tuple(I[m, t - 1]))
    return n


def _proverka_plana(plan, seed):
    I, por = plan(seed)
    assert all(klyuch(p) is not None for p in _tochki(I)), "в плане есть None-точка"
    n = _aktivnye(I, por)
    assert n[:6].min() >= 1, f"ось с допустимыми шагами без активных: {n}"
    assert n[6] == 0, "мёртвая ось получила активный шаг"


def kanon_plan(seed):
    return mv.traektorii_kanon(L, 12, klyuch, seed, 4)


@pytest.mark.parametrize("seed", SEEDS)
def test_vse_tochki_kanonicheskie_i_kazhdaya_os_aktivna(seed):
    _proverka_plana(kanon_plan, seed)


def test_mutatsiya_plan_s_none_tochkami_lovitsya():
    staryj = lambda seed: mv.traektorii(L, 12, seed, 4)               # noqa: E731 — прежний план без проверки канона
    with pytest.raises(AssertionError, match="None-точка"):
        for seed in SEEDS:
            _proverka_plana(staryj, seed)


def test_determinizm_po_seed_i_seed_menyaet_plan():
    a, b = kanon_plan(3), kanon_plan(3)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
    assert not (a[0] == kanon_plan(4)[0]).all()


def test_propusk_osi_ne_menyaet_tochku():
    """Ось 0 допустима только при 0: шагов на неё нет — точка повторяется (шаг не активен), None не появляется."""
    zapret = lambda idx: None if idx[0] == 1 else klyuch(idx)       # noqa: E731
    I, por = mv.traektorii_kanon(L, 12, zapret, 5, 4)
    assert all(zapret(p) is not None for p in _tochki(I)) and (I[:, :, 0] == 0).all()
    for m in range(len(I)):
        t = int(np.flatnonzero(por[m] == 0)[0]) + 1
        assert (I[m, t] == I[m, t - 1]).all()


def test_net_dopustimoy_startovoy_tochki_oshibka():
    with pytest.raises(ValueError, match="нет допустимой стартовой"):
        mv.traektorii_kanon(L, 3, lambda idx: None, 1, 2, popytok=20)


# ── статусы, мёртвые оси ────────────────────────────────────────────────────────────────────────────────────────────

DATY = np.arange(np.datetime64("2016-01-01"), np.datetime64("2019-01-01"))
SHUM = np.random.default_rng(7).normal(0, 0.01, DATY.size)
REZ = ps.Rezhim("v6", "train4", "seed", False, "t0d")


class Fake:
    """rel_signal=sektor → ne_schitana; rel_signal=rynok → исключение; иначе Y = шум + 0,01·(rel_signal=syroy ? 0 : 1)."""
    fdr = "t0d"

    def evaluate_v2(self, pt, period):
        rs = pt.get("rel_signal", "syroy")
        if rs == "sektor":
            return {"status": "ne_schitana"}, None, None
        if rs == "rynok":
            raise RuntimeError("сбой оценщика")
        return {"status": "ok"}, SHUM, None


def sreda(sid, ev=None):
    return ps.Sreda(ev or Fake(), "kod", "vhody", DATY, ["T"])


def test_status_okna_vse_chetyre_statusa():
    sr = sreda("S1")
    assert st.status_okna(sr, None, "train4")[1] == "none_kanon"
    assert st.status_okna(sr, {"rel_signal": "sektor"}, "train4")[1] == "ne_schitana"
    assert st.status_okna(sr, {}, "train4")[1] == "ok"
    sr2 = ps.Sreda(type("E", (), {"evaluate_v2": lambda s, p, per: ({"status": "inoy"}, None, None)})(), "k", "v", DATY, ["T"])
    assert st.status_okna(sr2, {}, "train4")[1] == "oshibka"


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "fabrika", sreda)
    sled = tmp_path / "fast.log"
    monkeypatch.setattr(ps, "fast_ok", lambda pt: True)

    @contextlib.contextmanager
    def patched():
        with open(sled, "a") as f:
            f.write("x\n")
        yield

    monkeypatch.setattr(fast_etap1, "patched", patched)
    return sled


def progon(tmp_path, **kw):
    return pm.strategiya("S1", argparse.Namespace(procs=2, prodolzhit=False, **kw), tmp_path, REZ, 20, mnozh=2)


def test_statusy_summiruyutsya_v_chislo_tochek_i_pishutsya_v_yaml(fake, tmp_path):
    i = progon(tmp_path)
    assert sum(i["statusy"].values()) == i["tochek"] and set(i["statusy"]) == set(st.STATUSY)
    assert i["statusy"]["ne_schitana"] > 0 and i["statusy"]["oshibka"] > 0 and i["statusy"]["ok"] > 0
    assert i["statusy"]["none_kanon"] == 0, "план не должен давать None-точек"
    assert i["oshibok"] == i["statusy"]["oshibka"]
    y = yaml.safe_load((tmp_path / "S1" / "zamorozka.yaml").read_text("utf-8"))
    assert y["statusy"] == i["statusy"] and y["osi_aktivnye"] and "mertvye_ne_ocenyeny" in y and "malo_shagov" in y


def test_bystryj_sloj_po_umolchaniyu_i_mutatsiya_bez_nego(fake, tmp_path):
    progon(tmp_path / "a")
    assert fake.exists() and fake.read_text().strip(), "драйвер Морриса не включил быстрый слой"
    fake.unlink()
    progon(tmp_path / "b", fast=False)
    assert not fake.exists(), "fast=False должен идти эталонным путём (мутация «без быстрого слоя» ловится первой проверкой)"


def _stat(n_akt, u95):
    return {w: {"n_akt": np.array(n_akt), "U95": np.array(u95), "mu_star": np.zeros(len(n_akt)), "sigma": np.zeros(len(n_akt))}
            for w in mz.OKNA}


OSI = [{"os": "mertv", "gruppa": "g", "novaya": True}, {"os": "malo", "gruppa": "g", "novaya": True},
       {"os": "nul", "gruppa": "g", "novaya": True}]


def _proverka_mertvoj(reshenie):
    stat = _stat([0, 5, 40], [np.nan, 0.0, 0.0])
    res = reshenie(OSI, stat)
    assert res["mertv"][0] is False and res["nul"][0] is True and res["malo"][0] is False
    pom = mz.pometki(OSI, stat)
    assert pom["mertv"]["pometka"] == mz.MERTVAYA and pom["malo"]["pometka"] == mz.MALO and pom["nul"]["pometka"] is None


def test_mertvaya_os_pomechena_i_ne_zamorozhena():
    _proverka_mertvoj(mz.reshenie)


def test_mutatsiya_mertvaya_os_zamorozhena_lovitsya():
    def mut(osi, stat):
        res = mz.reshenie(osi, stat)
        return {k: (True, "заморожена") if v[1].startswith("не измерено, " + mz.MERTVAYA) else v for k, v in res.items()}

    with pytest.raises(AssertionError):
        _proverka_mertvoj(mut)
