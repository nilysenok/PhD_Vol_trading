"""Моррис v6: база траектории — все оси-допуски «выкл.» и страж Д4 «0 сделок» (Ж-142 п. 2, решения 3 и 4 от 30.10 16:00).
Данных и сети нет: синтетический куб, синтетические Y; список допусков сверяется с кубами `filtr: true` и с eval_v6.

Мутации: база со случайными допусками (baza=()); нулевые пары (обе точки без сделок) засчитаны как «пустота» оси (nol=None)."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from fastbt import canon_v6, eval_v6, morris_v6 as mv, morris_v6_zamorozka as mz, progon_morris_v6 as pm, progon_morris_v6_status as st

L = [3, 2, 4, 2, 3, 2, 4, 2]
BAZA = [0, 2, 4, 6]            # оси-допуски синтетического куба
SEEDS = (1, 7, 20261101)


def klyuch(idx):
    """Запрещено: оси 1 и 3 вместе в «0» (часто попадает в старт) — старт заменяется, база обязана остаться."""
    return None if idx[1] == 0 and idx[3] == 0 else tuple(idx)


def plan(seed, baza=BAZA):
    return mv.traektorii_kanon(L, 16, klyuch, seed, 4, baza=baza)


def proverit_bazu(I, baza=BAZA):
    assert (I[:, 0][:, baza] == 0).all(), "в старте траектории есть включённый допуск"


@pytest.mark.parametrize("seed", SEEDS)
def test_baza_vse_dopuski_vykl_ostalnye_sluchajny_kanon_sohranyon(seed):
    I, por = plan(seed)
    proverit_bazu(I)
    ostal = [i for i in range(len(L)) if i not in BAZA]
    assert I[:, 0][:, ostal].any(), "остальные оси должны быть случайными, а не нулевыми"
    assert I[:, -1][:, BAZA].any(), "допуск обязан включаться на своём шаге"
    assert all(klyuch(tuple(int(i) for i in p)) is not None for tr in I for p in tr), "канон нарушен"
    assert all(sorted(p) == list(range(len(L))) for p in por)


def test_determinizm_po_seed():
    (a, pa), (b, pb) = plan(5), plan(5)
    assert (a == b).all() and (pa == pb).all()
    assert not (plan(6)[0] == a).all()


def test_baza_pustaya_ne_menyaet_staryj_plan():
    assert (plan(7, baza=())[0] == mv.traektorii_kanon(L, 16, klyuch, 7, 4)[0]).all()


def test_mutatsiya_baza_so_sluchajnymi_dopuskami_lovitsya():
    I, _ = plan(7, baza=())                                  # подложено: допуски не обнулены
    with pytest.raises(AssertionError, match="включённый допуск"):
        proverit_bazu(I)


def test_spisok_dopuskov_raven_filtr_true_kubov_i_podklyuchen():
    vse = set()
    for sid in canon_v6.SIDS:
        kub = canon_v6.load_kub(sid)
        vse |= {n for sec in canon_v6.NOVYE for n, v in kub.get(sec, {}).items() if v.get("filtr")}
    assert vse == set(mv.OSI_DOPUSK)
    assert set(mv.OSI_DOPUSK) <= eval_v6.PODKLYUCHENO


def test_plan_realnogo_kuba_baza_vykl():
    kub, ox = pm.setka("S1")
    baza = [i for i, (n, _, _) in enumerate(ox) if n in mv.OSI_DOPUSK]
    assert len(baza) >= 10
    I, _ = mv.traektorii_kanon([len(g) for _, g, _ in ox], 3, lambda idx: idx, mv.SEED, 2, baza=baza)
    proverit_bazu(I, baza)


def test_schetchik_nol_sdelok_i_vnimanie():
    Y = np.array([[0.0] * 3, [np.nan] * 3, [0.1, 0.0, 0.0], [0.0] * 3])
    nol, nl = st.nol_sdelok(Y, {"ok": 3})
    assert nol.tolist() == [True, False, False, True] and nl == {"chislo": 2, "ok": 3, "dolya": 0.6667}
    assert "ВНИМАНИЕ" in st.vnimanie("S1", nl) and "S1" in st.vnimanie("S1", nl)
    assert st.vnimanie("S1", {"chislo": 1, "ok": 2, "dolya": 0.5}) is None          # ровно 50 % — не больше
    assert st.nol_sdelok(np.zeros((0, 3)), {"ok": 0})[1]["dolya"] == 0.0


def test_nol_sdelok_v_yaml_i_csv(tmp_path):
    nl = {"chislo": 2, "ok": 3, "dolya": 0.6667}
    zam = mz.zapisat_yaml(tmp_path / "z.yaml", "2026-10-30", {}, {}, {"ok": 3}, {}, nl)
    assert zam == [] and yaml.safe_load((tmp_path / "z.yaml").read_text("utf-8"))["nol_sdelok"] == nl
    mz.zapisat_csv(tmp_path / "m.csv", {"nol_sdelok": nl}, [], {}, {})
    assert "# nol_sdelok: " in (tmp_path / "m.csv").read_text("utf-8")


def _statistika(Y, nol):
    r = len(Y)
    nom = np.arange(r * 3).reshape(r, 3)                       # ключи всех точек различны: все шаги активны
    por = np.tile([0, 1], (r, 1))
    one = dict(mv.statistika(mv.effekty(Y, nom, por, nol)), n_nol=mv.pary_nol(nom, por, nol) if nol is not None else np.zeros(2, int))
    return {w: one for w in mz.OKNA}


OSI = [{"os": "a", "gruppa": "g", "novaya": True}, {"os": "b", "gruppa": "g", "novaya": True}]


def test_os_s_effektom_tolko_na_nulevyh_parah_ne_zamorazhivaetsya():
    Y = np.tile([0.0, 0.0, 0.5], (30, 1))                      # ось a: пары без сделок; ось b: 0 → 0,5, эффект есть
    nol = Y == 0
    stat = _statistika(Y, nol)
    res, pom = mz.reshenie(OSI, stat), mz.pometki(OSI, stat)
    assert res["a"][0] is False and mz.NET_SDELOK in res["a"][1] and pom["a"]["pometka"] == mz.NET_SDELOK
    assert res["b"][0] is False and pom["b"]["pometka"] is None          # эффект 0,5 ≥ порога: не заморожена
    assert pom["a"]["n_akt"] == [0, 0, 0]


def test_mutatsiya_nulevye_pary_zasschitany_kak_pustota_lovitsya():
    Y = np.zeros((30, 3))
    mut = _statistika(Y, None)                                 # подложено: nol не учтён
    assert mz.reshenie(OSI, mut)["a"][0] is True, "мутация обязана замораживать по пустым парам"
    stat = _statistika(Y, np.ones((30, 3), bool))
    assert mz.reshenie(OSI, stat)["a"][0] is False and mz.reshenie(OSI, stat)["b"][0] is False
    assert mz.pometki(OSI, stat)["a"]["pometka"] == mz.NET_SDELOK
