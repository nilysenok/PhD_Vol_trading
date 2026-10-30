"""Моррис v6, PR #761: дополнительные тесты тестировщика (мутации, не пойманные tests/test_morris_baza.py) и свойство «база канонична».
Данных и сети нет. Мутации: (в) пара «одна точка без сделок» выброшена; (е) seed не влияет на починку старта; (г) ось с эффектом только на пустых парах заморожена."""
from __future__ import annotations

import numpy as np
import pytest

from fastbt import canon_v6, morris_v6 as mv, morris_v6_zamorozka as mz, progon_morris_v6 as pm


def effekty_mut_ili(Y, keys, por, nol):
    """Мутант (в): пара выбрасывается, если ХОТЯ БЫ одна точка без сделок."""
    dY = np.asarray(Y, float)[:, 1:] - np.asarray(Y, float)[:, :-1]
    ok = (keys[:, 1:] != keys[:, :-1]) & np.isfinite(dY) & ~(nol[:, 1:] | nol[:, :-1])
    E = np.full((len(Y), por.shape[1]), np.nan)
    np.put_along_axis(E, por, np.where(ok, dY, np.nan), axis=1)
    return E


def sintetika(r=12):
    """Ось 0: база со сделками (Y=0,4) → допуск убил все сделки (Y=0). Ось 1: 0 → 0 (обе без сделок). Ось 2 в той же траектории не нужна."""
    Y = np.tile([0.4, 0.0, 0.0], (r, 1))
    nol = Y == 0
    keys = np.tile([0, 1, 2], (r, 1)) + 3 * np.arange(r)[:, None]
    por = np.tile([0, 1], (r, 1))
    return Y, nol, keys, por


def test_vklyuchenie_dopuska_do_nulya_sdelok_popadaet_v_effekty():
    Y, nol, keys, por = sintetika()
    E = mv.effekty(Y, keys, por, nol)
    assert np.allclose(E[:, 0], -0.4), "шаг «база со сделками → 0 сделок» — информативный эффект, не нулевая пара"
    assert np.isnan(E[:, 1]).all(), "обе точки без сделок — не эффект"
    assert mv.pary_nol(keys, por, nol).tolist() == [0, 12]
    assert mv.statistika(E)["n_akt"].tolist() == [12, 0]


def test_obratnyj_shag_ot_nulya_sdelok_k_sdelkam_tozhe_effekt():
    Y, nol, keys, por = sintetika()
    Y, nol = Y[:, ::-1], nol[:, ::-1]
    E = mv.effekty(Y, keys, por, nol)
    assert np.isnan(E[:, 0]).all() and np.allclose(E[:, 1], 0.4)


def test_mutatsiya_odna_tochka_bez_sdelok_vybroshena_lovitsya():
    Y, nol, keys, por = sintetika()
    mut = effekty_mut_ili(Y, keys, por, nol)
    assert np.isnan(mut[:, 0]).all(), "мутант обязан терять эффект «допуск убил сделки»"
    real = mv.effekty(Y, keys, por, nol)
    assert not np.array_equal(np.isnan(real), np.isnan(mut))


def test_os_ubivayushchaya_sdelki_ne_zamorazhivaetsya_i_ne_pomechena_net_sdelok():
    Y, nol, keys, por = sintetika(30)
    one = dict(mv.statistika(mv.effekty(Y, keys, por, nol)), n_nol=mv.pary_nol(keys, por, nol))
    stat = {w: one for w in mz.OKNA}
    osi = [{"os": "a", "gruppa": "g", "novaya": True}, {"os": "b", "gruppa": "g", "novaya": True}]
    res, pom = mz.reshenie(osi, stat), mz.pometki(osi, stat)
    assert res["a"][0] is False and pom["a"]["pometka"] is None and pom["a"]["n_akt"] == [30] * 3
    assert pom["b"]["pometka"] == mz.NET_SDELOK


def test_seed_vliyaet_na_pochinku_starta_pri_odinakovom_otbore(monkeypatch):
    L = [3, 2, 4, 2, 3, 2]
    I0, por0 = mv.traektorii(L, 16, 1, 4)
    monkeypatch.setattr(mv, "traektorii", lambda *a, **k: (I0.copy(), por0))
    kl = lambda idx: None if idx[1] == 0 or idx[3] == 0 else tuple(idx)           # noqa: E731 — большинство стартов недопустимо
    a, b = (mv.traektorii_kanon(L, 16, kl, s, 4, baza=[0, 4])[0] for s in (1, 2))
    assert (a[:, 0][:, [0, 4]] == 0).all() and (b[:, 0][:, [0, 4]] == 0).all()
    assert not (a == b).all(), "seed обязан менять починку старта (мутация: rng не от seed)"


@pytest.mark.parametrize("sid", [s for s in canon_v6.SIDS if (canon_v6.KUB6 / f"kub_{s}.yaml").exists()])
def test_baza_realnogo_kuba_kanonicheskaya_i_filtry_vykl(sid, monkeypatch):
    kub, ox = pm.setka(sid)
    zapis = []
    ishod = mv.traektorii_kanon
    monkeypatch.setattr(mv, "traektorii_kanon", lambda *a, **k: zapis.append(ishod(*a, **k)) or zapis[-1])
    nom, _, pts = pm.plan(sid, kub, ox, 8, 2, int(kub["sobol"]["morris_seed"]))
    baza = [i for i, (n, _, _) in enumerate(ox) if n in mv.OSI_BAZA]
    assert baza and {n for n, _, _ in ox if n in mv.OSI_FILTR_V5} <= {ox[i][0] for i in baza}
    I = zapis[0][0]
    assert (I[:, 0][:, baza] == 0).all(), f"{sid}: в базе есть включённый допуск/фильтр"
    assert all(pts[nom[m, 0]] is not None for m in range(len(nom))), f"{sid}: база None-канон"
    assert (nom[:, 0] >= 0).all()
