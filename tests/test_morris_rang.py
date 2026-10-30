"""Моррис v6: база траектории не требует ранга; ранг NaN у всех стартов тренда → точка не считается; «0 сделок» ≠ «вес 0» (30.10, расследование s6nol).
Данных нет там, где не нужны (синтетика); тесты на реальных кубах v6 — без данных (только план, оценщик не вызывается); один тест на данных train4 (skip без них).

Мутации: (1) база без осей ранга (прежний OSI_BAZA); (2) база с m_sily ≠ "1"; (3) защита только для tikernyj; (4) нули без разделения по числу сделок."""
from __future__ import annotations

import numpy as np
import pytest

from fastbt import canon_v6, eval_v5, eval_v5_osi as eo, morris_v6 as mv, progon_morris_v6 as pm, progon_morris_v6_status as st

NAN = float("nan")
SIDS = canon_v6.SIDS
RANG_ZNACH = {"m_sily": ("lin", "kvadr"), "bezubytok": ("rang",), "tejk": ("sila",)}   # значения, включающие требование ранга (r.rang в eval_v5_osi)


def trebuet_ranga(pt: dict) -> bool:
    """Точка v6 (канон) требует ранг: m_sily ≠ 1, безубыток от ранга, тейк от силы."""
    return str(pt.get("m_sily", "1")) != "1" or str(pt.get("bezubytok", "net")).startswith("rang") or str(pt.get("tejk", "net")).startswith("sila")


def bazy_plana(sid: str, r: int = 2) -> list:
    """Базы (первые точки) r траекторий плана драйвера: seed куба, mnozh по умолчанию, как strategiya."""
    kub, ox = pm.setka(sid)
    nom, _, pts = pm.plan(sid, kub, ox, r, mv.MNOZH, int(kub["sobol"]["morris_seed"]))
    return [pts[i] for i in nom[:, 0]]


def proverit_bazy(sid: str, r: int = 2) -> None:
    for b in bazy_plana(sid, r):
        assert not trebuet_ranga(b), f"{sid}: база требует ранг ({ {k: b.get(k) for k in RANG_ZNACH} })"


@pytest.mark.parametrize("sid", SIDS)
def test_baza_ne_trebuet_ranga_tikera(sid):
    proverit_bazy(sid)


def test_osi_rang_ravny_osyam_kubov_s_rangom_i_pervoe_znachenie_bez_ranga():
    vse = set()
    for sid in SIDS:
        for n, g, _ in canon_v6.osi(canon_v6.load_kub(sid)):
            if n in RANG_ZNACH:
                assert any(str(v).startswith(RANG_ZNACH[n]) for v in g), (sid, n)       # ось действительно включает ранг
                assert not str(g[0]).startswith(RANG_ZNACH[n]), (sid, n)              # индекс 0 («выкл.» базы) — без ранга
                vse.add(n)
    assert vse == set(mv.OSI_RANG) and set(mv.OSI_RANG) <= set(mv.OSI_BAZA)


def test_mutatsiya_baza_bez_osej_ranga_lovitsya(monkeypatch):
    monkeypatch.setattr(mv, "OSI_BAZA", mv.OSI_DOPUSK + mv.OSI_FILTR_V5)                  # подложено: прежняя база
    with pytest.raises(AssertionError, match="база требует ранг"):
        for sid in SIDS:
            proverit_bazy(sid)


def test_mutatsiya_baza_s_m_sily_ne_1_lovitsya(monkeypatch):
    monkeypatch.setattr(mv, "OSI_BAZA", mv.OSI_DOPUSK + mv.OSI_FILTR_V5 + ("bezubytok", "tejk"))   # подложено: m_sily случаен
    with pytest.raises(AssertionError, match="база требует ранг"):
        for sid in SIDS:
            proverit_bazy(sid)


# --- защита: ранг NaN у всех стартов тренда ---
SIG = np.array([0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0])
RG_NET = np.full(7, NAN)
RG_EST = np.array([NAN, NAN, NAN, NAN, 0.4, NAN, NAN])                     # конечный ранг на старте второго сигнала (день 4)


def test_rang_nan_u_vseh_startov():
    assert eo.rang_nan_u_vseh([SIG], [RG_NET])
    assert not eo.rang_nan_u_vseh([SIG], [RG_EST])
    assert not eo.rang_nan_u_vseh([SIG, SIG], [RG_NET, RG_EST])               # хотя бы у одного тикера есть
    assert not eo.rang_nan_u_vseh([np.zeros(7)], [RG_NET])                    # стартов нет вовсе — это не «ранг NaN», а нет сигнала
    assert not eo.rang_nan_u_vseh([SIG], [np.array([NAN, 0.9, NAN, 0.5, NAN, 0.3, NAN])])   # ранг не в днях старта не считается


def _mut_tolko_tikernyj(sigs, rangi, tip="srednee"):
    return tip == "tikernyj" and eo.rang_nan_u_vseh(sigs, rangi)


def test_mutatsiya_zashchita_tolko_tikernyj_lovitsya():
    assert eo.rang_nan_u_vseh([SIG], [RG_NET]) is True
    assert _mut_tolko_tikernyj([SIG], [RG_NET], "srednee") is False


def _est_dannye() -> bool:
    from fastbt import daily_1850
    return daily_1850.out_path().exists()


@pytest.mark.skipif(not _est_dannye(), reason="нет дневного ряда 18:50 в FASTBT_MAIN")
def test_trend_srednee_m_sily_lin_ne_schitana_a_obshchiy_i_m1_schitaetsya(monkeypatch):
    """Редкий сигнал тренда (3 тикера S3, база куба): srednee × lin — ne_schitana с причиной; m_sily = 1 — точка как раньше (ranga нет в весе)."""
    from test_eval_v5 import _ocenshchiki, _sp5
    _, e5 = _ocenshchiki("S3", 3, semya="v5")
    pt = {**eo.yakor("S3"), **_sp5("S3"), "m_sily": "lin", "rang_tip": "srednee"}
    r, _, _ = e5.evaluate_v2(pt, "train4")
    assert r["status"] == "ne_schitana" and "ранга нет ни у одного старта" in r["prichina"], r.get("prichina")
    monkeypatch.setattr(eval_v5, "rang_nan_u_vseh", lambda *a: False)         # мутация: защиты нет → точка «ok» с весом 0
    assert e5.evaluate_v2(pt, "train4")[0]["status"] == "ok"
    monkeypatch.undo()
    r1, _, _ = e5.evaluate_v2({**pt, "m_sily": "1"}, "train4")
    assert r1["status"] == "ok"


# --- «0 сделок» ≠ «сделки есть, вес 0» ---
def test_nol_sdelok_razlichaet_nol_sdelok_i_ves0():
    Y = np.array([[0.0] * 3, [0.0] * 3, [0.1, 0.0, 0.0], [0.0] * 3, [NAN] * 3])
    ns = np.array([0.0, 120.0, 40.0, NAN, NAN])
    nol, nl = st.nol_sdelok(Y, {"ok": 4}, ns)
    assert nol.tolist() == [True, True, False, True, False]
    assert nl == {"chislo": 3, "ok": 4, "dolya": 0.75, "bez_sdelok": 1, "ves0": 1}        # третий нуль — ns неизвестно, ни туда ни туда
    assert st.nol_sdelok(Y, {"ok": 4})[1] == {"chislo": 3, "ok": 4, "dolya": 0.75}        # без ns — прежний вид (обратная совместимость)


def test_mutatsiya_nuli_bez_razdeleniya_lovitsya():
    Y = np.array([[0.0] * 3, [0.0] * 3])
    nl = st.nol_sdelok(Y, {"ok": 2}, np.array([0.0, 120.0]))[1]
    assert nl["bez_sdelok"] != nl["chislo"] and nl["ves0"] == 1                          # прежний счёт «всё нули = 0 сделок» дал бы 2


def test_baza_bez_sdelok_dva_scheta():
    nom = np.array([[0, 1], [2, 1], [3, 1]])
    nol = np.array([True, False, True, True])
    bz = st.baza_bez_sdelok(nol, nom, np.array([0.0, 5.0, 77.0, NAN]))
    assert bz == {"chislo": 3, "r": 3, "dolya": 1.0, "bez_sdelok": 1, "ves0": 1}


class _Evr:
    def __init__(self, r, ex):
        self.r, self.ex = r, ex

    def evaluate_v2(self, pt, period):
        return self.r, self.ex, None


def test_ocenka_beryot_chislo_sdelok_iz_sdelok_y():
    sr = type("S", (), {"evr": _Evr({"status": "ok", "sdelok_y": {"2015": 3, "2016": 4}}, None)})()
    assert pm.ocenka(sr, {}, "train4")[2] == 7.0
    assert pm.ocenka(sr, {}, "train4")[:2] == pm.status_okna(sr, {}, "train4")
    sr0 = type("S", (), {"evr": _Evr({"status": "ok"}, None)})()
    assert np.isnan(pm.ocenka(sr0, {}, "train4")[2])
    assert np.isnan(pm.ocenka(sr, None, "train4")[2])
