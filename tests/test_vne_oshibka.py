"""Б2 (замечание проверяющего #610): исполнение вне [low; high] бара в дневном автомате — ValueError, а не счётчик s.vne.
Падает до правки: _v_bare только копил нарушения. Мутация «снова счётчик без исключения» ловится G1 и G3.
Быстрый слой (fast_etap1/fast_exec, numba-ядро): счётчика нарушений там нет — зеркалить нечего (см. отчёт).
Синтетика, сеть не нужна."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fastbt import exec_e, progon_sobol as ps
from fastbt.exec_e_run import Run
from fastbt.exec_e_types import Day, Params

N_BAR, N_DEN = 4, 6


def _dni():
    """Плоские дни: бары [99; 101], close 100 — лимит L=99,75 внутри баров."""
    d = Day(np.full(N_BAR, 100.0), np.full(N_BAR, 101.0), np.full(N_BAR, 99.0), np.full(N_BAR, 100.0))
    return [d] * N_DEN


def _run(p=None):
    sig = np.array([0, 1, 1, 1, 0, 0], float)
    return Run("SYN", _dni(), sig, np.full(N_DEN, 0.01), p or Params(k=0.25, b=1.0))


def test_g1_vne_bara_dnevnoy_avtomat_valueerror():
    r = _run()
    with pytest.raises(ValueError, match=r"SYN день 2 бар 1 цена 97\.0 \[99\.0; 101\.0\]"):
        r._v_bare(2, 1, 97.0)


def test_g1_v_bare_vnutri_i_close_dnya_ne_padayut():
    r = _run()
    r._v_bare(2, 1, 99.0)
    r._v_bare(2, 1, 101.0)
    r._v_bare(2, 1, r.close_of(2))
    assert r.vne == []


def _mutant_stop(s, i, j, d, b, px, S):
    """Подложенное нарушение: выход ровно по цене вне бара."""
    if S is None:
        return False
    s._exit(i, j, b, 90.0, s.sr)
    return True


def test_g3_nasoznoe_narushenie_v_hode_prognoza(monkeypatch):
    monkeypatch.setattr(Run, "_stop_vhoda", _mutant_stop)
    p = Params(k=0.25, b=1.0)
    with pytest.raises(ValueError, match=r"вне \[low; high\]"):
        exec_e.run("SYN", _dni(), np.array([0, 1, 1, 1, 0, 0], float), np.full(N_DEN, 0.01), p)


def test_g3_tochka_s_narusheniem_stroka_oshibka_ostalnye_schitayutsya(monkeypatch):
    monkeypatch.setattr(ps.lm, "klyuch", lambda *a, **k: {})
    monkeypatch.setattr(ps.lm, "klyuch_sha", lambda *a, **k: "x")
    monkeypatch.setattr(ps, "fast_ok", lambda pt: False)
    monkeypatch.setattr(Run, "_stop_vhoda", _mutant_stop)

    def evaluate_v2(pt, period):
        if pt["plohaya"]:
            exec_e.run("SYN", _dni(), np.array([0, 1, 1, 1, 0, 0], float), np.full(N_DEN, 0.01), Params(k=0.25, b=1.0))
        return {"status": "ok", "prichina": None}, None, None

    sr = SimpleNamespace(evr=SimpleNamespace(evaluate_v2=evaluate_v2), kod="k", vhody="v")
    stroki = [ps.schitat_tochku(sr, "S", i, {"plohaya": i == 1}, "test", True, False)[0] for i in range(3)]
    assert [x["status"] for x in stroki] == ["ok", "oshibka", "ok"]
    assert "ValueError" in stroki[1]["prichina"] and "вне [low; high]" in stroki[1]["prichina"]


def _schetchik(s, i, b, px):
    """Мутация G5: прежнее поведение — копить в s.vne без исключения."""
    d = s.days[i]
    if px != s.close_of(i) and not (d.l[b] <= px <= d.h[b]):
        s.vne.append((i, b, px))


def test_g5_mutaciya_schetchik_bez_isklyucheniya_lovitsya(monkeypatch):
    monkeypatch.setattr(Run, "_v_bare", _schetchik)
    with pytest.raises(pytest.fail.Exception):
        test_g1_vne_bara_dnevnoy_avtomat_valueerror()
