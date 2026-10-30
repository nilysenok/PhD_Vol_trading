"""Зона wf в точках входа счёта (перенос на сервер, 31.10): `progon_morris_v6 --period wf` и `progon_server --period wf` ставят зону wf
(`zona.zona_perioda`), в том числе в дочерних процессах fork-пула; train4 с датой >= 2025 — по-прежнему PermissionError.
Данных нет: оценщик-заглушка сам проверяет `zona.check(max даты)`, как загрузчик (`fastbt/zona.py:51`), в родителе и в ребёнке."""
import concurrent.futures as cf
import contextlib
import multiprocessing as mp

import numpy as np
import pytest

from fastbt import progon_morris_v6 as pm, progon_server as srv, progon_sobol as ps, zona

DATY_WF = np.arange(np.datetime64("2014-06-02"), np.datetime64("2026-02-28"))      # 2014–2026: до 2026-02-27
DATY_WF = DATY_WF[np.is_busday(DATY_WF)]
DATY_V = DATY_WF[DATY_WF < np.datetime64("2020-01-01")]


class Fake:
    fdr = "t0d"

    def __init__(self, daty):
        self.daty = daty

    def evaluate_v2(self, pt, period):
        zona.check(self.daty.max(), "дневной ряд")                         # как загрузчик: в процессе, где идёт счёт
        x = np.random.default_rng(1).normal(0, 0.01, self.daty.size)
        return {"status": "ok", "by_year": {}, "T": x.size, "mediana_kv": 0.1}, x, np.tile(x, (17, 1))


@pytest.fixture(autouse=True)
def zona_dev():
    zona.set_zone("dev")
    yield
    zona.set_zone("dev")


def fabrika_s(daty):
    def f(sid, *a):
        zona.check(daty.max(), "дневной ряд")                              # построение входов — в родителе
        return ps.Sreda(Fake(daty), "kod", "vhody", daty, [f"T{i:02d}" for i in range(17)])
    return f


def morris(tmp_path, period):
    return pm.main(["--kub", "v6", "--period", period, "--procs", "1", "--r", "1", "--strategii", "S1", "--no-fast", "--out", str(tmp_path)])


def test_morris_wf_zona_wf_bez_permissionerror(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ps, "fabrika", fabrika_s(DATY_WF))
    assert morris(tmp_path, "wf") == 0
    assert "исключений 0" in capsys.readouterr().out                         # ребёнок пула тоже без PermissionError
    assert zona.name() == "dev"                                              # после счёта прежняя зона


def test_morris_train4_data_2025_po_prezhnemu_otkaz(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "fabrika", fabrika_s(np.arange(np.datetime64("2016-01-04"), np.datetime64("2025-06-01"))))
    with pytest.raises((PermissionError, ValueError)):
        morris(tmp_path, "train4")
    assert zona.name() == "dev"


def test_server_wf_zona_wf_bez_permissionerror(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "fabrika", fabrika_s(DATY_WF))
    monkeypatch.setattr(srv, "CHEKPOINT", 100)
    assert srv.main(["--kub", "v5", "--period", "wf", "--seed", "1", "--n1", "S1=1", "--strategii", "S1", "--procs", "1", "--out", str(tmp_path)]) == 0
    assert zona.name() == "dev"


def test_server_train4_data_2025_otkaz(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "fabrika", fabrika_s(np.arange(np.datetime64("2016-01-04"), np.datetime64("2025-06-01"))))
    try:
        rc = srv.main(["--kub", "v5", "--period", "train4", "--seed", "1", "--n1", "S1=1", "--strategii", "S1", "--procs", "1", "--out", str(tmp_path)])
    except (PermissionError, ValueError):
        rc = 2
    assert rc == 2                                                          # отказ (сбой куска или исключение), не успех
    assert zona.name() == "dev"


def _zona_rebenka() -> str:
    return zona.name()


def test_zona_wf_v_dochernem_processe_pula():
    with zona.zona_perioda("wf"):
        fk = mp.get_context("fork")
        with cf.ProcessPoolExecutor(1, mp_context=fk, initializer=ps._init, initargs=({}, fk.Value("i", 0))) as ex:
            assert ex.submit(_zona_rebenka).result() == "wf"
    assert zona.name() == "dev"


def test_zona_perioda_ne_wf_zonu_ne_trogaet():
    with zona.zona_perioda("train4"):
        assert zona.name() == "dev"


def test_mutatsiya_zona_ne_stavitsya_lovitsya(monkeypatch, tmp_path):
    monkeypatch.setattr(zona, "zona_perioda", lambda period: contextlib.nullcontext(), raising=False)  # мутация: зона не ставится
    monkeypatch.setattr(ps, "fabrika", fabrika_s(DATY_WF))
    with pytest.raises(PermissionError):
        morris(tmp_path, "wf")
