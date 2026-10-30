"""Мутации к #884 (тестировщик, 31.10): зона wf в точках входа счёта.
М1: `zona_perioda` не вызывается в `progon_server.main` / `progon_morris_v6.main` (по отдельности);
М2: зона не восстанавливается при исключении; М3: ekviv не делегирует в fastbt.zona; М4: контекст пула не fork (на Маке — spawn)."""
import contextlib
import multiprocessing as mp
import sys
from pathlib import Path

import pytest

from fastbt import progon_morris_v6 as pm, progon_server as srv, progon_sobol as ps, zona

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "server"))
import ekviv  # noqa: E402


@pytest.fixture(autouse=True)
def zona_dev():
    zona.set_zone("dev")
    yield
    zona.set_zone("dev")


class Stop(Exception):
    pass


def _otkaz(*a, **k):
    raise Stop


def _vyzovy(monkeypatch):
    zv = []
    real = zona.zona_perioda
    monkeypatch.setattr(zona, "zona_perioda", lambda p: (zv.append(p), real(p))[1])
    monkeypatch.setattr(ps, "fabrika", _otkaz)
    return zv


def test_morris_main_vyzyvaet_zona_perioda_wf(monkeypatch, tmp_path):
    zv = _vyzovy(monkeypatch)
    with pytest.raises(Stop):                                              # Моррис пробрасывает исключение; зона вернётся и тут
        pm.main(["--kub", "v6", "--period", "wf", "--procs", "1", "--r", "1", "--strategii", "S1", "--no-fast", "--out", str(tmp_path)])
    assert zv == ["wf"]                                                    # мутация «не вызывается» даст []
    assert zona.name() == "dev"


def test_server_main_vyzyvaet_zona_perioda_wf(monkeypatch, tmp_path):
    zv = _vyzovy(monkeypatch)
    srv.main(["--kub", "v5", "--period", "wf", "--seed", "1", "--n1", "S1=1", "--strategii", "S1", "--procs", "1", "--out", str(tmp_path)])
    assert zv == ["wf"]
    assert zona.name() == "dev"


def test_zona_vozvrashchaetsya_pri_isklyuchenii():
    with pytest.raises(Stop), zona.zona_perioda("wf"):
        assert zona.name() == "wf"
        raise Stop
    assert zona.name() == "dev"


def test_mutatsiya_zona_ne_vosstanavlivaetsya_lovitsya(monkeypatch):
    @contextlib.contextmanager
    def bez_finally(z):
        zona.set_zone(z)
        yield                                                              # мутация: нет try/finally
    monkeypatch.setattr(zona, "zone_ctx", bez_finally)
    with pytest.raises(Stop), zona.zona_perioda("wf"):
        raise Stop
    assert zona.name() == "wf"                                             # мутация видна: зона осталась wf (тест выше на ней упал бы)


def test_ne_wf_period_zonu_ne_trogaet_i_pri_isklyuchenii():
    for p in ("train4", "v2"):
        with pytest.raises(Stop), zona.zona_perioda(p):
            raise Stop
        assert zona.name() == "dev"


def test_ekviv_delegiruet_v_fastbt_zona(monkeypatch):
    zv = []
    monkeypatch.setattr(zona, "zona_perioda", lambda p: (zv.append(p), contextlib.nullcontext())[1])
    with ekviv.zona_perioda("wf"):
        pass
    assert zv == ["wf"]                                                    # своё определение в ekviv (Д7) — не вызвало бы


def _zona_rebenka() -> str:
    return zona.name()


def test_pul_morris_i_server_ispolzuyut_fork():
    for modul in (pm, srv):
        tekst = Path(modul.__file__).read_text(encoding="utf-8")
        assert 'get_context("fork")' in tekst                              # иначе на Маке (spawn) зона в детях не наследуется


def test_spawn_ne_nasleduet_zonu_wf():
    """Обоснование теста выше: при spawn ребёнок видит dev, а не wf — fork обязателен."""
    import concurrent.futures as cf
    with zona.zona_perioda("wf"), cf.ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        assert ex.submit(_zona_rebenka).result() == "dev"
