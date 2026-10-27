"""Э1-а (Ж-034/Ж-035): оценщик-сторона движка — кэш сигналов (Д5), DSR(N) строки, контракт строки v1.1.
Драйверы (OAT, плато, PBO, Моррис, Optuna) — модуль владельца `fastbt/sweep_drivers.py`: тесты в `test_sweep_modul.py`.
Детекторы `ok_*` — True, если свойство выполнено; мутации их роняют (`test_sweep_mutacii.py`)."""
import numpy as np
import pytest

from fastbt import metriki_e as me
from fastbt import sweep_stat as ss
from fastbt.sweep import SignalCache
from fastbt.sweep_synth import SynthEvaluator
from test_sweep_engine import P, mk

v11 = pytest.mark.v11


def _dsr_inputs():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0008, 0.01, 500)
    srs = [me.sr_dnevnoy(rng.normal(0, 0.01, 500)) for _ in range(39)] + [me.sr_dnevnoy(r)]
    return r, srs


def ok_dsr_n1_grows() -> bool:
    r, srs = _dsr_inputs()
    return ss.dsr_n(r, [me.sr_dnevnoy(r)]) > ss.dsr_n(r, srs) > ss.dsr_n(r, srs, mult=2)


def ok_cache_keys() -> bool:
    k = SignalCache.key("o", {"a": 5}, "d", "c")
    return len({k, SignalCache.key("o", {"a": 6}, "d", "c"), SignalCache.key("o", {"a": 5}, "d2", "c"),
                SignalCache.key("o", {"a": 5}, "d", "c2"), SignalCache.key("o2", {"a": 5}, "d", "c")}) == 5 \
        and k == SignalCache.key("o", {"a": 5}, "d", "c")


def ok_cache_miss_end_to_end(tmp_path, **ev_kw) -> bool:
    """Тот же cache_dir, оценщик с подменённым рядом/кодом: все 5 сигналов пересчитаны (промах), не взяты из кэша."""
    mk(SynthEvaluator(), cache_dir=tmp_path).run()
    sw = mk(SynthEvaluator(**ev_kw), cache_dir=tmp_path)
    sw.run()
    return sw.cache.misses == 5


def ok_cache_separate_from_pnl() -> bool:
    ev = SynthEvaluator()
    sw = mk(ev)
    sw.run()
    n = ev.signal_calls
    sw.objective({"a": 5, "w": 2.0})
    sw.objective({"a": 10, "w": 0.5})
    return n == 5 and ev.signal_calls == 5


# ---------- кэш сигналов (Д5, v1) ----------
def test_klyuch_kesha_menyaetsya_ot_obekta_parametra_ryada_koda():
    assert ok_cache_keys()


@pytest.mark.parametrize("kw", [dict(seed=1), dict(code="v2")])
def test_klyuch_kesha_promakh_pri_podmene_ryada_ili_koda(tmp_path, kw):
    assert ok_cache_miss_end_to_end(tmp_path, **kw)


def test_klyuch_kesha_popadanie_pri_tom_zhe_vkhode(tmp_path):
    mk(cache_dir=tmp_path).run()
    sw = mk(cache_dir=tmp_path)
    sw.run()
    assert sw.cache.misses == 0 and sw.cache.hits >= 5


def test_kesh_pnl_otdelen_ot_signalov():
    assert ok_cache_separate_from_pnl()


# ---------- контракт строки v1.1 (sweep_evaluate.md §6): красные на WIP ожидаемы ----------
ROW_COLS = ["v", "obj", "period", "okno_s", "okno_po", "zaderzhka", "chestnaya", "comm_pct", "sharpe", "msr",
            "sortino", "mdd", "oborot_y", "dni_v_rynke", "sdelok_y", "sharpe_2018", "sharpe_2019", "objective",
            "plateau", "n_sosedey", "N", "n_mult", "dsr", "pbo", "series_digest", "code_id"]


@v11
def test_kontrakt_stroka_imeet_kolonki_v11():
    row = mk().evaluate(P)
    assert [c for c in ROW_COLS if c not in row] == []


@v11
def test_kontrakt_sosedi_plato_idut_v_N():
    sw = mk()
    r = sw.evaluate(P)
    assert r["n_sosedey"] == 4 and r["N"] == 5 and r["n_mult"] == 1      # точка + 4 соседа
    sw.evaluate(P, comm_pct=0.035)
    assert len(sw.ledger) == 5                                           # пересчёт при 0,035 в N не входит


# ---------- DSR строки (v1) ----------
def test_dsr_n1_rastet_mutaciya_14():
    assert ok_dsr_n1_grows()


def test_dsr_n1_ravno_psr_i_N_schitaetsya_po_probam():
    r, srs = _dsr_inputs()
    assert ss.dsr_n(r, [me.sr_dnevnoy(r)]) == pytest.approx(me.psr(r, 0.0))
    assert ss.dsr_n(r, srs) == pytest.approx(me.dsr(r, 40, float(np.var(srs, ddof=1))))
    assert ss.dsr_n(r, srs, mult=2) == pytest.approx(me.dsr(r, 80, float(np.var(srs, ddof=1))))
