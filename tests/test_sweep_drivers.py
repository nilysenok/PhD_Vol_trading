"""Э1-а (Ж-034/Ж-035): стык оценщика и драйверов — кэш сигналов (Д5), ledger/N -> DSR модуля владельца, контракт строки.
Сам модуль `fastbt/sweep_drivers.py` — `test_sweep_modul.py`. Детекторы `ok_*` роняют мутации (`test_sweep_mutacii.py`).
`xfail(strict=True)` — найденное расхождение с контрактом (красное, названо в отчёте)."""
import pytest

from fastbt import sweep_drivers as sd
from fastbt.sweep_spec import SignalCache
from fastbt.sweep_synth import SynthEvaluator
from test_sweep_engine import A_GRID, P, mk

v11 = pytest.mark.v11


def run_grid(sw):
    for a in A_GRID:
        sw.evaluate({"a": a})
    return sw


def ok_cache_keys() -> bool:
    k = SignalCache.key("o", {"a": 5}, "d", "c")
    return len({k, SignalCache.key("o", {"a": 6}, "d", "c"), SignalCache.key("o", {"a": 5}, "d2", "c"),
                SignalCache.key("o", {"a": 5}, "d", "c2"), SignalCache.key("o2", {"a": 5}, "d", "c")}) == 5 \
        and k == SignalCache.key("o", {"a": 5}, "d", "c")


def ok_cache_miss_end_to_end(tmp_path, **ev_kw) -> bool:
    """Тот же cache_dir, оценщик с подменённым рядом/кодом: все 5 сигналов пересчитаны (промах), не взяты из кэша."""
    run_grid(mk(SynthEvaluator(), cache_dir=tmp_path))
    sw = run_grid(mk(SynthEvaluator(**ev_kw), cache_dir=tmp_path))
    return sw.cache.misses == 5


def ok_cache_separate_from_pnl() -> bool:
    ev = SynthEvaluator()
    sw = run_grid(mk(ev))
    sw.evaluate({"a": 5, "w": 2.0})
    sw.evaluate({"a": 10, "w": 0.5}, "train", 0.04)
    return ev.signal_calls == 5


def ok_dsr_from_ledger() -> bool:
    """DSR модуля по пробам оценщика: N=1 -> выше, чем после 5 проб; повторное открытие (N×2) -> ещё ниже."""
    one = mk().evaluate(P)
    sw = run_grid(mk())
    r = sw.evaluate(P)
    d1 = sd.dsr(one["sr_d"], [one["sr_d"]], one["T"], one["skew"], one["kurt"])["dsr"]
    d5 = sd.dsr(r["sr_d"], sw.trials_sr(), r["T"], r["skew"], r["kurt"])["dsr"]
    return one["N"] == 1 and r["N"] == 5 and d1 > d5


def test_klyuch_kesha_menyaetsya_ot_obekta_parametra_ryada_koda():
    assert ok_cache_keys()


@pytest.mark.parametrize("kw", [dict(seed=1), dict(code="v2")])
def test_klyuch_kesha_promakh_pri_podmene_ryada_ili_koda(tmp_path, kw):
    assert ok_cache_miss_end_to_end(tmp_path, **kw)


def test_klyuch_kesha_popadanie_pri_tom_zhe_vkhode(tmp_path):
    run_grid(mk(cache_dir=tmp_path))
    sw = run_grid(mk(cache_dir=tmp_path))
    assert sw.cache.misses == 0 and sw.cache.hits >= 5


def test_kesh_pnl_otdelen_ot_signalov():
    assert ok_cache_separate_from_pnl()


def test_dsr_modulya_po_probam_otsenshchika_N1_rastet():
    assert ok_dsr_from_ledger()


def test_pereschet_pri_drugoy_komissii_ne_dobavlyaet_N():
    sw = run_grid(mk())
    n = sw.evaluate(P, "train", 0.04)["N"]
    assert sw.evaluate(P, "train", 0.035)["N"] == n == 5


@pytest.mark.xfail(strict=True, reason="расхождение с контрактом §5: пересчёт при 0,035 перезаписывает SR пробы в ledger (отбор — 0,04)")
def test_pereschet_ne_perezapisyvaet_sr_v_ledger():
    sw = mk()
    sw.evaluate(P, "train", 0.04)
    before = list(sw.trials_sr())
    sw.evaluate(P, "train", 0.035)
    assert sw.trials_sr() == before


@v11
@pytest.mark.xfail(strict=True, reason="контракт §6 (формат v1.1): колонки v, obj, period-окна, zaderzhka, chestnaya, series_digest, code_id ещё не в строке")
def test_kontrakt_stroka_imeet_kolonki_v11():
    cols = ["v", "obj", "period", "okno_s", "okno_po", "zaderzhka", "chestnaya", "comm_pct", "sharpe", "msr", "sortino",
            "mdd", "oborot_y", "dni_v_rynke", "sdelok_y", "N", "n_mult", "series_digest", "code_id"]
    row = mk().evaluate(P)
    assert [c for c in cols if c not in row] == []
