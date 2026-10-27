"""К4 Ж-047, тест 2.14 на реальных данных 2018–2019: режим «якорь» = «A только Sk» через s9 (SkEvaluator.pnl =
ekviv_s9.run_sk), S1–S6, 2018 и 2019, 0,04 и 0,035 %, допуск 1e-12. Нужны 1-минутки (FASTBT_BARS1M) и данные s9;
нет данных — пропуск. Мутация: стороны последней строки окна (Н5) → 0 обязаны ронять якорь. Первая строка окна на реальных данных
пуста (позиций на 01.01 нет) — её правило проверено случайными окнами в test_etap1_run.py."""
import numpy as np
import pytest

from fastbt import bars5m

pytestmark = pytest.mark.skipif(not bars5m.SRC.exists(), reason="нет 1-минуток D25 (FASTBT_BARS1M)")
SIDS = ["S1", "S2", "S3", "S4", "S5", "S6"]
TOL = 1e-12


@pytest.fixture(scope="module")
def inputs():
    from fastbt import etap1
    return {s: etap1.build_inputs(s) for s in SIDS}


@pytest.mark.parametrize("sid", SIDS)
@pytest.mark.parametrize("comm", [0.0004, 0.00035])
def test_214_anchor_equals_s9(sid, comm, inputs):
    from fastbt import etap1
    from fastbt.etap1_eval import Evaluator
    d = etap1.anchor_s9_diff(sid, Evaluator(sid, inputs[sid], etap1.base_point(sid), comm), comm)
    assert max(d.values()) <= TOL, d


def test_214_mutation_last_row_sides_zero_caught(monkeypatch, inputs):
    from fastbt import etap1, etap1_eval, etap1_run
    from fastbt.etap1_eval import Evaluator

    def bad(*a, **kw):
        r = etap1_run.run_e1(*a, **kw)
        r.sides[-1] = 0.0                       # Н5: выход открытой позиции на последней строке окна не оплачен
        return r
    monkeypatch.setattr(etap1_eval, "run_e1", bad)
    d = etap1.anchor_s9_diff("S2", Evaluator("S2", inputs["S2"], etap1.base_point("S2")))
    assert max(d.values()) > TOL


FORBID = ("vpred", "ch4_dev", "sigma_pred")


def _guard(monkeypatch):
    """Любое открытие файла с запрещённым именем (open, read_parquet, read_table, read_csv, Path.read_*) → падение."""
    import builtins
    from pathlib import Path

    import pandas as pd
    import pyarrow.parquet as pq

    def wrap(f):
        def g(*a, **kw):
            for x in list(a[:1]) + list(kw.values()):
                if any(w in str(x) for w in FORBID):
                    raise AssertionError(f"этап 1 открыл прогнозный файл: {x}")
            return f(*a, **kw)
        return g
    monkeypatch.setattr(builtins, "open", wrap(builtins.open))
    for mod, name in ((pd, "read_parquet"), (pd, "read_csv"), (pq, "read_table")):
        monkeypatch.setattr(mod, name, wrap(getattr(mod, name)))
    for name in ("read_bytes", "read_text"):
        orig = getattr(Path, name)
        monkeypatch.setattr(Path, name, lambda self, *a, _o=orig, **kw: wrap(_o)(self, *a, **kw))


def test_no_forecast_files_opened_at_runtime(monkeypatch):
    """Ж-047 п. 1: fabrika("S2") + одна оценка не открывают vpred_aligned.parquet и прочие прогнозные файлы."""
    from fastbt import data, etap1
    data.vpred_df.cache_clear()
    _guard(monkeypatch)
    ev, chk = etap1.fabrika("S2")
    assert np.isfinite(ev({"k_s": 2.0, "tau": 5}, "train")["sharpe"]) and chk({}) is True


def test_guard_catches_vpred_load(monkeypatch):
    """Мутация: прежний путь SkEvaluator (load_ticker с align_sigma) открывает vpred — охрана обязана упасть."""
    from fastbt import data, etap1
    data.vpred_df.cache_clear()
    _guard(monkeypatch)
    monkeypatch.setattr(etap1, "load_ticker_nf", data.load_ticker)
    with pytest.raises(AssertionError, match="прогнозный"):
        etap1.sk_evaluator("S2")


def test_row_format_for_driver_k5(inputs):
    from fastbt import etap1
    from fastbt.etap1_eval import Evaluator
    ev = Evaluator("S2", inputs["S2"], etap1.base_point("S2"))
    pt = dict(k_s=2.0, R=2.0, tau=5, proverka="bars5m", vhod=0.5, sigma="ewma094")
    r = ev.evaluate(pt, "train")
    assert set(r["msr_by_year"]) == set(r["excess_d"]) == set(r["sdelok_y"]) == {2018, 2019}
    assert r["tselevaya"] == min(r["msr_by_year"].values())
    assert 0 < r["dolya_ispolneniya_limita"] <= 1 and r["v"] == "1.0"
    assert np.isfinite(np.concatenate(list(r["excess_d"].values()))).all()
    c = ev.evaluate(pt, "ctrl2019")
    assert set(c["excess_d"]) == {2019} and c["chestnaya"] == "kontrol"
    with pytest.raises(PermissionError):
        ev.evaluate(pt, "validation")


def test_weight_and_filters_change_result(inputs):
    from fastbt import etap1
    from fastbt.etap1_eval import Evaluator
    ev = Evaluator("S2", inputs["S2"], etap1.base_point("S2"))
    base = dict(k_s=2.0, tau=10)
    a = ev.evaluate(base, "train")["sharpe"]
    for ch in (dict(razmer=15), dict(rezhim="rank08"), dict(sobytiya="cbr"), dict(q_sily=0.7)):
        assert ev.evaluate({**base, **ch}, "train")["sharpe"] != a, ch
