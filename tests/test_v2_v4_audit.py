"""G1/G2/G3 (а), (б): аудит сохранённых прогнозов V4-0. Мутации: «M сдвинут на +1 день», «цель RV другого определения», «пары с t⁺ > d в обучении» — ловятся боевым путём аудита."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from models_v2 import har
from models_v2.s10 import hp_engine as E
from models_v2.s11 import v4_audit as AU, v4_bench as B, v4_io as IO, v4_oofcode as OC


def ar_series(n=600, seed=1):
    rng, x = np.random.default_rng(seed), np.zeros(n)
    for i in range(1, n):
        x[i] = 0.85 * x[i - 1] + rng.normal(0, 0.4)
    return np.exp(x)


def honest_forecast(rv):
    """Прогноз строки i известен на закрытии d_i: функция RV строки i−1 (последний известный день = цель строки i−1) и RV_{i−2}."""
    f = np.full(len(rv), np.nan)
    f[2:] = np.exp(0.8 * np.log(rv[1:-1]) + 0.1 * np.log(rv[:-2]))
    return f


def test_shift_profile_peaks_at_last_known_day_for_honest_forecast():
    rv = ar_series()
    f = honest_forecast(rv)
    p = AU.shift_profile(rv, f)
    assert max(p, key=p.get) == AU.LAST_KNOWN and AU.shift_ok(rv, f)


def test_shift_plus_one_synthetic_is_caught():
    rv = ar_series()
    f = honest_forecast(rv)
    leak = np.roll(f, -1)
    leak[-1] = np.nan
    assert not AU.shift_ok(rv, leak)


@pytest.mark.parametrize("t", ["SBER", "AFLT", "VTBR"])
def test_real_ticker_audit_ok(t):
    r = AU.audit_align(t)
    assert r["align_ok"] and r["oof_k2k3"] and r["status"] == "OOF-пригоден"
    assert r["rv_vs_daily_h"] == r["rv_vs_daily_k23"] == r["rv_vs_daily_d2"] == r["rv_vs_daily_eng"] == 0


def test_mutation_m_shifted_plus_one_day_is_caught_by_real_path(monkeypatch):
    """M (K2, K3) сдвинут на +1 день: значение дня d+1 записано на день d — боевой выровненный ряд и проверка сдвига это ловят."""
    orig = IO.load_k23

    def shifted(t):
        k = orig(t)
        k["k2"], k["k3"] = k["k2"].shift(-1), k["k3"].shift(-1)
        return k
    monkeypatch.setattr(IO, "load_k23", shifted)
    r = AU.audit_align("SBER")
    assert not r["shift_ok_k2"] and not r["shift_ok_k3"] and not r["align_ok"] and r["status"] != "OOF-пригоден"


def test_mutation_target_rv_other_definition_is_caught(monkeypatch):
    """Цель RV другого определения (колонка bv вместо rv) в ряде K1 — расходится с независимым дневным источником и с K2/K3."""
    orig, daily = IO.load_h, IO.load_daily("SBER")

    def other(t):
        h = orig(t)
        h["rv"] = daily["bv"].reindex(pd.DatetimeIndex(h["t_plus"])).to_numpy(float)
        return h
    monkeypatch.setattr(IO, "load_h", other)
    r = AU.audit_align("SBER")
    assert r["rv_vs_daily_h"] > 0 and r["rv_diff_h_k23"] > 0 and not r["align_ok"]


def test_mutation_tplus_not_next_day_is_caught(monkeypatch):
    orig = IO.load_k23

    def bad(t):
        k = orig(t)
        k["t_plus"] = k["t_plus"] + pd.Timedelta(days=1)
        return k
    monkeypatch.setattr(IO, "load_k23", bad)
    r = AU.audit_align("SBER")
    assert r["tplus_not_next_k23"] > 0 and not r["align_ok"]


def test_inventory_sha_matches_file_and_covers_four_models():
    rows = AU.inventory_rows("SBER")
    assert [r["model"] for r in rows] == ["K1", "K2", "K3", "LGBM-D2", "H∩K2∩K3"]
    p = IO.path_of("H", "SBER")
    assert rows[0]["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert rows[-1]["n"] <= min(r["n"] for r in rows[:3])


def test_oof_code_checks_pass():
    assert all(OC.run().values())


def test_mutation_pairs_with_future_target_in_k1_are_caught(monkeypatch):
    orig = har.pairs
    monkeypatch.setattr(har, "pairs", lambda p, i, cols, upto=None: orig(p, i, cols, upto=i + 1))
    assert not OC.check_k1()


def test_mutation_d2_pairs_closed_plus_one_is_caught(monkeypatch):
    monkeypatch.setattr(E, "pair_closed", lambda t_plus, d: t_plus <= d + np.timedelta64(1, "D"))
    assert not OC.check_d2()


def test_mutation_d2_choice_closed_plus_one_is_caught(monkeypatch):
    monkeypatch.setattr(E, "closed_mask", lambda date, t_plus, Y: pd.DatetimeIndex(date).year == Y - 1)
    assert not OC.check_closed_choice()


def test_sber_benchmark_reproduced_and_comparable():
    t, o = B.run()
    assert o["repro_ok"] and o["rv_def_same"] and o["xj_vs_k2_maxdiff"] == 0.0 and abs(t.iloc[0]["qlike"] - 0.20814004386759064) < 1e-9


def test_mutation_bench_wrong_series_and_other_rv_are_caught():
    ref = B.load_ref()
    wrong = ref.copy()
    wrong[B.SERIES] = wrong["HAR"]
    assert not B.bench_rows(wrong, IO.aligned("SBER"))[1]["repro_ok"]
    other = ref.copy()
    other["rv"] = other["rv"] * 1.001
    assert not B.bench_rows(other, IO.aligned("SBER"))[1]["rv_def_same"]
