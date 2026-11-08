"""V4-C: ключ кэша Z и C = хэш всех входов и кода (Д5) — подмена входа или кода даёт промах; сводки листа (шаги по блокам, вес по годам, Холм с N) на известном ответе."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_c_cache as CC, v4_c_stats as CS


def test_key_changes_when_input_or_code_changes(tmp_path):
    a, code = tmp_path / "in.parquet", tmp_path / "v4_z_x.py"
    a.write_bytes(b"1")
    code.write_text("x = 1")
    k0 = CC.keys("Z", extra_in=[a], extra_code=[code])
    a.write_bytes(b"2")
    assert CC.keys("Z", extra_in=[a], extra_code=[code])["key"] != k0["key"]            # подмена входа
    a.write_bytes(b"1")
    assert CC.keys("Z", extra_in=[a], extra_code=[code])["key"] == k0["key"]
    code.write_text("x = 2")
    assert CC.keys("Z", extra_in=[a], extra_code=[code])["key"] != k0["key"]            # правка кода


def test_cached_misses_on_other_key_and_hits_on_same(tmp_path, monkeypatch):
    monkeypatch.setattr(CC, "OUT", tmp_path)
    calls = []

    def build():
        calls.append(1)
        return {"": pd.DataFrame({"a": [len(calls)]})}
    assert CC.cached("C", "C1", "k1", build)[""]["a"][0] == 1
    assert CC.cached("C", "C1", "k1", build)[""]["a"][0] == 1 and len(calls) == 1       # тот же ключ — попадание
    assert CC.cached("C", "C1", "k2", build)[""]["a"][0] == 2                            # другой ключ — промах


def test_real_keys_include_new_modules_and_hyperparameters():
    names = {p.name for p in CC.c_code()}
    assert {"v4_c_fit.py", "v4_c_run.py", "v4_z_feat.py", "v4_z_shape.py", "v4_b_run.py", "v4_b_fit.py"} <= names and "v4_c_stage.py" in names
    assert {p.name for p in CC.z_inputs()} >= {"har_SBER.parquet", "k23_SBER.parquet", "daily_SBER.parquet", "SBER.parquet", "hyb_lgbm_p17.parquet"}


def test_block_steps_known_answer():
    crit = pd.Series({"k1": 0.2, "C1": 0.21, "C2": 0.2, "C3": 0.19, "C4": 0.18})
    s = CS.block_steps(crit)
    assert np.allclose(s["d_crit"], [0.01, -0.01, -0.01, -0.01]) and np.allclose(s["d_crit_pct_of_k1"], [5.0, -5.0, -5.0, -5.0])


def test_weight_by_year_known_answer():
    fc = pd.DataFrame({"date": pd.to_datetime(["2018-01-02", "2018-06-01", "2019-01-02"]), "w": [0.2, 0.4, 0.9]})
    assert CS.weight_by_year(fc).to_dict() == pytest.approx({2018: 0.3, 2019: 0.9})


def test_holm_wide_not_below_holm_and_mutation_n_one_caught():
    """p_holm_wide (N = 3094) ≥ p_holm (N = 694) ≥ p без поправки; мутация N → 1 снимает поправку (p_holm = dm_p) — тест ловит."""
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.bdate_range("2022-01-03", periods=n)
    rows = []
    for t in ["AFLT", "ALRS", "HYDR", "IRAO"]:
        base = np.exp(rng.normal(0, 0.3, n))
        rv = base * rng.exponential(1.0, n)
        rows.append(pd.DataFrame({"ticker": t, "group": "M16", "date": dates, "t_plus": dates + pd.Timedelta(days=1), "rv_tplus": rv, "k1": base, "k2": base * 1.1, "k3": base * 1.1, "d2": base * 0.9,
                                  "B6": base * 1.05, "C1": base * 0.95}))
    c = pd.concat(rows, ignore_index=True)
    t = CS.tests(c, ["C1"], 694, 3094)
    m = t[t["kind"] == "main"].dropna(subset=["p_holm"])
    assert (m["p_holm_wide"] >= m["p_holm"] - 1e-15).all() and (m["p_holm"] >= m["dm_p"] - 1e-15).all() and set(t["comparator"]) == set(CS.COMPS)
    t10 = CS.tests(c, ["C1"], 10, 10)                                                    # N = числу строк таблицы (мутация Холма, № 103): поправка слабее, p_holm не выше
    assert (t10.dropna(subset=["p_holm"])["p_holm"].to_numpy() <= m["p_holm"].to_numpy() + 1e-15).all()
    with pytest.raises(ValueError):                                                      # N меньше числа проверок — отказ, а не молчаливое занижение
        CS.tests(c, ["C1"], 1, 1)
