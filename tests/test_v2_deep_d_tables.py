"""G7/G10 D: таблицы волны D (состав, реестр, Холм, выбор лучшего ансамбля) по готовым файлам папки отчёта; без сети, без счёта."""
import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s5 import common
from models_v2.s5.c import space
from models_v2.s5.d import ens, select_d, stats_d, tab_d

REP = common.REP
ENS = REP / "ENSEMBLE_RESULTS.csv"
pytestmark = pytest.mark.skipif(not ENS.exists(), reason="стадия d1 не выполнена")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(ENS, comment="#")


def test_composition_26_rows_plus_e2_refs(df):
    d, ref = df[df["source"] == "D"], df[df["source"] != "D"]
    assert len(d) == 26 and d["row_id"].is_unique
    assert set(d["row_id"]) == set(tab_d.COMPONENTS) | set(ens.SPEC)
    assert set(ref["row_id"]) == {"c2_h3_roll_f4", "c2_h3_exp_f4"}
    assert sum(r.startswith("mean_") for r in d["row_id"]) == 10 and len(tab_d.D_TRIALS) == 16


def test_e2_refs_equal_c_all(df):
    c = pd.read_csv(REP / "C_ALL.csv", comment="#").set_index("row_id")
    for _, r in df[df["source"] != "D"].iterrows():
        assert abs(r["qlike_all"] - c.loc[r["row_id"], "qlike_all"]) < 1e-12


def test_trials_d_16_unique_and_within_budget():
    t = pd.read_csv(tab_d.TRIALS_D, dtype=str)
    assert len(t) == 16 and t["trial_id"].is_unique and len(t) <= tab_d.LIM_D
    assert set(t["направление"]) == {tab_d.DIRECTION} and set(t["волна"]) == {"D"}
    cfg = json.loads(t[t["конфигурация"].str.contains('"e4_roll"')]["конфигурация"].iloc[0])
    assert space.trial_id(cfg) in set(t["trial_id"])


def test_controls_are_not_trials_and_not_candidates(df):
    t = set(pd.read_csv(tab_d.TRIALS_D, dtype=str)["trial_id"])
    d = df[df["source"] == "D"]
    ctl = d[d["row_id"].str.startswith("mean_")]
    assert not (ctl["trial_id"].fillna("").isin(t)).any()
    best = select_d.best_ensemble(d, "q_int")
    assert best in ens.TRIALS and not best.startswith("mean_")
    assert d.set_index("row_id").loc[best, "q_int"] == d[d["row_id"].isin(ens.TRIALS)]["q_int"].min()


def test_holm_columns_monotone_and_grow_with_n(df):
    d = df[df["source"] == "D"]
    for c in ("p_holm_harj", "p_holm_b2"):
        assert (d[c] >= d[c.replace("holm_", "")] - 1e-12).all()
        assert (d[c + "_ref116"] >= d[c] - 1e-12).all()


def test_mutation_holm_n_one_is_caught(df):
    p = df[df["source"] == "D"]["p_b2"].to_numpy(float)
    ok, mut = stats_d.holm_n(p, tab_d.FAMILY), stats_d.holm_n(p, len(p))   # мутант: N не растёт с числом trials
    assert np.all(mut <= ok + 1e-12) and (mut < ok - 1e-6).any()


def test_registry_after_report_if_done():
    ck = json.loads((REP / "CHECKPOINT.json").read_text())
    if ck["waves"].get("D") != "done":
        pytest.skip("report не выполнен")
    allr = pd.read_csv(REP / "OPTUNA_TRIALS.csv", dtype=str)
    assert len(allr) == tab_d.FAMILY == 106 and allr["trial_id"].is_unique
    assert ck["trials_used"]["ensemble_calibration"] == 16 and sum(ck["trials_used"].values()) <= ck["trials_limit_total"]
    assert (allr["направление"] == tab_d.DIRECTION).sum() == 16
