"""B1-8 (Ж-204): таблицы B1 — колонки, пересчёт QLIKE из прогнозов, правило «перспективно», реестр trials. Без выходов счёта тесты пропускаются."""
import json

import numpy as np
import pandas as pd
import pytest

from models_v2 import metrics
from models_v2.s5 import common
from models_v2.s5.b1 import configs, tables, world

NEED = ["trial_id", "row_id", "direction", "q_int_kind", "q_int", "q_int_b2", "q_wf", "perspective", "qlike_all", "qlike_2017_2021", "qlike_2022",
        "qlike_2023_2025", "qlike_2026", "qlike_no_240222", "qlike_no_2022", "mean_r", "d_harj", "dm_harj", "p_harj", "d_harj_no_240222",
        "dm_harj_no_240222", "d_b2", "dm_b2", "p_b2", "d_b2_no_240222", "share_years_win_harj"]
TABS = {d: common.REP / f for d, f in configs.TABLE.items()}
pytestmark = pytest.mark.skipif(not all(p.exists() for p in TABS.values()), reason="счёт B1 не выполнен")


def b1_ids():
    """Строки B1 = контроли + row_id из реестра B1 (OPTUNA_TRIALS_B1.csv); строки C в этих таблицах проверяет tests/test_v2_deep_c_tables.py."""
    reg = pd.read_csv(tables.REG_B1, dtype=str)
    return set(tables.CONTROLS) | {json.loads(x)["row_id"] for x in reg["конфигурация"]}


def load(dirn):
    df = common.read_comment_csv(TABS[dirn])
    return df[df.row_id.isin(b1_ids())].reset_index(drop=True)


def fc(rid):
    return pd.read_parquet(world.B1_DIR / ("fc_%s.parquet" % rid))


@pytest.mark.parametrize("dirn", list(configs.TABLE))
def test_columns_controls_and_rows(dirn):
    df = load(dirn)
    assert set(NEED) <= set(df.columns)
    want = {"ctrl_b2", "ctrl_harj", "ctrl_b1"} | {r["row_id"] for r in configs.trials() if r["direction"] == dirn}
    assert set(df.row_id) == want and df.row_id.is_unique


def test_controls_values_and_grid():
    c = common.controls()
    df = load(configs.HAR).set_index("row_id")
    assert round(df.loc["ctrl_b2", "qlike_all"], 4) == 0.2205 and round(df.loc["ctrl_harj", "qlike_all"], 4) == 0.2337
    for rid in df.index:
        f = fc(rid)
        assert len(f) == common.N_GRID == len(c)
        assert (f["d"].to_numpy("datetime64[ns]") == c["d"].to_numpy("datetime64[ns]")).all()
        assert np.isfinite(f["sigma2"]).all() and (f["sigma2"] > 0).all()


def test_qlike_recomputed_from_forecasts():
    for dirn in configs.TABLE:
        for r in load(dirn).itertuples():
            f = fc(r.row_id)
            q = metrics.qlike(f["rv"], f["sigma2"])
            assert abs(q.mean() - r.qlike_all) < 1e-12
            keep = f["target_date"] != pd.Timestamp("2022-02-24")
            assert abs(q[keep].mean() - r.qlike_no_240222) < 1e-12 and keep.sum() == len(f) - 1


def test_perspective_flag_equals_rule():
    for dirn in configs.TABLE:
        df = load(dirn)
        df = df[df.direction == dirn]
        rule = (df.q_int_b2 - df.q_int > 1e-12) & (df.d_harj_no_240222 <= 0)
        assert (df.perspective.astype(bool).to_numpy() == rule.to_numpy()).all()


def test_registry_28_unique_and_within_limits():
    r = pd.read_csv(tables.REG_B1)
    assert len(r) == 28 and r["trial_id"].is_unique
    assert set(r["trial_id"]) == {configs.trial_id(x) for x in configs.trials()}
    cnt = r.groupby("направление").size().to_dict()
    assert all(cnt[k] <= configs.LIMITS[k] for k in cnt)
