"""G5 D: выбор лучшего ансамбля (MD11), тройка устойчивых (MD12), Q_wf_dev, LOYO по годам (MD14), топ-20, нижний дециль (MD13), бюджет (MD9)."""
import numpy as np
import pandas as pd
import pytest

from models_v2 import metrics
from models_v2.s5 import pit
from models_v2.s5.b1 import evalrows
from models_v2.s5.d import diag_d, ens, select_d, tab_d


def test_best_ensemble_uses_internal_criterion_not_description():
    df = pd.DataFrame({"row_id": ["e1_roll", "e4_roll", "e7_hx", "d_lgbm_f4"], "q_int": [0.19, 0.18, 0.20, 0.10], "qlike_all": [0.20, 0.25, 0.20, 0.10]})
    assert select_d.best_ensemble(df) == "e4_roll"                   # компонент (d_lgbm_f4) не кандидат, описание не используется (MD11)
    assert select_d.best_ensemble(df, "qlike_all") != "e4_roll"      # выбор по описанию дал бы другое — мутант отличим


def test_best_ensemble_tie_prefers_fewer_members_then_id():
    df = pd.DataFrame({"row_id": ["e4_roll", "e1_roll", "e1_exp"], "q_int": [0.18, 0.18 + 1e-13, 0.18 + 5e-13]})
    assert select_d.best_ensemble(df) == "e1_exp"
    with pytest.raises(ValueError):
        select_d.best_ensemble(pd.DataFrame({"row_id": ["d_lgbm_f4"], "q_int": [0.1]}))


def row(rid, q, **kw):
    r = dict(row_id=rid, q_int=q, q_int_b2=0.2, d_harj_no_240222=-0.01, perspective=True, seed_ok=True, plateau_ok=True, mcs_in=True)
    return {**r, **kw}


def test_top3_requires_all_flags_and_excludes_eval_selected():
    df = pd.DataFrame([row("a", 0.19), row("b", 0.18), row("c3_f3calm_rv", 0.10), row("c", 0.17, mcs_in=False), row("d", 0.17, seed_ok=False),
                       row("e", 0.17, plateau_ok=False), row("f", 0.17, perspective=False), row("g", 0.185), row("h", 0.186)])
    assert list(select_d.top3(df)["row_id"]) == ["b", "g", "h"]     # MD12: строка с выбором на оценочном периоде и без флагов не входит
    assert list(select_d.top3(df.iloc[[0]])["row_id"]) == ["a"] and select_d.top3(df.iloc[[2, 3]]).empty


def test_q_wf_dev_ignores_years_from_2022():
    td = pd.Series(pd.bdate_range("2017-01-02", "2026-02-20"))
    rng = np.random.default_rng(3)
    q = rng.random(len(td))
    q2 = q.copy()
    q2[(td.dt.year >= 2021).to_numpy()] += 5.0       # score_Y для Y <= 2021 использует цели <= 2020
    assert select_d.q_wf_dev(q, td) == select_d.q_wf_dev(q2, td) and evalrows.q_wf(q, td) != evalrows.q_wf(q2, td)


def test_loyo_years_excludes_each_year():
    td = pd.Series(pd.bdate_range("2017-01-02", "2026-02-20"))
    rng = np.random.default_rng(1)
    rv = np.exp(rng.normal(size=len(td)))
    c, h, b = rv * 1.1, rv * 1.5, rv * 1.2
    df = diag_d.loyo_years(rv, c, h, b, td)
    assert list(df.year_excluded) == list(range(2017, 2027))
    for r in df.itertuples():
        m = (td.dt.year != r.year_excluded).to_numpy()
        assert r.n == m.sum() and np.isclose(r.qlike, metrics.qlike(rv, c)[m].mean())   # MD14: год действительно исключён
    assert (df.n < len(td)).all()


def test_top20_known_answer():
    td = pd.Series(pd.bdate_range("2020-01-01", periods=300))
    rv = np.ones(300)
    base, cfg = np.ones(300) * 1.0, np.ones(300) * 1.0
    cfg[[5, 50, 100]] = [0.2, 3.0, 0.3]
    t = diag_d.top20(rv, cfg, base, td, k=3)
    assert set(t.target_date) == {td.iloc[i].strftime("%Y-%m-%d") for i in (5, 50, 100)} and np.isclose(t.attrs["sum_abs_share"], 1.0)
    assert np.isclose(t.attrs["sum_all"], t.attrs["sum_top"]) and t.attrs["mean_without_top"] == 0.0


def test_low_rv_mask_excludes_current_day_md13():
    rng = np.random.default_rng(2)
    x = np.exp(rng.normal(size=600))
    grid = np.arange(300, 600)
    ok = diag_d.low_rv_mask(x, grid)
    thr = pit.q_prior_nan(x, 0.1)
    assert np.array_equal(ok, x[grid] <= thr[grid]) and ok.any()
    x2 = x.copy()
    x2[450:] *= 10
    assert np.array_equal(diag_d.low_rv_mask(x2, grid)[:150], ok[:150])    # пороги до 450 не видят будущего
    k = 400
    x3 = x.copy()
    x3[k] *= 0.001
    assert diag_d.low_rv_thr(x3)[k] == diag_d.low_rv_thr(x)[k]               # порог дня k не зависит от x[k]
    assert diag_d.low_rv_thr(x3, include_current=True)[k] != diag_d.low_rv_thr(x, include_current=True)[k]   # мутант зависит — тест его различает


def test_low_rv_table_sign_and_counts():
    rv = np.array([1.0, 2.0, 3.0, 4.0])
    m = np.array([True, True, False, False])
    t = diag_d.low_rv_table(rv, {"harj": rv * 2, "b2": rv * 1.5, "cfg": rv * 1.0}, m).set_index("model")
    assert t.loc["cfg", "n"] == 2 and t.loc["cfg", "d_harj"] < 0 and t.loc["harj", "d_harj"] == 0 and np.isclose(t.loc["cfg", "mean_r"], 1.0)


def test_budget_md9_rejects_over_25_and_duplicate_ids():
    ok = [{"trial_id": "t%d" % i} for i in range(16)]
    assert tab_d.check_budget(ok, 0) is None
    with pytest.raises(SystemExit):
        tab_d.check_budget(ok + [{"trial_id": "x%d" % i} for i in range(10)], 0)      # 26 > 25
    with pytest.raises(SystemExit):
        tab_d.check_budget(ok + [{"trial_id": "t0"}], 0)                               # дубль id
    assert ens.TRIALS and len(tab_d.D_TRIALS) == 16
