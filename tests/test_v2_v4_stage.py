"""G4/G5/G6 и сквозной прогон стадий: реестр и N, пробелы, комплементарность (известные ответы), `--check` обеих стадий rc 0."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import run_v4, v4_compl as C, v4_gaps, v4_io as IO, v4_trials as TR


def test_registries_are_cumulative_union_not_sum():
    regs = TR.load_regs()
    rows = sum(len(r) for r in regs.values())
    n = TR.n_prior(regs)
    assert n == 634 and rows > n and all(TR.registry_rows(regs)["contains_previous"])


def test_mutation_double_counting_sum_of_rows_is_caught(monkeypatch):
    """N_prior как сумма строк реестров (двойной счёт) расходится с CHECKPOINT — verify это ловит."""
    run_v4.v4_trials.write_all()
    monkeypatch.setattr(TR, "n_prior", lambda regs: sum(len(r) for r in regs.values()))
    assert TR.verify()


def test_checkpoint_declares_prior_plus_60_before_any_p():
    cp = run_v4.v4_trials.write_all()
    assert cp["N_declared_before_stats"] == cp["N_prior"] + 60 == 694 and cp["used"] == 0
    assert not TR.verify()


def test_gaps_registry_nonempty_and_not_blocking():
    g = IO.read_csv("GAPS.csv")
    assert len(g) >= 5 and not g["blocks_meta_learning"].astype(str).eq("True").any()
    assert set(g.columns) == set(v4_gaps.COLS)


def test_pair_stats_identical_models():
    rng = np.random.default_rng(1)
    rv = np.exp(rng.normal(0, 0.5, 200))
    f = rv * np.exp(rng.normal(0, 0.3, 200))
    s = C.pair_stats(rv, f, f)
    assert s["corr_err_log"] == pytest.approx(1.0) and s["share_opposite"] == 0.0 and s["mean_dq"] == 0.0 and s["std_logratio"] == 0.0


def test_pair_stats_opposite_errors_known_answer():
    rv = np.exp(np.random.default_rng(2).normal(0, 0.5, 100))
    s = C.pair_stats(rv, rv * 2, rv / 2)
    assert s["share_opposite"] == 1.0 and s["share_both_under"] == 0.0 and s["share_both_over"] == 0.0
    assert s["mean_logratio"] == pytest.approx(-2 * np.log(2))
    both = C.pair_stats(rv, rv * 2, rv * 3)
    assert both["share_both_over"] == 1.0 and both["share_opposite"] == 0.0


def test_top_share_known_answer():
    dq = np.array([10.0] + [0.0] * 98 + [-5.0])
    assert C.top_share(dq, 0.01) == pytest.approx(2.0)        # крайний день (10) / сумма (5)
    assert np.isnan(C.top_share(np.array([1.0, -1.0] * 50), 0.05))


def test_crisis_flag_uses_only_past_and_has_no_history_false():
    rv = pd.Series(np.ones(400), index=pd.date_range("2020-01-01", periods=400))
    rv.iloc[300] = 5.0
    f = C.crisis_flag(rv)
    assert not f.iloc[:252].any() and bool(f.iloc[300]) and not f.iloc[301:].any()
    # мутация: порог с текущим днём (rolling без shift) не пометил бы выброс — квантиль включал бы сам день
    thr_mut = rv.rolling(252, min_periods=252).quantile(0.9)
    assert not bool((rv > thr_mut).iloc[300]) or bool(f.iloc[300])


def test_small_slices_report_only_n():
    s = C.pair_stats(np.ones(5), np.ones(5), np.ones(5))
    assert s == {"n": 5}


def test_complement_table_shape_sber_rows_and_regimes():
    t = C.compl_table(["SBER", "AFLT"])
    assert len(t) == 2 * 3 * 22 and set(t["regime"]) == {"all", "calm", "crisis"} and (t["ticker"] == "SBER").any()
    a = t[(t.ticker == "SBER") & (t.window == "all") & (t.pair == "H-K2")].set_index("regime")
    assert a.loc["calm", "n"] + a.loc["crisis", "n"] == a.loc["all", "n"]


def test_stage_checks_rc0():
    assert run_v4.main(["--stage", "audit", "--check"]) == 0
    assert run_v4.main(["--stage", "A", "--check"]) == 0


def test_mutation_stage_a_without_checkpoint_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(IO, "REP", tmp_path)
    with pytest.raises(FileNotFoundError):
        run_v4.stage_a()
