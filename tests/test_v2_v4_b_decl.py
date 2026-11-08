"""V4-B, G9/G11: порог «go» только по 2017–2021 (мутация: с 2022+), N_wide и время объявления, ключ кэша (Д5), правило B5 (мутация: B5 без правила)."""
import json

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_b_cache as CA, v4_b_data as D, v4_b_decl as DECL, v4_b_rule as RU, v4_io as IO


def frame(n_days=1700, tickers=("AFLT", "ALRS", "HYDR"), seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-12-30", periods=n_days)
    fr = []
    for t in tickers:
        k2 = np.exp(rng.normal(0, 0.4, n_days))
        f = k2 * np.exp(rng.normal(0, 0.15, n_days))
        fr.append(pd.DataFrame({"ticker": t, "date": dates, "t_plus": list(dates[1:]) + [dates[-1] + pd.Timedelta(days=1)], "rv": f * np.exp(rng.normal(-0.03, 0.3, n_days)), "h": f,
                                "k2": k2, "k3": k2 * 1.05, "d2": k2 * 1.1}))
    return D.eval_frame(pd.concat(fr, ignore_index=True))


def test_threshold_is_min_minus_margin_known_values():
    assert DECL.go_threshold({"k1": 0.3, "k2": 0.25, "k3": 0.26, "d2": 0.28}) == 0.25 - 0.0005


def future_blind(k) -> bool:
    """Порог не зависит от строк с целью 2022+ (подмена RV там его не меняет)."""
    k2 = k.copy()
    k2.loc[k2["t_plus"].dt.year >= 2022, "rv_tplus"] *= 5.0
    return abs(DECL.go_threshold(DECL.controls(k)) - DECL.go_threshold(DECL.controls(k2))) < 1e-15


def test_threshold_uses_only_2017_2021_targets():
    assert future_blind(frame())


def test_mutation_threshold_with_2022_plus_is_caught(monkeypatch):
    monkeypatch.setattr(S, "CRIT_YEARS", (2017, 2026))
    assert not future_blind(frame())


def test_controls_equal_crit_panel_function_not_own_mean():
    k = frame()
    ct = DECL.controls(k)
    assert ct == {c: float(S.crit_panel(k, DECL.CTRL).set_index("series").loc[c, "crit_panel"]) for c in DECL.CTRL}


def test_n_wide_counts_unique_trials_and_rule_text(monkeypatch, tmp_path):
    (tmp_path / "data/regen").mkdir(parents=True)
    pd.DataFrame({"model": ["lgbm"] * 4, "h": 1, "step": 6, "trial": [0, 1, 1, 2]}).to_csv(tmp_path / "data/regen/X_h1_trials.csv", index=False)
    pd.DataFrame({"model": ["lgbm"] * 2, "h": 1, "step": 1, "point": [0, 1]}).to_csv(tmp_path / "data/regen/Y_h1_trials.csv", index=False)
    monkeypatch.setattr(IO.P, "ROOT", tmp_path)
    n, parts = DECL.optuna_trials()
    assert n == 5 and parts == {"X_h1_trials.csv": 3, "Y_h1_trials.csv": 2} and "верхняя оценка" in DECL.N_WIDE_RULE


def test_declared_before_flags_files_not_newer_than_declaration(monkeypatch, tmp_path):
    cp = tmp_path / "CHECKPOINT.json"
    cp.write_text(json.dumps({"B_declared_at": "2026-01-01 00:00:00.000000"}))
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    old.write_text("x")
    new.write_text("x")
    import os
    os.utime(old, (1.0, 1.0))
    monkeypatch.setattr(DECL, "CP", cp)
    assert DECL.declared_before([old, new]) == [str(old)]


def test_cache_key_changes_with_any_input_or_code(tmp_path):
    a, b, c = tmp_path / "a.parquet", tmp_path / "b.parquet", tmp_path / "c.py"
    a.write_text("1")
    b.write_text("2")
    c.write_text("code")
    k0 = CA.keys([a, b], [c])["key"]
    a.write_text("1x")
    assert CA.keys([a, b], [c])["key"] != k0
    a.write_text("1")
    assert CA.keys([a, b], [c])["key"] == k0
    c.write_text("code2")
    assert CA.keys([a, b], [c])["key"] != k0


def rule_frame(c_bias, h_shift, m_shift, seed=2):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=1000)
    fr = []
    for t in IO.M16[:4]:
        F = np.exp(rng.normal(0, 0.3, 1000))
        rv = F * c_bias * np.exp(rng.normal(-0.045, 0.3, 1000))
        fr.append(pd.DataFrame({"ticker": t, "date": dates, "t_plus": dates, "rv": rv, "F": F, "h": rv * np.exp(h_shift), "k2": rv * np.exp(m_shift), "k3": rv * np.exp(m_shift)}))
    return pd.concat(fr, ignore_index=True)


def test_rule_passes_only_with_c_off_one_and_same_signs():
    ok = RU.rule_stats(rule_frame(1.3, -0.2, -0.1))            # RV > H, RV > M, c ≈ 1,3
    assert ok["c_differs"] and ok["same_sign"] and RU.rule_passes(ok) and "B5" in RU.cfgs_to_run(ok)
    unbiased = RU.rule_stats(rule_frame(1.0, -0.2, -0.1))
    assert not unbiased["c_differs"] and not RU.rule_passes(unbiased) and RU.cfgs_to_run(unbiased) == ["B1", "B2", "B3", "B4", "B6"]
    opposite = RU.rule_stats(rule_frame(1.3, -0.2, 0.1))
    assert opposite["c_differs"] and not opposite["same_sign"] and not RU.rule_passes(opposite)


def test_mutation_b5_without_rule_is_caught(monkeypatch):
    st = RU.rule_stats(rule_frame(1.0, -0.2, -0.1))
    assert RU.consistent(st, RU.cfgs_to_run(st))
    monkeypatch.setattr(RU, "rule_passes", lambda s: True)         # B5 считается без правила
    assert "B5" in RU.cfgs_to_run(st) and not RU.consistent(st, RU.cfgs_to_run(st))
    monkeypatch.undo()
    assert not RU.consistent(st, ["B1", "B2", "B3", "B4", "B5", "B6"])
