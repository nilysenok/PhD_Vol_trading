"""V6-3, G32–G35: правило E3 (автоматически: (E1 или E2 против A0 несёт новую информацию) и (A1 или A2 против A0 — по A_INFO)), реестр конфигураций (E1–E3 в списке плана, строки дописываются после A0–A3, C1–C4),
дизайн E1/E2 = S + R, E3 = S + R + Σ, изоляция (только SBER), регрессия: движок листов 1a–2 не менялся (побитно прежние прогнозы A0–A3, C1–C4 на диске — SHA в CHECKPOINT 1b/2)."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s13 import v6_0_cp as CP0, v6_0_io as IO, v6_1_reg as RG, v6_1_stage as S1, v6_1b_tab as TB, v6_3_stage as ST

CP = {"N": 722, "N_wide": 3122}


def synth(n: int = 1500, seed: int = 5):
    rng = np.random.default_rng(seed)
    tp = pd.bdate_range("2017-01-02", periods=n)
    mu = rng.normal(0, 0.3, n)
    rv = np.exp(mu + rng.normal(0, 0.5, n))
    w = pd.DataFrame({"rv": rv, "t_plus": tp, "K1": np.exp(mu) * 1.3, "K3": np.exp(mu) * 1.1}, index=tp - pd.offsets.BDay(1))
    return w, np.exp(mu + 0.125)                                  # точная условная средняя E[RV] = exp(μ + σ²/2)


def infos(w, fs):
    ie = TB.info_table(fs, [("E1", "A0"), ("E2", "A0"), ("E2", "E1")], w, CP)
    ia = TB.info_table(fs, [("A1", "A0"), ("A2", "A0")], w, CP)
    return ie, ia


def test_decide_e3_follows_rule_against_a0_both_parts():
    w, best = synth()
    bad = best * np.exp(0.4)
    for fs, want in (({"A0": bad, "E1": best, "E2": bad, "A1": bad, "A2": best}, True),         # E1 и A2 несут новую информацию против A0
                     ({"A0": bad, "E1": bad, "E2": best, "A1": best, "A2": bad}, True),         # E2 и A1
                     ({"A0": bad, "E1": bad, "E2": bad, "A1": best, "A2": best}, False),        # E не несёт: E3 не считается
                     ({"A0": bad, "E1": best, "E2": best, "A1": bad, "A2": bad}, False)):       # A не несёт: E3 не считается
        ie, ia = infos(w, fs)
        assert ST.decide_e3(ie, ia)["compute"] is want
    worse = best * np.exp(0.2)
    fs = {"A0": best, "E1": bad, "E2": worse, "A1": best, "A2": best}                           # E2 лучше E1, но хуже A0: правило — против A0, не против E1
    ie, ia = infos(w, fs)
    mut = ie[(ie["window"] == "2017_2021") & (ie["pair"] == "E2 vs E1")].iloc[0]
    assert bool(mut["new_information"]) is True and ST.decide_e3(ie, ia)["compute"] is False   # МУТАЦИЯ «E2 против E1 вместо A0» дала бы True


def test_decide_e3_reads_flags_from_csv_strings():
    w, best = synth()
    bad = best * np.exp(0.4)
    ie, ia = infos(w, {"A0": bad, "E1": best, "E2": bad, "A1": bad, "A2": best})
    ia = ia.assign(new_information=ia["new_information"].map(lambda v: str(v)))                  # CSV хранит True/False как текст
    assert ST.decide_e3(ie, ia)["compute"] is True and ST.flag("False") is False and ST.flag(np.True_) is True


def test_design_columns_s_r_and_sigma():
    idx = pd.bdate_range("2017-01-02", periods=5)
    S = pd.DataFrame(np.zeros((5, 2)), index=idx, columns=["s1", "s2"])
    parts = {"R": pd.DataFrame(np.ones((5, 2)), index=idx, columns=["rho1", "rho3"]), "sigma": pd.DataFrame(np.full((5, 1), 2.0), index=idx, columns=["a_PV"])}
    assert list(ST.design("E1", S, parts).columns) == ["s1", "s2", "rho1", "rho3"] == list(ST.design("E2", S, parts).columns)
    assert list(ST.design("E3", S, parts).columns) == ["s1", "s2", "rho1", "rho3", "a_PV"]
    assert ST.LEARNER == {"E1": "LIN", "E2": "GBM", "E3": "LIN"}


def test_registry_has_e_rows_and_trials_prefix_rule(tmp_path, monkeypatch):
    for c in ("E1", "E2", "E3"):
        assert c in RG.CONFIGS and c in RG.DECLARED
    f = tmp_path / "ALL_TRIALS.csv"
    monkeypatch.setattr(CP0, "ALL_TRIALS", f)
    prior = ["A0", "A1", "A2", "A3", "C1", "C2", "C3", "C4"]
    f.write_text(RG.trial_text(prior + ["E1", "E2"]), encoding="utf-8")
    assert RG.trials_ok(prior) and RG.trials_ok(prior + ["E1", "E2"], exact=True) and not RG.trials_ok(prior, exact=True)
    f.write_text(RG.trial_text(prior) + RG.trial_text(["E1"])[len(RG.trial_text([])):] * 2, encoding="utf-8")        # дубль строки E1 — недопустим
    assert not RG.trials_ok(prior)
    f.write_text(RG.trial_text(prior[:-1] + ["E1"]), encoding="utf-8")
    assert not RG.trials_ok(prior)


def test_leaf3_code_reads_only_sber_and_no_other_tickers():
    for p in sorted(Path(ST.__file__).parent.glob("v6_3_*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        lits = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert not (lits & {"GAZP", "LKOH", "GMKN", "IMOEX", "VTBR", "ROSN"}), p.name


def test_real_stage3_outputs_when_written():
    cp = CP0.read().get("3")
    if cp is None:
        pytest.skip("стадия 3 ещё не записана")
    assert cp["used"] in (10, 11) and cp["N"] == 722 and cp["N_wide"] == 3122 and cp["code_hash_v6_3"] == ST.code_hash()
    assert ("E3" in cp["configs"]) == (cp["E3"] == "посчитана по правилу") and (cp["used"] == 11) == ("E3" in cp["configs"])
    assert set(cp["r_kept"]) | set(cp["r_merged_with_sigma"]) == {"rho%d" % i for i in range(1, 7)}


def test_earlier_forecasts_on_disk_equal_checkpoint_sha():
    cp1b, cp2 = CP0.read().get("1b"), CP0.read().get("2")
    if cp1b is None or cp2 is None:
        pytest.skip("стадии 1b/2 ещё не записаны")
    for cp in (cp1b, cp2):
        for c, sha in cp["f_sha"].items():
            f = pd.read_parquet(IO.OUT / (c + ".parquet"))["F"].to_numpy()
            assert S1.sha_arr(f) == sha, c
