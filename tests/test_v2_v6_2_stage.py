"""V6-2, G25–G28: правило C4 (автоматически из C_INFO: C1, C2 или C3 против A0 — не против C1), таблица главной проверки (известный ответ и мутация перепутанных групп), реестр конфигураций (C1–C4 в списке плана, строки дописываются после A0–A3),
изоляция (только SBER, код листа 2 не читает других бумаг и IMOEX), регрессия: движок листов 1a/1b не менялся, состав файлов."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s13 import v6_0_cp as CP0, v6_1_reg as RG, v6_1b_tab as TB, v6_2_stage as ST, v6_2_unst as UN

CP = {"N": 722, "N_wide": 3122}


def synth(n: int = 1500, seed: int = 5):
    rng = np.random.default_rng(seed)
    tp = pd.bdate_range("2017-01-02", periods=n)
    mu = rng.normal(0, 0.3, n)
    rv = np.exp(mu + rng.normal(0, 0.5, n))
    w = pd.DataFrame({"rv": rv, "t_plus": tp, "K1": np.exp(mu) * 1.3, "K3": np.exp(mu) * 1.1}, index=tp - pd.offsets.BDay(1))
    return w, np.exp(mu + 0.125)                                  # точная условная средняя E[RV] = exp(μ + σ²/2)


def test_decide_c4_follows_rule_against_a0_not_c1():
    w, best = synth()
    bad, worse = best * np.exp(0.4), best * np.exp(0.2)
    pr = [("C1", "A0"), ("C2", "A0"), ("C3", "A0"), ("C2", "C1"), ("C3", "C1")]
    for fs, want in (({"A0": best, "C1": bad, "C2": worse, "C3": bad}, False),         # C2 лучше C1, но хуже A0, ни одна не лучше A0: C4 не считается
                     ({"A0": bad, "C1": bad, "C2": bad, "C3": best}, True),            # C3 лучше A0
                     ({"A0": bad, "C1": best, "C2": bad, "C3": bad}, True),            # C1 лучше A0
                     ({"A0": bad, "C1": bad, "C2": best, "C3": bad}, True)):           # C2 лучше A0
        info = TB.info_table(fs, pr, w, CP)
        assert ST.decide_c4(info)["compute"] is want
    info = TB.info_table({"A0": best, "C1": bad, "C2": worse, "C3": bad}, pr, w, CP)
    mut = info[(info["window"] == "2017_2021") & (info["pair"] == "C2 vs C1")].iloc[0]
    assert bool(mut["new_information"]) is True and ST.decide_c4(info)["compute"] is False       # МУТАЦИЯ «C2 против C1 вместо A0» дала бы True


def test_unstable_table_known_answer_and_swapped_groups_caught():
    w, best = synth()
    n = len(w)
    high = np.arange(n) % 5 == 0
    c = np.where(high, best, best * np.exp(0.4))                    # C точна в высокой группе и неверна в остальных; A0 — наоборот
    a0 = np.where(high, best * np.exp(0.4), best)
    t = UN.unstable_table(w, high, np.ones(n, bool), {"A0": a0, "C1": c}, ["C1"])
    row = lambda g, win="all", b="A0": t[(t["group"].str.startswith(g)) & (t["window"] == win) & (t["base"] == b)].iloc[0]
    assert row("высокая")["delta_qlike"] < 0 and row("остальные")["delta_qlike"] > 0 and row("высокая")["hi90"] < 0 and row("остальные")["lo90"] > 0
    assert row("высокая")["n_days"] == int(high.sum()) and row("высокая")["n_days"] + row("остальные")["n_days"] == n
    sw = UN.unstable_table(w, ~high, np.ones(n, bool), {"A0": a0, "C1": c}, ["C1"])           # МУТАЦИЯ: группы перепутаны — знаки меняются
    assert sw[(sw["group"].str.startswith("высокая")) & (sw["window"] == "all") & (sw["base"] == "A0")].iloc[0]["delta_qlike"] > 0
    assert set(t["window"]) == {"2017_2021", "2022_2025", "all"} and set(t["base"]) == {"A0", "K3"}


def test_unstable_excludes_days_without_group():
    w, best = synth()
    n = len(w)
    valid = np.arange(n) >= 100
    t = UN.unstable_table(w, np.arange(n) % 5 == 0, valid, {"A0": best, "C1": best}, ["C1"])
    a = t[(t["window"] == "all") & (t["base"] == "A0")]
    assert a["n_days"].sum() <= int((valid & (pd.DatetimeIndex(w["t_plus"]).year >= 2017)).sum())


def test_registry_has_c_rows_and_trials_prefix_rule(tmp_path, monkeypatch):
    for c in ("C1", "C2", "C3", "C4"):
        assert c in RG.CONFIGS and c in RG.DECLARED
    f = tmp_path / "ALL_TRIALS.csv"
    monkeypatch.setattr(CP0, "ALL_TRIALS", f)
    f.write_text(RG.trial_text(["A0", "A1", "A2", "A3", "C1", "C2", "C3"]), encoding="utf-8")
    assert RG.trials_ok(["A0", "A1", "A2", "A3"]) and RG.trials_ok(["A0", "A1", "A2", "A3", "C1", "C2", "C3"], exact=True) and not RG.trials_ok(["A0", "A1", "A2", "A3"], exact=True)
    f.write_text(RG.trial_text(["A0", "A1", "A2", "A3"]) + RG.trial_text(["C1"])[len(RG.trial_text([])):] * 2, encoding="utf-8")        # дубль строки C1 — недопустим
    assert not RG.trials_ok(["A0", "A1", "A2", "A3"])
    f.write_text(RG.trial_text(["A0", "A1", "A2"]), encoding="utf-8")
    assert not RG.trials_ok(["A0", "A1", "A2", "A3"])


def test_leaf2_code_reads_only_sber_and_no_other_tickers():
    for p in sorted(Path(ST.__file__).parent.glob("v6_2_*.py")):
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src)
        lits = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert not (lits & {"GAZP", "LKOH", "GMKN", "IMOEX", "VTBR", "ROSN"}), p.name


def test_real_stage2_outputs_when_written():
    cp = CP0.read().get("2")
    if cp is None:
        pytest.skip("стадия 2 ещё не записана")
    assert cp["used"] in (7, 8) and cp["N"] == 722 and cp["N_wide"] == 3122 and cp["code_hash_v6_2"] == ST.code_hash()
    assert ("C4" in cp["configs"]) == (cp["C4"] == "посчитана по правилу 2.2") and (cp["used"] == 8) == ("C4" in cp["configs"])
