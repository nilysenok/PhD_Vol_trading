"""V6-Z1 (Ж-230, G59 и пункт 5 задания): A0g в реестре ALL_TRIALS.csv и в N = 722 / N_wide = 3122; пороги и правила в коде равны объявленным в плане (коммит 99bd6681). Только чтение."""
from __future__ import annotations

import copy
import inspect
import subprocess

import pandas as pd
import pytest

from models_v2.s13 import v6_0_io as IO, v6_fin_asm as A, v6_fin_doc as W, v6_fin_z1 as Z1

T = W.load()
C0, C6 = T["cp"]["0"], T["cp"]["6"]


def test_a0g_row_in_registry_and_its_trial_is_inside_n_and_n_wide():
    tr = pd.read_csv(IO.REP / "ALL_TRIALS.csv")
    a = tr[tr["key"] == "Ж-227:A0g"]
    assert len(a) == 1 and a.iloc[0]["kind"] == "trial" and "ОТСТУПЛЕНИЕ" in a.iloc[0]["source"] and a.iloc[0]["direction"] == "A0g_ablation_deviation"
    assert len(tr) == C6["used"] == C0["budget_V6"] == 18 and "A0g" in C6["configs"] and "CB2" not in C6["configs"]
    assert (C0["N"], C0["N_wide"]) == (722, 3122) == (C0["N_V5_declared"] + len(tr), C0["N_wide_V5"] + len(tr))
    assert Z1.registry_fail(T) == []


def test_mutation_a0g_row_removed_or_trials_dropped_from_n_is_caught():
    t = dict(T)
    t["trials"] = T["trials"][T["trials"]["key"] != "Ж-227:A0g"]
    assert any("нет строки" in x for x in Z1.registry_fail(t))
    t = dict(T)
    t["cp"] = copy.deepcopy(T["cp"])
    t["cp"]["0"]["N"], t["cp"]["0"]["N_wide"] = C0["N_V5_declared"], C0["N_wide_V5"]
    assert Z1.registry_fail(t) != []
    t["cp"] = copy.deepcopy(T["cp"])
    t["cp"]["6"]["configs"] = [c for c in C6["configs"] if c != "A0g"]
    assert any("A0g не в ключе 6" in x for x in Z1.registry_fail(t))


def test_holm_in_g_info_equals_holm_with_n_722_and_3122():
    assert Z1.holm_fail(T) == []
    g = T["info"][T["info"]["src"] == "G_INFO.csv"]
    assert len(g) >= 20 and g["p_holm_N"].max() > g["dm_p_one"].min()


def test_mutation_holm_with_row_count_instead_of_n_is_caught():
    t = dict(T)
    i = T["info"].copy()
    g = i["src"] == "G_INFO.csv"
    i.loc[g, "p_holm_N"] = (i.loc[g, "dm_p_one"] * 4).clip(upper=1.0)
    t["info"] = i
    assert any("Холм" in x for x in Z1.holm_fail(t))
    assert A.claims_fail(t) != []
    with pytest.raises(ValueError):
        A.build(t)


def test_thresholds_in_code_equal_plan_99bd6681_and_gates():
    assert Z1.thresholds_fail() == []
    assert C0["thresholds"]["delta"] == 0.0005 and C0["thresholds"]["cond5_years_better_min"] == 3
    gates = (IO.REP / "GATES_V6.md").read_text(encoding="utf-8")
    assert "δ = 0,0005" in gates and "Холм N 722 / 3122" in gates
    go = pd.read_csv(IO.REP / "A0g_GO.csv", comment="#")
    assert list(go["cond"]) == [1, 2, 3, 4, 5] and go.iloc[4]["threshold"] == 3


def test_plan_file_is_the_one_committed_in_99bd6681():
    try:
        out = subprocess.run(["git", "show", "99bd6681:docs/otchety/HARJ_HYBRID_SBER_V6_PLAN.md"], cwd=IO.ROOT, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git недоступен")
    if out.returncode != 0:
        pytest.skip("коммит 99bd6681 недоступен")
    assert out.stdout.decode("utf-8") == Z1.PLAN.read_text(encoding="utf-8")


def test_mutations_delta_p_three_of_five_and_bootstrap_are_caught(monkeypatch):
    plan = Z1.PLAN.read_text(encoding="utf-8")
    src = inspect.getsource(Z1.EV.new_information)
    assert Z1.thresholds_fail(plan, src) == []
    assert any("p < 0,05" in x for x in Z1.thresholds_fail(plan, src.replace("p1 < 0.05", "p1 < 0.10")))
    assert any("3 из 5" in x for x in Z1.thresholds_fail(plan, src.replace("neg >= 3", "neg >= 2")))
    assert any("в плане нет" in x for x in Z1.thresholds_fail(plan.replace("p < 0,05 без поправки", "p < 0,10 без поправки"), src))
    monkeypatch.setattr(Z1.Q, "MARGIN", 0.001)
    assert any("δ" in x for x in Z1.thresholds_fail(plan, src))
    monkeypatch.setattr(Z1.Q, "MARGIN", 0.0005)
    monkeypatch.setattr(Z1.EV, "B", 1000)
    assert any("бутстрап EV" in x for x in Z1.thresholds_fail(plan, src))
