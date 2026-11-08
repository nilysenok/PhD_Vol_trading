"""V4-RQC (G35–G36): ключ кэша (Д5), реестр через `replace_rows`, независимый пересчёт заморозки против боевого пути, свойства RQC3 на мутациях. Данные проекта не читаются."""
from __future__ import annotations

from unittest import mock

import numpy as np

from models_v2.s11 import v4_rqc_cache as RC, v4_rqc_chk35 as C35, v4_rqc_r3 as R3, v4_rqc_run as R, v4_rqc_stage as ST, v4_trials as TR
from tests.test_v2_v4_rqc_run import synth


def test_key_changes_with_input_code_and_constants(tmp_path):
    a, c = tmp_path / "a.txt", tmp_path / "c.py"
    a.write_text("1")
    c.write_text("x")
    k0 = RC.keys([a], [c])["key"]
    a.write_text("2")
    assert RC.keys([a], [c])["key"] != k0
    a.write_text("1")
    c.write_text("y")
    assert RC.keys([a], [c])["key"] != k0
    c.write_text("x")
    assert RC.keys([a], [c])["key"] == k0
    with mock.patch.object(R3, "STEP", 0.05):
        assert RC.keys([a], [c])["key"] != k0


def test_trials_rows_replace_keep_others_and_no_duplicates(tmp_path):
    f = tmp_path / "ALL_TRIALS.csv"
    f.write_text(TR.HEADER + "Ж-222:CLD1:aa,p,s,d,cfg,trial\nЖ-222:E1:bb,p,s,d,cfg,trial\n")
    for _ in range(2):
        TR.replace_rows(["Ж-222:%s:" % c for c in ST.CFGS], ST.trials_rows("k" * 64), path=f)
    keys = [x.split(",")[0] for x in f.read_text().splitlines()[1:]]
    assert sorted(k.split(":")[1] for k in keys) == sorted(["CLD1", "E1"] + ST.CFGS)


def test_expected_frozen_independent_recount_equals_production():
    P, Z = synth(n=400, seed=3)
    Ps = C35.sorted_panel(P, Z)
    out = R.forecast("RQC3", P, Z)[0]
    exp = C35.expected_frozen(Ps)
    assert exp.any() and np.array_equal(out["frozen"].to_numpy(), exp)
    assert C35.rqc3_props(out, exp) == []


def test_props_catch_step_and_freeze_mutations_on_synthetic():
    P, Z = synth(n=400, seed=3, ratio=lambda i, d: 1.0 + 0.5 * (i > 100))
    exp = C35.expected_frozen(C35.sorted_panel(P, Z))
    for name, val in (("STEP", 0.05), ("FREEZE", 0)):
        with mock.patch.object(R3, name, val):
            assert C35.rqc3_props(R.forecast("RQC3", P, Z)[0], exp), name
    assert C35.rqc3_props(R.forecast("RQC3", P, Z)[0], exp) == []
