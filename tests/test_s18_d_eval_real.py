"""S-D-EVAL: боевая проверка и мутации (E.G3; mutations.md № 125, 160, 123, 158): подложенная ошибка в таблице, отчёте или пороге обязана ловиться независимым пересчётом `d_eval_chk` / `d_doc`."""
import copy
import shutil

import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import d_doc as DOC
from models_v2.s18 import d_eval as E
from models_v2.s18 import d_eval_chk as C
from models_v2.s18 import d_eval_tab as T


@pytest.fixture(scope="module")
def real():
    inp = E.load_inputs()
    t = E.compute(inp)
    return inp, t, E.texts(t, inp["tp"])


def test_honest_tables_and_verdicts_are_clean(real):
    inp, t, tx = real
    assert C.p_const() == [] and C.p_qlike(t, inp) == [] and C.p_verdict(t) == [] and C.p_years_slices(t, inp) == [] and C.p_pred(t) == []
    assert all((E.REP / n).read_text(encoding="utf-8") == v for n, v in tx.items())
    assert t["table"].set_index("series").loc["S-RCK", "qlike"] == pytest.approx(0.215026781817, rel=1e-10)


def test_mutation_sign_of_delta_flipped_is_caught(real):
    inp, t, _ = real
    m = copy.deepcopy(t)
    m["delta"]["delta_pct"] = -m["delta"]["delta_pct"]
    assert C.p_delta(m, inp) != []


def test_mutation_base_of_percent_swapped_is_caught(real):
    inp, t, _ = real
    m = copy.deepcopy(t)
    q = m["delta"][["q_series", "q_base"]]
    m["delta"]["delta_pct"] = 100.0 * (q["q_series"] - q["q_base"]) / q["q_series"]
    assert C.p_delta(m, inp) != []


def test_mutation_trimmed_dates_in_main_delta_is_caught(real):
    inp, t, _ = real
    m = copy.deepcopy(t)
    ls = T.losses(inp["rv"], inp["F"])
    keep = T.trim_keep(ls["S-RCKf"] - ls["S-RCK"])
    d = m["delta"]
    i = d.index[(d.series == "S-RCKf") & (d.base == "S-RCK")][0]
    d.loc[i, "delta_pct"] = 100.0 * (ls["S-RCKf"][keep] - ls["S-RCK"][keep]).sum() / ls["S-RCK"][keep].sum()
    assert C.p_delta(m, inp) != []


def test_mutation_qlike_as_mean_of_yearly_means_is_caught(real):
    inp, t, _ = real
    m = copy.deepcopy(t)
    y = inp["tp"].year.to_numpy()
    m["table"]["qlike"] = [float(np.mean([T.ql(inp["rv"], inp["F"][s])[y == k].mean() for k in sorted(set(y))])) for s in m["table"].series]
    assert C.p_qlike(m, inp) != []


def test_mutation_verdict_typed_by_hand_is_caught(real):
    _, t, _ = real
    m = copy.deepcopy(t)
    m["elements"].loc[1, "code"] = "needed"
    assert C.p_verdict(m) != []
    m = copy.deepcopy(t)
    m["elements"].loc[m["elements"].index[-1], "verdict"] = T.SUMMARY["both_needed"]
    assert C.p_verdict(m) != []


def test_mutation_prediction_threshold_taken_from_result_is_caught(real, monkeypatch):
    _, t, _ = real
    monkeypatch.setattr(T, "THR_PF", abs(float(t["pred"].set_index("id").loc["P-f", "value"])) + 1e-6)
    assert C.p_const() != []
    m = copy.deepcopy(t)
    m["pred"].loc[m["pred"].id == "P-f2", "fulfilled"] = "выполнено"
    monkeypatch.undo()
    assert C.p_pred(m) != []


def _copy_rep(tmp_path):
    for n in E.OUTS + ("S_D_REPORT.md",):
        shutil.copy2(E.REP / n, tmp_path / n)
    shutil.copy2(E.CKPT, tmp_path / "CHECKPOINT.json")
    return tmp_path, tmp_path / "CHECKPOINT.json"


def test_report_equals_tables_and_mutations_number_and_better_worse_are_caught(tmp_path):
    rep, ck = _copy_rep(tmp_path)
    assert DOC.check(rep, ck) == []
    p = rep / "S_D_REPORT.md"
    txt = p.read_text(encoding="utf-8")
    p.write_text(txt.replace("+0.1504", "+0.1405", 1), encoding="utf-8")
    assert DOC.check(rep, ck) != []
    p.write_text(txt.replace("без элемента хуже на", "без элемента лучше на", 1), encoding="utf-8")
    assert DOC.check(rep, ck) != []
    p.write_text(txt.replace("Знаменатель Δ", "Δ", 1), encoding="utf-8")
    assert DOC.check(rep, ck) != []
    p.write_text(txt, encoding="utf-8")
    assert txt.splitlines()[0].startswith("диагностика вклада элементов R, исследовательский результат, не OOS") and DOC.check(rep, ck) == []


def test_check_that_writes_is_caught_by_hash(tmp_path, monkeypatch):
    rep, ck = _copy_rep(tmp_path)
    orig = C.problems

    def mut(t, inp, rep_, ckpt, tx):
        (rep_ / "S_D_TABLE.csv").write_text("x", encoding="utf-8")
        return orig(t, inp, rep_, ckpt, tx)
    monkeypatch.setattr(C, "problems", mut)
    assert any("изменил выходы" in b for b in E.check(rep, ck))


def test_key_written_before_tables_and_mutated_order_is_caught(real, tmp_path):
    inp, t, tx = real
    rep, ck = _copy_rep(tmp_path)
    assert C.p_key(t, rep, ck, tx) == []
    import json
    cp = json.loads(ck.read_text(encoding="utf-8"))
    cp["S.D.eval"]["written_at"] = "11.11.2026 23:59:59"
    ck.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    assert any("позже самой ранней" in b for b in C.p_key(t, rep, ck, tx))
    cp["S.D.eval"]["written_at"] = "11.11.2026 20:00:00"
    ck.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    assert any("не позже S.D.registered" in b for b in C.p_key(t, rep, ck, tx))
    assert pd.Timestamp("2026-11-11") > pd.Timestamp("2018-01-04")
