"""V7 лист 1b-C, G25: отчёт `LEAF1B_REPORT.md` целиком из таблиц; CHECKPOINT `1b.eval` — числа без p-значений и записан до таблиц с p; пересчёт `check_eval` сравнивает с файлами.
Мутации: число в тексте подменено; число вписано вручную (не следует за таблицей); p-значения в `1b.eval`; запись `1b.eval` после таблиц. Синтетика, файлы SBER не читаются."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s14 import v7_0_io as IO, v7_1b_ev as EV, v7_1b_evrep as REP, v7_1b_evrun as R

IDX = pd.bdate_range("2018-01-03", "2026-02-26")


@pytest.fixture(scope="module")
def world():
    g = np.random.default_rng(1)
    rv = np.exp(g.normal(0, 0.5, len(IDX)))
    s = {k: rv * np.exp(g.normal(0, 0.15, len(IDX))) for k in EV.REFS + EV.CFGS}
    inp = dict(idx=IDX, rv=rv, tp=IDX.to_numpy("datetime64[ns]"), series=s, shas={"V6:A0": "x"})
    t = R.compute(inp)
    e0 = pd.DataFrame([dict(kind=k, seed=s_, component=c, r2_weighted=0.1 * c + 0.01 * s_) for k in ("tcn", "cnn") for s_ in range(5) for c in range(1, 5)])
    cp = dict(choice3=dict(choice="tcn", qlike=dict(tcn=0.1, cnn=0.2)), E0=dict(both_no_new_info=False), rules=R.rules(t))
    return inp, t, e0, cp


def test_render_is_deterministic_and_contains_table_numbers(world):
    inp, t, e0, cp = world
    a, b = REP.render(t, e0, inp, cp), REP.render(t, e0, inp, cp)
    assert a == b and a.count("\n") > 60
    m = t["METRICS"]
    assert REP.f(m[(m["series"] == "cfg1") & (m["window"] == "2018_2021")]["qlike"].iloc[0], 6) in a
    assert "потолок" in a and "Холм N=733" in a


def test_mutation_number_substituted_in_report_is_caught(world):
    inp, t, e0, cp = world
    good = REP.render(t, e0, inp, cp)
    v = REP.f(t["METRICS"].query("series == 'K2' and window == '2018_2021'")["qlike"].iloc[0], 6)
    assert v in good and good.replace(v, v[:-1] + ("1" if v[-1] != "1" else "2"), 1) != good
    assert good.replace("False", "True", 1) != good


def test_mutation_number_typed_by_hand_does_not_follow_table(world):
    """Любая ячейка таблицы, попавшая в отчёт, меняет текст: чисел «вписанных вручную» нет."""
    inp, t, e0, cp = world
    base = REP.render(t, e0, inp, cp)
    for key, col in (("METRICS", "qlike"), ("TESTS", "delta_pct"), ("TESTS", "lo90"), ("TESTS", "p_one"), ("TESTS", "p_holm_N"), ("TESTS", "p_holm_Nwide"), ("GO", "value"), ("NEWINFO", "delta"), ("MCS", "mean_qlike"), ("STRESS", "qlike")):
        t2 = {k: v.copy() for k, v in t.items()}
        t2[key][col] = t2[key][col].astype(float) + 0.123456
        assert REP.render(t2, e0, inp, cp) != base, (key, col)
    e2 = e0.assign(r2_weighted=e0["r2_weighted"] + 0.5)
    assert REP.render(t, e2, inp, cp) != base


def test_numbers_have_no_pvalues_and_mutation_with_p_is_detected(world):
    inp, t, *_ = world
    n = R.numbers(t, inp)
    dump = json.dumps(n, ensure_ascii=False)
    assert "p_one" not in dump and "p_holm" not in dump and "p_dm" not in dump and "no_new" not in dump
    assert json.loads(dump) == n                                                             # JSON-круг без потерь (сверка с CHECKPOINT)
    n["delta_pct"]["cfg1"]["2018_2021|V6:A0"].append(float(t["TESTS"]["p_one"].iloc[0]))     # мутация: p-значение попало в ключ eval
    assert len(n["delta_pct"]["cfg1"]["2018_2021|V6:A0"]) == 4 and len(R.numbers(t, inp)["delta_pct"]["cfg1"]["2018_2021|V6:A0"]) == 3


def test_write_eval_puts_eval_before_any_table_with_p_values(world, monkeypatch, tmp_path):
    inp, t, e0, cp = world
    events = []
    monkeypatch.setattr(R, "load", lambda: inp)
    monkeypatch.setattr(R.CR, "update_1b", lambda sub, val: events.append("cp:" + sub))
    monkeypatch.setattr(R.CR, "read_cp", lambda: {"1b": cp})
    monkeypatch.setattr(R.IO, "write_csv", lambda df, name, note, status=None: events.append("csv:" + name))
    monkeypatch.setattr(R.IO, "read_csv", lambda name: e0)
    monkeypatch.setattr(R.IO, "REP", tmp_path)
    assert R.write_eval() == 0
    assert events[0] == "cp:eval" and events.index("cp:eval") < events.index("csv:1b_TESTS.csv") < events.index("cp:rules")
    assert (tmp_path / "LEAF1B_REPORT.md").read_text(encoding="utf-8") == REP.render(t, e0, inp, dict(cp, rules=R.rules(t)))


def test_verdict_confirmed_only_with_holm_and_go():
    assert REP.verdict(False, True) == "go не выполнен"
    assert REP.verdict(True, False) == "статистически неподтверждённое историческое улучшение"
    assert "подтверждено" in REP.verdict(True, True)


def test_same_frame_detects_changed_cell(world):
    _, t, *_ = world
    x = t["METRICS"].copy()
    assert R.same_frame(x, t["METRICS"])
    x.loc[0, "qlike"] += 1e-9
    assert not R.same_frame(x, t["METRICS"])
