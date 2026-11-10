"""Y-B: условие Y5/Y6 (известный ответ, мутация «по окну 2022+»), отказ Y5/Y6, отчёт против таблиц (подменённое число), --check только читает (хэш выходов)."""
import json

import numpy as np
import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_reg as R, x1_yb_doc as DOC, x1_yb_eval as EV, x1_yb_stage as S


def ra(y1):
    return pd.DataFrame([dict(y="Y1", rule62=y1), dict(y="Y2", rule62="нет")])


def rb(y3):
    return pd.DataFrame([dict(y="Y3", rule62=y3), dict(y="Y4", rule62="нет")])


@pytest.mark.parametrize("y1,y3,ans", [("нет", "нет", "нет"), ("да", "нет", "да"), ("нет", "да", "да"), ("да", "да", "да")])
def test_condition_known_answer(y1, y3, ans):
    c = EV.condition_table(ra(y1), rb(y3)).set_index("item")["rule62"]
    assert c["условие Y5/Y6"] == ans and c["Y1"] == y1 and c["Y3"] == y3
    assert S.condition({"CONDITION": EV.condition_table(ra(y1), rb(y3))})["condition"] == ans


def test_y4_and_y2_do_not_enter_condition():
    a = pd.DataFrame([dict(y="Y1", rule62="нет"), dict(y="Y2", rule62="да")])
    b = pd.DataFrame([dict(y="Y3", rule62="нет"), dict(y="Y4", rule62="да")])
    assert EV.condition_table(a, b).set_index("item").at["условие Y5/Y6", "rule62"] == "нет"


def test_rule62_window_2022_plus_mutation_caught():
    rng = np.random.default_rng(0)
    tp = pd.bdate_range("2018-01-04", "2025-12-30")
    rv = np.exp(rng.normal(0, .5, len(tp)))
    x1 = rv * np.exp(rng.normal(0, .3, len(tp)))
    y = np.where(tp >= pd.Timestamp("2022-01-01"), rv * np.exp(rng.normal(0, .05, len(tp))), x1)       # лучше X1 только с 2022
    honest = EV.ST.new_information(rv, y, x1, tp)
    assert honest["new_information"] is False or not honest["new_information"]
    mutant = EV.ST.new_information(rv, y, x1, tp - pd.DateOffset(years=4))                                # окно сдвинуто на 2022+
    assert mutant["new_information"]


def test_y5_y6_always_refused():
    for c in ("Y5", "Y6"):
        with pytest.raises(SystemExit):
            R.guard(c)
        with pytest.raises(SystemExit):
            S.main(["--stage", c])


def _tables():
    t = {k: pd.read_csv(S.REP / (f + ".csv"), comment="#") for f, k in S.TABLES.items()}
    ev = json.loads((S.REP / "CHECKPOINT.json").read_text(encoding="utf-8"))["YB.eval"]["run"]
    return t, ev


def test_report_equals_generator_and_substituted_number_caught():
    t, num = _tables()
    txt = DOC.build(t, num)
    assert txt == S.REPORT.read_text(encoding="utf-8")
    m = t["METRICS"]
    v = "%.4f" % m[(m["series"] == "Y3") & (m["window"] == "all")]["qlike"].iloc[0]
    assert v in txt
    assert txt.replace(v, "9.9999", 1) != S.REPORT.read_text(encoding="utf-8")
    assert "НЕ ВЫПОЛНЕНО" in txt or "ВЫПОЛНЕНО" in txt and "не запускаются без отдельного разрешения владельца" in txt


def test_check_that_writes_is_caught_by_output_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "REP", tmp_path / "rep")
    monkeypatch.setattr(S, "OUT", tmp_path / "out")
    monkeypatch.setattr(S, "YA_OUT", tmp_path / "ya")
    monkeypatch.setattr(S.SA, "WATCH", [])
    for d in ("rep", "out", "ya"):
        (tmp_path / d).mkdir()
    (tmp_path / "rep" / "YB_X.csv").write_text("a\n", encoding="utf-8")
    before = S.outputs_hash()
    assert S.outputs_hash() == before
    (tmp_path / "rep" / "YB_X.csv").write_text("b\n", encoding="utf-8")                  # «проверка», которая пишет
    assert S.outputs_hash() != before
    (tmp_path / "out" / "new.parquet").write_text("c", encoding="utf-8")
    assert S.outputs_hash() != before
