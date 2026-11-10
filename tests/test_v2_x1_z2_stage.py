"""Z2: `--check` только читает (пересчёт = файлы, подмена числа в отчёте и таблицах ловится, запись чужого ловится), порядок ключей CHECKPOINT, критерии K1–K5 (известный ответ, границы, каждое условие закрывает), Z2_CRIT.csv."""
import shutil

import pandas as pd
import pytest

from models_v2.s16 import x1_z2_eval as EZ, x1_z2_io as IO, x1_z2_stage as STG


@pytest.fixture(scope="module")
def h0():
    return STG.compute_h0()


@pytest.fixture()
def copy(tmp_path, monkeypatch, h0):
    rep = tmp_path / "harj_x1"
    shutil.copytree(IO.REP, rep)
    for mod in (IO, STG):
        monkeypatch.setattr(mod, "REP", rep)
    monkeypatch.setattr(IO, "CP_FILE", rep / "CHECKPOINT.json")
    for mod in (IO, STG):
        monkeypatch.setattr(mod, "REPORT", rep / "Z2_REPORT.md")
    monkeypatch.setattr(STG, "OUT", tmp_path / "Z2_empty")
    monkeypatch.setattr(STG, "compute_h0", lambda: h0)
    return rep


def test_check_clean_and_substituted_report_or_table_or_foreign_write_caught(copy, monkeypatch):
    assert STG.check() == []
    rep = copy / "Z2_REPORT.md"
    txt = rep.read_text(encoding="utf-8")
    rep.write_text(txt.replace("Тождество e_d", "Тождество  e_d", 1), encoding="utf-8")
    assert any("Z2_REPORT.md ≠ генератору" in b for b in STG.check())
    rep.write_text(txt, encoding="utf-8")
    lr = copy / "Z2_H0_LR.csv"
    s = lr.read_text(encoding="utf-8")
    lr.write_text(s.replace(",152,", ",153,", 1), encoding="utf-8")
    assert any("Z2_H0_LR.csv" in b for b in STG.check())
    lr.write_text(s, encoding="utf-8")
    real = STG.compute_h0

    def writing():
        (copy / "Z_FOREIGN.txt").write_text("чужой выход", encoding="utf-8")
        return real()
    monkeypatch.setattr(STG, "compute_h0", writing)
    assert any("изменил чужие выходы" in b for b in STG.check())


def test_check_flags_forecasts_or_eval_keys_when_condition_false(copy, tmp_path):
    (tmp_path / "Z2_empty").mkdir()
    (tmp_path / "Z2_empty" / "h1.parquet").write_bytes(b"x")
    assert any("условие ложно, но есть прогнозы" in b for b in STG.check())


def cp_with(order=(0, 1, 2, 3), p_in_eval=False):
    t = ["10.11.2026 17:00:00", "10.11.2026 17:01:00", "10.11.2026 17:02:00", "10.11.2026 17:03:00"]
    t = [t[i] for i in order]
    return {"Z2.condition": {"written_at": t[0]}, "Z2": {"computed_at": t[1]}, "Z2.eval": {"written_at": t[2], "numbers": {"p_holm_N": 1} if p_in_eval else {"delta": 1}}, "Z2.rules": {"written_at": t[3]}}


def test_key_order_condition_forecast_eval_rules_and_no_p_in_eval():
    assert STG.order_ok(cp_with())
    assert not STG.order_ok(cp_with((1, 0, 2, 3)))        # прогноз раньше условия
    assert not STG.order_ok(cp_with((0, 1, 3, 2)))        # Z2.rules раньше Z2.eval
    assert not STG.order_ok(cp_with(p_in_eval=True))      # p в Z2.eval


def mk(k1=-1.0, k2=0.0, k5=-0.05, k4=-1.0, yneg=6, rule=True, holm=0.01):
    pairs = pd.DataFrame([dict(y="H1", base="X1", window="all", delta_pct=k1, delta_q=k1 / 100, p_holm_N=0.5), dict(y="H1", base="X1", window="2018_2021", delta_pct=k5, delta_q=k5 / 100, p_holm_N=holm),
                          dict(y="H1", base="Hc", window="all", delta_pct=k4, delta_q=k4 / 100, p_holm_N=0.5)])
    states = pd.DataFrame([dict(y="H1", state="N", window="all", delta_pct=k2)])
    years = pd.DataFrame([dict(y="H1", base="X1", year=2018 + i, delta_q=-1.0 if i < yneg else 1.0) for i in range(8)])
    return EZ.crit_table(pairs, states, years, dict(delta=-0.01, new_information=rule))


def passed(t):
    return t.set_index("crit")["passed"].to_dict()


def test_k1_k5_known_answer_boundaries_and_each_condition_alone_closes():
    ok = passed(mk())
    assert all(ok[k] for k in ("K1", "K2", "K3", "K4", "K5", "итог", "6.2", "Холм"))
    assert passed(mk(k2=0.10))["K2"] and not passed(mk(k2=0.1001))["K2"]                # ≤ +0,10 % включительно
    assert passed(mk(k5=0.10))["K5"] and not passed(mk(k5=0.1001))["K5"]
    assert passed(mk(yneg=5))["K3"] and not passed(mk(yneg=4))["K3"]                    # ≥ 5 из 8
    assert not passed(mk(k1=0.0))["K1"] and not passed(mk(k4=0.0))["K4"]                # строго < 0
    for kw in (dict(k1=0.5), dict(k2=0.5), dict(yneg=2), dict(k4=0.5), dict(k5=0.5)):
        assert not passed(mk(**kw))["итог"]
    assert not passed(mk(rule=False))["6.2"] and not passed(mk(holm=0.05))["Холм"]      # подтверждение — отдельно от «продолжить»
    assert passed(mk(rule=False))["итог"]


def test_substituted_number_in_z2_crit_csv_caught(tmp_path):
    t = {"CRIT": mk()}
    (tmp_path / "Z2_CRIT.csv").write_text(IO.csv_text(t["CRIT"]), encoding="utf-8")
    sha = {"Z2_CRIT": STG.SA.sha_file(tmp_path / "Z2_CRIT.csv")}
    assert STG.table_diffs({"Z2_CRIT": "CRIT"}, t, tmp_path, sha) == []
    s = (tmp_path / "Z2_CRIT.csv").read_text(encoding="utf-8")
    (tmp_path / "Z2_CRIT.csv").write_text(s.replace("-1.0,0.0,True", "-0.5,0.0,True", 1), encoding="utf-8")
    assert STG.table_diffs({"Z2_CRIT": "CRIT"}, t, tmp_path, sha) == ["Z2_CRIT.csv ≠ пересчёту"]
