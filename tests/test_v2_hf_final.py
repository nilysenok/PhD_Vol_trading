"""Ж-211 (GATES_HF G1/HF10/HF11): стадия итога. N поправок = число строк ALL_TRIALS.csv (мутация «n_final() = 12» ловится), p Холма в таблицах
соответствуют этому N; метки «потолок» в LEVELS/TABLE17; постселекционная оговорка в каждой строке STATISTICAL_TESTS."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s5.d import stats_d
from models_v2.s7 import hf_ledger as L, hf_levels as V

REP = L.REP


def _read(name):
    return pd.read_csv(REP / name, comment="#")


def test_n_final_equals_registry_rows_synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "ALL", tmp_path / "ALL.csv")
    monkeypatch.setattr(L, "CHECK", tmp_path / "CHECK.json")
    rows = [("k%d" % i, "Ж-204", "s", "d", "c", "trial") for i in range(7)] + [("p%d" % i, "Ж-200/202", "s", "d", "c", "pilot_forecast") for i in range(3)]
    pd.DataFrame(rows, columns=L.COLS).to_csv(L.ALL, index=False)
    assert L.n_final() == 10
    pd.concat([L.load(), pd.DataFrame([("n1", "Ж-211", "s", "bounded", "c", "trial")], columns=L.COLS)]).to_csv(L.ALL, index=False)
    assert L.n_final() == 11 == len(L.load())


def test_n_final_is_real_registry_521():
    assert L.n_final() == len(L.load()) == 521


def test_stat_tests_use_registry_n_and_holm_matches():
    n = L.n_final()
    t = _read("STATISTICAL_TESTS.csv")
    assert (t["N"] == n).all() and n != 12
    a = t[t["period"] == "all"]
    assert len(a) > 0
    exp = stats_d.holm_n(a["p_raw"].fillna(1.0).to_numpy(), n)
    assert np.allclose(a["p_holm"].to_numpy(), exp, rtol=1e-12, atol=0)
    with pytest.raises(ValueError):                                                          # N=12 < числа тестов: стадия итога упала бы
        stats_d.holm_n(a["p_raw"].fillna(1.0).to_numpy(), 12)
    assert t.loc[t["period"] != "all", "p_holm"].isna().all()


def test_levels_l1d_follows_holm_with_registry_n():
    t, lv = _read("STATISTICAL_TESTS.csv"), _read("LEVELS.csv")
    for _, r in lv.iterrows():
        assert bool(r["L1_d"]) == bool(V.pick(t, r["model"], "A", "all")["p_holm"] < 0.10)


def test_ceiling_note_labels():
    assert "потолок: выбор по crit_wf на всей истории" in V.ceiling_note("H1: A1_roll_126")
    assert "признаки по EX1" not in V.ceiling_note("H1: A1_roll_126")
    for lab in ("H3: xgb_abl_S34_noF3", "H4: lgbm_abl_S34_noF3"):
        assert "+ признаки по EX1" in V.ceiling_note(lab)
    assert V.ceiling_note("HAR-J") == "" and V.ceiling_note("XGBoost") == ""


def test_output_tables_carry_ceiling_and_postselection_marks():
    lv, t17, st = _read("LEVELS.csv"), _read("TABLE17_SBER_H1.csv"), _read("STATISTICAL_TESTS.csv")
    assert lv["ceiling_note"].str.contains("потолок: выбор по crit_wf на всей истории").all()
    h34 = lv[lv["model"].str[:2].isin(["H3", "H4"])]
    assert len(h34) and h34["ceiling_note"].str.contains("+ признаки по EX1", regex=False).all()
    assert not lv[lv["model"].str[:2].isin(["H1", "H2", "H7"])]["ceiling_note"].str.contains("EX1").any()
    fin = t17[~t17["Модель"].isin(["HAR-J", "XGBoost", "LightGBM"])]
    assert fin["Пометка"].str.contains("потолок: выбор по crit_wf на всей истории").all()
    assert (t17.loc[t17["Модель"].isin(["HAR-J", "XGBoost", "LightGBM"]), "Пометка"].isna().all())
    assert st["post_selection"].str.contains("постселекционно").all() and st["post_selection"].notna().all()
    assert (REP / "HYBRID_HAR_BOOSTING_FULL_RESEARCH_FINAL.md").read_text(encoding="utf-8").count("потолок: выбор по crit_wf на всей истории") >= len(fin)
