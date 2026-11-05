"""B6 (Ж-206, волна B, поток XGB): объявленные конфигурации, trial_id = хэш, дубли Ж-204 не trial, реестр только дописывается, бюджет, правило условных.
Мутации: дубль учтён как trial; повторная запись удваивает; счётчик выше 45 не останавливает; имя входит в хэш."""
import json

import pandas as pd
import pytest

from models_v2.s6.xgb import cfgx, runx


def test_declared_set_and_unique_deterministic_ids():
    assert len(cfgx.NEW) == 9 and len(cfgx.COND) == 1               # вторая условная строка строится по результату (см. test_conditional_rule)
    ids = [cfgx.trial_id(c) for c in cfgx.NEW + cfgx.COND]
    assert len(set(ids)) == len(ids) and ids == [cfgx.trial_id(c) for c in cfgx.NEW + cfgx.COND]
    assert all(len(i) == 12 for i in ids)
    for c in cfgx.NEW[:5]:                                           # любой объектив × F4+F2 не совпадает с объявленными строками
        r = cfgx.mk("cond", c["loss"], c["gw"], ("F4", "F2"))
        assert cfgx.trial_id(r) not in ids or c["loss"] == "l2" and c["gw"] == 0.0


def test_name_is_not_in_hash_but_every_forecast_field_is():
    a = cfgx.mk("a", "qlike", 0.5)
    assert cfgx.trial_id(a) == cfgx.trial_id(cfgx.mk("b", "qlike", 0.5))
    for other in (cfgx.mk("a", "gamma", 0.5), cfgx.mk("a", "qlike", 1.0), cfgx.mk("a", "qlike", 0.5, ("F4", "F2")), cfgx.mk("a", "qlike", 0.5, drop="no_imoex")):
        assert cfgx.trial_id(other) != cfgx.trial_id(a)


def test_prior_duplicates_are_references_not_trials():
    assert cfgx.is_duplicate(cfgx.CTRL) == ("b2_f4", "344764e88789")
    assert cfgx.is_duplicate(cfgx.mk("x", groups=("F1", "F4"))) == ("c1_f4_f1", "365963b63196")      # порядок групп не меняет конфигурацию
    assert all(cfgx.is_duplicate(c) is None for c in cfgx.NEW + cfgx.COND)
    assert cfgx.is_duplicate(cfgx.mk("x", groups=("F4", "F1"), gw=0.5)) is None                     # другое взвешивание — не дубль


def test_register_appends_only_new_and_is_idempotent(tmp_path):
    p = tmp_path / "OPTUNA_TRIALS.csv"
    rows = [{k: "v" for k in cfgx.TRIAL_HEAD} | {"trial_id": t} for t in ("aaa", "bbb")]
    assert cfgx.register(p, rows) == 2
    assert cfgx.register(p, rows) == 0                                  # мутация «повторная запись удваивает» — ловится
    assert cfgx.register(p, rows + [{k: "v" for k in cfgx.TRIAL_HEAD} | {"trial_id": "ccc"}]) == 1
    df = pd.read_csv(p)
    assert list(df["trial_id"]) == ["aaa", "bbb", "ccc"] and list(df.columns) == cfgx.TRIAL_HEAD


def test_checkpoint_counts_registry_rows_and_stops_above_45(tmp_path):
    ck, tr = tmp_path / "CHECKPOINT.json", tmp_path / "T.csv"
    ck.write_text(json.dumps({"used": {"obj_feat": 0, "hp": 0}}))
    cfgx.register(tr, [{k: "v" for k in cfgx.TRIAL_HEAD} | {"trial_id": "t%d" % i} for i in range(9)])
    out = cfgx.sync_checkpoint(ck, tr)
    assert out["used"] == {"obj_feat": 9, "hp": 0}
    cfgx.register(tr, [{k: "v" for k in cfgx.TRIAL_HEAD} | {"trial_id": "u%d" % i} for i in range(40)])
    with pytest.raises(ValueError):
        cfgx.sync_checkpoint(ck, tr)


def row(beats, crit_pen):
    return {"beats_f4_ref": beats, "crit_pen": crit_pen}


def test_conditional_rule():
    rows = {"x3_gamma_f4": row(False, 0.150), "x4_qlike_f4": row(True, 0.155), "x5_l2_g05_f4": row(True, 0.152), "f4_f2": row(True, 0.151)}
    d = runx.decide_cond(rows)
    assert d == {"best_obj": "x5_l2_g05_f4", "f4_f2_beats": True, "run_cond_obj": True, "run_f1_f2": True}
    rows["f4_f2"] = row(False, 0.16)
    d = runx.decide_cond(rows)
    assert not d["run_cond_obj"] and not d["run_f1_f2"]                  # F2 не прирост — условные не запускаются
    d = runx.decide_cond({"f4_f2": row(True, 0.1), "x3_gamma_f4": row(False, 0.15)})
    assert d["best_obj"] is None and not d["run_cond_obj"] and d["run_f1_f2"]
