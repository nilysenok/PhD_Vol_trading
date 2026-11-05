"""A3: BASELINE_MANIFEST.json полон и сверяется с файлами."""
import json

from models_v2.s5 import common, manifest

M = json.loads((common.REP / "BASELINE_MANIFEST.json").read_text())


def test_spec_b2_polnaya():
    b = M["b2"]
    assert b["n_features"] == 28 and len(b["features"]) == 28 and all(f["definition"] for f in b["features"])
    for k in ("target", "library", "hyperparameters_by_year", "schedule", "early_stopping", "s2", "hp_selection", "grid"):
        assert b[k]
    ch = common.read_comment_csv(common.TAB / "fix_i2_choice_g132.csv")
    ch = ch[ch.chosen]
    assert sorted(map(int, b["hyperparameters_by_year"])) == list(range(2017, 2027)) == sorted(ch.Y)
    for r in ch.itertuples():
        h = b["hyperparameters_by_year"][str(r.Y)]
        assert (h["depth"], h["min_child_weight"], h["reg_lambda"], h["eta"], h["subsample"], h["colsample_bytree"], h["max_trees"]) == (
            r.depth, r.min_data, r.l2, r.lr, r.subsample, r.colsample, r.max_trees)
    assert M["harj_b"]["method"] and len(M["harj_b"]["features"]) == 4


def test_sha_perescheityvayutsya():
    for f, s in M["sha_code"].items():
        assert common.sha_file(common.ROOT / f) == s, f
    for f, s in M["sha_forecasts"].items():
        assert common.sha_file(common.OUT / f) == s, f
    assert common.sha_file(common.IMOEX) == M["sha_inputs"]["imoex_bars"] and len(M["sha_inputs"]["papers16"]) == 16
    assert set(M["sha_forecasts"]) == set(manifest.FORECASTS)


def test_qlike_i_vosproizvedenie():
    q = M["qlike_recomputed"]
    fm = common.read_comment_csv(common.TAB / "fix_metrics.csv")
    ref = float(fm[(fm.model == common.B2) & (fm.slice == "all")].qlike.iloc[0])
    assert abs(q["b2"] - ref) <= 1e-12 and round(q["b2"], 4) == 0.2205 and round(q["harj"], 4) == 0.2337 and q["n"] == 2271
    assert M["reproduced"] is True and len(M["reproduction"]["dates"]) == 3
    assert all(x["rel_b2"] <= 1e-9 and x["rel_harj"] <= 1e-12 for x in M["reproduction"]["dates"])
