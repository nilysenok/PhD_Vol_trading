"""V7 лист 1b-A, G22–G23: артефакты Stage A (хэш содержимого, детерминизм, PIT z), решение E0 (все компоненты), слияние подразделов CHECKPOINT 1b, отказ `--check` без артефактов. Синтетика, tmp_path."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s14 import v7_0_io as IO, v7_1b_e0 as EZ, v7_1b_stage as ST, v7_1b_sta as STA, v7_1e_rep as REP, v7_1e_syn as SYN

Y, KW = 2020, dict(max_epochs=2, min_rows=30)


@pytest.fixture(scope="module")
def pack():
    return SYN.synth_pack(n=130, W=5, seed=0, start="2019-08-01")


def test_content_hash_sensitive_and_order_free():
    a = dict(w=np.arange(4.0), z=np.ones((2, 2)))
    assert STA.content_hash(a) == STA.content_hash(dict(z=a["z"], w=a["w"]))
    b = dict(a, w=a["w"].copy())
    b["w"][2] += 1e-12
    assert STA.content_hash(a) != STA.content_hash(b)


def test_compute_bitwise_same_seed_and_dates_not_after_year_end(pack):
    a, b = STA.compute("tcn", Y, 0, pack, **KW), STA.compute("tcn", Y, 0, pack, **KW)
    assert STA.content_hash(a) == STA.content_hash(b)
    assert STA.content_hash(STA.compute("tcn", Y, 1, pack, **KW)) != STA.content_hash(a)
    d = pd.DatetimeIndex(a["days"].astype("datetime64[ns]"))
    assert d.max() < pd.Timestamp(Y + 1, 1, 1) and a["z"].shape == (len(d), 4) and np.isfinite(a["z"]).all()


def test_mutation_pca_over_whole_sample_is_caught(pack, monkeypatch):
    honest = STA.compute("cnn", Y, 0, pack, **KW)["z"]
    monkeypatch.setattr(REP, "_past_mask", lambda dates, year: np.ones(len(dates), bool))
    assert not np.array_equal(STA.compute("cnn", Y, 0, pack, **KW)["z"], honest)


def test_save_load_roundtrip_and_tamper_changes_hash(tmp_path, pack):
    a = STA.compute("tcn", Y, 0, pack, **KW)
    h = STA.save(a, tmp_path / "x" / "a.npz")
    got = STA.load(tmp_path / "x" / "a.npz")
    assert STA.content_hash(got) == h and not list((tmp_path / "x").glob("tmp_*"))
    got["w__" + next(k[3:] for k in got if k.startswith("w__"))][...] += 1.0
    assert STA.content_hash(got) != h


def test_decide_requires_all_components():
    ok = [np.full(4, 0.95)] * 5
    assert EZ.decide(ok)["no_new_info"] is True
    low = [np.array([0.99, 0.99, 0.99, 0.5])] * 5
    assert EZ.decide(low)["no_new_info"] is False                      # мутант «хотя бы одна» дал бы True
    mix = [np.full(4, 1.0)] * 2 + [np.full(4, 0.0)] * 3                # среднее по seeds 0,4 < 0,9
    assert EZ.decide(mix)["no_new_info"] is False and EZ.decide(mix)["r2_mean"][0] == pytest.approx(0.4)


def test_update_1b_merges_subsections(tmp_path, monkeypatch):
    monkeypatch.setattr(IO, "REP", tmp_path)
    (tmp_path / "CHECKPOINT.json").write_text(json.dumps({"1a": {"a": 1}, "1b": {"stageA": {"n": 1}}}), encoding="utf-8")
    ST.update_1b("E0", {"x": 2})
    cp = json.loads((tmp_path / "CHECKPOINT.json").read_text(encoding="utf-8"))
    assert cp["1a"] == {"a": 1} and cp["1b"] == {"stageA": {"n": 1}, "E0": {"x": 2}}


def test_check_without_artifacts_refuses_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(IO, "OUT", tmp_path / "out")
    assert ST.run(True) == 1 and "нет артефактов Stage A" in capsys.readouterr().out and not (tmp_path / "out").exists()
