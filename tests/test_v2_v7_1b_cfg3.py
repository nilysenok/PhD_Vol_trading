"""V7 лист 1b-C, G24 (конфигурация 3): энкодер только из заранее записанного `1b.choice3`, окно 1 день (файлы `*_w1_*`, не `*_w5_*`), тот же протокол, что у 1–2 (`run_kind` с window=1).
Мутации: окно 5 вместо 1; выбор энкодера без записи choice3. Данные SBER не читаются (синтетика, загрузчик подменён)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s14 import v7_1b_cfg as CFG, v7_1b_cfg3 as C3, v7_1b_sta as STA

TAB_IDX = pd.bdate_range("2017-01-02", "2019-12-31")


def tab():
    g = np.random.default_rng(0)
    n = len(TAB_IDX)
    H = np.exp(g.normal(0, 0.3, n))
    return pd.DataFrame({"H": H, "K2": H, "K3": H, "rv_next": H * np.exp(g.normal(0, 0.1, n)), "t_plus": list(TAB_IDX[1:]) + [TAB_IDX[-1] + pd.Timedelta(days=1)]}, index=TAB_IDX)


def recorder(seen):
    def load(path):
        seen.append(path.name)
        year = int(path.name.removesuffix(".npz").split("_")[2])
        days = TAB_IDX[TAB_IDX < pd.Timestamp(year + 1, 1, 1)]
        i = np.arange(len(days))
        return dict(days=days.to_numpy("datetime64[ns]").astype(np.int64), z=np.column_stack([np.sin(i / 9.0), np.cos(i / 13.0), np.zeros(len(i)), np.zeros(len(i))]))
    return load


def test_names_window_default_is_5_and_w1_for_config3():
    assert STA.name("tcn", 2018, 0) == "tcn_w5_2018_s0.npz" and STA.name("tcn", 2018, 0, 1) == "tcn_w1_2018_s0.npz"
    assert C3.file3("cnn", 2020, 4).name == "cnn_w1_2020_s4.npz" and len(C3.tasks3("tcn")) == 45


def test_run_kind_window1_reads_only_w1_artifacts_and_window5_mutation_is_caught():
    seen = []
    df = CFG.run_kind("tcn", tab(), load=recorder(seen), seeds=(0,), years=(2018, 2019), window=1)
    assert len(df) > 0 and seen and all("_w1_" in n for n in seen)
    seen5 = []
    CFG.run_kind("tcn", tab(), load=recorder(seen5), seeds=(0,), years=(2018, 2019))
    assert seen5 and all("_w5_" in n for n in seen5)                                    # умолчание (конфигурации 1–2) не изменилось
    assert set(seen) != set(seen5)                                                       # мутация «окно 5 вместо 1» дала бы второй набор файлов


def test_config3_same_protocol_as_1_2_F_positive_and_g0_gives_H():
    df = CFG.run_kind("tcn", tab(), load=recorder([]), seeds=(0, 1), years=(2018, 2019), window=1)
    assert (df["F"] > 0).all() and np.array_equal(df["F"][df["g"] == 0], df["H"][df["g"] == 0])
    assert df["d"].iloc[0] >= pd.Timestamp("2018-01-03")


def test_kind3_requires_choice_written_before_and_mutation_without_record_stops():
    assert C3.kind3({"1b": {"choice3": {"choice": "cnn", "written_before_cfg3": True}}}) == "cnn"
    for cp in ({}, {"1b": {}}, {"1b": {"choice3": {"choice": "tcn"}}}, {"1b": {"choice3": {"choice": "xyz", "written_before_cfg3": True}}}):
        with pytest.raises(SystemExit):
            C3.kind3(cp)


def test_check3_reports_missing_checkpoint_and_files(tmp_path, monkeypatch):
    assert C3.check3({}) == ["CHECKPOINT 1b: нет cfg3/stageA_w1"]
    monkeypatch.setattr(STA, "directory", lambda: tmp_path)
    cp = {"cfg3": {"encoder": "tcn"}, "stageA_w1": {"hashes": {}}, "choice3": {"choice": "tcn"}}
    bad = C3.check3(cp)
    assert bad and "нет файлов" in bad[0]
    assert "cfg3: энкодер ≠ choice3" in C3.check3(dict(cp, choice3={"choice": "cnn"}))
