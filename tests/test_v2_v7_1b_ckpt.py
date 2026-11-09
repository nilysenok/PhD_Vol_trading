"""V7 лист 1b, G21: запись стадии сливает ключи CHECKPOINT.json (класс № 117; Ж-235 п. 6). Реальный путь `run_v7.write_stage` на заглушках данных, файлы — в tmp_path.
Мутация «перезапись одним ключом» (старое поведение: `{"0": ...}` целиком) обязана ловиться."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s14 import run_v7 as RUN, v7_0_io as IO

OTHER = {"1a": {"N_holm": 733, "written_at": "T"}, "1b": {"x": 1}}


def overwrite_mutant(key, value, path=None):
    """Старое поведение стадии 0: файл целиком одним ключом."""
    path = path or IO.REP / "CHECKPOINT.json"
    path.write_text(json.dumps({key: value}, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return {key: value}


@pytest.fixture()
def stage(tmp_path, monkeypatch):
    rep, out = tmp_path / "rep", tmp_path / "out"
    rep.mkdir()
    monkeypatch.setattr(IO, "REP", rep)
    monkeypatch.setattr(IO, "OUT", out)
    monkeypatch.setattr(IO, "write_csv", lambda *a, **k: None)
    monkeypatch.setattr(IO, "code_hash", lambda: "c")
    monkeypatch.setattr(IO, "inputs_hash", lambda: "i")
    monkeypatch.setattr(RUN, "gate_status", lambda a: dict(G1_controls_equal=True, G3_identity_ok=True))
    monkeypatch.setattr(RUN, "sequences", lambda store, W: dict(X=np.zeros(2)))
    a = dict(ctrl=None, passport=None, samples=None, sinfo=dict(first_a="x", first_b="y"), reuse_md="r", audit_md="a",
             store=dict(raw=np.zeros((2, 2)), present=np.ones((2, 2), bool)), days=pd.DatetimeIndex(["2020-01-02", "2020-01-03"]))
    return rep / "CHECKPOINT.json", a


def seeded(path):
    path.write_text(json.dumps(OTHER), encoding="utf-8")


def test_stage0_write_keeps_other_keys(stage):
    path, a = stage
    seeded(path)
    RUN.write_stage(a)
    cp = json.loads(path.read_text(encoding="utf-8"))
    assert cp["1a"] == OTHER["1a"] and cp["1b"] == OTHER["1b"] and "0" in cp and cp["0"]["code"] == "c"


def test_stage0_write_without_file_creates_key_zero(stage):
    path, a = stage
    RUN.write_stage(a)
    assert list(json.loads(path.read_text(encoding="utf-8"))) == ["0"]


def test_rewrite_of_same_key_replaces_only_it(stage):
    path, a = stage
    seeded(path)
    RUN.write_stage(a)
    IO.merge_checkpoint("0", dict(code="new"), path)
    cp = json.loads(path.read_text(encoding="utf-8"))
    assert cp["0"] == dict(code="new") and cp["1a"] == OTHER["1a"]


def test_mutation_overwrite_by_one_key_is_caught(stage, monkeypatch):
    path, a = stage
    seeded(path)
    monkeypatch.setattr(IO, "merge_checkpoint", overwrite_mutant)
    RUN.write_stage(a)
    cp = json.loads(path.read_text(encoding="utf-8"))
    assert not ({"1a", "1b"} <= set(cp)), "мутант должен терять чужие ключи, иначе тест слеп"
