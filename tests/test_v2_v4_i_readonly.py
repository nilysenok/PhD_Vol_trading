"""V4-I (G38 для стадии I): `run_v4 --stage I --check` только читает — ни один файл выходов и CHECKPOINT.json не меняется (хэши до и после); проверка не вызывает запись (`write_all`, `DECL.update`, `write_trials`);
реестр: строки I1–I3 пишутся только через `v4_trials.replace_rows` и не затирают чужие; ключ кэша меняется при подмене входа и кода (Д5); `--check` прежних стадий допускает строки I. Выходы стадии должны уже лежать на диске."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from models_v2.s11 import run_v4, v4_b_decl as DECL, v4_i_cache as IC_, v4_i_stage as ST, v4_trials as TR
from tests.test_v2_v4_check_readonly import ROOT, changed, snapshot


def test_stage_i_check_does_not_change_any_file():
    before = snapshot()
    r = subprocess.run([sys.executable, "-W", "ignore", "-m", "models_v2.s11.run_v4", "--stage", "I", "--check"], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "ИТОГ: OK" in r.stdout, r.stdout[-800:] + r.stderr[-800:]
    assert changed(before, snapshot()) == []


def test_stage_i_check_never_calls_write_paths(monkeypatch, capsys):
    from models_v2.s11 import v4_i_write as W
    calls = []
    monkeypatch.setattr(W, "write_all", lambda o: calls.append("write_all"))
    monkeypatch.setattr(DECL, "update", lambda **kw: calls.append(("update", sorted(kw))))
    monkeypatch.setattr(ST, "write_trials", lambda key: calls.append("write_trials"))
    assert run_v4.LATE["I"] == ("v4_i_main", "main_i")
    assert run_v4.main(["--stage", "I", "--check"]) == 0
    assert calls == []


def test_cache_miss_in_check_mode_raises_and_writes_nothing(tmp_path, monkeypatch):
    """Мутация «проверка пишет кэш»: при write=False промах ключа — исключение, файл не появляется."""
    monkeypatch.setattr(IC_, "OUT", tmp_path)
    with pytest.raises(RuntimeError):
        IC_.cached("I9", "k", lambda: {"": None}, write=False)
    assert list(Path(tmp_path).iterdir()) == []


def test_replace_rows_keeps_other_stages_and_has_no_duplicates(tmp_path):
    f = tmp_path / "ALL_TRIALS.csv"
    f.write_text(TR.HEADER + "Ж-222:H1:aaa,x\nЖ-222:I1:old,x\nЖ-222:K8:bbb,x\n")
    TR.replace_rows(["Ж-222:%s:" % c for c in ST.CFGS], ST.trials_rows("key" * 8), path=f)
    TR.replace_rows(["Ж-222:%s:" % c for c in ST.CFGS], ST.trials_rows("key" * 8), path=f)         # повтор идемпотентен
    keys = [x.split(",")[0].split(":")[1] for x in f.read_text().splitlines()[1:]]
    assert sorted(keys) == sorted(["H1", "K8", "I1", "I2", "I3"]) and "old" not in f.read_text()


def test_cache_key_changes_with_input_and_code(tmp_path):
    a, b, c = tmp_path / "a.parquet", tmp_path / "b.py", tmp_path / "c.py"
    a.write_bytes(b"1")
    b.write_bytes(b"x")
    c.write_bytes(b"y")
    k0 = IC_.keys(extra_in=[a], extra_code=[b])["key"]
    a.write_bytes(b"2")
    assert IC_.keys(extra_in=[a], extra_code=[b])["key"] != k0                    # подмена входа -> промах кэша
    k1 = IC_.keys(extra_in=[a], extra_code=[b])["key"]
    b.write_bytes(b"z")
    assert IC_.keys(extra_in=[a], extra_code=[b])["key"] != k1                    # правка кода -> промах кэша
    assert IC_.keys(extra_in=[a], extra_code=[b, c])["key"] != IC_.keys(extra_in=[a], extra_code=[b])["key"]
