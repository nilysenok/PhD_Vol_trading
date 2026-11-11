"""W1.8: `--check` только читает (хэш всех выходов и CHECKPOINT до и после равен); ключ W1 — слиянием, прочие ключи побайтно; ALL_TRIALS.csv не трогается; обучения нет (AST). Мутации: --check, который пишет; запись, затирающая прочие ключи."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from models_v2.s16 import x1_w1 as W

CK = json.dumps({"FWD": {"n": 1, "x": [1, 2]}, "P0": {"код": "b6d1", "sha256": {"a": "1"}}, "ZE": {"k": 0.1}}, indent=1, sort_keys=True, ensure_ascii=False)
MODULES = ("x1_w1.py", "x1_w1_join.py", "x1_w1_stat.py", "x1_w1_st.py", "x1_w1_doc.py")


@pytest.fixture(scope="module")
def real():
    return W.compute()


def others_preserved(old: str, new: str) -> bool:
    """Все ключи, кроме W1, в новом файле — побайтно как в старом (текст без W1 = старый текст)."""
    d = json.loads(new)
    d.pop("W1", None)
    return json.dumps(d, indent=1, sort_keys=True, ensure_ascii=False) == old and all(json.dumps(json.loads(old)[k], sort_keys=True) == json.dumps(json.loads(new)[k], sort_keys=True) for k in json.loads(old))


def test_checkpoint_merge_adds_only_w1_and_is_idempotent(real, tmp_path) -> None:
    ck = tmp_path / "CHECKPOINT.json"
    ck.write_text(CK, encoding="utf-8")
    texts = W.files(real)
    W.merge_checkpoint(ck, real, texts)
    new = ck.read_text(encoding="utf-8")
    assert others_preserved(CK, new) and set(json.loads(new)) == {"FWD", "P0", "ZE", "W1"}
    w1 = json.loads(new)["W1"]
    assert w1["configurations"] == 0 and w1["forecasts_written"] == 0 and w1["n_holm"] == 751 and set(w1["outputs"]) == set(texts) and len(w1["inputs"]) == 14 == len(W.input_shas())
    W.merge_checkpoint(ck, real, texts)
    assert ck.read_text(encoding="utf-8") == new                                                # повторная запись — тот же файл


def test_mutation_overwrite_by_one_key_is_caught(real, tmp_path) -> None:
    ck = tmp_path / "CHECKPOINT.json"
    ck.write_text(CK, encoding="utf-8")
    bad = json.dumps({"W1": {"x": 1}}, indent=1, sort_keys=True)                                 # мутант: файл затёрт одним ключом
    assert not others_preserved(CK, bad)
    W.merge_checkpoint(ck, real, W.files(real))
    assert others_preserved(CK, ck.read_text(encoding="utf-8"))


def test_check_reads_only_and_mutation_writing_check_is_caught(real, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(W, "compute", lambda: real)
    ck = tmp_path / "CHECKPOINT.json"
    ck.write_text(CK, encoding="utf-8")
    W.write(tmp_path, ck)
    h0 = W.tree_hash(tmp_path, ck)
    assert W.run_check(tmp_path, ck) == [] and W.tree_hash(tmp_path, ck) == h0                   # --check ничего не меняет
    assert set(h0) == set(W.OUTS) | {"CHECKPOINT.json"}

    def writing_check(out: Path = tmp_path) -> list:                                             # мутант: проверка пишет служебный ключ в CHECKPOINT
        ck.write_text(ck.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return []
    monkeypatch.setattr(W, "check", writing_check)
    assert W.run_check(tmp_path, ck) == ["--check изменил выходы"]
    monkeypatch.undo()


def test_real_check_changes_no_output_and_checkpoint() -> None:
    h0 = W.tree_hash()
    assert W.run_check() == []
    assert W.tree_hash() == h0 and set(W.OUTS) <= set(h0)


def test_modules_do_not_touch_registry_or_train() -> None:
    """W1 не регистрирует испытаний и не обучает: нет ALL_TRIALS, импортов регистра и обучателей, вызовов fit/train."""
    for m in MODULES:
        src = (Path(W.__file__).parent / m).read_text(encoding="utf-8")
        tree = ast.parse(src)
        names = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        assert "ALL_TRIALS" not in src and not any(x in names for x in ("x1_reg", "xgboost", "lightgbm", "torch")), m
        calls = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "") for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert not calls & {"fit", "train", "refit", "to_parquet"}, m                            # прогнозов не пишется
        assert len(src.splitlines()) <= 200, m
