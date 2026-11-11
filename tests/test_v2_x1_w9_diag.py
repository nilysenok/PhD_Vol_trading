"""W9, диагностика: пересчёт `W9_DIAG.csv` равен файлу побайтно (тело, без строк шапки); `--check` только читает — хэши всех файлов
`docs/otchety/harj_x1` и `data/regen/harj_x1` до и после равны; мутация (один бит тела) ловится; sha256 файла — 9e46efd2328084b2…
Один пересчёт на модуль (около 16 с)."""
from __future__ import annotations

import hashlib
import sys

import pytest

from models_v2.s16 import x1_w9_diag as D
from models_v2.s16 import x1_w9_diag_core as K

DIRS = (K.ROOT / "docs/otchety/harj_x1", K.ROOT / "data/regen/harj_x1")


def tree_hash() -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for d in DIRS for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def run():
    before = tree_hash()
    df, _ = D.compute()
    ok, sha = D.check()
    return before, tree_hash(), df, ok, sha


def test_recompute_equals_csv_bytewise(run) -> None:
    _, _, df, ok, sha = run
    assert ok                                                                  # check(): тело файла == fmt(пересчёт)
    assert D.body_of(D.OUT.read_text(encoding="utf-8")) == D.fmt(df)
    assert sha.startswith("9e46efd2328084b2")


def test_check_only_reads_hashes_equal(run) -> None:
    before, after, _, _, _ = run
    assert len(before) > 150 and before == after                                # 198 файлов, ни один не изменён ни `compute`, ни `check`


def test_mutation_one_value_caught(run) -> None:
    _, _, df, _, _ = run
    body = D.fmt(df)
    lines = D.OUT.read_text(encoding="utf-8").splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("T,W9A,T1,,d_X1_pct,"))
    head, val = lines[i].rsplit(",", 1)
    lines[i] = head + "," + "%.10g" % (float(val) + 1e-6)
    assert D.body_of("\n".join(lines) + "\n") != body                          # правка 1e-6 в одной ячейке копии — расхождение (файл не пишется)


def test_cli_modes_check_and_usage_write_nothing(run, monkeypatch, capsys) -> None:
    before = run[0]
    monkeypatch.setattr(sys, "argv", ["x1_w9_diag"])                           # без аргументов — только справка, `--write` не включается
    D.main()
    assert "--check" in capsys.readouterr().out
    assert tree_hash() == before
