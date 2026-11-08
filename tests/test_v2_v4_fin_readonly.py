"""V4-FIN (G47): `run_v4 --stage FIN --check` только ЧИТАЕТ — ни файла выходов (docs/otchety/harj_hybrid_v4, data/regen/harj_hybrid_v4), ни итогового документа, ни CHECKPOINT.json (хэши до и после); проверка не вызывает пути записи;
FIN не импортирует oracle/диагностику путями прогноза; прежние parquet и таблицы этапов побитно прежние (хэши при объявлении = текущие); мутация «проверка пишет» ловится детектором. Тест запускает настоящую проверку подпроцессом (в сеть не ходит)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

from models_v2.s11 import run_v4, v4_b_decl as DECL, v4_fin_data as FD, v4_fin_stage as ST, v4_fin_write as W, v4_isolation as ISO
from tests.test_v2_v4_check_readonly import DIRS, ROOT, changed, snapshot


def snap() -> dict:
    s = snapshot(DIRS)
    s[str(W.FINAL)] = hashlib.sha256(W.FINAL.read_bytes()).hexdigest()
    return s


def test_stage_fin_check_does_not_change_any_file():
    before = snap()
    r = subprocess.run([sys.executable, "-W", "ignore", "-m", "models_v2.s11.run_v4", "--stage", "FIN", "--check"], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "ИТОГ: OK" in r.stdout, r.stdout[-800:] + r.stderr[-800:]
    assert changed(before, snap()) == []


def test_stage_fin_check_never_calls_write_paths(monkeypatch):
    from models_v2.s11 import v4_fin_write
    calls = []
    monkeypatch.setattr(v4_fin_write, "write_docs", lambda o: calls.append("write_docs"))
    monkeypatch.setattr(ST, "write_tables", lambda o: calls.append("write_tables"))
    monkeypatch.setattr(DECL, "update", lambda **kw: calls.append(("update", sorted(kw))))
    monkeypatch.setattr(ST, "declare", lambda: calls.append("declare"))
    assert run_v4.LATE["FIN"] == ("v4_fin_main", "main_fin")
    assert run_v4.main(["--stage", "FIN", "--check"]) == 0
    assert calls == []


def test_detector_catches_check_that_writes(tmp_path):
    d = tmp_path / "o"
    d.mkdir()
    (d / "FIN_ALL.csv").write_text("a\n1\n")
    before = snapshot([d])
    (d / "FIN_ALL.csv").write_text("a\n2\n")                                      # мутация: «проверка» переписала таблицу
    assert changed(before, snapshot([d])) == ["%s" % (d / "FIN_ALL.csv")]


def test_prior_forecasts_and_stage_tables_are_bit_identical_to_declared():
    cp = json.loads(DECL.CP.read_text())
    assert cp["FIN_prior_parquet_sha"] == FD.parquet_sha() and cp["FIN_prior_tables_sha"] == FD.tables_sha()
    assert cp["FIN_configs_spent"] == 0


def test_fin_modules_do_not_import_oracle_and_write_no_parquet():
    assert ISO.violators() == []
    import pathlib
    for p in pathlib.Path(ROOT / "models_v2/s11").glob("v4_fin_*.py"):
        src = p.read_text(encoding="utf-8")
        assert "to_parquet" not in src and "v4_oracle" not in src, p.name
        assert len(src.splitlines()) <= 200, p.name
