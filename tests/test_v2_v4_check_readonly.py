"""V4, G38: `run_v4 --stage X --check` только ЧИТАЕТ: ни один файл выходов (docs/otchety/harj_hybrid_v4, data/regen/harj_hybrid_v4) и CHECKPOINT.json не меняются (хэши до и после) — для всех стадий, в т. ч. DX.
Дефект до исправления: `--stage DX --check` пересчитывал стадию и перезаписывал `DX_runtime_sec` в CHECKPOINT. Мутация «проверка пишет служебный ключ» ловится детектором (хэши).
Тест запускает настоящие проверки подпроцессами (в сеть не ходит, данные ≥ 2022 как входы стадий — только чтение, как в самих стадиях); прежние выходы должны уже лежать на диске."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from models_v2.s11 import run_v4, v4_b_decl as DECL, v4_io as IO

ROOT = Path(__file__).resolve().parents[1]
DIRS = [IO.REP, ROOT / "data/regen/harj_hybrid_v4"]
STAGES = ["audit", "A", "B", "Ca", "Cb", "E", "DX", "CS", "NTP", "CLD", "RQC", "H"]


def snapshot(dirs=None) -> dict:
    """{путь: sha256} всех файлов каталогов выходов (вложенные — тоже)."""
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for d in (dirs or DIRS) for p in sorted(Path(d).rglob("*")) if p.is_file()}


def changed(a: dict, b: dict) -> list:
    return sorted({k for k in set(a) | set(b) if a.get(k) != b.get(k)})


def test_detector_catches_service_key_written_by_check(tmp_path):
    """Мутация: «проверка» дописывает служебный ключ в CHECKPOINT (как делал DX_runtime_sec) — хэши до/после различаются; честная проверка (только чтение) — нет."""
    d = tmp_path / "out"
    d.mkdir()
    cp = d / "CHECKPOINT.json"
    cp.write_text(json.dumps({"N_prior": 634}))
    before = snapshot([d])
    json.loads(cp.read_text())                                                    # честная проверка: читает
    assert changed(before, snapshot([d])) == []
    cp.write_text(json.dumps({**json.loads(cp.read_text()), "DX_runtime_sec": 17.4}))      # мутация: пишет служебный ключ
    assert changed(before, snapshot([d])) == [str(cp)]


def test_dx_check_does_not_call_stage_or_update(tmp_path, monkeypatch, capsys):
    """Боевой путь DX на копии CHECKPOINT: `main_dx(check=True)` не вызывает ни `stage_dx`, ни `DECL.update`; байты копии не меняются. Мутация (старый код: пересчёт стадии в проверке) меняет копию."""
    from models_v2.s11 import v4_dx_stage
    cp = tmp_path / "CHECKPOINT.json"
    shutil.copy(DECL.CP, cp)
    monkeypatch.setattr(DECL, "CP", cp)
    calls = []
    real_update = DECL.update
    monkeypatch.setattr(DECL, "update", lambda **kw: (calls.append(kw), real_update(**kw)))
    monkeypatch.setattr(v4_dx_stage, "stage_dx", lambda: calls.append("stage_dx"))
    before = cp.read_bytes()
    assert run_v4.main_dx(True) == 0
    assert calls == [] and cp.read_bytes() == before
    DECL.update(DX_runtime_sec=1.0)                                               # мутация: старая проверка писала время счёта
    assert cp.read_bytes() != before and calls == [{"DX_runtime_sec": 1.0}]


def test_sandbox_redirects_writes_and_restores_paths():
    """Песочница `--check`: запись выходов стадией внутри неё не меняет проект; пути модулей возвращаются; временный каталог удалён. Мутация «запись мимо песочницы» (по абсолютному пути проекта) ловится хэшами."""
    import pandas as pd
    from models_v2.s11 import v4_b_cache as BC, v4_sandbox as SB
    rep0, out0, cp0 = IO.REP, BC.OUT, DECL.CP
    before = snapshot()
    with SB.sandbox() as tmp:
        assert IO.REP == tmp / "rep" and BC.OUT == tmp / "out" and DECL.CP == tmp / "rep" / "CHECKPOINT.json"
        IO.write_csv(pd.DataFrame({"a": [1]}), "SANDBOX_PROBE.csv", "проба")
        DECL.update(PROBE_KEY=1)
        assert (tmp / "rep" / "SANDBOX_PROBE.csv").exists()
    assert (IO.REP, BC.OUT, DECL.CP) == (rep0, out0, cp0) and not tmp.exists()
    assert changed(before, snapshot()) == []
    (IO.REP / "SANDBOX_PROBE.csv").write_text("мимо песочницы")                  # мутация: запись по пути проекта
    try:
        assert changed(before, snapshot()) == [str(IO.REP / "SANDBOX_PROBE.csv")]
    finally:
        (IO.REP / "SANDBOX_PROBE.csv").unlink()


@pytest.mark.parametrize("stage", STAGES)
def test_check_of_every_stage_changes_no_output_file_and_no_checkpoint(stage):
    before = snapshot()
    r = subprocess.run([sys.executable, "-W", "ignore", "-m", "models_v2.s11.run_v4", "--stage", stage, "--check"], cwd=str(ROOT), capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stdout[-600:] + r.stderr[-600:]
    assert "ИТОГ: OK" in r.stdout or "все проверки пройдены" in r.stdout
    assert changed(before, snapshot()) == []
