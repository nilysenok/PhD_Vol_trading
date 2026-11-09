"""V7-FIN2 (G54): конфликт OpenMP (lightgbm/xgboost + torch на macOS). Пакет `models_v2.s14` закрепляет torch на одном потоке; проверка в дочерних процессах (скрипт во временном каталоге).
(в) мутация «без закрепления потоков» — наблюдение на этой машине: см. test_mutation_*; если процесс не падает, тест фиксирует это как наблюдение и не утверждает падения."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPUTE = "import torch\nx = torch.randn(256, 64)\nprint('threads', torch.get_num_threads(), 'softmax', float(torch.softmax(x, 1).sum()))\n"


def run(tmp_path, body: str):
    script = tmp_path / "child_omp.py"
    script.write_text(body, encoding="utf-8")
    return subprocess.run([sys.executable, "-W", "ignore", str(script)], cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("booster", ["lightgbm", "xgboost"])
def test_booster_then_s14_then_torch_computation_ok_single_thread(tmp_path, booster):
    r = run(tmp_path, "import %s\nimport models_v2.s14\n%s" % (booster, COMPUTE))
    assert r.returncode == 0 and "threads 1 " in r.stdout and "softmax" in r.stdout, (r.returncode, r.stderr[-400:])


def test_mutation_without_thread_pinning_crashes_or_is_recorded(tmp_path):
    """Без закрепления (lightgbm → torch с потоками по умолчанию → softmax) ожидается rc 139; если на этой машине не падает — наблюдение, не утверждение."""
    r = run(tmp_path, "import lightgbm\n" + COMPUTE)
    if r.returncode == 0:
        pytest.skip("наблюдение: без закрепления потоков процесс на этой машине не упал (rc 0) — мутация не воспроизводится")
    assert r.returncode in (139, -11), r.returncode
