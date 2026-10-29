"""Находка #639: `tests/test_svod_k1_prefiks.py` импортировал `tests.test_svod_server_v5` и не собирался, если корень репозитория
не в sys.path (запуск `pytest` из другого каталога). Тест запускает сбор в отдельном процессе: fastbt виден через симлинк,
корня репозитория и cwd в пути нет. Мутация: вернуть `from tests.test_svod_server_v5 import …` — тест падает."""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _sobrat(tmp: Path, fajl: str) -> subprocess.CompletedProcess:
    pkg = tmp / "pkg"
    pkg.mkdir()
    (pkg / "fastbt").symlink_to(REPO / "fastbt")
    rab = tmp / "rab"
    rab.mkdir()
    env = {**os.environ, "PYTHONPATH": str(pkg)}
    return subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(REPO / "tests" / fajl)],
                          cwd=rab, env=env, capture_output=True, text=True)


def test_svod_k1_prefiks_sobiraetsya_ne_iz_kornya(tmp_path):
    r = _sobrat(tmp_path, "test_svod_k1_prefiks.py")
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-400:]
    assert "test_svod_k1_prefiks.py::test_" in r.stdout
