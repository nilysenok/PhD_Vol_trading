"""Чистый процесс: замыкание импортов цепочки открытия (как `fwd_frz_parts.import_chain`) → JSON {путь: sha256} в stdout. Нужен tests/test_fwd_l5_freeze.py, чтобы
результат не зависел от модулей, которые импортировали другие тесты процесса pytest."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: F401,E402  (первым: общий OpenMP)
from models_v2.s17 import fwd_frz_parts as P  # noqa: E402

if __name__ == "__main__":
    P.import_chain()
    print(json.dumps(P.loaded_files()))
