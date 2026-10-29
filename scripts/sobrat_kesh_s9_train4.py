"""Пересборка эталонного кэша s9 train4 (путь A, без σ-источников) на текущей базе (после #678).
Генератор тот же, что собирал кэш 28.10: s9.build_cache при set_period("train4"), scope dev, σ h1 final2 (суффикс пуст).
Запуск: python3 scripts/sobrat_kesh_s9_train4.py [--workers 4]. Пишет cache_<ключ>.pkl и указатель cache_latest_train4.txt в s9.OUT.
Старый кэш — в data/fastbt/staryj_kesh_s9_2026-10-29_div/ (SHA256.txt, PRICHINA.txt)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastbt import s9  # noqa: E402


def main(workers: int) -> Path:
    s9.use_sigma_h1("final2")
    s9.set_period("train4")
    assert s9._latest_name() == "cache_latest_train4.txt", s9._latest_name()
    return s9.build_cache(workers=workers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if not 1 <= a.workers <= 4:
        raise SystemExit("workers: 1..4 (лимит процессов задания)")
    print(main(a.workers))
