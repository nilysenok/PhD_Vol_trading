"""Раннер аудита fastbt: каждый модуль test_NN_*.py — чистый код (ждём OK) и каждая его мутация
из MUTATIONS (ждём падения). Печатает таблицу «проверка → чистый код / мутация поймана».

python3 tests/audit_fastbt/run_mutations.py [--md out.md]
Код возврата 0 — чистый код зелёный и все мутации пойманы.
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(path: Path, mutation: str | None) -> tuple[int, str]:
    env = dict(os.environ)
    env.pop("AUDIT_MUTATION", None)
    if mutation:
        env["AUDIT_MUTATION"] = mutation
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(path)],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["?"])[-1]
    return r.returncode, last


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", help="записать таблицу в markdown-файл")
    ap.add_argument("--only", default="test_*.py",
                    help="маска модулей (FA2b: --only 'test_2[0-9]_*.py')")
    a = ap.parse_args(argv)
    sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "scripts" / "analysis")]
    rows, ok_all = [], True
    for path in sorted(HERE.glob(a.only)):
        mod = importlib.import_module(path.stem)
        title = (mod.__doc__ or path.stem).strip().splitlines()[0]
        rc, last = run(path, None)
        clean = rc == 0
        ok_all &= clean
        for m in mod.MUTATIONS:
            mrc, mlast = run(path, m)
            caught = mrc != 0
            ok_all &= caught
            rows.append((title, "OK" if clean else f"ПАДАЕТ ({last})", m,
                         "поймана" if caught else "НЕ ПОЙМАНА", mlast))
    head = "| проверка | чистый код | мутация | результат | pytest на мутации |\n|---|---|---|---|---|"
    lines = [head] + [f"| {t} | {c} | `{m}` | {r} | {x} |" for t, c, m, r, x in rows]
    n_m = len(rows)
    n_c = sum(r[3] == "поймана" for r in rows)
    lines.append(f"\nмутаций: {n_m}, поймано: {n_c}; чистый код: "
                 f"{'все модули OK' if all(r[1] == 'OK' for r in rows) else 'есть падения'}")
    out = "\n".join(lines)
    print(out)
    if a.md:
        Path(a.md).write_text(out + "\n", encoding="utf-8")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
