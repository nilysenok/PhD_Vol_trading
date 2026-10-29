#!/usr/bin/env python3
"""Версии пакетов в окружении = пины requirements (Ж-131 п. 3: те же версии, что на Маке). Только стандартная библиотека.
    python3 versii.py requirements_server.txt   ->  OK versii: python …, numba …, numpy …, pandas …, pyarrow …, scipy …  | FAIL …"""
import platform
import sys
from importlib import metadata

GLAVNYE = ("numba", "llvmlite", "numpy", "pandas", "pyarrow", "scipy")


def piny(path: str) -> dict:
    out = {}
    for s in open(path, encoding="utf-8"):
        s = s.split("#", 1)[0].strip()
        if s:
            if "==" not in s:
                raise SystemExit(f"FAIL versii: строка без пина == : {s}")
            k, v = s.split("==", 1)
            out[k.strip()] = v.strip()
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p, bad = piny(argv[0]), []
    for k, v in p.items():
        try:
            got = metadata.version(k)
        except metadata.PackageNotFoundError:
            got = "нет"
        if got != v:
            bad.append(f"{k}: нужно {v}, стоит {got}")
    py = platform.python_version()
    if not py.startswith("3.9."):
        bad.append(f"python {py}, нужен 3.9.x (Мак 3.9.6)")
    if bad:
        print("FAIL versii: " + "; ".join(bad))
        return 1
    print(f"OK versii: python {py} ({platform.machine()}), " + ", ".join(f"{k} {p[k]}" for k in GLAVNYE if k in p)
          + f"; пинов {len(p)}, все совпали")
    return 0


if __name__ == "__main__":
    sys.exit(main())
