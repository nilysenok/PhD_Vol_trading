"""G8 (в): oracle (`v4_oracle`) — только диагностика; пути прогноза (`models_v2/s10`, мета-слой и прочие модули `s11`) его не импортируют. Разрешены: сам модуль, `v4_a` (сводка oracle), `run_v4` (драйвер)."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = {"v4_oracle.py", "v4_a.py", "run_v4.py"}
NAME = "v4_oracle"


def imports_oracle(source: str) -> bool:
    """True, если код импортирует модуль oracle любым способом (import / from … import, в т. ч. относительный)."""
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Import) and any(NAME in a.name.split(".") for a in n.names):
            return True
        if isinstance(n, ast.ImportFrom) and (NAME in (n.module or "").split(".") or any(a.name == NAME for a in n.names)):
            return True
    return False


def violators() -> list:
    """Файлы путей прогноза, импортирующие oracle: все `s10/*.py` и `s11/*.py` вне ALLOWED."""
    bad = []
    for d in ("s10", "s11"):
        for p in sorted((ROOT / "models_v2" / d).glob("*.py")):
            if p.name not in ALLOWED and imports_oracle(p.read_text(encoding="utf-8")):
                bad.append("%s/%s" % (d, p.name))
    return bad
