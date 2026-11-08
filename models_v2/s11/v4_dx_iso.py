"""V4-DX, G26: диагностика (`v4_dx_*`) — задним числом по факту RV; пути прогноза (`models_v2/s10`, остальные модули `s11`) её не импортируют. Разрешены: сами модули `v4_dx_*` и драйвер `run_v4`.
Дополняет `v4_isolation` (oracle), не меняя его."""
from __future__ import annotations

import ast

from models_v2.s11.v4_isolation import ROOT

PREFIX = "v4_dx_"


def imports_dx(source: str) -> bool:
    """True, если код импортирует модуль `v4_dx_*` любым способом (import / from … import, в т. ч. относительный)."""
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Import) and any(PREFIX in p for a in n.names for p in a.name.split(".") if p.startswith(PREFIX)):
            return True
        if isinstance(n, ast.ImportFrom) and (any(p.startswith(PREFIX) for p in (n.module or "").split(".")) or any(a.name.startswith(PREFIX) for a in n.names)):
            return True
    return False


def violators() -> list:
    """Файлы путей прогноза, импортирующие диагностику DX: все `s10/*.py` и `s11/*.py` вне `v4_dx_*.py` и `run_v4.py`."""
    bad = []
    for d in ("s10", "s11"):
        for p in sorted((ROOT / "models_v2" / d).glob("*.py")):
            if not p.name.startswith(PREFIX) and p.name != "run_v4.py" and imports_dx(p.read_text(encoding="utf-8")):
                bad.append("%s/%s" % (d, p.name))
    return bad
