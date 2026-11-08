"""G38: `run_v4 --stage X --check` не меняет ни одного файла проекта. Стадии audit, A, B, Ca, Cb, E, CS, NTP, CLD, RQC устроены как «записать выходы -> проверить»; их `--check` выполняется в ПЕСОЧНИЦЕ:
каталоги выходов (docs/otchety/harj_hybrid_v4, data/regen/harj_hybrid_v4) копируются во временный каталог, пути модулей `s11` (REP, OUT, CP) на время проверки указывают на копию, стадия пересчитывается и проверяется там, копия удаляется.
Входы (сохранённые прогнозы, M10) читаются как есть. Стадии DX и H проверяют только чтение с диска без пересчёта и песочницы не используют."""
from __future__ import annotations

import contextlib
import shutil
import sys
import tempfile
from pathlib import Path

from models_v2.s11 import v4_b_cache as BC, v4_b_decl as DECL, v4_io as IO

SANDBOXED = ("audit", "A", "B", "Ca", "Cb", "E", "CS", "NTP", "CLD", "RQC")


def _swap(mapping: dict) -> None:
    """Во всех модулях `models_v2.s11` заменить атрибуты-пути, равные ключам mapping, на значения."""
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("models_v2.s11"):
            for name, val in list(vars(mod).items()):
                if isinstance(val, Path) and val in mapping:
                    setattr(mod, name, mapping[val])


@contextlib.contextmanager
def sandbox():
    tmp = Path(tempfile.mkdtemp(prefix="v4_check_"))
    real = {IO.REP: tmp / "rep", BC.OUT: tmp / "out"}
    for src, dst in real.items():
        shutil.copytree(src, dst, copy_function=shutil.copy2)       # mtime сохраняются: проверки порядка «объявлено раньше результатов» работают как на оригинале
    real[DECL.CP] = tmp / "rep" / DECL.CP.name
    back = {v: k for k, v in real.items()}
    _swap(real)
    try:
        yield tmp
    finally:
        _swap(back)
        shutil.rmtree(tmp, ignore_errors=True)
