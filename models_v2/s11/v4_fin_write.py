"""V4-FIN: запись документов (только `--stage FIN`, не `--check`): `docs/otchety/HARJ_HYBRID_V4_FINAL.md` и `docs/otchety/harj_hybrid_v4/REPRODUCTION_V4.md`. Документ собирается из `FIN_*.csv` на диске (`v4_fin_doc.tables_from_disk`)."""
from __future__ import annotations

from models_v2.s11 import v4_fin_build as B, v4_fin_doc as DOC, v4_io as IO

FINAL = IO.REP.parent / "HARJ_HYBRID_V4_FINAL.md"
REPRO = IO.REP / "REPRODUCTION_V4.md"


def write_docs(o: dict) -> None:
    t = DOC.tables_from_disk()
    FINAL.write_text(B.build(t), encoding="utf-8")
    REPRO.write_text(B.reproduction(), encoding="utf-8")
