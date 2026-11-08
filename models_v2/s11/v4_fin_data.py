"""V4-FIN: реестр 46 конфигураций программы и чтение их СОХРАНЁННЫХ прогнозов (`data/regen/harj_hybrid_v4/*.parquet`). Ничего не обучает и не пишет parquet.
Конфигурации — по этапам: B 5 (B5 не считался, правило), C 10, E 6, CS 2, NTP 4, CLD 4, K8 + RQC 3, H 8, I 3 = 46 (K8 — одна из 46, она же контроль-ориентир K8). Контроли: k1 (HAR-J), k2 (XGB), k3 (LGBM), d2 (LGBM-D2)."""
from __future__ import annotations

import hashlib

import pandas as pd

from models_v2.s11 import v4_b_cache as BC, v4_b_data as D, v4_io as IO

STAGES = {"B": ["B1", "B2", "B3", "B4", "B6"], "C": ["C%d" % i for i in range(1, 11)], "E": ["E%d" % i for i in range(1, 7)], "CS": ["CS1", "CS2"], "NTP": ["NTP%d" % i for i in range(1, 5)],
          "CLD": ["CLD%d" % i for i in range(1, 5)], "RQC": ["K8", "RQC1", "RQC2", "RQC3"], "H": ["H%d" % i for i in range(1, 9)], "I": ["I1", "I2", "I3"]}
CFGS = [c for v in STAGES.values() for c in v]
CTRL = ["k1", "k2", "k3", "d2"]
STAGE_OF = {c: s for s, v in STAGES.items() for c in v}
OUT = BC.OUT


def _sha_of(files) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(f.name.encode())
        h.update(IO.sha256(f).encode())
    return h.hexdigest()


def parquet_sha() -> str:
    """Хэш ВСЕХ parquet программы (прогнозы, веса, кэш): FIN их не пишет и не меняет — сверка в `--check`."""
    return _sha_of(OUT.glob("*.parquet"))


def tables_sha() -> str:
    """Хэш прежних таблиц этапов (`docs/otchety/harj_hybrid_v4/*.csv` кроме FIN_*): FIN их не меняет."""
    return _sha_of(f for f in IO.REP.glob("*.csv") if not f.name.startswith("FIN_"))


def file_of(c: str) -> str:
    return "B_%s.parquet" % c if c.startswith("B") else "%s.parquet" % c


def load_all() -> dict:
    """{конфигурация: прогнозы (ticker, date, F, …)} — только чтение сохранённых parquet."""
    return {c: pd.read_parquet(OUT / file_of(c)) for c in CFGS}


def frame(P: pd.DataFrame | None = None) -> tuple:
    """-> (панель P, широкая таблица оценки k: контроли + 46 рядов на тех же строках)."""
    P = D.panel() if P is None else P
    return P, D.eval_frame(P, load_all())
