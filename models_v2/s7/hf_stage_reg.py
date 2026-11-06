"""Стадия reg: свод реестров Ж-197…Ж-210 -> ALL_TRIALS.csv, CHECKPOINT.json (N_prior объявляется до E-статистики)."""
from __future__ import annotations

from models_v2.s7 import hf_ledger as L


def stage(_a=None) -> int:
    c = L.init()
    print("REG:", {k: c[k] for k in ("n_zh204", "n_zh206_210", "n_pilot", "N_prior", "N_ref_288", "used_total", "N_final")}, flush=True)
    return 0
