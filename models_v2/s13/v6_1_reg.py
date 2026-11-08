"""V6-1a: реестр конфигураций и `ALL_TRIALS.csv` (строка = конфигурация `Ж-227:<id>`; чувствительность λ/деревьев — соседние точки, не конфигурации, в реестр и N не входят), шапки таблиц листа 1a, хэш кода v6_1 (Д5).
Следующий лист добавляет СТРОКУ в `CONFIGS` (набор признаков, обучатель, база правила 2.2) и не переписывает движок."""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import pandas as pd

from models_v2.s13 import v6_0_cp as CP0, v6_0_io as IO

CODE_VERSION = "v6.1a"
STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-227, лист 1a (SBER, h = 1; инфраструктура и абляционная база A0)"
PLAN = "docs/otchety/HARJ_HYBRID_SBER_V6_PLAN.md"
CONFIGS = {"A0": dict(direction="A0_baseline", features=["S"], learner="LIN", base_rule22="K1 (справочно; в листе 1b база A1/A2 — A0)", source=PLAN + " п. 3 (A0), 2.1, 2.2", kind="trial")}
CTRL = ["K1", "K2", "K3", "K8", "V4:NTP2"]


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    """Хэш кода листов 1+: все `v6_1_*.py` и `run_v6.py` (Д5)."""
    h = hashlib.sha256()
    d = Path(__file__).parent
    for f in sorted(d.glob("v6_1_*.py")) + [d / "run_v6.py"]:
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def header(note: str) -> str:
    return "# %s; код %s %s; входы %s; %s; %s\n" % (STATUS, CODE_VERSION, code_hash(), IO.inputs_hash(), IO.CAVEAT, note)


def write_csv(df: pd.DataFrame, name: str, note: str) -> Path:
    IO.REP.mkdir(parents=True, exist_ok=True)
    p = IO.REP / name
    with open(p, "w", encoding="utf-8") as f:
        f.write(header(note))
        df.to_csv(f, index=False, lineterminator="\n")
    return p


def trial_text(ids: list) -> str:
    """Текст ALL_TRIALS.csv: заголовок листа 0 и строки конфигураций ids (в порядке ids)."""
    rows = []
    for i in ids:
        c = CONFIGS[i]
        cfg = json.dumps({"cfg": i, "features": c["features"], "learner": c["learner"], "base_rule22": c["base_rule22"], "base": "H=K1", "form": "F=H*exp(clip(g,-1,1))"}, ensure_ascii=False)
        rows.append(pd.DataFrame([["Ж-227:%s" % i, "Ж-227", c["source"], c["direction"], cfg, c["kind"]]]).to_csv(header=False, index=False, lineterminator="\n"))
    return IO.HEADER_TRIALS + "".join(rows)


def write_trials(ids: list) -> None:
    CP0.ALL_TRIALS.write_text(trial_text(ids), encoding="utf-8")


def trials_ok(ids: list) -> bool:
    return CP0.ALL_TRIALS.exists() and CP0.ALL_TRIALS.read_text(encoding="utf-8") == trial_text(ids)
