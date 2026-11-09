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
CONFIGS = {"A0": dict(direction="A0_baseline", features=["S"], learner="LIN", base_rule22="K1 (справочно; в листе 1b база A1/A2 — A0)", source=PLAN + " п. 3 (A0), 2.1, 2.2", kind="trial"),
           # лист 1b: строки конфигураций A1–A3 (набор признаков, обучатель, база правила 2.2); движок не менялся
           "A1": dict(direction="A_path_signature", features=["S", "Sigma"], learner="LIN", base_rule22="A0", source=PLAN + " п. 3 (A1), 2.2", kind="trial"),
           "A2": dict(direction="A_path_signature", features=["S", "Sigma"], learner="GBM", base_rule22="A0 и A1", source=PLAN + " п. 3 (A2), 2.2", kind="trial"),
           "A3": dict(direction="A_path_signature", features=["S", "Sigma", "Sigma3"], learner="LIN", base_rule22="A1 (условная: A1 или A2 несёт новую информацию против A0)", source=PLAN + " п. 3 (A3), 2.2", kind="trial"),
           # лист 2: строки конфигураций C1–C4 (набор признаков, обучатель, база правила 2.2); движок не менялся
           "C1": dict(direction="C_measurement_uncertainty", features=["S", "Q"], learner="LIN", base_rule22="A0", source=PLAN + " п. 4 (C1), 2.2", kind="trial"),
           "C2": dict(direction="C_measurement_uncertainty", features=["S", "Q"], learner="GBM", base_rule22="A0 и C1", source=PLAN + " п. 4 (C2), 2.2", kind="trial"),
           "C3": dict(direction="C_measurement_uncertainty", features=["S", "Q", "u*(s1,s2,s3)"], learner="LIN", base_rule22="C1 (и A0)", source=PLAN + " п. 4 (C3), 2.2", kind="trial"),
           "C4": dict(direction="C_measurement_uncertainty", features=["S", "Q", "u*(s1,s2,s3)"], learner="GBM", base_rule22="C3 и A0 (условная: C1, C2 или C3 несёт новую информацию против A0)", source=PLAN + " п. 4 (C4), 2.2", kind="trial"),
           # лист 3: строки конфигураций E1–E3 (набор признаков, обучатель, база правила 2.2); движок не менялся
           "E1": dict(direction="BE_time_reversal_shock_recovery", features=["S", "R"], learner="LIN", base_rule22="A0", source=PLAN + " п. 5 (E1), 2.2", kind="trial"),
           "E2": dict(direction="BE_time_reversal_shock_recovery", features=["S", "R"], learner="GBM", base_rule22="A0 и E1", source=PLAN + " п. 5 (E2), 2.2", kind="trial"),
           "E3": dict(direction="BE_time_reversal_shock_recovery", features=["S", "R", "Sigma"], learner="LIN", base_rule22="A1 и E1 (условная: E1 или E2 и A1 или A2 несут новую информацию против A0)", source=PLAN + " п. 5 (E3), 2.2", kind="trial"),
           # лист 4: строки конфигураций D1–D3 (состояние Z_D, аналоги без обучателя, база правила 2.2 — контроль K9, не строка реестра); движок не менялся
           "D1": dict(direction="D_historical_analog", features=["Z_D"], learner="analog_k50", base_rule22="K9 (контроль без бюджета)", source=PLAN + " п. 6 (D1), 2.2", kind="trial"),
           "D2": dict(direction="D_historical_analog", features=["Z_D"], learner="analog_kernel_k100", base_rule22="K9 (контроль без бюджета)", source=PLAN + " п. 6 (D2), 2.2", kind="trial"),
           "D3": dict(direction="D_historical_analog", features=["Z_D"], learner="analog_shrink_kappa50", base_rule22="K9 (контроль без бюджета)", source=PLAN + " п. 6 (D3), 2.2", kind="trial")}
DECLARED = ["A0", "A1", "A2", "A3", "C1", "C2", "C3", "C4", "E1", "E2", "E3", "D1", "D2", "D3", "CB1", "CB2", "F1", "F2"]      # 18 id плана, п. 8
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


def trials_ok(ids: list, exact: bool = False) -> bool:
    """exact — файл = строки ids; иначе — строки ids стоят первыми, дальше только строки следующих листов из списка плана (лист 2 дописывает C1–C4 после A0–A3)."""
    if not CP0.ALL_TRIALS.exists():
        return False
    txt = CP0.ALL_TRIALS.read_text(encoding="utf-8")
    return txt == trial_text(ids) if exact else txt.startswith(trial_text(ids)) and not trials_bad(txt)


def trials_bad(text: str, ids_ok=DECLARED) -> list:
    """Строки ALL_TRIALS: заголовок прежний; ключи только `Ж-227:<id>` из списка плана (id в CONFIGS: текст строки побитно прежний), без дублей. -> список нарушений (пусто = ок)."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0] != IO.HEADER_TRIALS:
        return ["заголовок ALL_TRIALS изменён"]
    keys = [ln.split(",", 1)[0] for ln in lines[1:]]
    bad = ["строка вне списка плана: %s" % k for k in keys if not (k.startswith("Ж-227:") and k[6:] in ids_ok)]
    bad += ["дубль: %s" % k for k in sorted(set(keys)) if keys.count(k) > 1]
    for ln, k in zip(lines[1:], keys):
        if k[6:] in CONFIGS and ln != trial_text([k[6:]])[len(IO.HEADER_TRIALS):]:
            bad.append("текст строки %s ≠ реестру" % k)
    return bad
