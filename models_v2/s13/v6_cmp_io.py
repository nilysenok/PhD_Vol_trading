"""V6 лист Z2: пути, шапки таблиц `cmp/`, хэши кода и входов, суммарный хэш выходов (для «до = после» у `--check`), покрытие N Холма реестрами. Запись — только `write_*` (вызывает `run_v6_cmp` без `--check`)."""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import pandas as pd

from models_v2.s13 import v6_0_io as IO

CMP = IO.REP / "cmp"
DOC = IO.ROOT / "docs/otchety/HARJ_SBER_V1_V6_SRAVNENIE.md"
CP = CMP / "CHECKPOINT_CMP.json"
TABLES = ["CMP_SERIES", "CMP_REPRO", "CMP_YEARS", "CMP_WINDOWS", "CMP_TESTS", "CMP_MCS", "CMP_BEST"]
VERSION = "v6.cmp"
STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-230, лист Z2 (сравнение SBER V1–V6, h = 1; новых прогнозов и конфигураций нет)"
FILES = ["v6_cmp_reg.py", "v6_cmp_tab.py", "v6_cmp_best.py", "v6_cmp_repro.py", "v6_cmp_io.py", "v6_cmp_fmt.py", "v6_cmp_doc.py", "v6_cmp_doc2.py", "run_v6_cmp.py"]


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    h = hashlib.sha256()
    for f in FILES:
        p = Path(__file__).parent / f
        h.update(f.encode() + (p.read_bytes() if p.exists() else b""))
    return h.hexdigest()[:16]


def header() -> str:
    return "# %s; код %s %s; входы %s; %s\n" % (STATUS, VERSION, code_hash(), IO.inputs_hash(), IO.CAVEAT)


def write_csv(df: pd.DataFrame, name: str) -> None:
    CMP.mkdir(parents=True, exist_ok=True)
    with open(CMP / (name + ".csv"), "w", encoding="utf-8") as f:
        f.write(header())
        df.to_csv(f, index=False, lineterminator="\n")


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(CMP / (name + ".csv"), comment="#")


def digest() -> str:
    """Суммарный хэш выходов листа Z2 (таблицы cmp/ и документ): у `--check` до и после обязан совпасть."""
    h = hashlib.sha256()
    for f in sorted(CMP.rglob("*")) + [DOC]:
        if f.is_file():
            h.update(str(f).encode() + IO.sha_file(f).encode())
    return h.hexdigest()


def coverage() -> pd.DataFrame:
    """Что покрывают N = 722 и N_wide = 3122: реестры прошлых программ (`harj_hybrid_v5/SEARCH_HISTORY.csv`, ключи по объединению, не по сумме строк) + бюджеты V5 (24) и V6 (18); N_wide = N + 2400 проб Optuna h = 1 (K2/K3)."""
    h = pd.read_csv(IO.V5REP / "SEARCH_HISTORY.csv", comment="#")
    reg = h[h["item"] == "registry"].set_index("name")
    n = h[h["item"] == "N"].set_index("name")["rows"]
    prior_n, decl = int(n["N_prior (объединение ключей, до V5)"]), int(n["N_declared (N_prior + бюджет V5 24)"])
    cp5 = json.loads((IO.V5REP / "CHECKPOINT.json").read_text(encoding="utf-8"))
    names = [("V1 (Ж-211)", "hybrid_final_v1"), ("V1 (Ж-212)", "hybrid_struct_v1"), ("V2", "harj_hybrid_v2"), ("V3", "harj_panel_hybrid"), ("V4", "harj_hybrid_v4")]
    rows = [dict(row=r, registry=k, unique_keys=int(reg.loc[k, "unique_keys"]), in_N722=True, in_N3122=True, note="входит в N_prior = %d (объединение ключей)" % prior_n) for r, k in names]
    rows += [dict(row="V5", registry="бюджет V5", unique_keys=decl - prior_n, in_N722=True, in_N3122=True, note="N_declared = %d" % decl),
             dict(row="V6", registry="бюджет V6", unique_keys=722 - decl, in_N722=True, in_N3122=True, note="N = %d; N_wide = N + %d проб Optuna h = 1 (дизайн K2/K3)" % (722, int(cp5["N_wide_optuna_trials"])))]
    t = pd.DataFrame(rows)
    assert decl + 18 == 722 and decl + int(cp5["N_wide_optuna_trials"]) + 18 == 3122, "N = 722 / N_wide = 3122 не сходятся с реестрами"
    return t
