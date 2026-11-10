"""Лист ZE (Ж-247 п. 4), реестр конфигураций ZE, ZEp (план HARJ_HYBRID_SBER_ZE_PLAN, разд. 3) — дописывается в `ALL_TRIALS.csv` (прежние строки не трогаются, реестр только дополняется; порядок строк не проверяется).
Запись — ОТДЕЛЬНОЙ командой ДО любого прогноза ZE и только после ключа `ZE.precondition` (ok) в CHECKPOINT: `python3 -W ignore -m models_v2.s16.x1_ze_reg`.
`guard` отказывает конфигурации вне ZE/ZEp (в т. ч. Y5, Y6, любой сверх двух), с чужим статусом и записанной ПОЗЖЕ прогноза (`computed_at`). Время — по часам машины."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from models_v2.s16 import x1_reg as R

TRIALS, COLS, now_text, _parse = R.TRIALS, R.COLS, R.now_text, R._parse
POST = "после просмотра Y1–Y4, Z, ZD (Ж-247)"
ROWS = [
    ("ZE", "Е: частичная подстройка к уровню дня d на экстремальных днях", "F = X1·exp(g), g = e_d·(b + a·s_d + c·G_d), G_d = ln RV_d − ln X1_d (RV_d — Σ r² баров дня d склада M10 V7; X1_d — прогноз X1 на t⁺ с закрытия d); c ∈ [0, 1]; протокол ZB",
     "b, a, c (3); λ ∈ {0; 0,001; 0,01; 0,1}; V_e = 60", "зарегистрирована; " + POST),
    ("ZEp", "Контроль (б) к ZE: G от d − 252", "как ZE, G_d → G_{d−252} циклическим сдвигом по ряду общих дат; s_d настоящий", "как ZE", "зарегистрирована; контроль к ZE, критериев не получает; " + POST),
]
IDS = tuple(r[0] for r in ROWS)
CP_FILE = R.REP / "CHECKPOINT.json"


def precondition_ok(cp: dict) -> bool:
    return bool(cp.get("ZE.precondition", {}).get("ok"))


def register(now: str | None = None, trials: Path = TRIALS) -> int:
    """Дописывает ОТСУТСТВУЮЩИЕ строки ZE в конец файла (существующие — байт в байт прежние). -> число добавленных строк."""
    cur = pd.read_csv(trials, comment="#", dtype=str, keep_default_na=False)
    new = [dict(zip(COLS, (i, n, f, p, st, now or now_text()))) for i, n, f, p, st in ROWS if i not in set(cur["id"])]
    if not new:
        return 0
    text = trials.read_text(encoding="utf-8")
    with trials.open("a", encoding="utf-8", newline="") as fh:
        fh.write(("" if text.endswith("\n") else "\n") + pd.DataFrame(new)[COLS].to_csv(index=False, header=False, lineterminator="\n"))
    return len(new)


def guard(cfg: str, computed_at: str | None = None, trials: Path = TRIALS) -> None:
    """Отказ (SystemExit): конфигурации нет в реестре ZE (Y5, Y6, любые сверх двух); статус не «зарегистрирована…»; запись реестра позже прогноза."""
    reg = pd.read_csv(trials, comment="#", dtype=str, keep_default_na=False) if trials.exists() else pd.DataFrame(columns=COLS)
    row = reg[reg["id"] == cfg]
    if cfg not in IDS or not len(row):
        raise SystemExit("%s: не зарегистрирована в реестре ZE (ALL_TRIALS.csv) — счёт отказан" % cfg)
    if not str(row.iloc[0]["status"]).startswith("зарегистрирована"):
        raise SystemExit("%s: статус «%s», не «зарегистрирована»" % (cfg, row.iloc[0]["status"]))
    if computed_at is not None and _parse(row.iloc[0]["registered_at"]) > _parse(computed_at):
        raise SystemExit("%s: запись реестра (%s) позже прогноза (%s)" % (cfg, row.iloc[0]["registered_at"], computed_at))


def main() -> int:
    cp = json.loads(CP_FILE.read_text(encoding="utf-8")) if CP_FILE.exists() else {}
    if not precondition_ok(cp):
        raise SystemExit("реестр ZE отказан: нет ключа ZE.precondition (ok) в CHECKPOINT — сначала `x1_ze_stage --precondition`")
    print("реестр ZE: добавлено строк %d" % register())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
