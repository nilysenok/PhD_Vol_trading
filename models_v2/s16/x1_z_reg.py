"""Этап Z (Ж-246), реестр конфигураций ZA, ZB, ZB0, ZC, ZCp (план HARJ_HYBRID_SBER_Z_PLAN, раздел 3) — дописывается в `ALL_TRIALS.csv` (строки Y1–Y6 не трогаются; `x1_reg.py` не правится: он в хэше кода YA).
Хэш `ALL_TRIALS.csv` в сверку P0/YA/YB не входит (там только `guard` по id, статусу и времени), поэтому отдельный `Z_TRIALS.csv` не нужен. Запись реестра — ОТДЕЛЬНОЙ командой ДО любого прогноза Z:
`python3 -W ignore -m models_v2.s16.x1_z_reg`. `guard` отказывает конфигурации, которой нет в реестре, и записанной ПОЗЖЕ прогноза (`computed_at`). Время — по часам машины."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from models_v2.s16 import x1_reg as R

TRIALS, COLS, now_text, _parse = R.TRIALS, R.COLS, R.now_text, R._parse
POST = "зарегистрирована; предложена после просмотра Y1–Y4 (Ж-246)"
CTRL = "зарегистрирована; контроль к %s, критериев не получает"
ROWS = [
    ("ZA", "А: общий корректор с весом ω", "F = X1·exp(g), g = ω·(b + a·s_d); (b, a) — по всем закрытым парам с L2 λ; ω ∈ {0; 0,25; 0,5; 0,75; 1} по валидации", "b, a (2); λ ∈ {0; 0,001; 0,01; 0,1}; ω (выбор)", "зарегистрирована"),
    ("ZB", "Б: специализированный корректор области e_d = 1", "F = X1·exp(g), g = e_d·(b + a·s_d); обучение и валидация только на закрытых парах с e_d = 1; включение при ≥ 30 таких парах", "b, a (2); λ", POST),
    ("ZB0", "Контроль (а) к ZB: без сигнала", "F = X1·exp(g), g = e_d·b", "b (1); λ", CTRL % "ZB"),
    ("ZC", "В: совместное обучение области и поправки", "F = X1·exp(g), g = σ(γ0 + γᵀz̃_d)·(b + a·s_d), z_d = (e_d, x_d, s_d, v_d); x_d = ln max_k r²_k − ln(RV_d/n_d), v_d = ln RV_d − ln mean(RV_{d−1..d−20})",
     "γ0, γ (4), b, a (7); старт γ0 = −2, остальные 0; L2 λ на γ, b, a", POST),
    ("ZCp", "Контроль (б) к ZC: затвор от z_{d−252}", "как ZC, z_d → z_{d−252} циклическим сдвигом по ряду общих дат; поправка b + a·s_d — с настоящим s_d", "как ZC", CTRL % "ZC"),
]
IDS = tuple(r[0] for r in ROWS)


def register(now: str | None = None, trials: Path = TRIALS) -> int:
    """Дописывает ОТСУТСТВУЮЩИЕ строки Z в конец файла (существующие строки — байт в байт прежние). -> число добавленных строк."""
    cur = pd.read_csv(trials, comment="#", dtype=str, keep_default_na=False)
    new = [dict(zip(COLS, (i, n, f, p, st, now or now_text()))) for i, n, f, p, st in ROWS if i not in set(cur["id"])]
    if not new:
        return 0
    text = trials.read_text(encoding="utf-8")
    with trials.open("a", encoding="utf-8", newline="") as fh:
        fh.write(("" if text.endswith("\n") else "\n") + pd.DataFrame(new)[COLS].to_csv(index=False, header=False, lineterminator="\n"))
    return len(new)


def guard(cfg: str, computed_at: str | None = None, trials: Path = TRIALS) -> None:
    """Отказ (SystemExit): конфигурации нет в реестре Z (в т. ч. Y5/Y6 и любые сверх пяти); статус не «зарегистрирована…»; запись реестра позже прогноза (`computed_at`)."""
    reg = pd.read_csv(trials, comment="#", dtype=str, keep_default_na=False) if trials.exists() else pd.DataFrame(columns=COLS)
    row = reg[reg["id"] == cfg]
    if cfg not in IDS or not len(row):
        raise SystemExit("%s: не зарегистрирована в реестре Z (ALL_TRIALS.csv) — счёт отказан" % cfg)
    if not str(row.iloc[0]["status"]).startswith("зарегистрирована"):
        raise SystemExit("%s: статус «%s», не «зарегистрирована»" % (cfg, row.iloc[0]["status"]))
    if computed_at is not None and _parse(row.iloc[0]["registered_at"]) > _parse(computed_at):
        raise SystemExit("%s: запись реестра (%s) позже прогноза (%s)" % (cfg, row.iloc[0]["registered_at"], computed_at))


if __name__ == "__main__":
    print("реестр Z: добавлено строк %d" % register())
