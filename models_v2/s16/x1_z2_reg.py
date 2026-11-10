"""Этап Z2 (Ж-246 п. 6), реестр H1, Hc, H2 (план `docs/otchety/HARJ_HYBRID_SBER_Z2_HAWKES_DRAFT.md`, раздел 3) — дописывается в `ALL_TRIALS.csv`; строки Y и Z не трогаются (`x1_reg.py`, `x1_z_reg.py` не правятся).
Запись реестра — ОТДЕЛЬНОЙ командой ДО любого счёта Z2: `python3 -W ignore -m models_v2.s16.x1_z2_reg`. `guard` отказывает: конфигурации нет в реестре Z2; статус не «зарегистрирована…» (H2 — «не считается без решения владельца» — отказ ВСЕГДА);
запись реестра позже прогноза (`computed_at`). Время — по часам машины."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from models_v2.s16 import x1_reg as R

TRIALS, COLS, now_text, _parse = R.TRIALS, R.COLS, R.now_text, R._parse
POST = "зарегистрирована; условная (L1–L3); после просмотра Y1–Y4, ZD"
ROWS = [
    ("H1", "Z2-H1: перенос внутридневного шока (Hawkes) над X1", "F = X1·exp(g), g = a·ln(1 + ξ_d); ξ_d — остаточная интенсивность событий |r| > Q0,99 прошлых баров на закрытии d; ступень 1: θ = (μ, n, β, η, ρ) по правдоподобию баров, ступень 2: a по QLIKE",
     "μ, n, β, η, ρ (5, ступень 1 без цели RV); a (1), λ ∈ {0; 0,001; 0,01; 0,1}", POST),
    ("Hc", "Контроль (б) к H1: перестановка (N, m) внутри дня", "как H1; внутри каждого дня набор (N, m) переставлен по присутствующим слотам (default_rng(0), дни по порядку); ступени 1 и 2 переоцениваются", "как H1",
     POST),
    ("H2", "Z2-H2: асимметрия знака (Hawkes 2 × 2)", "g = a⁻·ln(1 + ξ⁻) + a⁺·ln(1 + ξ⁺); не считается", "9 + 2", "не считается без решения владельца"),
]
IDS = tuple(r[0] for r in ROWS)
COUNTED = ("H1", "Hc")


def register(now: str | None = None, trials: Path = TRIALS) -> int:
    """Дописывает ОТСУТСТВУЮЩИЕ строки Z2 в конец файла (существующие строки — байт в байт прежние). -> число добавленных строк."""
    cur = pd.read_csv(trials, comment="#", dtype=str, keep_default_na=False)
    new = [dict(zip(COLS, (i, n, f, p, st, now or now_text()))) for i, n, f, p, st in ROWS if i not in set(cur["id"])]
    if not new:
        return 0
    text = trials.read_text(encoding="utf-8")
    with trials.open("a", encoding="utf-8", newline="") as fh:
        fh.write(("" if text.endswith("\n") else "\n") + pd.DataFrame(new)[COLS].to_csv(index=False, header=False, lineterminator="\n"))
    return len(new)


def guard(cfg: str, computed_at: str | None = None, trials: Path = TRIALS) -> None:
    """Отказ (SystemExit): нет в реестре Z2 или вне {H1, Hc}; статус не «зарегистрирована…» (H2); запись реестра позже прогноза."""
    reg = pd.read_csv(trials, comment="#", dtype=str, keep_default_na=False) if trials.exists() else pd.DataFrame(columns=COLS)
    row = reg[reg["id"] == cfg]
    if cfg not in IDS or not len(row):
        raise SystemExit("%s: не зарегистрирована в реестре Z2 (ALL_TRIALS.csv) — счёт отказан" % cfg)
    if cfg not in COUNTED or not str(row.iloc[0]["status"]).startswith("зарегистрирована"):
        raise SystemExit("%s: статус «%s» — не считается без решения владельца" % (cfg, row.iloc[0]["status"]))
    if computed_at is not None and _parse(row.iloc[0]["registered_at"]) > _parse(computed_at):
        raise SystemExit("%s: запись реестра (%s) позже прогноза (%s)" % (cfg, row.iloc[0]["registered_at"], computed_at))


if __name__ == "__main__":
    print("реестр Z2: добавлено строк %d" % register())
