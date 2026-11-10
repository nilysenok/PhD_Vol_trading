"""X1-пилот, реестр конфигураций Y1–Y6 (Ж-243; план HARJ_HYBRID_SBER_X1_PILOT_PLAN, раздел 3) — `ALL_TRIALS.csv`, ТОЛЬКО ДОПОЛНЯЕТСЯ. Запись реестра — отдельной командой ДО любого прогноза:
`python3 -W ignore -m models_v2.s16.x1_reg`. `guard` отказывает конфигурации, которой нет в реестре, записанной ПОЗЖЕ прогноза (`computed_at`), и Y5/Y6 всегда (условные, без отдельного разрешения владельца).
Время — по часам машины (`datetime.now`)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "docs/otchety/harj_x1"
TRIALS = REP / "ALL_TRIALS.csv"
COLS = ["id", "name", "formula", "params", "status", "registered_at"]
OK, COND = "зарегистрирована", "условная, не запускается без отдельного разрешения владельца (Ж-243)"
ROWS = [
    ("Y1", "М1: затухание закрытия, по состоянию", "F = X1·exp(g), g = (a_n·(1 − e_d) + a_e·e_d)·s_d·exp(−κ·Δ_d); s_d = ln(RV^last_d·n_d/n^last_d) − ln RV_d; Δ_d = дни(t⁺ − d) − 1; e_d — колонка s файла x1.parquet",
     "a_n, a_e, κ ≥ 0 (3); L2 на a; старт a = 0, κ = 0,5", OK),
    ("Y2", "М1 без затухания (абляция Y1)", "как Y1 при κ ≡ 0", "a_n, a_e (2); старт a = 0", OK),
    ("Y3", "М2: обучаемая агрегация под остаток X1", "F = X1·exp(θ·[ln B_d(w) − ln RV_d]), B_d = Σ_k w_k·r²_k, w_k = n_d·softmax_k(γᵀφ_k)", "γ (4), θ (1); старт θ = 0,01, γ ~ N(0; 0,1²)", OK),
    ("Y4", "М2 с фиксированными весами (абляция Y3)", "как Y3, w — средний профиль слота по прошлому (PIT), γ не обучается", "θ (1)", OK),
    ("Y5", "М3: совместные компоненты", "F = X1·exp(g(Z)), F_j = F·p_j(Z), Σp_j = 1 (3 трети сессии); L = QLIKE(F, RV) + α·Σ_j QLIKE(F_j, RV_j)", "g (10), p (2 × 10); α ∈ {0,3; 1}", COND),
    ("Y6", "Комбинация лучшей из Y1, Y3, Y5 и второй по QLIKE 2018–2021", "суммой g", "параметры обоих", COND),
]
RUNNABLE = ("Y1", "Y2", "Y3", "Y4")


def now_text() -> str:
    return dt.datetime.now().strftime("%d.%m.%Y %H:%M:%S")


def read() -> pd.DataFrame:
    return pd.read_csv(TRIALS, comment="#", dtype=str, keep_default_na=False) if TRIALS.exists() else pd.DataFrame(columns=COLS)


def register(now: str | None = None, trials: Path = TRIALS) -> int:
    """Дописывает ОТСУТСТВУЮЩИЕ строки Y1–Y6; существующие не трогаются и не перезаписываются. -> число добавленных строк."""
    cur = pd.read_csv(trials, comment="#", dtype=str, keep_default_na=False) if trials.exists() else pd.DataFrame(columns=COLS)
    new = [dict(zip(COLS, (i, n, f, p, st, now or now_text()))) for i, n, f, p, st in ROWS if i not in set(cur["id"])]
    if not new:
        return 0
    out = pd.concat([cur, pd.DataFrame(new)], ignore_index=True)[COLS]
    trials.parent.mkdir(parents=True, exist_ok=True)
    trials.write_text("# Пилот X1, реестр конфигураций (Ж-243): только дополняется; статус записан до любого прогноза Y\n" + out.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
    return len(new)


def _parse(t: str) -> dt.datetime:
    return dt.datetime.strptime(t, "%d.%m.%Y %H:%M:%S")


def guard(cfg: str, computed_at: str | None = None, trials: Path = TRIALS) -> None:
    """Отказ (SystemExit): Y5/Y6 всегда; конфигурации нет в реестре; статус не «зарегистрирована»; запись реестра позже прогноза (`computed_at`)."""
    if cfg in ("Y5", "Y6"):
        raise SystemExit("%s: %s — счёт отказан" % (cfg, COND))
    reg = pd.read_csv(trials, comment="#", dtype=str, keep_default_na=False) if trials.exists() else pd.DataFrame(columns=COLS)
    row = reg[reg["id"] == cfg]
    if cfg not in RUNNABLE or not len(row):
        raise SystemExit("%s: не зарегистрирована в ALL_TRIALS.csv — счёт отказан" % cfg)
    if row.iloc[0]["status"] != OK:
        raise SystemExit("%s: статус «%s», не «зарегистрирована»" % (cfg, row.iloc[0]["status"]))
    if computed_at is not None and _parse(row.iloc[0]["registered_at"]) > _parse(computed_at):
        raise SystemExit("%s: запись реестра (%s) позже прогноза (%s)" % (cfg, row.iloc[0]["registered_at"], computed_at))


if __name__ == "__main__":
    print("реестр Y1–Y6: добавлено строк %d" % register())
