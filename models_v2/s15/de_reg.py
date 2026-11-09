"""Deep Extension: реестр 5 мест (Ж-240; план раздел 3) — `ALL_TRIALS.csv`, ТОЛЬКО ДОПОЛНЯЕТСЯ. Запись реестра — до любого прогноза: бегун (`guard`) отказывает конфигурации, которой нет в реестре,
записанной позже прогноза (`computed_at`), и X5 без выполненного условия (CHECKPOINT ключ `E5.condition`, bool True). Время — по часам машины (`datetime.now`)."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from models_v2.s15 import de0_io as IO

TRIALS = IO.REP / "ALL_TRIALS.csv"
COLS = ["id", "name", "formula", "condition", "status", "registered_at"]
STAGE_CFG = {"E1": "X1", "E2": "X2", "E3": "X3", "E4": "X4", "E5": "X5"}
X5_COND = "у лучшего из X2–X4 по QLIKE 2018–2021 правило 6.2 против A0g = «да» (CHECKPOINT E5.condition = true)"
ROWS = [
    ("X1", "D1 — доверие по состоянию", "F = M^(1−π_t)·C^(π_t); C = cfg8 V7, M = √(K2·K3); π_t = π^(s_t); s_t = e_d (экстремальный M10-бар дня d, PIT-порог 99 % |r| по барам сессий < d); π^(s) — fixed share (α^(s) ∈ {0,01; 0,02; 0,05}, внутренняя валидация 126 закрытых пар, помесячно), старт 1/2", "безусловная — основная"),
    ("X2", "D2 — хвостовая смесь с меткой стресса", "F = H·[(1 − p_t)·c_n(Z) + p_t·c_s(Z)]; p_t — логистика на метке «RV(t⁺) > PIT-перцентиль 90»; c_n, c_s — QLIKE", "безусловная"),
    ("X3", "B1 — устойчивая + адаптивная поправка", "F = H·exp(g_s(Z) + a_t·g_a(Z)); g_s = A0g; g_a — QLIKE-гребень на последних 250 закрытых парах; a_t ∈ [0; 1] — CUSUM", "безусловная"),
    ("X4", "C1 — затвор по неопределённости", "F = H·exp(α_t·ḡ(Z)); ḡ — среднее 5 QLIKE-GBM на разных прошлых окнах; α_t = exp(−κ·sd_r(g_r)/s̄); κ — внутренняя валидация", "безусловная"),
    ("X5", "Комбинация: лучший из X2–X4 как C в D1", "D1 с C = лучший из X2–X4 по QLIKE 2018–2021", X5_COND),
]


def now_text() -> str:
    return dt.datetime.now().strftime("%d.%m.%Y %H:%M:%S")


def read() -> pd.DataFrame:
    return pd.read_csv(TRIALS, comment="#", dtype=str, keep_default_na=False) if TRIALS.exists() else pd.DataFrame(columns=COLS)


def register(now: str | None = None) -> int:
    """Дописывает ОТСУТСТВУЮЩИЕ места X1–X5 (статус «зарегистрирована»); существующие строки не трогаются и не перезаписываются. -> число добавленных строк."""
    cur = read()
    new = [dict(zip(COLS, (i, n, f, c, "зарегистрирована", now or now_text()))) for i, n, f, c in ROWS if i not in set(cur["id"])]
    if not new:
        return 0
    out = pd.concat([cur, pd.DataFrame(new)], ignore_index=True)[COLS]
    IO.REP.mkdir(parents=True, exist_ok=True)
    TRIALS.write_text("# Deep Extension, реестр мест (Ж-240): только дополняется; статус «зарегистрирована» записан до любого прогноза\n" + out.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
    return len(new)


def _parse(t: str) -> dt.datetime:
    return dt.datetime.strptime(t, "%d.%m.%Y %H:%M:%S")


def guard(cfg: str, cp: dict | None = None, computed_at: str | None = None) -> None:
    """Отказ (SystemExit): конфигурации нет в реестре / запись позже прогноза / X5 без условия. `computed_at` — время прогноза (для проверки порядка «реестр раньше»)."""
    reg = read()
    row = reg[reg["id"] == cfg]
    if cfg not in set(x[0] for x in ROWS) or not len(row):
        raise SystemExit("%s: не зарегистрирована в ALL_TRIALS.csv — счёт отказан" % cfg)
    if row.iloc[0]["status"] != "зарегистрирована":
        raise SystemExit("%s: статус «%s», не «зарегистрирована»" % (cfg, row.iloc[0]["status"]))
    if computed_at is not None and _parse(row.iloc[0]["registered_at"]) > _parse(computed_at):
        raise SystemExit("%s: запись реестра (%s) позже прогноза (%s)" % (cfg, row.iloc[0]["registered_at"], computed_at))
    if cfg == "X5" and not (cp if cp is not None else IO.read_cp()).get("E5", {}).get("condition") is True:
        raise SystemExit("X5: условие (%s) не выполнено" % X5_COND)
