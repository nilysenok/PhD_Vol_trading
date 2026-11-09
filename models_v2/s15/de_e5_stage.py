"""E5: условие X5 (Ж-239 п. 6, Ж-240; план раздел 3) — МЕХАНИЧЕСКОЕ, по таблицам листов E2–E4, ничего не считается и не обучается.
Условие: лучший из X2–X4 по QLIKE на окне разработки 2018–2021 (`WINDOW`; окна 2022+ в условие не входят) и у него правило 6.2 против A0g = «да».
`write()` пишет CHECKPOINT ключ `E5` (condition: bool) и строку реестра «X5 — не считается: условие ложно» (реестр только дополняется). `check()` только читает.
Бегун отказывает X5 (`de_reg.guard`: условие не True). Реестр пишет этот модуль, а не `de_reg.py`: хэш кода E1 включает `de_reg.py`, его правка меняла бы байты E1."""
from __future__ import annotations

import json

import pandas as pd

from models_v2.s15 import de0_io as IO, de_reg as R

WINDOW = "2018_2021"
CONF = ("X2", "X3", "X4")
BASE = "V6:A0g"
NOT_COUNTED = "не считается"


def _tab(name: str) -> pd.DataFrame:
    p = IO.REP / (name + ".csv")
    if not p.exists():
        raise SystemExit("X5: условие не вычислено — нет таблицы %s.csv (листы E2–E4)" % name)
    return pd.read_csv(p, comment="#")


def condition_from(metrics: dict, pairs: dict, window: str | None = None) -> dict:
    """metrics/pairs: имя конфигурации → таблица (`E*_METRICS`, `E*_PAIRS`). Берёт строки ТОЛЬКО окна `window` (по умолчанию WINDOW)."""
    w = WINDOW if window is None else window
    q = {}
    for c in CONF:
        m = metrics[c]
        q[c] = float(m.loc[(m["series"] == c) & (m["window"] == w), "qlike"].iloc[0])
    best = min(q, key=lambda c: (q[c], c))
    p = pairs[best]
    r = p[(p["x"] == best) & (p["y"] == BASE) & (p["window"] == w)].iloc[0]
    cond = bool(str(r["rule62"]) == "да")
    return dict(condition=cond, best=best, window=w, qlike=q, rule62=str(r["rule62"]), delta_pct=float(r["delta_pct"]), lo90=float(r["lo90"]), hi90=float(r["hi90"]), p_one=float(r["p_one"]),
                years_neg=int(r["years_neg"]), n_years=int(r["n_years"]), base=BASE)


def compute() -> dict:
    mt = {c: _tab("E%d_METRICS" % (i + 2)) for i, c in enumerate(CONF)}
    pr = {c: _tab("E%d_PAIRS" % (i + 2)) for i, c in enumerate(CONF)}
    out = condition_from(mt, pr)
    out["sha256"] = {n: IO.sha_file(IO.REP / (n + ".csv")) for i in range(2, 5) for n in ("E%d_METRICS" % i, "E%d_PAIRS" % i)}
    return out


def row_text(c: dict) -> str:
    return "X5 — не считается: условие %s (лучший из X2–X4 по QLIKE %s — %s; правило 6.2 против A0g — «%s»)" % ("ложно" if not c["condition"] else "истинно", c["window"].replace("_", "–"), c["best"], c["rule62"])


def add_registry_row(c: dict) -> None:
    """Реестр только дополняется; строка «не считается» — один раз."""
    cur = R.read()
    if ((cur["id"] == "X5") & (cur["status"] == NOT_COUNTED)).any():
        return
    row = pd.DataFrame([dict(id="X5", name="результат X5", formula=row_text(c), condition="", status=NOT_COUNTED, registered_at=R.now_text())])
    R.TRIALS.write_text(R.TRIALS.read_text(encoding="utf-8") + row.to_csv(index=False, header=False, lineterminator="\n"), encoding="utf-8")


def write() -> dict:
    """Условие → CHECKPOINT `E5` → строка реестра. Истинное условие здесь не обслуживается: X5 считается только своим листом (его нет)."""
    c = compute()
    if c["condition"]:
        raise SystemExit("X5: условие истинно — счёт X5 в отдельном листе, здесь не реализован")
    IO.merge_checkpoint("E5", dict(c, computed_at=R.now_text(), x5_computed=False))
    add_registry_row(c)
    return c


def _json(x):
    return json.loads(json.dumps(x))


def check() -> list:
    """Только читает: пересчёт условия = CHECKPOINT[E5]; реестр содержит строку «не считается»; бегун отказывает X5; CHECKPOINT и реестр не изменены."""
    before = (IO.sha_file(IO.CP_FILE), IO.sha_file(R.TRIALS))
    cp, bad = IO.read_cp().get("E5"), []
    if not cp:
        return ["нет ключа E5 в CHECKPOINT"]
    c = compute()
    if _json(c) != _json({k: v for k, v in cp.items() if k not in ("computed_at", "x5_computed")}):
        bad.append("E5 ≠ пересчёту условия по таблицам E2–E4")
    if cp.get("condition") is not False or cp.get("x5_computed") is not False:
        bad.append("E5.condition должен быть false, X5 не считалась")
    reg = R.read()
    if not ((reg["id"] == "X5") & (reg["status"] == NOT_COUNTED)).any():
        bad.append("в реестре нет строки «X5 — не считается»")
    elif reg[(reg["id"] == "X5") & (reg["status"] == NOT_COUNTED)].iloc[0]["formula"] != row_text(c):
        bad.append("строка «X5 — не считается» ≠ условию")
    try:
        R.guard("X5")
        bad.append("бегун не отказал X5")
    except SystemExit:
        pass
    if (IO.sha_file(IO.CP_FILE), IO.sha_file(R.TRIALS)) != before:
        bad.append("--check изменил выходы")
    return bad
