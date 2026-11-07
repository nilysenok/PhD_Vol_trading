"""P3c, G13 (R15): стадия `diag`. Запуск: `run_hp --stage diag [--check]`. Только чтение обучения на 17 (`*_p17`, `k23_<T>`); обучений нет.
Пишет `diagnostics/CAUSES.csv` (шапка — правила вывода и SHA, объявленные в `CHECKPOINT.json` ДО счёта) и вспомогательные таблицы. Статус: исследовательский, HISTORICAL WALK-FORWARD, потолок."""
from __future__ import annotations

import json
import os
import time

import pandas as pd

from models_v2.s10 import hp_diag_ab as AB, hp_diag_data as D, hp_diag_dh as DH, hp_diag_rules as R, hp_paths as P

OUT = P.REP / "diagnostics"
STAT = "исследовательский, HISTORICAL WALK-FORWARD, не независимый OOS; обучение на 17 (Ж-220); выбор — внутренняя валидация M16 × 2017–2021; цифры — потолок"


def declared() -> dict:
    cp = json.loads((P.REP / "CHECKPOINT.json").read_text())
    if cp.get("diag_rules_sha") != R.rules_sha() or "diag_rules_declared_at" not in cp:
        raise SystemExit("правила R15 не объявлены в CHECKPOINT.json до счёта или изменены после (SHA %s)" % R.rules_sha())
    return cp


def compute(df: pd.DataFrame, loto: pd.DataFrame) -> tuple:
    rows, met, aux = [], {}, {}
    for name, fn, args in (("a", AB.cause_a, (df,)), ("b", AB.cause_b, (df,)), ("c", AB.cause_c, (df,)), ("d", DH.cause_d, (df, loto)),
                           ("e", DH.cause_e, (df,)), ("f", DH.cause_f, (df,)), ("g", DH.cause_g, (df,)), ("h", DH.cause_h, (df,))):
        r, m, a = fn(*args)
        rows += r
        met.update(m)
        if a is not None:
            aux[name] = a
    for c, (s, rule) in R.CAUSES.items():
        rows.append(D.row(c, "M16", "см. правило", "ВЫВОД", s, None, None, verdict=R.verdict(c, met)))
    return pd.DataFrame(rows), met, aux


def write(tab: pd.DataFrame, aux: dict, cp: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    head = ["# P3c, R15 (Ж-217 §9); %s" % STAT, "# правила вывода объявлены %s, SHA %s (CHECKPOINT.json), до расчёта; M16 — вывод, SBER и E11 (перенос) — строки рядом" % (cp["diag_rules_declared_at"], R.rules_sha())]
    head += ["# " + line for line in R.rules_text().splitlines()]
    with open(OUT / "CAUSES.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(head) + "\n")
        tab.to_csv(f, index=False, lineterminator="\n")
    names = {"a": "GAMMA_TICKER.csv", "d": "LOTO_TICKER.csv", "f": "YEAR_GAIN.csv", "g": "CHOICE_DIST.csv"}
    for k, v in aux.items():
        with open(OUT / names[k], "w", encoding="utf-8") as f:
            f.write("# P3c, R15 (%s); вспомогательная таблица; %s\n" % (k, STAT))
            v.to_csv(f, index=False, lineterminator="\n")


def check() -> list:
    bad = []
    cp = json.loads((P.REP / "CHECKPOINT.json").read_text())
    p = OUT / "CAUSES.csv"
    if not p.exists():
        return ["нет CAUSES.csv"]
    t = pd.read_csv(p, comment="#")
    v = t[t["item"] == "ВЫВОД"]
    bad += [] if sorted(v["cause"]) == list("abcdefgh") else ["выводы не по всем пунктам а–з: %s" % sorted(v["cause"])]
    bad += ["вывод вне {подтверждена, не подтверждена, не измеримо}" ] if not v["verdict"].isin(["подтверждена", "не подтверждена", "не измеримо"]).all() else []
    bad += ["SHA правил в шапке ≠ CHECKPOINT"] if cp.get("diag_rules_sha") != R.rules_sha() or R.rules_sha() not in p.read_text(encoding="utf-8") else []
    bad += ["правила объявлены не раньше записи CAUSES.csv"] if time.mktime(time.strptime(cp["diag_rules_declared_at"], "%Y-%m-%d %H:%M:%S")) >= os.path.getmtime(p) else []
    bad += ["значения нефинитны в основных строках"] if t[t["item"] != "ВЫВОД"]["value"].isna().all() else []
    return bad + check_docs()


def check_docs() -> list:
    """Итоговые документы G13 есть и несут статус «HISTORICAL WALK-FORWARD, не независимый OOS» (Ж-219 п. 8)."""
    docs = [P.ROOT / "docs/otchety/HARJ_PANEL_HYBRID_FINAL.md", P.REP / "REPRODUCTION.md", OUT / "FINAL_TABLES.md", P.REP / "FINAL_PANEL_CONFIG.json"]
    bad = ["нет %s" % d.name for d in docs if not d.exists()]
    for d in docs[:2] + docs[3:]:
        if d.exists() and ("HISTORICAL WALK-FORWARD" not in d.read_text(encoding="utf-8") or "OOS" not in d.read_text(encoding="utf-8")):
            bad.append("в %s нет статуса HISTORICAL WALK-FORWARD / не независимый OOS" % d.name)
    return bad


def stage(check_only: bool) -> int:
    if not check_only:
        cp = declared()
        D.check_names(D.input_names())
        tab, met, aux = compute(D.load(), D.load_loto())
        write(tab, aux, cp)
        from models_v2.s10 import hp_diag_tab as TB
        TB.run()
        print("метрики вывода:", {k: round(v, 5) for k, v in met.items()})
    bad = check()
    print("\n".join(bad))
    print("ИТОГ:", "OK" if not bad else "FAIL")
    return 1 if bad else 0
