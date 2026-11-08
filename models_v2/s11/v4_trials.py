"""V4-0, G4: реестр ранее просмотренных конфигураций h = 1 (N Холма) и CHECKPOINT до любого p. Правило: N_prior = число УНИКАЛЬНЫХ ключей `key` в объединении реестров программ."""
from __future__ import annotations

import datetime as dt
import glob
import json

import pandas as pd

from models_v2.s11 import v4_io as IO

REG = IO.P.ROOT / "docs/otchety"
NAMES = ["hybrid_final_v1", "hybrid_struct_v1", "harj_hybrid_v2", "harj_panel_hybrid"]       # в порядке добавления (Ж-211, Ж-212, Ж-214, Ж-217…220)
BUDGET = 60
HEADER = "key,program,source,direction,config,kind\n"
RULE = "N_prior = |объединение ключей `key` реестров|; реестры кумулятивны (каждый следующий содержит предыдущий), поэтому суммировать строки нельзя (двойной счёт)"


def load_regs() -> dict:
    return {n: pd.read_csv(REG / n / "ALL_TRIALS.csv") for n in NAMES}


def registry_rows(regs: dict) -> pd.DataFrame:
    keys = {n: set(r["key"]) for n, r in regs.items()}
    rows, prev = [], set()
    for n in NAMES:
        r = regs[n]
        rows.append({"item": "registry", "name": n, "rows": len(r), "unique_keys": len(keys[n]), "dup_rows": len(r) - len(keys[n]), "n_trial": int((r["kind"] == "trial").sum()),
                     "n_pilot_forecast": int((r["kind"] == "pilot_forecast").sum()), "contains_previous": bool(prev <= keys[n]), "new_vs_previous": len(keys[n] - prev),
                     "programs": ";".join("%s:%d" % (p, c) for p, c in r.groupby("program", sort=False).size().items()), "note": ""})
        prev = keys[n]
    return pd.DataFrame(rows)


def n_prior(regs: dict) -> int:
    return len(set().union(*[set(r["key"]) for r in regs.values()]))


def other_h1_files() -> pd.DataFrame:
    """Файлы `data/regen/*h1*trials*.csv` — прежние Optuna-поиски h = 1 диссертации (другие модели/программа). В реестры гибридов не входят."""
    rows = []
    for p in sorted(glob.glob(str(IO.P.ROOT / "data/regen/*h1*trials*.csv"))):
        d = pd.read_csv(p)
        rows.append({"item": "other_h1_file", "name": p.split("/")[-1], "rows": len(d), "unique_keys": "", "dup_rows": "", "n_trial": len(d), "n_pilot_forecast": "", "contains_previous": "", "new_vs_previous": "",
                     "programs": "model=" + ";".join(sorted(d["model"].astype(str).unique())) if "model" in d else "", "note": "вне реестров гибридов; в N_prior не входит (см. GAPS.csv)"})
    return pd.DataFrame(rows)


def checkpoint(regs: dict) -> dict:
    n = n_prior(regs)
    p0 = json.loads((REG / "harj_panel_hybrid/CHECKPOINT.json").read_text())
    return {"program": "HAR-J Hybrid V4 (Ж-221 + Ж-222), h=1, 17 бумаг", "N_prior": n, "budget": BUDGET, "N_declared_before_stats": n + BUDGET, "rule": RULE,
            "sources": {k: len(set(r["key"])) for k, r in regs.items()}, "panel_declared": {"N_declared_before_stats": p0.get("N_declared_before_stats"), "N_declared_before_stats_p17": p0.get("N_declared_before_stats_p17")},
            "used": 0, "written_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "note": "записано до любого p; аудит и этап A конфигураций не расходуют"}


def replace_rows(prefixes, rows: list, path=None) -> None:
    """Общий способ записи реестра стадией (G22): строки с ключом, начинающимся с одного из `prefixes`, заменяются на `rows` (на месте первой прежней; нет прежних — в конец); все остальные строки сохраняются."""
    f = path or IO.REP / "ALL_TRIALS.csv"
    lines = f.read_text().splitlines()[1:] if f.exists() else []
    mine = tuple(prefixes)
    out, done = [], False
    for x in lines:
        if x.startswith(mine):
            if not done:
                out += list(rows)
                done = True
            continue
        out.append(x)
    if not done:
        out += list(rows)
    f.write_text(HEADER + "".join(x + "\n" for x in out))


def write_all() -> dict:
    regs = load_regs()
    t = pd.concat([registry_rows(regs), other_h1_files()], ignore_index=True)
    t = pd.concat([t, pd.DataFrame([{"item": "N_prior", "name": "union_unique_keys", "rows": n_prior(regs), "unique_keys": n_prior(regs), "note": RULE}])], ignore_index=True)
    IO.write_csv(t, "PRIOR_TRIALS.csv", "реестры гибридов h=1 (Ж-211/212/214/217–220); правило подсчёта в строке N_prior")
    cp, f = checkpoint(regs), IO.REP / "CHECKPOINT.json"
    old = json.loads(f.read_text()) if f.exists() else None
    if old and old["N_prior"] == cp["N_prior"] and old["N_declared_before_stats"] == cp["N_declared_before_stats"]:
        return old        # повторный запуск не сдвигает время объявления
    f.write_text(json.dumps(cp, ensure_ascii=False, indent=1))
    (IO.REP / "ALL_TRIALS.csv").write_text(HEADER)       # шапка; конфигурации V4 пока не испытывались (0 из 60)
    return cp


def verify() -> list:
    """Нарушения: CHECKPOINT согласован с реестрами, объявлен до любых p (файлов с p нет), ALL_TRIALS без строк конфигураций."""
    bad, regs = [], load_regs()
    cp = json.loads((IO.REP / "CHECKPOINT.json").read_text())
    if cp["N_prior"] != n_prior(regs) or cp["N_declared_before_stats"] != cp["N_prior"] + BUDGET:
        bad.append("CHECKPOINT: N_prior / N_declared_before_stats не сходятся с реестрами")
    rows = [x for x in (IO.REP / "ALL_TRIALS.csv").read_text().strip().splitlines()[1:] if not x.startswith(("Ж-222:B", "Ж-222:C", "Ж-222:E", "Ж-222:NTP", "Ж-222:K8", "Ж-222:RQC", "Ж-222:H", "Ж-222:I"))]       # отступление (листы H, I: + Ж-222:I*)       # отступление (лист H): + Ж-222:H*       # (+ K8, RQC — лист K8-RQC); строки этапов B, C, E (Ж-222:B*, C*, E*) допустимы — их пишут стадии B, Ca, Cb, E
    if rows:
        bad.append("ALL_TRIALS.csv: есть строки конфигураций вне этапа B")
    if not all(registry_rows(regs)["contains_previous"]):
        bad.append("реестры не кумулятивны — правило объединения пересмотреть")
    return bad
