"""V5-0, G10: реестр испытаний и CHECKPOINT. N Холма = |объединение ключей `key` реестров| (реестры кумулятивны — суммировать строки нельзя, мутация № 114): 4 реестра до V4 (634) ∪ V4 (46 ключей) = 680;
+ бюджет V5 24 = 704; N_wide = 704 + 2400 проб Optuna h = 1 (файлы `data/regen/*h1*trials*.csv`) = 3104. Пороги «go» направлений (Ж-226 п. 4): три условия с K8 в планке."""
from __future__ import annotations

import datetime as dt
import io
import json

import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_e_go as GO, v4_fin_data as FD, v4_trials as TR
from models_v2.s12 import v5_0_io as IO

BUDGET = 24
HEADER = "key,program,source,direction,config,kind\n"
V4_NAME, V5_NAME = "harj_hybrid_v4", "harj_hybrid_v5"
ALL_TRIALS = IO.REP / "ALL_TRIALS.csv"
CP = IO.REP / "CHECKPOINT.json"


def n_unique(regs: dict) -> int:
    """N по реестрам — объединение ключей (не сумма строк)."""
    return len(set().union(*[set(r["key"]) for r in regs.values()])) if regs else 0


def load_regs() -> tuple:
    prior = TR.load_regs()
    v4 = pd.read_csv(IO.ROOT / "docs/otchety" / V4_NAME / "ALL_TRIALS.csv")
    v5 = pd.read_csv(ALL_TRIALS) if ALL_TRIALS.exists() else pd.read_csv(io.StringIO(HEADER))
    return prior, v4, v5


def history_table(prior: dict, v4: pd.DataFrame, v5: pd.DataFrame) -> tuple:
    """SEARCH_HISTORY: реестры до V4 (по `v4_trials.registry_rows`), V4, V5, прочие Optuna-файлы h = 1, итоговые числа."""
    rows = TR.registry_rows(prior)
    k_prior = set().union(*[set(r["key"]) for r in prior.values()])
    allk = k_prior | set(v4["key"])
    v4r = dict(item="registry", name=V4_NAME, rows=len(v4), unique_keys=v4["key"].nunique(), dup_rows=len(v4) - v4["key"].nunique(), n_trial=int((v4["kind"] == "trial").sum()), n_pilot_forecast=int((v4["kind"] == "pilot_forecast").sum()),
               contains_previous=False, new_vs_previous=len(set(v4["key"]) - k_prior), programs=";".join("%s:%d" % (p, c) for p, c in v4.groupby("program", sort=False).size().items()), note="V4 не содержит прежних ключей: не кумулятивный, ключи добавляются к объединению")
    v5r = dict(item="registry", name=V5_NAME, rows=len(v5), unique_keys=v5["key"].nunique(), dup_rows=0, n_trial=0, n_pilot_forecast=0, contains_previous=False, new_vs_previous=len(set(v5["key"]) - allk), programs="", note="лист 0: конфигураций нет")
    opt = TR.other_h1_files()
    n_opt = int(opt["rows"].sum())
    tot = pd.DataFrame([dict(item="N", name="N_prior (объединение ключей, до V5)", rows=len(allk), note="сумма строк реестров %d — НЕ N" % (sum(len(r) for r in prior.values()) + len(v4))),
                        dict(item="N", name="N_declared (N_prior + бюджет V5 %d)" % BUDGET, rows=len(allk) + BUDGET), dict(item="N", name="N_wide (N_declared + пробы Optuna h=1)", rows=len(allk) + BUDGET + n_opt)])
    out = pd.concat([rows, pd.DataFrame([v4r, v5r]), opt, tot], ignore_index=True)
    for c in ("unique_keys", "dup_rows", "n_trial", "n_pilot_forecast", "new_vs_previous"):
        out[c] = pd.to_numeric(out[c], errors="coerce")       # пустые ячейки «—» не должны делать колонку смешанного типа
    return out, dict(n_prior=len(allk), n_opt=n_opt, opt_parts={r["name"]: int(r["rows"]) for _, r in opt.iterrows()}, sources={n: r["key"].nunique() for n, r in prior.items()} | {V4_NAME: v4["key"].nunique()})


def go_thresholds(k: pd.DataFrame, cp: pd.Series) -> dict:
    """Пороги трёх условий «go» с K8 в планке: (1) crit_panel ≤ min(K1,K2,K3,D2,K8) − m; (2) без 2017 ≤ min(K2,K3,D2,K8) − m; (3) без 1 % крайних дат ≤ min(K2,K3,D2,K8) − m, m = 0,0005."""
    g = GO.rules(k, ["k1"] + FD.CFGS, cp, 0.0)
    n17 = {**{r: g["ref"][r]["crit_no2017"] for r in GO.REF}, "K8": g["cfg"]["K8"]["crit_no2017"], "k1": g["cfg"]["k1"]["crit_no2017"]}
    tr = {**{r: g["ref"][r]["crit_trim1"] for r in GO.REF}, "K8": g["cfg"]["K8"]["crit_trim1"], "k1": g["cfg"]["k1"]["crit_trim1"]}
    m = S.MARGIN
    c1 = {s: float(cp[s]) for s in ("k1", "k2", "k3", "d2", "K8")}
    return dict(margin=m, cond1_crit_panel=min(c1.values()) - m, cond1_min_of=min(c1, key=c1.get), cond2_crit_no2017=min(n17[s] for s in ("k2", "k3", "d2", "K8")) - m, cond3_crit_trim1=min(tr[s] for s in ("k2", "k3", "d2", "K8")) - m,
                values=dict(crit_panel=c1, crit_no2017={s: float(v) for s, v in n17.items()}, crit_trim1={s: float(v) for s, v in tr.items()}))


def checkpoint(info: dict, thr: dict, cp: pd.Series) -> dict:
    n = info["n_prior"]
    return {"program": "HAR-J Hybrid V5 (Ж-226), h=1, 17 бумаг", "N_prior": n, "budget": BUDGET, "N_declared_before_stats": n + BUDGET, "N_wide": n + BUDGET + info["n_opt"], "N_wide_optuna_trials": info["n_opt"], "N_wide_optuna_parts": info["opt_parts"],
            "sources": info["sources"], "rule": TR.RULE, "go_rule": "три условия одновременно (GATES_V5.md, Договорённости): crit_panel; crit_panel без 2017; crit_panel без 1 % крайних дат; планка включает K8", "go_thresholds": thr,
            "controls_crit_panel": {s: float(cp[s]) for s in ("k1", "k2", "k3", "d2", "K8")}, "used": 0, "note": "лист 0: конфигурации не расходуются; записано до первой конфигурации и до любого p",
            "written_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def write_checkpoint(cpd: dict) -> None:
    """Время записи сохраняется при повторном запуске (идемпотентно): если файл есть и числа те же, не переписывается."""
    if CP.exists():
        old = json.loads(CP.read_text())
        if {k: v for k, v in old.items() if k != "written_at"} == {k: v for k, v in cpd.items() if k != "written_at"}:
            return
    CP.write_text(json.dumps(cpd, ensure_ascii=False, indent=1))
