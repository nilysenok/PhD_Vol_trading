"""V5 лист FIN (Ж-226): данные итога — СОХРАНЁННЫЕ прогнозы GS1–GS5, SP1–SP4 (кэш листов 1a и 2) и контроли листа 0 на общем кадре. Ничего не обучает и не пишет parquet;
если кэш отсутствует или устарел — загрузка отказывает (итог не пересчитывает прогнозы). Ряды итога: K1, K2, K3, M, D2, K8, GS1–GS5, SP1–SP4 (H3 V4 — только для абляции)."""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s12 import v5_0_io as IO, v5_gs_eval as EV, v5_gs_panel as PA, v5_gs_run as GSR, v5_sp_run as SPR

GS, SP = list(EV.GS), list(SPR.CFGS)
BASE6 = ["k1", "k2", "k3", "M", "d2", "K8"]
SER = BASE6 + GS + SP
NAME = {"k1": "K1", "k2": "K2", "k3": "K3", "d2": "D2"}
STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-226, лист FIN (итог, новых прогнозов и конфигураций нет)"
CODE_FILES = ["v5_fin_data.py", "v5_fin_tabs.py", "v5_fin_joint.py", "v5_fin_cost.py"]
FIN_TABLES = ["FIN_ALL", "FIN_TICKERS", "FIN_YEARS", "FIN_SBER", "FIN_TESTS", "FIN_MCS", "FIN_JOINT", "FIN_ENCOMPASS", "FIN_ABLATION", "FIN_COST"]


def nm(s: str) -> str:
    return NAME.get(s, s)


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(f.encode() + (Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:16]


def header(note: str) -> str:
    return "# %s; код v5.fin %s; входы %s; %s\n" % (STATUS, code_hash(), IO.inputs_hash(), note)


def write_csv(df: pd.DataFrame, name: str, note: str) -> None:
    with open(IO.REP / (name + ".csv"), "w", encoding="utf-8") as f:
        f.write(header(note))
        df.to_csv(f, index=False, lineterminator="\n")


def checkpoint() -> dict:
    return json.loads((IO.REP / "CHECKPOINT.json").read_text())


def merged(k: pd.DataFrame, runs: dict) -> pd.DataFrame:
    """Общий кадр листа 0 + прогноз F каждой конфигурации на строках (бумага, d)."""
    k = k.copy()
    for g, r in runs.items():
        k = k.merge(r["main"][["ticker", "date", "F"]].rename(columns={"F": g}), on=["ticker", "date"], how="left")
    return k


def load() -> dict:
    """-> dict(P, k, kk, cc, runs, cp, best_gs, best_sp, cached). cc — общие строки (все ряды итога и H3 V4 конечны, > 0)."""
    P, k, cp = PA.build(), IO.frame(), checkpoint()
    runs = {g: GSR.get(P, g, False) for g in GS}
    runs.update({g: SPR.get(P, g, False) for g in SP})
    kk = merged(k, runs)
    cc = S.common(kk, SER + ["H3"]).reset_index(drop=True)
    return dict(P=P, k=k, kk=kk, cc=cc, runs=runs, cp=cp, best_gs=cp["GS"]["best_gs"], best_sp=cp["SP"]["best_sp"], cached={g: bool(r["cached"]) for g, r in runs.items()},
                keys={g: r["key"] for g, r in runs.items()})
