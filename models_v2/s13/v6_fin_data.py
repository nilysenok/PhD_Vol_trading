"""V6 лист FIN (Ж-227): данные итога — СОХРАНЁННЫЕ прогнозы 18 конфигураций (`data/regen/harj_hybrid_v6/*.parquet`), контроли листа 0 (K1, K2, K3, M, K8, D2, V4:NTP2, V5 GS4/SP1) и K9 на 2271 общих датах.
Ничего не обучает и не пишет parquet. Контроль D2 (панельный V4/V5) в таблицах итога называется `D2ctl`, чтобы не путать с конфигурацией D2 (аналоги листа 4)."""
from __future__ import annotations

import functools
import hashlib
from pathlib import Path

import pandas as pd

from models_v2.s13 import v6_0_cp as CP0, v6_0_io as IO, v6_0_load as LD, v6_1_stage as S1, v6_4_stage as S4

CFG = ["A0", "A1", "A2", "A3", "C1", "C2", "C3", "C4", "E1", "E2", "E3", "D1", "D2", "D3", "CB1", "F1", "F2", "A0g"]
CTRL = ["K1", "K2", "K3", "M", "K8", "D2ctl", "V4:NTP2", "V5:GS4", "V5:SP1", "K9"]
SER = CTRL + CFG
FROM_WIDE = {"M": "M", "D2ctl": "D2", "V5:GS4": "V5:GS4", "V5:SP1": "V5:SP1"}
CODE_VERSION = "v6.fin"
STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-227, лист FIN (итог программы V6; новых прогнозов и конфигураций нет)"
CODE_FILES = ["v6_fin_data.py", "v6_fin_tabs.py", "v6_fin_abl.py"]
TABLES = ["FIN_ALL", "FIN_YEARS", "FIN_TESTS", "FIN_MCS", "FIN_ABLATION", "FIN_STRESS", "FIN_ENCOMPASS", "FIN_INFO_ALL", "FIN_LEVELS"]
INFO_FILES = ["A_INFO", "C_INFO", "D_INFO", "E_INFO", "F_INFO", "G_INFO"]
LEVEL_FILES = ["A_LEVELS", "C_LEVELS", "D_LEVELS", "E_LEVELS", "F_LEVELS", "G_LEVELS"]
NOTE = "итог программы V6, SBER, h = 1, 2271 общих дат"


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(f.encode() + (Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:16]


def header(note: str) -> str:
    return "# %s; код %s %s; входы %s; %s; %s\n" % (STATUS, CODE_VERSION, code_hash(), IO.inputs_hash(), IO.CAVEAT, note)


def write_csv(df: pd.DataFrame, name: str, note: str = NOTE) -> None:
    with open(IO.REP / (name + ".csv"), "w", encoding="utf-8") as f:
        f.write(header(note))
        df.to_csv(f, index=False, lineterminator="\n")


def load() -> dict:
    """-> dict(w, fs, cp, cp0): w — rv, t⁺, K1, K2, K3, K8, V4:NTP2 (лист 0); fs — имя ряда -> прогноз на тех же датах (все ряды SER)."""
    wide, _ = LD.load()
    w = S1.controls()
    fs = {k: w[k].to_numpy(float) for k in ("K1", "K2", "K3", "K8", "V4:NTP2")}
    fs.update({k: wide[v].reindex(w.index).to_numpy(float) for k, v in FROM_WIDE.items()})
    fs["K9"] = pd.read_parquet(IO.OUT / "K9.parquet").set_index("d")["F"].reindex(w.index).to_numpy(float)
    fs.update(S4.prior_forecasts(w, CFG))
    cp = CP0.read()
    return dict(w=w, fs={k: fs[k] for k in SER}, cp=cp, cp0=cp["0"])


def read_info() -> pd.DataFrame:
    """Все `*_INFO.csv` листов одной таблицей (правило п. 2.2 по всем парам); копия «прежний вывод против A0» из G_INFO не дублируется."""
    out = []
    for n in INFO_FILES:
        t = IO.read_csv(n + ".csv")
        if "role" in t.columns:
            t = t[~t["role"].astype(str).str.startswith("прежний вывод")]
        out.append(t.assign(src=n + ".csv"))
    return pd.concat(out, ignore_index=True)


def read_levels() -> pd.DataFrame:
    t = pd.concat([IO.read_csv(n + ".csv").assign(src=n + ".csv") for n in LEVEL_FILES], ignore_index=True)
    return t.drop_duplicates(["config", "level"], keep="first").reset_index(drop=True)
