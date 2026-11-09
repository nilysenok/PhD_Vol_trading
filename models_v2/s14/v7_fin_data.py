"""V7 лист FIN (Ж-232 п. 16): данные итога — СОХРАНЁННЫЕ прогнозы конфигураций 1–8, 10 (версия 2) и контроли на одинаковых датах (`v7_4_evrun.load`), таблицы листов 1b–4 и DX с диска, реестр, журнал.
Ничего не обучает и не пишет parquet. Конфигурации не расходуются (0)."""
from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from models_v2.s14 import v7_0_io as IO

CFG = ["cfg%d" % i for i in (1, 2, 3, 4, 5, 6, 7, 8, 10)]
CTRL = ["K1", "K2", "K3", "M", "K8", "V4:NTP2", "V6:A2", "V6:A0", "V6:A0g"]
SER = CTRL + CFG
BASES = ["K2", "K3", "V4:NTP2", "V6:A2"]                       # XGBoost, LightGBM, прежние гибриды NTP2 и A2
CODE_VERSION = "v7.fin"
STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-232 п. 16, лист FIN (итог программы V7; новых прогнозов и конфигураций нет)"
TABLES = ["FIN_ALL", "FIN_YEARS", "FIN_TESTS", "FIN_MCS", "FIN_STRESS", "FIN_AFTER", "FIN_PAIRS", "FIN_GO", "FIN_WEIGHTS", "FIN_SPEC", "FIN_V1", "FIN_LAMSWEEP", "FIN_DATA", "FIN_CHANNELS", "FIN_DX", "FIN_REGISTRY", "FIN_DEPARTURES"]
NOTE = "итог программы V7, SBER, h = 1, 2018–2026 общие даты"
JOURNAL = IO.ROOT / "docs/resheniya/resheniya.md"
HEAD = re.compile(r"^(\*\*)?(↺ — \[V7|Моя ошибка \[V7)")
CODE_DIR = Path(__file__).parent


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(CODE_DIR.glob("v7_fin_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def write_csv(df: pd.DataFrame, name: str) -> None:
    with open(IO.REP / (name + ".csv"), "w", encoding="utf-8") as f:
        f.write("# %s; код %s %s; входы %s; %s; %s\n" % (STATUS, CODE_VERSION, code_hash(), IO.inputs_hash(), IO.CAVEAT, NOTE))
        df.to_csv(f, index=False, lineterminator="\n")


def read_cp() -> dict:
    return json.loads((IO.REP / "CHECKPOINT.json").read_text(encoding="utf-8"))


def departures(path: Path = JOURNAL) -> pd.DataFrame:
    """Все записи журнала «↺ — [V7]» и «Моя ошибка [V7]» дословно (заголовок и пункты под ним, до 900 знаков)."""
    lines, rows = path.read_text(encoding="utf-8").splitlines(), []
    for i, ln in enumerate(lines):
        if HEAD.match(ln):
            body = [ln.strip()] + [x.strip() for x in _bullets(lines[i + 1:])]
            rows.append(dict(line=i + 1, kind="ошибка дирижёра" if "ошибка" in ln[:26] else "отступление", text=" ".join(body)[:900]))
    return pd.DataFrame(rows)


def _bullets(rest: list) -> list:
    out = []
    for x in rest:
        if x.startswith(("- ", "  ", "* ")):
            out.append(x)
        else:
            break
    return out


def registry() -> pd.DataFrame:
    """11 мест плана: название конструкции, итоговый статус (последняя строка результата, кроме архива версии 1), версия 1 в архиве."""
    t = pd.read_csv(IO.REP / "ALL_TRIALS.csv")
    rows = []
    for n in range(1, 12):
        reg = t[t["key"] == "Ж-233:%d" % n].iloc[0]
        res = t[t["key"].str.startswith("Ж-233:%d:" % n) & ~t["key"].str.contains("_v1_архивирован")]
        arch = bool((t["key"].str.startswith("Ж-233:%d:" % n) & t["key"].str.contains("_v1_архивирован")).any())
        rows.append(dict(place=n, name=json.loads(reg["config"])["name"], registered_at=reg["registered_at"], status=res.iloc[-1]["status"] if len(res) else reg["status"], n_rows=int(len(res)), v1_archived=arch))
    return pd.DataFrame(rows)


def sheets() -> dict:
    """Таблицы листов, читаемые «как есть» (общие для FIN и проверки)."""
    r = IO.read_csv
    w2 = r("2_WEIGHTS.csv").assign(quantity=lambda d: "pi" + d["k"].astype(str)).rename(columns={"mean_pi": "mean"})[["config", "quantity", "period", "n", "mean"]]
    p = r("M10_PASSPORT.csv")
    smp = r("SAMPLES.csv")
    smp = smp[smp["section"] == "month"].assign(year=lambda d: d["date"].str[:4].astype(int)).groupby("year").head(1)[["year", "date", "n_a", "n_b"]]
    yr = p[p["section"] == "year"].assign(year=lambda d: d["year"].astype(int))[["year", "n_days", "n_bars", "bars_mode", "days_not_mode", "zero_return_share"]]
    ls = r("DX_LAMSWEEP.csv").groupby(["cfg", "lam_g"], as_index=False).agg(red_in_pct=("red_in_pct", "mean"), val_q_mean=("val_q_mean", "mean"), n_years=("date", "count"))
    return dict(FIN_PAIRS=pd.concat([r(n + "_PAIRS.csv").assign(src=n + "_PAIRS.csv") for n in ("2", "3", "4")], ignore_index=True),
                FIN_GO=pd.concat([r(n + "_GO.csv").assign(src=n + "_GO.csv") for n in ("2", "3", "4")], ignore_index=True),
                FIN_WEIGHTS=pd.concat([w2, r("3_WEIGHTS.csv"), r("4_WEIGHTS.csv")], ignore_index=True), FIN_SPEC=r("2_SPEC.csv"), FIN_V1=r("2_V1.csv"), FIN_LAMSWEEP=ls,
                FIN_DATA=yr.merge(smp, on="year", how="left"), FIN_CHANNELS=p[p["section"] == "channel"][["channel", "available", "note"]].reset_index(drop=True),
                FIN_DX=dx_table(read_cp()["DX"]), FIN_REGISTRY=registry(), FIN_DEPARTURES=departures())


def dx_table(dx: dict) -> pd.DataFrame:
    rows = []
    for k, v in dx["numbers"].items():
        for kk, vv in (v.items() if isinstance(v, dict) else [("", v)]):
            rows.append(dict(key=k + ("." + kk if kk else ""), value=float(vv), text="—"))
    return pd.concat([pd.DataFrame(rows), pd.DataFrame([dict(key="verdict." + k, value=float("nan"), text=v) for k, v in dx["verdicts"].items()])], ignore_index=True)
