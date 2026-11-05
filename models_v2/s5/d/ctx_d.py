"""D: кэш ежедневных прогнозов волны D (ключ = входы + код C + код счёта d + HP/seed/lib/окно/полураспад), хранилище JSON стадий, запись csv с шапкой."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from models_v2.s5 import common
from models_v2.s5.c import ctx_c, engine_c
from models_v2.s5.d import engine_d

HERE = Path(__file__).resolve().parent
DATA = common.DEEP / "d"
FIT_FILES = ("lgbfit.py", "engine_d.py")   # код, влияющий на прогнозы; правка отчётных модулей кэш не сбрасывает
HEAD = "# разработка; волна D Ж-204; до аудита; оценка 2017–02.2026 (2022+ — просмотр контроля гл. 3 по SBER, не независимый OOS); код d sha=%s; %s\n"


def code_sha() -> str:
    h = hashlib.sha256(ctx_c.code_sha().encode())
    for f in FIT_FILES:
        h.update(f.encode() + (HERE / f).read_bytes())
    return h.hexdigest()[:16]


def cached(p, C: dict, tag: str, name: str, hp_y: dict, seed: int, lib: str, win: int, hl: float) -> tuple:
    """Прогноз бустера (engine_d.daily) с кэшем; -> (DataFrame d/target_date/sigma2/rv, статистика)."""
    DATA.mkdir(parents=True, exist_ok=True)
    s = C["L"]["key"] + code_sha() + name + engine_c.hp_key(hp_y, seed) + "%s|%d|%g" % (lib, win, hl)
    k = hashlib.sha256(s.encode()).hexdigest()[:14]
    fp, sp = DATA / ("fc_%s_s%d_%s.parquet" % (tag, seed, k)), DATA / ("st_%s_s%d_%s.json" % (tag, seed, k))
    if fp.exists() and sp.exists():
        return pd.read_parquet(fp), json.loads(sp.read_text())
    fc, st = engine_d.daily(p, C["w"], name, hp_y, seed, lib, win, hl)
    fc.to_parquet(fp)
    sp.write_text(json.dumps(st))
    return fc, st


def load_json(name: str, default):
    f = DATA / name
    return json.loads(f.read_text()) if f.exists() else default


def save_json(name: str, obj) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / name).write_text(json.dumps(obj, ensure_ascii=False, default=float))


def put_row(rid: str, row: dict, trial: dict | None) -> None:
    rows = load_json("rows.json", {})
    rows[rid] = row
    save_json("rows.json", rows)
    if trial is not None:
        tr = load_json("trials_rows.json", {})
        tr[trial["trial_id"]] = trial
        save_json("trials_rows.json", tr)


def write(df: pd.DataFrame, path, note: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(HEAD % (code_sha(), note))
        df.to_csv(f, index=False, lineterminator="\n")
