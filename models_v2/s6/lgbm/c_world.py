"""Волна C, поток LGBM: счёт пакета конфигураций с кэшем в `data/.../lightgbm/c/` (ключ = trial + код ядра), строки таблиц, реестр trials волны C
(только hp-направление, лимит потока 40 из 55 − 15 B), CHECKPOINT. Результаты B читаются, не пишутся."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s6 import paths6
from models_v2.s6.lgbm import crit, job, tab, world
from models_v2.s6.lgbm.world import Item

HERE = Path(__file__).resolve().parent
DATA_C = paths6.OUT / "lightgbm" / "c"
CORE = ("fitl.py", "job.py", "grid.py", "crit.py", "world.py", "tab.py", "c_hp.py", "c_world.py")
LIMIT_C, LIMIT_HP = 40, 55
COLS = ("trial_id", "wave", "direction", "config", "metric_inner", "seed", "time")


def c_sha() -> str:
    h = hashlib.sha256()
    for n in CORE:
        h.update(n.encode() + (HERE / n).read_bytes())
    return h.hexdigest()[:16]


class CItem(Item):
    def key(self) -> str:
        return hashlib.sha256((self.tid() + c_sha()).encode()).hexdigest()[:16]


def paths(name: str, root: Path = DATA_C) -> tuple:
    return root / ("fc_%s.parquet" % name), root / ("st_%s.json" % name)


def cached(it: Item) -> bool:
    fp, sp = paths(it.name)
    return fp.exists() and sp.exists() and json.loads(sp.read_text()).get("key") == it.key()


def get(name: str) -> tuple:
    """Прогноз и статистика: сначала каталог C, иначе B (baseline и конфигурации волны B — только чтение)."""
    for root in (DATA_C, tab.DATA):
        fp, sp = paths(name, root)
        if fp.exists() and sp.exists():
            return pd.read_parquet(fp), json.loads(sp.read_text())
    raise FileNotFoundError(name)


def compute(W: dict, it: CItem, p) -> None:
    t0 = time.time()
    c = {"crit_pen": float("nan"), "crit_mean": float("nan"), "worst_fold": float("nan"), "by_year": {}}
    if it.folds:
        c = crit.crit(crit.frame(job.run_folds(p, it.name, it.hp_y), it.name))
    res = job.run_days(p, W["F"], it.name, it.hp_y, W["grid"])
    fc, st = job.assemble(W["F"], it.spec, W["eps"], W["grid"], res, it.name)
    fp, sp = paths(it.name)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(fp)
    st.update(key=it.key(), trial_id=it.tid(), crit=c, hp_by_year={str(Y): h.label() for Y, h in sorted(it.hp_y.items())}, seed=it.seed,
              kind=it.spec.kind, wt=list(it.spec.wt), direction=it.direction, desc=it.desc, secs_total=time.time() - t0)
    sp.write_text(json.dumps(st, ensure_ascii=False, default=float))


def run_batch(W: dict, items: list, procs: int = 4) -> None:
    todo = [it for it in items if not cached(it)]
    if not todo:
        return
    with job.pool(W["F"], {it.name: it.spec for it in todo}, W["eps"], procs) as p:
        for it in todo:
            compute(W, it, p)
            print("готово", it.name, "crit_pen %.6f" % get(it.name)[1]["crit"]["crit_pen"], flush=True)


def rows(W: dict, names: list, base: str, ref: str = "lb0_inh") -> pd.DataFrame:
    """Строки описания (колонки B + статус); DM и «перспективно» — к base; дополнительно Δ к ref (baseline LGBM) и к HAR-J(б) уже в строке."""
    fb, sb = get(base)
    fr, sr = get(ref)
    out = []
    for n in names:
        fc, st = get(n)
        meta = {"trial_id": st["trial_id"], "direction": st["direction"], "crit_pen": st["crit"]["crit_pen"],
                "crit_pen_base": sb["crit"]["crit_pen"], "crit_mean": st["crit"]["crit_mean"], "worst_fold": st["crit"]["worst_fold"], "desc": st["desc"]}
        r = tab.row(n, st["kind"], fc, W["har"], fb, st, meta)
        q, qr = _q(fc), _q(fr)
        r.update(d_crit_pen_vs_base=meta["crit_pen"] - meta["crit_pen_base"], d_crit_pen_vs_ref=meta["crit_pen"] - sr["crit"]["crit_pen"],
                 d_qlike_vs_ref=float((q - qr).mean()), seed=st["seed"], hp=json.dumps(st["hp_by_year"], ensure_ascii=False), base=base, ref=ref)
        out.append(r)
    return pd.DataFrame(out)


def _q(fc: pd.DataFrame) -> np.ndarray:
    return np.asarray(metrics.qlike(fc["rv"], fc["sigma2"]))


def write(df: pd.DataFrame, path: Path, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# %s; волна C Ж-206 поток LGBM; %s; DEVELOPMENT: колонки 2017–2021 — потолок (кандидаты видели 2017–2021), чистые 2022–2026; "
                "критерий crit_pen = среднее годовых LOYO + 0,25·(худший − среднее)\n" % (paths6.STATUS, note))
        df.to_csv(f, index=False, lineterminator="\n")


def register(names: list) -> int:
    """В реестр попадают только direction 'hp' (контроли — 'control' — не trials); дубль trial_id не считается."""
    new = []
    for n in names:
        st = get(n)[1]
        if st["direction"] == "hp":
            m = "crit_pen=%.6f;crit_mean=%.6f" % (st["crit"]["crit_pen"], st["crit"]["crit_mean"])
            new.append({"trial_id": st["trial_id"], "wave": "C", "direction": "hp", "config": "%s|%s" % (n, st["desc"]), "metric_inner": m, "seed": 42,
                        "time": "%.1f" % st["secs_total"]})
    return tab.register(new)


def checkpoint(path: Path = tab.CHECK, trials: Path = tab.TRIALS) -> dict:
    """Пересчёт счётчиков из реестра; волна C: hp (wave C) <= 40, всего hp <= 55."""
    c = json.loads(path.read_text())
    t = pd.read_csv(trials)
    hp = t[t["direction"] == "hp"]
    c["used"]["hp"], c["used"]["obj_feat"] = int(len(hp)), int((t["direction"] != "hp").sum())
    c["used_stream_c"] = {"hp": int((hp["wave"] == "C").sum())}
    c["limits_stream_c"] = {"hp": LIMIT_C}
    c["updated"] = time.strftime("%Y-%m-%d %H:%M")
    path.write_text(json.dumps(c, ensure_ascii=False, indent=1))
    if c["used_stream_c"]["hp"] > LIMIT_C or c["used"]["hp"] > LIMIT_HP or t["trial_id"].duplicated().any():
        raise ValueError("превышен бюджет потока LGBM (C) или дубль trial_id: %s" % c["used_stream_c"])
    return c
