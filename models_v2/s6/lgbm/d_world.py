"""Волна D, поток LGBM: счёт пакета временных/режимных конфигураций с кэшем в `data/.../lightgbm/d/` (ключ = trial + код ядра D), строки таблиц,
реестр trials (direction temporal | regime, лимит потока 30 из 60), CHECKPOINT. Результаты B и C читаются, не пишутся."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s6 import paths6
from models_v2.s6.lgbm import c_world, d_comp, d_fit, d_reg, d_win, job, tab
from models_v2.s6.lgbm.d_win import Tm
from models_v2.s6.lgbm.grid import trial_id

HERE = Path(__file__).resolve().parent
DATA_D = paths6.OUT / "lightgbm" / "d"
CORE = ("fitl.py", "job.py", "grid.py", "crit.py", "world.py", "tab.py", "c_hp.py", "d_win.py", "d_fit.py", "d_reg.py", "d_world.py")
LIMIT_D, DIRS = 30, ("temporal", "regime")
BASE, REF = "C0_inh_F4", "lb0_inh"


def d_sha() -> str:
    h = hashlib.sha256()
    for n in CORE:
        h.update(n.encode() + (HERE / n).read_bytes())
    return h.hexdigest()[:16]


@dataclass
class DItem:
    name: str
    spec: job.Spec
    hp_y: dict
    tm: Tm
    direction: str            # temporal | regime | control
    desc: str = ""
    crit_from: str = ""       # частота: фолды LOYO не чувствительны к ней — crit копируется из ежедневной конфигурации того же окна

    def tid(self) -> str:
        hp = {str(Y): h.label() for Y, h in sorted(self.hp_y.items())}
        feats = [self.spec.X.shape[1], hashlib.sha256(np.ascontiguousarray(self.spec.X).tobytes()).hexdigest()[:12]]
        return trial_id(self.spec.kind, hp, feats, "", {"tm": asdict(self.tm), "name": self.name, "seed": 42})

    def key(self) -> str:
        return hashlib.sha256((self.tid() + d_sha()).encode()).hexdigest()[:16]


def paths(name: str, root: Path = DATA_D) -> tuple:
    return root / ("fc_%s.parquet" % name), root / ("st_%s.json" % name)


def cached(it: DItem) -> bool:
    fp, sp = paths(it.name)
    return fp.exists() and sp.exists() and json.loads(sp.read_text()).get("key") == it.key()


def get(name: str) -> tuple:
    for root in (DATA_D, c_world.DATA_C, tab.DATA):
        fp, sp = paths(name, root)
        if fp.exists() and sp.exists():
            return pd.read_parquet(fp), json.loads(sp.read_text())
    raise FileNotFoundError(name)


def folds_of(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_D / ("fold_%s.parquet" % name))


def compute(W: dict, it: DItem, p) -> None:
    t0, F = time.time(), W["F"]
    c, fdf = {"crit_pen": float("nan"), "crit_mean": float("nan"), "worst_fold": float("nan"), "by_year": {}}, None
    if it.crit_from:
        c = get(it.crit_from)[1]["crit"]
    else:
        fdf = d_comp.fold_frame(d_fit.run_folds(p, it.name, it.hp_y, it.tm))
        c = d_comp.crit_of(fdf, it.name)
    flip = d_reg.flip_on_grid(d_reg.stress_rows(F.dates), W["grid"]) if it.tm.freq == "adapt" else None
    bl = d_win.blocks(W["grid"], d_win.block_ids(F.dates[W["grid"]], it.tm.freq, flip))
    fc, st = job.assemble(F, it.spec, W["eps"], W["grid"], d_fit.run_blocks(p, F, it.name, it.hp_y, it.tm, bl), it.name)
    st.update(n_fits=len(bl), secs_total=time.time() - t0)
    save(it.name, fc, st, dict(key=it.key(), trial_id=it.tid(), crit=c, hp_by_year={str(Y): h.label() for Y, h in sorted(it.hp_y.items())},
                               tm=asdict(it.tm), direction=it.direction, desc=it.desc, kind=it.spec.kind, seed=42), fdf)


def save(name: str, fc: pd.DataFrame, st: dict, extra: dict, fdf=None) -> None:
    fp, sp = paths(name)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(fp)
    if fdf is not None:
        fdf.to_parquet(DATA_D / ("fold_%s.parquet" % name))
    sp.write_text(json.dumps(dict(st, **extra), ensure_ascii=False, default=float))


def run_batch(W: dict, items: list, procs: int = 4) -> None:
    todo = [it for it in items if not cached(it)]
    if not todo:
        return
    with job.pool(W["F"], {it.name: it.spec for it in todo}, W["eps"], procs) as p:
        for it in todo:
            compute(W, it, p)
            print("готово", it.name, "crit_pen %.6f" % get(it.name)[1]["crit"]["crit_pen"], flush=True)


def rows(W: dict, names: list, base: str = BASE, ref: str = REF) -> pd.DataFrame:
    """Строки описания (колонки B/C + статус); DM — к HAR-J(б) и к base (tab.row), к ref (baseline LGBM) — колонки dloss_ref_*; crit_wf; время и число обучений."""
    fb, sb = get(base)
    fr, sr = get(ref)
    cwb = d_comp.crit_wf(fb)
    out = []
    for n in names:
        fc, st = get(n)
        meta = {"trial_id": st["trial_id"], "direction": st["direction"], "crit_pen": st["crit"]["crit_pen"], "crit_pen_base": sb["crit"]["crit_pen"],
                "crit_mean": st["crit"]["crit_mean"], "worst_fold": st["crit"]["worst_fold"], "desc": st["desc"]}
        r = tab.row(n, st["kind"], fc, W["har"], fb, st, meta)
        rr = tab.row(n, st["kind"], fc, W["har"], fr, st, meta)
        cw = d_comp.crit_wf(fc)
        r.update(dloss_ref_mean=rr["dloss_base_mean"], dloss_ref_dm_p=rr["dloss_base_dm_p"], crit_wf=cw, d_crit_wf_vs_base=cw - cwb,
                 d_crit_pen_vs_base=meta["crit_pen"] - meta["crit_pen_base"], d_crit_pen_vs_ref=meta["crit_pen"] - sr["crit"]["crit_pen"],
                 d_qlike_vs_ref=float(_dq(fc, fr)), tm=json.dumps(st.get("tm", {})), n_fits=st.get("n_fits", np.nan), base=base, ref=ref)
        out.append(r)
    return pd.DataFrame(out)


def _dq(a: pd.DataFrame, b: pd.DataFrame) -> float:
    return float((c_world._q(a) - c_world._q(b)).mean())


def write(df: pd.DataFrame, path: Path, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# %s; волна D Ж-206 поток LGBM; %s; DEVELOPMENT: колонки 2017–2021 — потолок (кандидаты видели 2017–2021); "
                "критерий crit_pen = среднее годовых LOYO + 0,25·(худший − среднее); crit_wf — по реализованным прошлым годам ряда\n" % (paths6.STATUS, note))
        df.to_csv(f, index=False, lineterminator="\n")


def register_rows(new: list, path: Path = tab.TRIALS) -> int:
    """Только direction temporal | regime (контроли — нет); дубль trial_id не считается; сверх лимита потока — ошибка до записи."""
    old = pd.read_csv(path)
    seen = set(old["trial_id"].astype(str))
    fresh = []
    for r in new:
        if r["direction"] in DIRS and r["trial_id"] not in seen:
            seen.add(r["trial_id"])
            fresh.append(r)
    if int(old["direction"].isin(DIRS).sum()) + len(fresh) > LIMIT_D:
        raise ValueError("превышен бюджет потока LGBM (D): %d + %d > %d" % (int(old["direction"].isin(DIRS).sum()), len(fresh), LIMIT_D))
    return tab.register(fresh, path)


def register(names: list) -> int:
    new = []
    for n in names:
        st = get(n)[1]
        m = "crit_pen=%.6f;crit_mean=%.6f" % (st["crit"]["crit_pen"], st["crit"]["crit_mean"])
        new.append({"trial_id": st["trial_id"], "wave": "D", "direction": st["direction"], "config": "%s|%s" % (n, st["desc"]), "metric_inner": m,
                    "seed": 42, "time": "%.1f" % st.get("secs_total", 0.0)})
    return register_rows(new)


def checkpoint(path: Path = tab.CHECK, trials: Path = tab.TRIALS) -> dict:
    c = json.loads(path.read_text())
    t = pd.read_csv(trials)
    tr, hp = t["direction"].isin(DIRS), t["direction"] == "hp"
    c["used"].update(hp=int(hp.sum()), obj_feat=int((~hp & ~tr).sum()), temporal_regime=int(tr.sum()))
    c["used_stream_d"], c["limits_stream_d"] = {"temporal_regime": int((tr & (t["wave"] == "D")).sum())}, {"temporal_regime": LIMIT_D}
    c["updated"] = time.strftime("%Y-%m-%d %H:%M")
    path.write_text(json.dumps(c, ensure_ascii=False, indent=1))
    if c["used_stream_d"]["temporal_regime"] > LIMIT_D or t["trial_id"].duplicated().any():
        raise ValueError("превышен бюджет потока LGBM (D) или дубль trial_id: %s" % c["used_stream_d"])
    return c
