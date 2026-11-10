"""ZD, стадия (Ж-246): compute() — всё в памяти; write() — ZD_*.csv, ZD_REPORT.md, ZD_MANIFEST.json (sha входов и выходов); check() — ТОЛЬКО читает (пересчёт, сверка, хэш чужих выходов до и после).
CHECKPOINT.json и ALL_TRIALS.csv не пишутся и не читаются (их ведёт лист Z). Конфигураций 0, прогнозов RV нет, в `x1_reg` не входим. Вход: `python3 -W ignore -m models_v2.s16.x1_zd_stage [--check]`."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import models_v2.s16  # noqa: F401  (torch первым)
from models_v2.s16 import x1_ya_feat as FE, x1_zd_causes as CA, x1_zd_core as CO, x1_zd_doc as DOC, x1_zd_tab as TB

ROOT = Path(__file__).resolve().parents[2]
REP, DATA = ROOT / "docs/otchety/harj_x1", ROOT / "data/regen"
INPUTS = {"x1": DATA / "harj_deep/E1/x1.parquet", "y1": DATA / "harj_x1/YA/y1.parquet", "y2": DATA / "harj_x1/YA/y2.parquet", "y3": DATA / "harj_x1/YB/y3.parquet",
          "store_raw": DATA / "harj_hybrid_v7/store_raw.npz", "YA_PARAMS": REP / "YA_PARAMS.csv", "YB_PARAMS": REP / "YB_PARAMS.csv"}
TABLES = {"ZD_STATES": "STATES", "ZD_PRED": "PRED", "ZD_CALIB": "CALIB", "ZD_CAUSES": "CAUSES"}
REPORT, MANIFEST = REP / "ZD_REPORT.md", REP / "ZD_MANIFEST.json"
WATCH = [DATA / "harj_deep", DATA / "harj_hybrid_v7", DATA / "harj_x1/YA", DATA / "harj_x1/YB"] + [REP / n for n in ("YA_*", "YB_*", "P0_*", "T0_*")]
HEAD = "# диагностика, не отбор; историческая; без конфигураций; Ж-246, лист ZD (SBER, h = 1); код zd %s\n"
YS = {"Y1": "y1", "Y2": "y2", "Y3": "y3"}


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("x1_zd_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def digest(patterns) -> str:
    h, files = hashlib.sha256(), []
    for p in patterns:
        files += [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else sorted(p.parent.glob(p.name))
    for f in sorted(set(files)):
        h.update(str(f).encode() + sha_file(f).encode())
    return h.hexdigest()


def read_params(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, skiprows=1)


def load() -> dict:
    """X1, Y1–Y3 на общих датах (сверка d, t⁺, rv, X1), склад M10 (только чтение), s_d Y-A, z_d."""
    x1 = pd.read_parquet(INPUTS["x1"])
    ys = {y: pd.read_parquet(INPUTS[k]) for y, k in YS.items()}
    for y, f in ys.items():
        ok = pd.DatetimeIndex(f["d"]).equals(pd.DatetimeIndex(x1["d"])) and pd.DatetimeIndex(f["t_plus"]).equals(pd.DatetimeIndex(x1["t_plus"]))
        if not (ok and np.allclose(f["rv"], x1["rv"]) and np.allclose(f["X1"], x1["F"])):
            raise SystemExit("%s не на общих датах / не тот X1 или RV" % y)
    d, tp, e = pd.DatetimeIndex(x1["d"]), pd.DatetimeIndex(x1["t_plus"]), x1["s"].to_numpy(int)
    store = FE.load_store()
    s = FE.features(store, d, tp)["s"]
    return dict(d=d, tp=tp, e=e, ys=ys, Z=CO.zfeatures(store, d, e, s), ya=read_params(INPUTS["YA_PARAMS"]), yb=read_params(INPUTS["YB_PARAMS"]), shas={k: sha_file(p) for k, p in INPUTS.items()})


def compute(inp: dict | None = None) -> tuple:
    inp = inp or load()
    d, tp, e, n = inp["d"], inp["tp"], inp["e"], len(inp["d"])
    year = np.asarray(tp.year)
    Ds = {y: CO.dloss(f["rv"].to_numpy(float), f["F"].to_numpy(float), f["X1"].to_numpy(float)) for y, f in inp["ys"].items()}
    preds, act, pools = {}, np.zeros(n, bool), None
    for y, D in Ds.items():
        w = CO.walk(inp["Z"], D, d, tp)
        preds[y] = {k: np.full(n, np.nan) for k in ("pa", "pb", "pc", "pg")}
        for k in preds[y]:
            preds[y][k][w["idx"].to_numpy()] = w[k].to_numpy()
        act[w["idx"].to_numpy()] = True
        pools = w.assign(year=d[w["idx"].to_numpy()].year)       # пулы одни и те же для Y1–Y3 (зависят только от дат)
    st = TB.states_table(Ds, e, year)
    pr = TB.pred_table(preds, Ds, e, year, act)
    t = dict(STATES=st, PRED=pr, CALIB=TB.calib_table(preds["Y2"]["pa"], Ds["Y2"], act), CAUSES=CA.causes_table(st, pr, pools, inp["ya"], inp["yb"]))
    num = dict(n_dates=n, first=str(d[0].date()), last=str(d[-1].date()), code=code_hash(), identity_bad=TB.identity(st, Ds), n_active=int(act.sum()), n_e_days=int(e.sum()), configs_spent=0, input_sha256=inp["shas"])
    return t, num


def csv_text(df: pd.DataFrame) -> str:
    return HEAD % code_hash() + df.to_csv(index=False, lineterminator="\n", float_format="%.10g")


def manifest(num: dict) -> dict:
    return dict(code=num["code"], configs_spent=0, input_sha256=num["input_sha256"], n_dates=num["n_dates"], n_active=num["n_active"], identity_bad=num["identity_bad"],
                output_sha256={**{k: sha_file(REP / (k + ".csv")) for k in TABLES}, "ZD_REPORT": sha_file(REPORT)})


def write() -> None:
    t, num = compute()
    if num["identity_bad"]:
        raise SystemExit("тождество Σ D нарушено: " + "; ".join(num["identity_bad"]))
    for k, v in TABLES.items():
        (REP / (k + ".csv")).write_text(csv_text(t[v]), encoding="utf-8")
    REPORT.write_text(DOC.build(t, num), encoding="utf-8")
    MANIFEST.write_text(json.dumps(manifest(num), ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def check() -> list:
    """ТОЛЬКО читает. -> расхождения (пусто = 0)."""
    before, bad = digest(WATCH), []
    t, num = compute()
    bad += num["identity_bad"]
    for k, v in TABLES.items():
        p = REP / (k + ".csv")
        if not p.exists() or p.read_text(encoding="utf-8") != csv_text(t[v]):
            bad.append("%s.csv ≠ пересчёту" % k)
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8") != DOC.build(t, num):
        bad.append("ZD_REPORT.md ≠ генератору")
    if not MANIFEST.exists():
        bad.append("нет ZD_MANIFEST.json")
    elif not bad and json.loads(MANIFEST.read_text(encoding="utf-8")) != json.loads(json.dumps(manifest(num))):
        bad.append("ZD_MANIFEST.json ≠ пересчёту / sha")
    if digest(WATCH) != before:
        bad.append("--check изменил чужие выходы")
    return bad


def main(argv: list) -> int:
    if "--check" not in argv:
        write()
        print("ZD: ZD_*.csv, ZD_REPORT.md, ZD_MANIFEST.json записаны")
        return 0
    bad = check()
    print("\n".join(bad) if bad else "OK: таблицы, отчёт, манифест = пересчёт; тождество Σ D выполнено; чужие выходы не изменены")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
