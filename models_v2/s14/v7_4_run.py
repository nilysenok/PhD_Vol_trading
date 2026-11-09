"""V7-4a: запуск конфигурации 10 (стадия `run_v7 --stage 4`, G45). Рабочий процесс `python3 -W ignore -m models_v2.s14.v7_4_run <seed> …` считает годы 2019…2026 для своих seeds и пишет ЧАСТИ
`data/regen/harj_hybrid_v7/4/parts/s<seed>_<год>.npz` (G, π, g_k, λ по строкам года; возобновляемо — готовая часть пропускается) и веса годовых энкодеров Stage B `4/encB/tcn_w5_<год>_s<seed>.npz` (+ sha256 содержимого).
Seeds независимы (свой энкодер Stage A, своя цепочка, один поток torch) — результат не зависит от того, в каком процессе и в каком порядке они считаются. `write_cfg10` (бегун) досчитывает недостающие части, собирает
`4/cfg10.parquet` (d, t_plus, rv, H, G, F, fitted, pi0–pi2, g1, g2, stageA_2018; среднее по seeds; 2018 — копия строк конфигурации 6 с пометкой), пишет CHECKPOINT `4.cfg10` (sha256, календарь) и строку результата 10 в ALL_TRIALS.csv.
Здесь только QLIKE 2018–2021 без p; оценка — `v7_4_evrun`; проверка — `v7_4_chk`."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_io as IO, v7_1b_cfg as CFG1, v7_1b_cfgrun as CR, v7_1b_sta as STA, v7_1m_fit as FT, v7_1m_reg as RG, v7_2_run as R2, v7_4_cfg as C4

STATUS_4 = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-235/Ж-236, лист 4a (конфигурация 10: совместное обучение Stage B по календарю; QLIKE 2018–2021 без p; оценка — v7_4_evrun)"
COLS = ["d", "t_plus", "rv", "H", "G", "F", "fitted", "pi0", "pi1", "pi2", "g1", "g2", "stageA_2018"]
CFG6 = IO.OUT / "2" / "cfg6.parquet"


def path():
    return IO.OUT / "4" / "cfg10.parquet"


def enc_path(year: int, seed: int):
    return IO.OUT / "4" / "encB" / STA.name("tcn", year, seed)


def part_path(seed: int, year: int):
    return IO.OUT / "4" / "parts" / ("s%d_%d.npz" % (seed, year))


def update_4(sub: str, value: dict) -> None:
    cur = dict(CR.read_cp().get("4", {}))
    cur[sub] = value
    IO.merge_checkpoint("4", cur)


def ready(cp_path=None) -> bool:
    p = cp_path or RG.CHECKPOINT
    return p.exists() and "cfg10" in json.loads(p.read_text(encoding="utf-8")).get("4", {})


def inputs():
    """-> (tab, pack, seq, Zg): данные SBER читаются только здесь."""
    from models_v2.s14 import v7_1e_pre as PRE, v7_1m_data as DATA, v7_1m_feat as FE
    tab, pack = DATA.load_tab(), PRE.pack_from_npz(STA.WINDOW, IO.OUT)
    return tab, pack, C4.seq_matrix(pack, tab.index), FE.build_Z(tab).to_numpy(float)


def save_year(seed: int, year: int, out: dict, seconds: float) -> dict:
    """Часть года и веса энкодера. -> sha256 содержимого обоих (в CHECKPOINT)."""
    bl = [k for k in out if k != "enc"]
    rows, ks = np.concatenate([np.arange(a, a + len(out[a]["G"])) for a in bl]), lambda f: np.concatenate([f(out[a]) for a in bl])
    part = dict(rows=rows, G=ks(lambda r: r["G"]), pi=np.vstack([out[a]["pi"] for a in bl]), gk=np.vstack([out[a]["gk"] for a in bl]), lam=ks(lambda r: np.tile(np.array(r["lam"], float), (len(r["G"]), 1))).reshape(-1, 2),
                joint=ks(lambda r: np.full(len(r["G"]), float(r is out[bl[0]]))), seconds=np.array([seconds]))
    e = out["enc"]
    arrs = {"w__" + k: v for k, v in e["state"].items()}
    arrs.update({"pca_" + k: np.asarray(e["pca"][k], float) for k in ("mu", "sd", "comps", "ssd")}, meta=np.array([year, seed, e["pca"]["n_past"]], float))
    return dict(part=STA.save(part, part_path(seed, year)), enc=STA.save(arrs, enc_path(year, seed)))


def worker(seeds, years=C4.YEARS_B, log=print) -> int:
    tab, pack, seq, Zg = inputs()
    plan, done = C4.plan(tab.index), 0
    for seed in seeds:
        for y in years:
            if part_path(seed, y).exists() and enc_path(y, seed).exists():
                continue
            t0 = time.time()
            out = C4.run_year(tab, Zg, seq, pack, y, seed, [b for b in plan if b["year"] == y])
            save_year(seed, y, out, time.time() - t0)
            done += 1
            log("год %d seed %d готово, %.0f с" % (y, seed, time.time() - t0))
    return done


def load_parts(seeds=FT.SEEDS, years=C4.YEARS_B) -> dict:
    miss = [(s, y) for s in seeds for y in years if not part_path(s, y).exists()]
    if miss:
        raise SystemExit("нет частей конфигурации 10: %s" % miss[:5])
    return {(s, y): STA.load(part_path(s, y)) for s in seeds for y in years}


def assemble(tab: pd.DataFrame, df6: pd.DataFrame, parts: dict, seeds=FT.SEEDS) -> pd.DataFrame:
    """Среднее по seeds: G (→ F = H·exp(G)), π, g_k по строкам 2019+; строки до 2019 — копия конфигурации 6, stageA_2018 = 1."""
    n, S = len(tab), len(seeds)
    G, pi, gk = np.full((S, n), np.nan), np.full((S, n, 3), np.nan), np.full((S, n, 2), np.nan)
    for (s, _), p in parts.items():
        i = seeds.index(s)
        G[i, p["rows"]], pi[i, p["rows"]], gk[i, p["rows"]] = p["G"], p["pi"], p["gk"]
    pos, a18 = tab.index.get_indexer(pd.DatetimeIndex(df6["d"])), np.asarray(pd.DatetimeIndex(df6["d"]) < C4.FIRST_B)
    mean = lambda a: a.mean(axis=0)[pos]
    H, Gm = df6["H"].to_numpy(float), np.where(a18, df6["G"].to_numpy(float), mean(G))
    out = pd.DataFrame({"d": df6["d"], "t_plus": df6["t_plus"], "rv": df6["rv"], "H": H, "G": Gm, "F": np.where(a18, df6["F"].to_numpy(float), H * np.exp(Gm)), "fitted": np.where(a18, df6["fitted"].to_numpy(float), 1.0)})
    for j, c in enumerate(("pi0", "pi1", "pi2")):
        out[c] = np.where(a18, df6[c].to_numpy(float), mean(pi)[:, j])
    for j, c in enumerate(("g1", "g2")):
        out[c] = np.where(a18, df6[c].to_numpy(float), mean(gk)[:, j])
    out["stageA_2018"] = a18.astype(float)
    if not np.isfinite(out[COLS[4:]].to_numpy(float)).all():
        raise ValueError("конфигурация 10: нечисловые значения G, F, π или g_k")
    return out[COLS]


def record(df: pd.DataFrame, parts: dict, shas: dict, seconds: float) -> dict:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    q, n = CFG1.qlike_choice(df)
    lam = {}
    for pt in parts.values():
        for g, b in np.unique(pt["lam"][pt["joint"] == 1.0], axis=0):
            lam["%g/%g" % (g, b)] = lam.get("%g/%g" % (g, b), 0) + 1
    return dict(file=str(p.relative_to(IO.ROOT)), encoder="tcn", window=CFG1.WINDOW_SIZE, sha256_content=CR.content_hash(df), sha256_file=IO.sha_file(p), n_rows=len(df), first=str(df["d"].iloc[0].date()), last=str(df["d"].iloc[-1].date()),
                qlike_2018_2021=q, n_dates_2018_2021=n, params=C4.PARAMS_HEAD + 905, router="B", grid=[list(g) for g in C4.GRID10], seeds=list(FT.SEEDS), lam_e=1e-2, joint_months=list(C4.JOINT_MONTHS), joint_years=list(C4.YEARS_B),
                stageA_2018_rows=int(df["stageA_2018"].sum()), lam_selected_joint=lam, enc_sha256=shas["enc"], part_sha256=shas["part"], seconds_cpu=round(seconds, 1))


def write_cfg10() -> int:
    import shutil
    if shutil.disk_usage(IO.ROOT).free / 1e9 < 40:
        raise SystemExit("свободно < 40 ГБ: запись отменена")
    old = CR.read_cp().get("4", {}).get("cfg10")
    if old and path().exists() and IO.sha_file(path()) == old["sha256_file"]:
        print("конфигурация 10: уже посчитана, пропуск")
        return 0
    worker(FT.SEEDS)                                                          # досчитывает недостающее (готовые части пропускаются)
    tab, df6, parts = inputs()[0], pd.read_parquet(CFG6), load_parts()
    df = assemble(tab, df6, parts)
    if not R2.same_dates(df):
        raise SystemExit("конфигурация 10: даты ≠ датам конфигурации 1: запись отменена")
    shas = dict(enc={"%d_s%d" % (y, s): STA.content_hash(STA.load(enc_path(y, s))) for (s, y) in parts}, part={"%d_s%d" % (y, s): STA.content_hash(p) for (s, y), p in parts.items()})
    sec = float(sum(p["seconds"][0] for p in parts.values()))
    rec = record(df, parts, shas, sec)
    update_4("cfg10", dict(rec, written_at=RG.now()))
    RG.add_result("10", dict(stage="4a", qlike_2018_2021=rec["qlike_2018_2021"], sha256_content=rec["sha256_content"], n_rows=len(df)))
    print("конфигурация 10: %d строк, QLIKE 2018–2021 %.6f, %.0f с процессорного времени" % (len(df), rec["qlike_2018_2021"], sec))
    return 0


def connect(runners: dict) -> None:
    """Подключает счёт 10 к бегуну; 11 — обработчика нет (часть 4b, только при `4.cond11` и `cond_11` = true)."""
    runners.setdefault("10", write_cfg10)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    print("посчитано частей: %d" % worker(tuple(int(x) for x in sys.argv[1:])))
    sys.exit(0)
