"""V7-1b-C: конфигурация 3 — лучший из 1 и 2 (`1b.choice3`, записан в CHECKPOINT ДО этого счёта) с окном 1 день, тот же протокол (G24).
Предобучение энкодера окна 1: годы 2018…2026 × seeds 0–4, только бары < 01.01.Y (`v7_1b_sta.compute(..., window=1)`, `X_w1.npz`); артефакты `stageA/<kind>_w1_<Y>_s<seed>.npz`, sha256 содержимого — в `1b.stageA_w1`.
Корректор g(z) и walk-forward — функции `v7_1b_cfg` (window=1). Прогноз `1b/cfg3.parquet`, CHECKPOINT `1b.cfg3`, строка результата в ALL_TRIALS.csv. Другие конфигурации здесь не считаются; `check3()` ничего не пишет."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_io as IO, v7_1b_cfg as CFG, v7_1b_cfgrun as CR, v7_1b_sta as STA, v7_1m_reg as RG, v7_fx_warm as WM

W3, CFG3 = 1, "3"
PATH = IO.OUT / "1b" / "cfg3.parquet"


def kind3(cp: dict) -> str:
    """Энкодер конфигурации 3 — только из записанного заранее `1b.choice3` (не пересчитывается и не выбирается здесь)."""
    ch = cp.get("1b", {}).get("choice3")
    if not ch or not ch.get("written_before_cfg3") or ch["choice"] not in STA.KINDS:
        raise SystemExit("1b.choice3 не записан до счёта конфигурации 3: стоп")
    return ch["choice"]


def tasks3(kind: str) -> list:
    return [(kind, y, s) for y in STA.YEARS for s in STA.SEEDS]


def file3(kind: str, y: int, s: int):
    return STA.directory() / STA.name(kind, y, s, W3)


def train_missing3(kind: str, log=print) -> int:
    from models_v2.s14 import v7_1e_pre as PRE
    pack, done = PRE.pack_from_npz(W3, IO.OUT), 0
    for k, y, s in tasks3(kind):
        if file3(k, y, s).exists():
            continue
        STA.save(STA.compute(k, y, s, pack, window=W3), file3(k, y, s))
        done += 1
        log("%s w1 %d seed %d готово" % (k, y, s))
    return done


def hashes3(kind: str) -> dict:
    return {file3(*t).name: STA.content_hash(STA.load(file3(*t))) for t in tasks3(kind)}


def save3(df: pd.DataFrame, kind: str, seconds: float) -> dict:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PATH, index=False)
    q, n = CFG.qlike_choice(df)
    return dict(file=str(PATH.relative_to(IO.ROOT)), encoder=kind, window=W3, sha256_content=CR.content_hash(df), sha256_file=IO.sha_file(PATH), n_rows=len(df), first=str(df["d"].iloc[0].date()),
                last=str(df["d"].iloc[-1].date()), qlike_2018_2021=q, n_dates_2018_2021=n, params=5, grid_lam_g=[g[0] for g in CFG.GRID], lam_lb_lam_s="не применяются (нет маршрутизатора)",
                seeds=list(CFG.FT.SEEDS), seconds=round(seconds, 1))


def write3() -> int:
    import shutil
    from models_v2.s14 import v7_1m_data as DATA
    if shutil.disk_usage(IO.ROOT).free / 1e9 < 40:
        raise SystemExit("свободно < 40 ГБ: запись отменена")
    cp = CR.read_cp()
    kind, old = kind3(cp), cp["1b"].get("cfg3")
    if old and PATH.exists() and IO.sha_file(PATH) == old["sha256_file"]:
        print("конфигурация 3: уже посчитана, пропуск")
        return 0
    t0 = time.time()
    train_missing3(kind)
    CR.update_1b("stageA_w1", dict(encoder=kind, window=W3, years=list(STA.YEARS), seeds=list(STA.SEEDS), hashes=hashes3(kind), note="конфигурация 3: энкодер окна 1 день, бары < 01.01.Y"))
    df = CFG.run_kind(kind, DATA.load_tab(), window=W3, warm=WM.load_warm())
    rec = save3(df, kind, time.time() - t0)
    CR.update_1b("cfg3", rec)
    RG.add_result(CFG3, dict(stage="1b-C", qlike_2018_2021=rec["qlike_2018_2021"], sha256_content=rec["sha256_content"], n_rows=len(df)))
    print("конфигурация 3 (%s, w1): %d строк, QLIKE 2018–2021 %.6f, %.0f с" % (kind, len(df), rec["qlike_2018_2021"], time.time() - t0))
    return 0


def check3(cp: dict) -> list:
    """-> замечания (пусто = ок): sha256 энкодеров и прогноза, F > 0 и g = 0 ⇒ F = H, один энкодер и одна переоценка побитно. Ничего не пишет."""
    if "cfg3" not in cp or "stageA_w1" not in cp:
        return ["CHECKPOINT 1b: нет cfg3/stageA_w1"]
    kind, bad = cp["cfg3"]["encoder"], []
    if kind != cp["choice3"]["choice"]:
        bad.append("cfg3: энкодер ≠ choice3")
    miss = [file3(*t).name for t in tasks3(kind) if not file3(*t).exists()]
    if miss or not PATH.exists():
        return bad + ["нет файлов конфигурации 3: %s" % (miss[:2] or PATH.name)]
    bad += ["%s: хэш содержимого ≠ CHECKPOINT" % n for n, h in hashes3(kind).items() if h != cp["stageA_w1"]["hashes"].get(n)]
    df = pd.read_parquet(PATH)
    if IO.sha_file(PATH) != cp["cfg3"]["sha256_file"] or CR.content_hash(df) != cp["cfg3"]["sha256_content"]:
        bad.append("cfg3.parquet: sha256 ≠ CHECKPOINT")
    if not ((df["F"] > 0).all() and np.array_equal(df["F"][df["g"] == 0], df["H"][df["g"] == 0])):
        bad.append("cfg3.parquet: F ≤ 0 или g = 0, но F ≠ H")
    from models_v2.s13 import v6_1_wf as W6
    from models_v2.s14 import v7_1e_pre as PRE, v7_1m_data as DATA, v7_1m_wf as WF
    new = STA.compute(kind, 2018, 0, PRE.pack_from_npz(W3, IO.OUT), window=W3)
    if STA.content_hash(new) != STA.content_hash(STA.load(file3(kind, 2018, 0))):
        bad.append("%s w1 2018 seed 0: пересчёт не совпал побитно" % kind)
    tab = DATA.load_tab()
    a = next(i for i in W6.refit_positions(tab.index, WF.FIRST_REFIT) if tab.index[i] >= CR.CHECK_FROM)
    got = CFG.refit_block(kind, tab, a, window=W3, warm=WM.load_warm())
    if not np.array_equal(got, df.set_index("d")["g"].reindex(tab.index[a:a + len(got)]).to_numpy(float)):
        bad.append("переоценка %s (конфигурация 3): пересчёт не совпал побитно с cfg3.parquet" % tab.index[a].date())
    return bad
