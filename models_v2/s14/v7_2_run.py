"""V7-2a: запуск конфигураций 4–7 (стадия `run_v7 --stage 2`, G27). Пишет `data/regen/harj_hybrid_v7/2/cfg4..7.parquet` (d, t_plus, rv, H, G, F, fitted; для MoE ещё pi0, pi1, pi2, g1, g2 — среднее по seeds),
CHECKPOINT ключ `2`, подраздел `cfgN` (sha256 содержимого и файла), строки результата в ALL_TRIALS.csv и `2a_QLIKE.csv`. Только QLIKE 2018–2021: оценка против контролей, p-значения и окна 2022+ — часть 2b.
Энкодер — только из записанного `1b.choice3` (TCN). `check_run()` ничего не пишет: sha256, F > 0, G = 0 ⇒ F = H, веса π в симплексе, те же даты, что у конфигурации 1, одна переоценка каждой конфигурации побитно."""
from __future__ import annotations

import collections
import time

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_io as IO, v7_1b_cfg as CFG1, v7_1b_cfg3 as C3, v7_1b_cfgrun as CR, v7_1m_fit as FT, v7_1m_forms as FM, v7_1m_reg as RG, v7_2_cfg as C2

STATUS_2A = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-233, лист 2a (конфигурации 4–7; QLIKE 2018–2021 без p; окна 2022+ и сравнения с контролями не оценивались)"
PARAMS = {"4": 76, "5": 51 + 10, "6": 67 + 10, "7": 31 + 10}        # плоский 76; маршрутизатор A/B/C + эксперты 10 (план п. 4; сверка с FM.counts() — тест)
CHECK_FROM = CR.CHECK_FROM


def path(cfg: str):
    return IO.OUT / "2" / ("cfg%s.parquet" % cfg)


def update_2(sub: str, value: dict) -> None:
    cur = dict(CR.read_cp().get("2", {}))
    cur[sub] = value
    IO.merge_checkpoint("2", cur)


def lam_counts(per: list) -> dict:
    c = collections.Counter("%g/%g" % tuple(f["lam"]) for p in per for f in p["fits"] if f["fitted"])
    return dict(sorted(c.items()))


def record(cfg: str, df: pd.DataFrame, per: list, kind: str, seconds: float) -> dict:
    p = path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    q, n = CFG1.qlike_choice(df)
    return dict(file=str(p.relative_to(IO.ROOT)), encoder=kind, window=CFG1.WINDOW_SIZE, sha256_content=CR.content_hash(df), sha256_file=IO.sha_file(p), n_rows=len(df), first=str(df["d"].iloc[0].date()),
                last=str(df["d"].iloc[-1].date()), qlike_2018_2021=q, n_dates_2018_2021=n, params=PARAMS[cfg], router=C2.SPECS[cfg].get("router", "—"), grid=[list(g) for g in C2.GRIDS[cfg]],
                lam_lb="не применяется (нет маршрутизатора)" if cfg == "4" else "в решётке", seeds=list(FT.SEEDS), fitted_share=float(df["fitted"].mean()), lam_selected=lam_counts(per), seconds=round(seconds, 1))


def same_dates(df: pd.DataFrame) -> bool:
    ref = pd.read_parquet(CR.path("tcn"))
    return bool(np.array_equal(df["d"].to_numpy("datetime64[ns]"), ref["d"].to_numpy("datetime64[ns]")))


def write_cfgs(ids=C2.IDS) -> int:
    import shutil
    from models_v2.s14 import v7_1m_data as DATA
    if shutil.disk_usage(IO.ROOT).free / 1e9 < 40:
        raise SystemExit("свободно < 40 ГБ: запись отменена")
    kind, tab, done = C3.kind3(CR.read_cp()), DATA.load_tab(), {}
    for cfg in ids:
        old = CR.read_cp().get("2", {}).get("cfg" + cfg)
        if old and path(cfg).exists() and IO.sha_file(path(cfg)) == old["sha256_file"]:
            print("конфигурация %s: уже посчитана, пропуск" % cfg)
            continue
        t0 = time.time()
        df, per = C2.run_prod(cfg, tab, kind)
        if not same_dates(df):
            raise SystemExit("конфигурация %s: даты ≠ датам конфигурации 1: запись отменена" % cfg)
        rec = record(cfg, df, per, kind, time.time() - t0)
        update_2("cfg" + cfg, rec)
        RG.add_result(cfg, dict(stage="2a", qlike_2018_2021=rec["qlike_2018_2021"], sha256_content=rec["sha256_content"], n_rows=len(df)))
        done[cfg] = rec
        print("конфигурация %s: %d строк, QLIKE 2018–2021 %.6f, %.0f с" % (cfg, len(df), rec["qlike_2018_2021"], rec["seconds"]))
    cp2 = CR.read_cp().get("2", {})                                    # таблица — по всем записанным конфигурациям (бегун вызывает по одной)
    if done or not (IO.REP / "2a_QLIKE.csv").exists():
        tbl = pd.DataFrame([dict(config=c, router=r["router"], params=r["params"], qlike_2018_2021=r["qlike_2018_2021"], n_dates=r["n_dates_2018_2021"], fitted_share=r["fitted_share"], seconds=r["seconds"])
                            for c, r in ((c, cp2[k]) for c in C2.IDS for k in ["cfg" + c] if k in cp2)])
        IO.write_csv(tbl, "2a_QLIKE.csv", "2a: средняя QLIKE на датах целей 2018–2021, конфигурации 4–7 (без p)", status=STATUS_2A)
    return 0


def check_cfg(cfg: str, cp2: dict, kind: str, tab, **kw) -> list:
    """kw (load, seeds) — только для тестов на синтетике."""
    rec, p = cp2.get("cfg" + cfg), path(cfg)
    if rec is None or not p.exists():
        return ["конфигурация %s: нет в CHECKPOINT или нет файла" % cfg]
    df, bad = pd.read_parquet(p), []
    if IO.sha_file(p) != rec["sha256_file"] or CR.content_hash(df) != rec["sha256_content"]:
        bad.append("cfg%s.parquet: sha256 ≠ CHECKPOINT" % cfg)
    if not ((df["F"] > 0).all() and np.array_equal(df["F"][df["G"] == 0], df["H"][df["G"] == 0])):
        bad.append("cfg%s: F ≤ 0 или G = 0, но F ≠ H" % cfg)
    if not same_dates(df):
        bad.append("cfg%s: даты ≠ датам конфигурации 1" % cfg)
    if C2.SPECS[cfg]["kind"] == "moe" and not np.allclose(df.loc[df["fitted"] > 0, list(C2.PI_COLS)].sum(axis=1), 1.0, rtol=0, atol=1e-12):
        bad.append("cfg%s: π не в симплексе" % cfg)
    from models_v2.s13 import v6_1_wf as W6
    from models_v2.s14 import v7_1m_wf as WF
    a = next(i for i in W6.refit_positions(tab.index, WF.FIRST_REFIT) if tab.index[i] >= CHECK_FROM)
    got = C2.refit_block(cfg, tab, kind, a, **kw)
    ref = df.set_index("d").reindex(tab.index[a:a + len(got["G"])])
    cols = {"G": "G", "pi": list(C2.PI_COLS), "gk": list(C2.GK_COLS)}
    for k, c in cols.items():
        if k in got and not np.array_equal(got[k], ref[c].to_numpy(float)):
            bad.append("cfg%s: переоценка %s (%s) не совпала побитно с parquet" % (cfg, tab.index[a].date(), k))
    return bad


def check(cp: dict, ids=C2.IDS) -> list:
    """-> замечания (пусто = ок). Читает файлы, ничего не пишет."""
    bad = C2.declared_problems()
    cp2 = cp.get("2", {})
    from models_v2.s14 import v7_1m_data as DATA
    kind, tab = C3.kind3(cp), DATA.load_tab()
    for cfg in ids:
        bad += check_cfg(cfg, cp2, kind, tab)
    return bad


def check_run(ids) -> int:
    before = IO.outputs_digest()
    cp = CR.read_cp()
    bad = check(cp, tuple(ids))
    if tuple(ids) == C2.IDS:                                            # 2b: таблицы оценки 1–7 (G28–G30) — пересчёт и сверка, без записи
        from models_v2.s14 import v7_2_evrun as EVR
        bad += EVR.check_eval(cp)
    if IO.outputs_digest() != before:
        bad.append("--check изменил выходы (хэш до ≠ после)")
    print("\n".join(bad) if bad else "OK: G27 конфигурации %s по sha256 CHECKPOINT, F > 0, G = 0 ⇒ F = H, π в симплексе, даты = датам конфигурации 1, переоценка каждой побитно; %sвыходы не изменены"
          % (", ".join(ids), "G28–G30 таблицы оценки, отчёт, 2.eval/rules/cond9 равны пересчёту; " if tuple(ids) == C2.IDS else ""))
    return 1 if bad else 0


def connect(runners: dict) -> None:
    """Подключает счёт 4–7 к бегуну стадий; уже стоящий обработчик не заменяется."""
    for cfg in C2.IDS:
        runners.setdefault(cfg, lambda c=cfg: write_cfgs((c,)))
