"""V7-3: запуск конфигурации 8 (стадия `run_v7 --stage 3`, G42). Пишет `data/regen/harj_hybrid_v7/3/cfg8.parquet` (d, t_plus, rv, H, ell, w, G, F, fitted, pi0–pi2, g1, g2 — среднее по seeds), CHECKPOINT ключ `3`, подраздел `cfg8`
(sha256 содержимого и файла), строку результата 8 и строку «9 — не считается: условие ложно (2.cond9)» в ALL_TRIALS.csv. Конфигурацию 9 бегун отказывает (`v7_1m_reg.gate`: в CHECKPOINT нет `cond_9 = true`); отказ ожидаемый — rc 0.
Здесь только QLIKE 2018–2021 без p; оценка против контролей и окна 2022+ — `v7_3_evrun`. `check_run()` ничего не пишет: sha256, F > 0, G = 0 ⇒ F = H, π в симплексе, ℓ = ln(M/H) даты d, даты как у конфигурации 1, одна переоценка побитно."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_io as IO, v7_1b_cfg as CFG1, v7_1b_cfg3 as C3, v7_1b_cfgrun as CR, v7_1m_fit as FT, v7_1m_reg as RG, v7_2_cfg as C2, v7_2_run as R2, v7_3_cfg as C8

STATUS_3 = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-236, лист 3 (конфигурация 8; QLIKE 2018–2021 без p; оценка — v7_3_evrun)"
CHECK_FROM = CR.CHECK_FROM


def path():
    return IO.OUT / "3" / "cfg8.parquet"


def update_3(sub: str, value: dict) -> None:
    cur = dict(CR.read_cp().get("3", {}))
    cur[sub] = value
    IO.merge_checkpoint("3", cur)


def ready(cp_path=None) -> bool:
    """В CHECKPOINT (по умолчанию — реестра) есть `3.cfg8`: конфигурация посчитана."""
    p = cp_path or RG.CHECKPOINT
    return p.exists() and "cfg8" in json.loads(p.read_text(encoding="utf-8")).get("3", {})


def refusal_ok(refused: list, cp_path=None) -> bool:
    """Ожидаемый отказ: отказана только 9, `2.cond9.cond9` ложно, 8 посчитана."""
    p = cp_path or RG.CHECKPOINT
    return bool(refused == ["9"] and ready(p) and json.loads(p.read_text(encoding="utf-8")).get("2", {}).get("cond9", {}).get("cond9") is False)


def record(df: pd.DataFrame, per: list, kind: str, seconds: float) -> dict:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    q, n = CFG1.qlike_choice(df)
    return dict(file=str(p.relative_to(IO.ROOT)), encoder=kind, window=CFG1.WINDOW_SIZE, sha256_content=CR.content_hash(df), sha256_file=IO.sha_file(p), n_rows=len(df), first=str(df["d"].iloc[0].date()),
                last=str(df["d"].iloc[-1].date()), qlike_2018_2021=q, n_dates_2018_2021=n, params=C8.PARAMS8, router="B", anchor="w·ln(M/H), M = sqrt(K2·K3)", grid=[list(g) for g in C8.GRID8],
                seeds=list(FT.SEEDS), fitted_share=float(df["fitted"].mean()), lam_selected=R2.lam_counts(per), seconds=round(seconds, 1))


def write_cfg8() -> int:
    import shutil
    from models_v2.s14 import v7_1m_data as DATA
    if shutil.disk_usage(IO.ROOT).free / 1e9 < 40:
        raise SystemExit("свободно < 40 ГБ: запись отменена")
    cp, tab = CR.read_cp(), DATA.load_tab()
    old = cp.get("3", {}).get("cfg8")
    if old and path().exists() and IO.sha_file(path()) == old["sha256_file"]:
        print("конфигурация 8: уже посчитана, пропуск")
        return 0
    kind, t0 = C3.kind3(cp), time.time()
    df, per = C8.run_prod(tab, kind)
    if not R2.same_dates(df):
        raise SystemExit("конфигурация 8: даты ≠ датам конфигурации 1: запись отменена")
    rec = record(df, per, kind, time.time() - t0)
    update_3("cfg8", rec)
    RG.add_result("8", dict(stage="3", qlike_2018_2021=rec["qlike_2018_2021"], sha256_content=rec["sha256_content"], n_rows=len(df)))
    RG.add_result("9", dict(stage="3", reason="не считается: условие 2.cond9 ложно", cond9=cp["2"]["cond9"]["cond9"]), suffix=RG.NOT_RUN_SUFFIX)
    print("конфигурация 8: %d строк, QLIKE 2018–2021 %.6f, %.0f с" % (len(df), rec["qlike_2018_2021"], rec["seconds"]))
    return 0


def check_cfg(cp3: dict, kind: str, tab, **kw) -> list:
    """kw (load, seeds) — только для тестов на синтетике."""
    rec, p = cp3.get("cfg8"), path()
    if rec is None or not p.exists():
        return ["конфигурация 8: нет в CHECKPOINT или нет файла"]
    df, bad = pd.read_parquet(p), []
    if IO.sha_file(p) != rec["sha256_file"] or CR.content_hash(df) != rec["sha256_content"]:
        bad.append("cfg8.parquet: sha256 ≠ CHECKPOINT")
    if list(df.columns) != C8.COLS:
        bad.append("cfg8: колонки ≠ объявленным")
    if not ((df["F"] > 0).all() and np.array_equal(df["F"][df["G"] == 0], df["H"][df["G"] == 0])):
        bad.append("cfg8: F ≤ 0 или G = 0, но F ≠ H")
    if not R2.same_dates(df):
        bad.append("cfg8: даты ≠ датам конфигурации 1")
    if not np.allclose(df.loc[df["fitted"] > 0, list(C2.PI_COLS)].sum(axis=1), 1.0, rtol=0, atol=1e-12):
        bad.append("cfg8: π не в симплексе")
    if not np.array_equal(df["ell"].to_numpy(float), C8.ell_of(tab).take(tab.index.get_indexer(pd.DatetimeIndex(df["d"])))):
        bad.append("cfg8: ℓ ≠ ln(M/H) даты d (сохранённые прогнозы K2, K3 той же даты)")
    return bad + _recount(df, kind, tab, **kw)


def _recount(df, kind, tab, **kw) -> list:
    from models_v2.s13 import v6_1_wf as W6
    from models_v2.s14 import v7_1m_wf as WF
    a = next(i for i in W6.refit_positions(tab.index, WF.FIRST_REFIT) if tab.index[i] >= CHECK_FROM)
    got = C8.refit_block(tab, kind, a, **kw)
    ref = df.set_index("d").reindex(tab.index[a:a + len(got["G"])])
    cols = {"G": "G", "w": "w", "pi": list(C2.PI_COLS), "gk": list(C2.GK_COLS)}
    return ["cfg8: переоценка %s (%s) не совпала побитно с parquet" % (tab.index[a].date(), k) for k, c in cols.items() if not np.array_equal(got[k], ref[c].to_numpy(float))]


def check(cp: dict) -> list:
    """-> замечания (пусто = ок). Читает файлы, ничего не пишет."""
    from models_v2.s14 import v7_1m_data as DATA
    bad = C8.declared_problems() + check_cfg(cp.get("3", {}), C3.kind3(cp), DATA.load_tab())
    rows = {r[0]: r for r in RG.read()[1:]}
    if "Ж-233:8:результат" not in rows or "Ж-233:9" + RG.NOT_RUN_SUFFIX not in rows:
        bad.append("ALL_TRIALS.csv: нет строки результата 8 или строки «9 — не считается»")
    return bad


def check_run(ok: list, refused: list) -> int:
    before = IO.outputs_digest()
    cp = CR.read_cp()
    bad = [] if refusal_ok(refused) else ["бегун: отказ должен быть ровно у 9 при ложном 2.cond9, получено %s" % refused]
    bad += check(cp) if ok == ["8"] else ["бегун допустил %s, нужна только 8" % ok]
    if "eval" in cp.get("3", {}):                                      # оценка записана: таблицы, отчёт и ключи 3.eval / 3.rules равны пересчёту
        from models_v2.s14 import v7_3_evrun as EVR
        bad += EVR.check_eval(cp)
    if IO.outputs_digest() != before:
        bad.append("--check изменил выходы (хэш до ≠ после)")
    print("\n".join(bad) if bad else "OK: G42 конфигурация 8 по sha256 CHECKPOINT, F > 0, G = 0 ⇒ F = H, π в симплексе, ℓ = ln(M/H) даты d, переоценка побитно, 9 отказана (2.cond9 ложно); %sвыходы не изменены"
          % ("G43 таблицы оценки, отчёт, 3.eval/rules равны пересчёту; " if "eval" in cp.get("3", {}) else ""))
    return 1 if bad else 0


def connect(runners: dict) -> None:
    """Подключает счёт 8 к бегуну; уже стоящий обработчик не заменяется. Для 9 обработчика нет (отказ)."""
    runners.setdefault("8", write_cfg8)
