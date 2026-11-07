"""P3: стадии `stats` и `loto` (G11, G12). `stats --part sel` — выбор и объявление N (ПЕРВЫМ, без p); `loto --module xgb|lgbm` — счёт LOTO (фон); `stats --part tab` — таблицы и статистика (после sel и ПОСЛЕ LOTO: уровень 4);
`stats --check` / `loto --check` — проверка выхода (код возврата 0 — гейт закрыт)."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from models_v2.s10 import hp_hyb_out as O, hp_paths as P, hp_stats_cmp as CM, hp_stats_sel as S, hp_stats_tst as T, hp_stats_val as V


def _log(m: str) -> None:
    print(time.strftime("%H:%M:%S"), m, flush=True)


def loto_names(module: str) -> list:
    """Конфигурации LOTO модуля из объявленного списка (SELECTION.json; объявлен до p)."""
    sel = json.loads((S.STATS_DIR / "SELECTION.json").read_text())
    return [c.split("_", 1)[1] for c in sel["loto"] if c.startswith(module + "_")]


def declared_loto() -> list:
    return json.loads((S.STATS_DIR / "SELECTION.json").read_text())["loto"]


def load_loto(write: bool = False) -> pd.DataFrame:
    """Объединённый LOTO (`loto.parquet`): все объявленные прогоны, по 28 бумаг; неполный — ошибка."""
    parts = [pd.read_parquet(P.OUT / ("loto_%s.parquet" % m)) for m in ("xgb", "lgbm")]
    df = pd.concat(parts, ignore_index=True)
    bad = [c for c in declared_loto() if df[df["config"] == c]["ticker"].nunique() != 28]
    if bad:
        raise SystemExit("LOTO неполный (нужны 28 бумаг): %s" % bad)
    if write:
        df.to_parquet(P.OUT / "loto.parquet")
    return df


def tab() -> None:
    n = T.declared_n()
    k, loto = S.load(), load_loto(True)
    pt = CM.run(k, _log)
    tests, mcs = T.run_tests(k, loto, n, _log), T.run_mcs(k, _log)
    T.write(tests, mcs, n)
    c = S.common(k, CM.SIGS)
    V.write(V.harj_value(pt, tests), V.levels(pt, tests, set(loto["config"].unique())), V.loto_table(c, loto, tests))
    _log("таблицы записаны: %s" % S.STATS_DIR)


def check_stats() -> list:
    bad, cp = [], json.loads((P.REP / "CHECKPOINT.json").read_text())
    for f in ("SELECTION", "COMPARISON", "TESTS", "MCS", "PER_TICKER", "HARJ_VALUE", "LEVELS", "LOTO"):
        bad += [] if (S.STATS_DIR / (f + ".csv")).exists() else ["нет statistics/%s.csv" % f]
    if bad:
        return bad
    n_loto = len(declared_loto())
    bad += [] if cp.get("N_declared_before_stats") == 585 + 20 + n_loto else ["N_declared %s != 585 + 20 + %d" % (cp.get("N_declared_before_stats"), n_loto)]
    t = pd.read_csv(S.STATS_DIR / "TESTS.csv", comment="#")
    m = t[t["kind"] == "main"]
    bad += [] if len(m) > 0 and (m["p_holm"] >= m["dm_p"] - 1e-12).all() and m["p_holm"].notna().all() else ["Холм: p_holm не заполнен или меньше p"]
    sel = pd.read_csv(S.STATS_DIR / "SELECTION.csv", comment="#")
    bad += [] if len(sel) == 25 and sel["crit_panel"].notna().all() else ["SELECTION: ожидалось 25 рядов с crit_panel"]
    lv = pd.read_csv(S.STATS_DIR / "LEVELS.csv", comment="#")
    bad += [] if len(lv) == 40 else ["LEVELS: строк %d != 40" % len(lv)]
    mc = pd.read_csv(S.STATS_DIR / "MCS.csv", comment="#")
    bad += [] if mc["in_mcs90"].any() else ["MCS: пустое множество"]
    print("N=%s; TESTS %d строк (основное семейство %d p); LEVELS %d (макс. уровень %d); MCS %d; LOTO-прогонов %d" % (cp.get("N_declared_before_stats"), len(t), len(m), len(lv), lv["level"].max(), len(mc), n_loto))
    return bad


def check_loto() -> list:
    bad = []
    try:
        df = load_loto()
    except (SystemExit, FileNotFoundError) as e:
        return ["LOTO: %s" % e]
    ok = np.isfinite(df["sigma2"]) & (df["sigma2"] > 0)
    bad += [] if ok.all() else ["LOTO: σ̂² не конечен или ≤ 0"]
    reg = [r for r in O.REG.read_text(encoding="utf-8").splitlines() if r.startswith(("Ж-217-LOTO:"))]
    bad += [] if len(reg) == len(declared_loto()) else ["реестр: LOTO-строк %d != %d" % (len(reg), len(declared_loto()))]
    bad += [] if (P.OUT / "loto.parquet").exists() and (S.STATS_DIR / "LOTO.csv").exists() else ["нет loto.parquet или statistics/LOTO.csv (стадия stats --part tab)"]
    print("LOTO: прогонов %d, строк %d, бумаг по прогону %s, строк реестра %d" % (len(declared_loto()), len(df), df.groupby("config")["ticker"].nunique().to_dict(), len(reg)))
    return bad


def stage(name: str, check: bool, part: str, module: str | None = None) -> int:
    bad = O.check_hashes() + ["%s: SHA != %s" % (f, h) for f, h in S.HASHES.items() if P.sha16(P.OUT / f) != h]
    if bad:
        print("стоп: входы не совпали:", bad)
        return 1
    if check:
        bad = check_stats() if name == "stats" else check_loto()
        print("\n".join(bad))
        print("ИТОГ:", "OK" if not bad else "FAIL")
        return 1 if bad else 0
    if name == "stats" and part == "sel":
        S.run(_log)
    elif name == "stats" and part == "tab":
        tab()
    elif name == "loto" and module:
        from models_v2.s10 import hp_loto
        hp_loto.run_module(module, loto_names(module), 4, _log)
    else:
        print("часть %s %s не задана" % (name, part))
        return 1
    return 0
