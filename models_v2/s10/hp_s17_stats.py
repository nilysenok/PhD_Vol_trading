"""P3-17, G17 (Ж-220 п. 9, У-22): статистика РАЗДЕЛЬНО для двух совокупностей — «17» (M16 и SBER отдельно, 17 целиком; обучение на 17) и «E11» (перенос зафиксированной модели, обученной на 17; E11 ранее
анализировались, не полностью независимая выборка). Каждая совокупность считается на своих строках (общие даты, топ-5 дат, бутстрап дат, MCS — внутри совокупности), смешивания нет.
Формулы и таблицы — `hp_stats_cmp/tst/val` без копирования; здесь — подмножества строк, переименование области (all28 → «17»), Холм (N из `N_declared_before_stats_p17`), запись в `statistics_p17/`.
`stats` — совокупность 17 (нужен LOTO на 17), `e11` — перенос; общие таблицы собираются, когда есть обе (уровень 4 R14 — E11 и LOTO). Результаты на 28 не трогаются."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s5.d import stats_d as SD
from models_v2.s10 import hp_paths as P, hp_s17_sel as SEL, hp_stats_cmp as CM, hp_stats_sel as S, hp_stats_tst as T, hp_stats_val as V

POPS = {"17": ("M16", "SBER"), "E11": ("E11",)}
TABLES = ("COMPARISON", "TESTS", "MCS", "PER_TICKER", "HARJ_VALUE")


def declared_n() -> int:
    cp = json.loads((P.REP / "CHECKPOINT.json").read_text())
    if "N_declared_before_stats_p17" not in cp:
        raise SystemExit("N не объявлено в CHECKPOINT.json: сначала `s17 --part sel`")
    return int(cp["N_declared_before_stats_p17"])


def population(k: pd.DataFrame, pop: str) -> pd.DataFrame:
    """Строки совокупности: «17» — M16 и SBER; «E11» — только E11 (без смешивания)."""
    return k[k["group"].isin(POPS[pop])].reset_index(drop=True)


def relabel(df: pd.DataFrame, pop: str, col: str = "scope") -> pd.DataFrame:
    """«all28» (все бумаги совокупности) → «17»; в E11 строки all28 — дубль области E11, убираются; колонка `population`."""
    df = df.copy()
    if pop == "17":
        df[col] = df[col].astype(str).str.replace("all28", "17", regex=False)
    else:
        df = df[~df[col].astype(str).str.endswith("all28")]
    return df.assign(population=pop)


def tests_pop(kp: pd.DataFrame, c: pd.DataFrame, loto: pd.DataFrame | None, pop: str, n: int) -> pd.DataFrame:
    """TESTS совокупности: гибриды × {K1, K4, K5} (все срезы), K2/K3 на бумагах с ними, контроли K4–K7 к K1, LOTO к K1 и пулу (только «17»). main — M16 × (подтверждение, вся история) для «17»;
    для «E11» те же окна на E11 — kind `transfer` (отдельное семейство Холма с тем же N)."""
    wm, sc = CM.slice_masks(c), T.scope_masks(c)
    wins = {w: wm.get(w, wm["all"]) for w in T.WINS}
    rows = T.rows_for(c, CM.SIGS, [(h, b) for h in T.HYB for b in ("k1", "k4", "k5")], sc, wins, "aux")
    c2 = S.common(CM.with_k23(kp)[kp["ticker"].isin(CM.k23_tickers()).to_numpy()], CM.SIGS + ["k2", "k3"])
    rows += T.k23_rows(c2) if len(c2) else []
    rows += T.rows_for(c, CM.SIGS, [(s, "k1") for s in ("k4", "k5", "k6", "k7")], sc, wins, "control")
    df = pd.DataFrame(rows)
    scope, kind = ("M16", "main") if pop == "17" else ("E11", "transfer")
    w_main = set(T.MAIN[1]) | {CM.K23_PREF + w for w in T.MAIN[1]}
    df.loc[(df["kind"].isin(["aux", "main"])) & (df["scope"] == scope) & df["window"].isin(w_main), "kind"] = kind
    if loto is not None:
        df = pd.concat([df, pd.DataFrame(T.loto_rows(c, loto))], ignore_index=True)
    return holm(df, n)


def holm(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """p Холма с N объявленным (`holm_n`): отдельно семейство main («17») и transfer («E11»); N < числа p — ошибка. Остальные строки — NaN."""
    df["p_holm"] = np.nan
    for kind in ("main", "transfer"):
        sel = (df["kind"] == kind) & df["dm_p"].notna()
        if sel.any():
            df.loc[sel, "p_holm"] = SD.holm_n(df.loc[sel, "dm_p"].to_numpy(), n)
    return df


def compute_pop(k: pd.DataFrame, pop: str, loto: pd.DataFrame | None, n: int, log=print) -> dict:
    kp = population(k, pop)
    pt, cmp_, c = CM.tables(kp)
    tests = tests_pop(kp, c, loto, pop, n)
    mcs = T.run_mcs(kp, log, ("M16", "all28") if pop == "17" else ("E11",))
    out = {"pt": pt, "tests": tests, "cmp": cmp_, "mcs": mcs, "hv": V.harj_value(pt, tests), "c": c}
    if loto is not None:
        out["lt"] = V.loto_table(c, loto, tests)
    log("совокупность %s: бумаг %d, общих строк %d, TESTS %d, main %d, transfer %d" % (pop, c["ticker"].nunique(), len(c), len(tests), int((tests["kind"] == "main").sum()), int((tests["kind"] == "transfer").sum())))
    return out


def _path(pop: str, name: str):
    return P.OUT / ("s17_%s_%s.parquet" % (pop, name))


def save(pop: str, parts: dict) -> None:
    for name, df in parts.items():
        if name != "c":
            df.to_parquet(_path(pop, name))


def load_loto17() -> pd.DataFrame:
    """LOTO на 17 всех объявленных прогонов; неполный (не 17 бумаг у прогона) — ошибка."""
    df = pd.concat([pd.read_parquet(P.OUT / ("loto_p17_%s.parquet" % m)) for m in ("xgb", "lgbm") if (P.OUT / ("loto_p17_%s.parquet" % m)).exists()], ignore_index=True)
    decl = json.loads((SEL.D17 / "SELECTION.json").read_text())["loto"]
    bad = [c for c in decl if df[df["config"] == c]["ticker"].nunique() != 17]
    if bad:
        raise SystemExit("LOTO на 17 неполный (нужны 17 бумаг у прогона): %s" % bad)
    return df[df["config"].isin(decl)].reset_index(drop=True).rename(columns={"sigma2": "sigma2"})


def _write(name: str, df: pd.DataFrame, note: str) -> None:
    P.write_csv(df, "statistics_p17/%s.csv" % name, "%s; %s" % (note, SEL.STAT))


def assemble(n: int, log=print) -> None:
    """Общие таблицы `statistics_p17/*.csv` из сохранённых частей. Уровни R14 — по M16 («17»); уровень 4 — E11 и LOTO (если частей нет — соответствующие колонки пусты)."""
    have = [p for p in POPS if all(_path(p, nm).exists() for nm in ("pt", "tests", "cmp", "mcs", "hv"))]
    rd = lambda p, nm: pd.read_parquet(_path(p, nm))
    cat = lambda nm, col="scope": pd.concat([relabel(rd(p, nm), p, col) for p in have], ignore_index=True)
    pt, tests = pd.concat([rd(p, "pt") for p in have], ignore_index=True), pd.concat([rd(p, "tests") for p in have], ignore_index=True)
    k23 = "K2/K3 на %d бумагах: %s" % (len(CM.k23_tickers()), ",".join(CM.k23_tickers()))
    (SEL.D17).mkdir(parents=True, exist_ok=True)
    _write("COMPARISON", cat("cmp"), "QLIKE, RMSE, MAE, R² по бумаге/группам/срезам, раздельно «17» (M16, SBER, 17) и «E11» (перенос); %s" % k23)
    _write("PER_TICKER", pd.concat([rd(p, "pt").assign(population=p) for p in have], ignore_index=True), "метрики по бумаге и срезам; %s" % k23)
    _write("TESTS", cat("tests"), "Δ = QLIKE(гибрид) − QLIKE(база) по датам; N Холма = %d (объявлено до p; main — «17»: M16 × окна; transfer — «E11»); %s" % (n, k23))
    _write("MCS", cat("mcs"), "MCS 90 %, T_max, бутстрап дат (блок 20, 5000, seed 42), внутри совокупности")
    _write("HARJ_VALUE", cat("hv"), "добавленная ценность HAR-J (У-15): Δ к K4/K5 своего модуля; A1 — также к K1; раздельно «17» и «E11»")
    loto_cfgs = set()
    if "17" in have and _path("17", "lt").exists():
        lt = relabel(rd("17", "lt"), "17", "name")
        lt = lt[lt["n_tickers"].fillna(1) > 0] if "n_tickers" in lt else lt
        _write("LOTO", lt, "LOTO на 17 (У-22): обучение на 16, выбор по 16; по бумагам и группам; панельные DM — в TESTS (config__LOTO)")
        loto_cfgs = set(tests[tests["config"].str.endswith("__LOTO")]["config"].str.replace("__LOTO", "", regex=False))
    if "17" in have:
        _write("LEVELS", V.levels(pt, tests, loto_cfgs).assign(population="17; l4_e11 — перенос на E11"), "уровни R14; evidence: confirm (M16 × 2022–02.2026), hist (M16 × вся история); уровень 4 — E11 и LOTO; потолок, ACCEPTED не присваивается")
    log("таблицы записаны: %s (совокупности %s)" % (SEL.D17, have))


def run_part(part: str, log=print) -> None:
    """`stats` — «17» (после LOTO на 17), `e11` — перенос; затем сборка общих таблиц."""
    bad = SEL.check_inputs()
    if bad:
        raise SystemExit("стоп: входы не совпали: %s" % bad)
    n, k = declared_n(), SEL.load17()
    pop = "17" if part == "stats" else "E11"
    save(pop, compute_pop(k, pop, load_loto17() if pop == "17" else None, n, log))
    assemble(n, log)
