"""P17, G16 (Ж-220, У-19…У-23): стадия `p17` — панель с поперечными признаками по 17, K4/K5 и 20 гибридов с обучением на 17 (прогноз — 28 бумаг).
Запуск: `run_hp --stage p17 --part panel|k45|hyb|summary [--module xgb|lgbm] [--check]`; `--check` без `--part` — полная проверка выхода. Порядок: panel, k45 (оба модуля), hyb (оба модуля), summary.
Не пересчитываются: дневные M10-величины, K1, K6, K7, K2/K3 (У-23); результаты на 28 не перезаписываются (SHA сверяется)."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from models_v2.s10 import hp_ctrl as CT, hp_engine as E, hp_hyb_out as O, hp_hyb_run as HR, hp_p17 as X, hp_p17_hyb as H17, hp_p17_out as OUT, hp_paths as P

KEY = ["ticker", "date"]


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def part_panel() -> None:
    daily = {t: pd.read_parquet(P.OUT / ("daily_%s.parquet" % t)) for t in P.TICKERS}
    X.build_panel(daily, pd.read_parquet(P.OUT / "daily_IMOEX.parquet")).to_parquet(X.ENGINE_P17)
    OUT.set_checkpoint(panel_sha=P.sha16(X.ENGINE_P17))


def k45_compute(module: str, nthread: int, log=_log) -> None:
    """K4 (xgb) / K5 (lgbm) без HAR-J: обучение на 17, λ ≡ 1, выбор (глубина, c) по 17; прогноз — 28."""
    name = {"xgb": "k4", "lgbm": "k5"}[module]
    panel = pd.read_parquet(X.ENGINE_P17).reset_index(drop=True)
    pn, feats, base = CT.k45_inputs(panel)
    t0 = time.time()
    runs = X.cached_runs(module, "k45", pn, feats, base, P.sha16(X.ENGINE_P17), "qlike", None, E.DEPTHS, nthread, log)
    fc, ch = X.finalize(pn, base, runs, "qlike", lams=(1.0,))
    f = fc[fc["g"].notna()].rename(columns={"sigma2": "sigma2_" + name, "depth": name + "_depth", "c": name + "_c", "lam": name + "_lam", "n_train": name + "_ntrain"})
    f[KEY + ["sigma2_" + name, name + "_depth", name + "_c", name + "_lam", name + "_ntrain"]].to_parquet(P.OUT / ("k45_p17_%s.parquet" % module))
    O.merge_csv(ch, "P17_K45_CHOICE.csv", "P17: выбор (глубина, c) K4/K5 по годам, по 17 бумагам, λ ≡ 1 (У-20)", module)
    O.merge_csv(pd.DataFrame([dict(stage=name, wall_sec=round(time.time() - t0, 1))]), "P17_K45_TIME.csv", "P17: время счёта K4/K5, с", module)
    assemble_k45(panel)


def assemble_k45(panel: pd.DataFrame | None = None) -> None:
    """`k45_p17.parquet` = строки панели × K4, K5 (когда готовы оба модуля)."""
    parts = [P.OUT / ("k45_p17_%s.parquet" % m) for m in ("xgb", "lgbm")]
    if not all(p.exists() for p in parts):
        return
    panel = pd.read_parquet(X.ENGINE_P17) if panel is None else panel
    out = panel[KEY + ["t_plus", "group", "gap", "rv_tplus", "sigma2_k1"]].copy()
    for p in parts:
        out = out.merge(pd.read_parquet(p), on=KEY, how="left")
    out.sort_values(KEY).reset_index(drop=True).to_parquet(P.OUT / "k45_p17.parquet")


def hyb_compute(module: str, nthread: int, log=_log) -> None:
    r = H17.Run17(module, nthread, log)
    for n in HR.ORDER:
        r.one(n)
    ch = pd.concat([c.assign(config=n) for n, c in r.choice.items()], ignore_index=True)
    O.merge_csv(ch, "P17_CHOICE.csv", "P17: выбор параметров гибридов по годам, по 17 бумагам (У-20); пусто = параметр не применяется", module)
    O.merge_csv(pd.DataFrame(r.time.values()), "P17_TIME.csv", "P17: время счёта конфигураций, с", module)
    OUT.set_checkpoint(**{"hyb_" + module: P.sha16(P.OUT / ("hyb_%s_p17.parquet" % module))})
    OUT.write_status()


def part_summary() -> None:
    P.write_csv(OUT.summary(OUT.load_all()), "P17_SUMMARY.csv", "P17: QLIKE (среднее по бумагам группы) K1, K4, K5 и 20 гибридов с обучением на 17; общие строки; описательно, без выбора (G16)")


def check_panel() -> list:
    new, old = pd.read_parquet(X.ENGINE_P17), pd.read_parquet(P.OUT / "engine_panel.parquet")
    d = X.panel_diff(new, old)
    print("engine_panel_p17: строк %d (engine_panel %d, общих %d), бумаг %d; вне поперечных колонок расхождений: %s; поперечных колонок %d, из них без изменений %s" % (
        d["n_new"], d["n_old"], d["n_common"], new["ticker"].nunique(), d["differs_noncross"] or "нет", d["cross_n"], d["cross_unchanged"] or "нет"))
    bad = [] if d["n_common"] == d["n_old"] == d["n_new"] and new["ticker"].nunique() == 28 else ["состав строк не совпал с engine_panel"]
    bad += ["колонки вне поперечных расходятся: %s" % d["differs_noncross"]] if d["differs_noncross"] else []
    bad += ["поперечные колонки не изменились: %s" % d["cross_unchanged"]] if d["cross_unchanged"] else []
    u17 = {t for t in P.TICKERS if t in X.TRAIN}
    same_cal = set(new[new["ticker"].isin(u17)]["date"]) == set(new["date"])
    print("календарь месяцев: даты 17 = даты 28: %s" % same_cal)
    return bad + ([] if same_cal else ["даты 17 ≠ даты 28: месячные старты переобучения различаются"])


def check_k45() -> list:
    k, old = pd.read_parquet(P.OUT / "k45_p17.parquet"), pd.read_parquet(P.OUT / "k4567.parquet")
    bad = ["дубли (бумага, дата)"] if k.duplicated(KEY).any() else []
    for s in ("sigma2_k4", "sigma2_k5"):
        v = k[s].dropna()
        print("%s: значений %d, бумаг %d" % (s, len(v), k.loc[v.index, "ticker"].nunique()))
        bad += [] if len(v) and (np.isfinite(v) & (v > 0)).all() and k.loc[v.index, "ticker"].nunique() == 28 else ["%s: нет значений / не конечны / не по 28 бумагам" % s]
    m = k.merge(old[KEY + ["k4_ntrain", "k5_ntrain"]], on=KEY, suffixes=("", "_28"))
    for n in ("k4", "k5"):
        ok = m[n + "_ntrain"].notna() & m[n + "_ntrain_28"].notna()
        a, b = m.loc[ok, n + "_ntrain"], m.loc[ok, n + "_ntrain_28"]
        less = bool((a <= b).all() and (a < b).any())
        print("%s: обучающих пар на 17 <= на 28 во всех %d строках, строго меньше в %d: %s" % (n, int(ok.sum()), int((a < b).sum()), less))
        bad += [] if less else ["%s: в обучении на 17 пар больше, чем на 28, или не меньше нигде" % n]
    return bad


def check_hyb(module: str) -> list:
    k = pd.read_parquet(P.OUT / ("hyb_%s_p17.parquet" % module))
    bad = ["дубли (бумага, дата)"] if k.duplicated(KEY).any() else []
    for n in OUT.ORDER:
        v = k.get("sigma2_" + n)
        ok = v is not None and v.notna().sum() > 0 and bool((v.dropna() > 0).all() and np.isfinite(v.dropna()).all()) and k.loc[v.notna(), "ticker"].nunique() == 28
        print("%s %s: значений %d" % (module, n, 0 if v is None else v.notna().sum()))
        bad += [] if ok else ["%s %s: нет значений / не конечны / ≤ 0 / не по 28 бумагам" % (module, n)]
    z = k[k["a1_lam"] == 0]
    eq = bool((z["sigma2_A1"] == z["sigma2_k1"]).all())
    print("%s: A1 при λ = 0: строк %d, побитно K1: %s" % (module, len(z), eq))
    return bad + ([] if eq else ["%s: A1 при λ = 0 ≠ K1 побитно" % module])


def check_registry() -> list:
    rows = [r for r in O.REG.read_text(encoding="utf-8").splitlines() if r.startswith("Ж-220-p17:")]
    used = json.loads(O.CP.read_text())["used"]
    st = pd.read_csv(P.REP / "ALL_TRIALS_STATUS.csv")
    n17 = int(st["key"].str.startswith("Ж-220-p17:").sum())
    print("строк реестра Ж-220-p17: %d, used %d, строк статуса %d (из них p17 %d)" % (len(rows), used, len(st), n17))
    return [] if len(rows) == 20 and used == 20 and n17 == 20 and len(st) == 43 else ["реестр/used/статус: %d строк p17, used %d, статус %d/%d" % (len(rows), used, len(st), n17)]


def check_all() -> list:
    bad = OUT.check_28_unchanged() + check_panel() + check_k45() + check_hyb("xgb") + check_hyb("lgbm") + check_registry()
    s = pd.read_csv(P.REP / "P17_SUMMARY.csv", comment="#")
    print("P17_SUMMARY.csv: строк %d, групп %s" % (len(s), sorted(s["group"].unique())))
    return bad + ([] if set(s["group"]) == {"M16", "SBER", "E11", "17", "28"} else ["P17_SUMMARY: не все группы"])


def stage(part: str, module: str | None, check: bool, nthread: int = 5) -> int:
    if not check:
        if part in ("k45", "hyb") and module is None:
            raise SystemExit("--part %s требует --module xgb|lgbm" % part)
        todo = {"panel": part_panel, "summary": part_summary, "k45": lambda: k45_compute(module, nthread), "hyb": lambda: hyb_compute(module, nthread)}
        if part not in todo:
            raise SystemExit("--part: panel|k45|hyb|summary")
        todo[part]()
        return 0
    bad = check_all() if part == "all" else {"panel": check_panel, "k45": check_k45, "hyb": lambda: check_hyb(module) if module else check_hyb("xgb") + check_hyb("lgbm")}[part]()
    bad = OUT.check_28_unchanged() + bad if part != "all" else bad
    print("\n".join(bad))
    print("ИТОГ:", "OK" if not bad else "FAIL")
    return 1 if bad else 0
