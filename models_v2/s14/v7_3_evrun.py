"""V7-3, G43: чтение рядов, запись CHECKPOINT `3.eval` (QLIKE, Δ с интервалами, пары, условия «go», веса — БЕЗ p-значений) ДО таблиц с p, затем таблицы `3_*.csv`, `3.rules` (6.2, статус «go»; p) и отчёт `LEAF3_REPORT.md`.
Запуск записи: `python3 -W ignore -m models_v2.s14.v7_3_evrun`. `check_eval` (из `run_v7 --stage 3 --check`) ничего не пишет: пересчитывает таблицы и отчёт из сохранённых прогнозов и сверяет с файлами и CHECKPOINT.
Конфигурации 1–7 — версия 2 (`1b/cfg1..3`, `2/cfg4..7`), 8 — `3/cfg8.parquet`; файлы сверяются по sha256 CHECKPOINT. Новых счётов нет."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_ctrl as C, v7_0_io as IO, v7_1b_cfgrun as CR, v7_1b_ev as EV, v7_1b_evrep as REP1, v7_1b_evrun as R1, v7_1m_data as DATA, v7_2_evrun as EV2, v7_3_ev as E3, v7_3_evrep as REP, v7_3_run as R3

STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-236, лист 3 (конфигурация 8; окна 2022+ — только оценка, выбора по ним нет; 9 не считается)"
NAMES = {"METRICS": "3_METRICS.csv", "TESTS": "3_TESTS.csv", "PAIRS": "3_PAIRS.csv", "GO": "3_GO.csv", "YEARS": "3_YEARS.csv", "STRESS": "3_STRESS.csv", "MCS": "3_MCS.csv", "WEIGHTS": "3_WEIGHTS.csv"}
NOTES = {"METRICS": "3 G43: QLIKE, RMSE, MAE, R², bias, 17 рядов, одинаковые даты", "TESTS": "3 G43: Δ конфигурации 8 в % уровня K1 (90 %, блок 20, 2000, seed 0), DM лаг 5, Холм N = 733 / 3166",
         "PAIRS": "3 G43: 8 против 6 (вклад якоря), правило 6.2, стресс (без 1 % крайних, без 2018)", "GO": "3 G43: «go» 6.1, шесть условий, конфигурация 8", "YEARS": "3: годовой вклад 8 в условия 4–5",
         "STRESS": "3: стресс — без 1 % крайних дат, без 2018", "MCS": "3: MCS 90 %", "WEIGHTS": "3 G43: средние π_0, π_1, π_2 и w по годам цели, до и после 2021"}


def load() -> dict:
    """-> dict(idx, rv, tp, series (17), df8, shas). Файлы — по sha256 CHECKPOINT; даты одинаковые; rv и H равны таблице SBER."""
    tab, w, cp = DATA.load_tab(), C.aligned(), CR.read_cp()
    paths = {**EV2.paths_v2(), "cfg8": R3.path()}
    raw, shas = {k: pd.read_parquet(p) for k, p in paths.items()}, {}
    for k, p in paths.items():
        shas[k] = IO.sha_file(p)
        if shas[k] != (cp["3"][k] if k == "cfg8" else cp["1b" if k in ("cfg1", "cfg2", "cfg3") else "2"][k])["sha256_file"]:
            raise SystemExit("%s: sha256 файла ≠ CHECKPOINT" % k)
    if not EV.same_dates({k: v.set_index("d")[["F"]] for k, v in raw.items()}):
        raise SystemExit("даты конфигураций 1–8 не совпадают")
    idx, t = pd.DatetimeIndex(raw["cfg1"]["d"]), tab.loc[pd.DatetimeIndex(raw["cfg1"]["d"])]
    for k, v in raw.items():
        if not (np.array_equal(v["rv"].to_numpy(float), t["rv_next"].to_numpy(float)) and np.array_equal(v["H"].to_numpy(float), t["H"].to_numpy(float))):
            raise SystemExit("%s: rv или H не равны таблице SBER" % k)
    ctl = EV2._controls(tab, w, idx)
    series = {**ctl["refs"], **{k: raw[k]["F"].to_numpy(float) for k in E3.CFGS}}
    bad = [k for k, v in series.items() if not (np.isfinite(v).all() and (v > 0).all())]
    if bad:
        raise SystemExit("нет значений на датах конфигураций: %s" % bad)
    return dict(idx=idx, rv=t["rv_next"].to_numpy(float), tp=t["t_plus"].to_numpy(), series=series, df8=raw["cfg8"], shas={**shas, **ctl["shas"]})


def compute(inp: dict) -> dict:
    return E3.main_tables(inp["rv"], inp["series"], inp["tp"], inp["df8"])


def numbers(t: dict, inp: dict) -> dict:
    """`3.eval`: QLIKE по окнам, Δ с интервалами, пары (Δ, интервал, годы), условия «go», веса и w. Колонок p здесь нет."""
    m, tt, pr = t["METRICS"], t["TESTS"], t["PAIRS"]
    ql = {s: {w: float(g.loc[g["window"] == w, "qlike"].iloc[0]) for w in g["window"]} for s, g in m.groupby("series")}
    dl = {"%s|%s" % (r.window, r.base): [float(r.delta_pct), float(r.lo90), float(r.hi90)] for r in tt.itertuples()}
    pair = {"%s|%s|%s|%s" % (r.hyp, r.x, r.y, r.window): [float(r.delta_pct), float(r.lo90), float(r.hi90), float(r.delta_q), int(r.years_neg), int(r.n_years), r.rule62] for r in pr.itertuples()}
    go = {str(int(r.cond)): [float(r.value), bool(r.passed)] for r in t["GO"].itertuples()}
    wt = {"%s|%s" % (r.quantity, r.period): float(r.mean) for r in t["WEIGHTS"].itertuples()}
    return dict(n_dates=len(inp["idx"]), first=str(inp["idx"][0].date()), last=str(inp["idx"][-1].date()), qlike=ql, delta_pct_cfg8=dl, pairs=pair, go_cfg8=go, weights=wt, input_sha256=inp["shas"], status=STATUS,
                p_values="не входят: записаны в 3.rules после этого ключа")


def rules(t: dict) -> dict:
    """6.2 (с p) по паре 8 против 6 и статус «go» 6.1 для 8: «подтверждено» — только при p Холма (N = 733) < 0,05 против K2 и K3 на 2018–2021."""
    g, tt = t["GO"], t["TESTS"][(t["TESTS"]["window"] == "2018_2021") & t["TESTS"]["base"].isin(["K2", "K3"])]
    is_go, conf = bool(g.loc[g["cond"] == 0, "passed"].iloc[0]), bool((tt["p_holm_N"] < 0.05).all())
    return dict(cfg8=dict(go=is_go, p_holm_N_vs_K2_K3=[float(x) for x in tt["p_holm_N"]], status=REP1.verdict(is_go, conf)),
                pairs_62={"%s|%s|%s|%s" % (r.hyp, r.x, r.y, r.window): [float(r.delta_q), float(r.p_one), int(r.years_neg), r.rule62, float(r.p_holm_N)] for r in t["PAIRS"].itertuples()})


def update_3(sub, val):
    R3.update_3(sub, val)


def write_eval() -> int:
    inp = load()
    t = compute(inp)
    update_3("eval", numbers(t, inp))                                  # ДО записи и вывода p-значений
    for k, name in NAMES.items():
        IO.write_csv(t[k], name, NOTES[k], status=STATUS)
    update_3("rules", rules(t))
    (IO.REP / "LEAF3_REPORT.md").write_text(REP.render(t, inp, CR.read_cp()["3"]), encoding="utf-8")
    print("оценка 8: %d дат; таблицы %s; отчёт LEAF3_REPORT.md; go = %s" % (len(inp["idx"]), ", ".join(NAMES.values()), CR.read_cp()["3"]["rules"]["cfg8"]["status"]))
    return 0


def check_eval(cp: dict) -> list:
    """-> замечания (пусто = ок). Читает файлы, ничего не пишет."""
    cp3 = cp.get("3", {})
    if not all(k in cp3 for k in ("eval", "rules")):
        return ["CHECKPOINT 3: нет eval/rules"]
    inp = load()
    t, bad = compute(inp), []
    if numbers(t, inp) != cp3["eval"]:
        bad.append("3.eval: пересчёт чисел ≠ CHECKPOINT")
    if rules(t) != cp3["rules"]:
        bad.append("3.rules: пересчёт ≠ CHECKPOINT")
    for k, name in NAMES.items():
        if not (IO.REP / name).exists() or not R1.same_frame(t[k], IO.read_csv(name)):
            bad.append("%s: нет или не совпал с пересчётом" % name)
    p = IO.REP / "LEAF3_REPORT.md"
    if not p.exists() or p.read_text(encoding="utf-8") != REP.render(t, inp, cp3):
        bad.append("LEAF3_REPORT.md: нет или не равен тексту из таблиц")
    return bad


if __name__ == "__main__":
    sys.exit(write_eval())
