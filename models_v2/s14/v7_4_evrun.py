"""V7-4a, G46: чтение рядов, запись CHECKPOINT `4.eval` (QLIKE, Δ с интервалами, пары, «go», веса — БЕЗ p) ДО таблиц с p, затем `4_*.csv`, `4.rules` (6.2, «go»; p), `4.cond11` (условие конфигурации 11), `4_DRIFT.csv` + `4.drift`
(диагностика дрейфа энкодера, `v7_4_dx`) и отчёт `LEAF4_REPORT.md`. Запуск: `python3 -W ignore -m models_v2.s14.v7_4_evrun`. `check_eval` (из `run_v7 --stage 4 --check`) ничего не пишет: пересчитывает и сверяет с файлами и CHECKPOINT.
Конфигурации 1–7 — версия 2, 8 — `3/cfg8`, 10 — `4/cfg10`; файлы сверяются по sha256 CHECKPOINT. Если `4.cond11` ложно — в реестр пишется «11 — не считается: условие ложно (4.cond11)»; конфигурация 11 здесь не считается."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_ctrl as C, v7_0_io as IO, v7_1b_cfgrun as CR, v7_1b_ev as EV, v7_1b_evrun as R1, v7_1m_data as DATA, v7_1m_reg as RG, v7_1b_sta as STA, v7_2_evrun as EV2, v7_3_run as R3, v7_4_dx as DX, v7_4_ev as E4, v7_4_evrep as REP, v7_4_run as R4

STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-235/Ж-236, лист 4a (конфигурация 10; окна 2022+ — только оценка, выбора по ним нет; 11 не считается)"
NAMES = {"METRICS": "4_METRICS.csv", "TESTS": "4_TESTS.csv", "PAIRS": "4_PAIRS.csv", "GO": "4_GO.csv", "YEARS": "4_YEARS.csv", "STRESS": "4_STRESS.csv", "MCS": "4_MCS.csv", "WEIGHTS": "4_WEIGHTS.csv"}
NOTES = {"METRICS": "4 G46: QLIKE, RMSE, MAE, R², bias, 18 рядов, одинаковые даты", "TESTS": "4 G46: Δ конфигурации 10 в % уровня K1 (90 %, блок 20, 2000, seed 0), DM лаг 5, Холм N = 733 / 3166",
         "PAIRS": "4 G46: 10 против 6 (правило 6.2; условие 11) и 10 против 8 (справочно), стресс (без 1 % крайних, без 2018)", "GO": "4 G46: «go» 6.1, шесть условий, конфигурация 10", "YEARS": "4: годовой вклад 10 в условия 4–5",
         "STRESS": "4: стресс — без 1 % крайних дат, без 2018", "MCS": "4: MCS 90 %", "WEIGHTS": "4 G46: средние π_0, π_1, π_2 по годам цели, до и после 2021 (строки Stage B)"}
DRIFT = "4_DRIFT.csv"


def _cp_of(cp: dict, k: str) -> dict:
    return cp["3"] if k == "cfg8" else cp["4"] if k == "cfg10" else cp["1b"] if k in ("cfg1", "cfg2", "cfg3") else cp["2"]


def load() -> dict:
    """-> dict(idx, rv, tp, series (18), df10, shas). Файлы — по sha256 CHECKPOINT; даты одинаковые; rv и H равны таблице SBER."""
    tab, w, cp = DATA.load_tab(), C.aligned(), CR.read_cp()
    paths = {**EV2.paths_v2(), "cfg8": R3.path(), "cfg10": R4.path()}
    raw, shas = {k: pd.read_parquet(p) for k, p in paths.items()}, {}
    for k, p in paths.items():
        shas[k] = IO.sha_file(p)
        if shas[k] != _cp_of(cp, k)[k]["sha256_file"]:
            raise SystemExit("%s: sha256 файла ≠ CHECKPOINT" % k)
    if not EV.same_dates({k: v.set_index("d")[["F"]] for k, v in raw.items()}):
        raise SystemExit("даты конфигураций 1–8 и 10 не совпадают")
    idx, t = pd.DatetimeIndex(raw["cfg1"]["d"]), tab.loc[pd.DatetimeIndex(raw["cfg1"]["d"])]
    for k, v in raw.items():
        if not (np.array_equal(v["rv"].to_numpy(float), t["rv_next"].to_numpy(float)) and np.array_equal(v["H"].to_numpy(float), t["H"].to_numpy(float))):
            raise SystemExit("%s: rv или H не равны таблице SBER" % k)
    ctl = EV2._controls(tab, w, idx)
    series = {**ctl["refs"], **{k: raw[k]["F"].to_numpy(float) for k in E4.CFGS}}
    bad = [k for k, v in series.items() if not (np.isfinite(v).all() and (v > 0).all())]
    if bad:
        raise SystemExit("нет значений на датах конфигураций: %s" % bad)
    return dict(idx=idx, rv=t["rv_next"].to_numpy(float), tp=t["t_plus"].to_numpy(), series=series, df10=raw["cfg10"], shas={**shas, **ctl["shas"]})


def compute(inp: dict) -> dict:
    return E4.main_tables(inp["rv"], inp["series"], inp["tp"], inp["df10"])


def numbers(t: dict, inp: dict) -> dict:
    """`4.eval`: QLIKE по окнам, Δ с интервалами, пары (Δ, интервал, годы), «go», веса. Колонок p здесь нет."""
    m, tt, pr = t["METRICS"], t["TESTS"], t["PAIRS"]
    ql = {s: {w: float(g.loc[g["window"] == w, "qlike"].iloc[0]) for w in g["window"]} for s, g in m.groupby("series")}
    dl = {"%s|%s" % (r.window, r.base): [float(r.delta_pct), float(r.lo90), float(r.hi90)] for r in tt.itertuples()}
    pair = {"%s|%s|%s|%s" % (r.hyp, r.x, r.y, r.window): [float(r.delta_pct), float(r.lo90), float(r.hi90), float(r.delta_q), int(r.years_neg), int(r.n_years), r.rule62] for r in pr.itertuples()}
    go = {str(int(r.cond)): [float(r.value), bool(r.passed)] for r in t["GO"].itertuples()}
    wt = {"%s|%s" % (r.quantity, r.period): (float(r.mean) if np.isfinite(r.mean) else None) for r in t["WEIGHTS"].itertuples()}         # 2018 — прогноз Stage A, весов Stage B нет (None, не NaN: NaN ≠ NaN при сверке с CHECKPOINT)
    return dict(n_dates=len(inp["idx"]), first=str(inp["idx"][0].date()), last=str(inp["idx"][-1].date()), qlike=ql, delta_pct_cfg10=dl, pairs=pair, go_cfg10=go, weights=wt, input_sha256=inp["shas"], status=STATUS,
                p_values="не входят: записаны в 4.rules и 4.cond11 после этого ключа")


def rules(t: dict) -> dict:
    """6.2 (с p) по парам 10 против 6 и 8 и статус «go» 6.1 для 10: «подтверждено» — только при p Холма (N = 733) < 0,05 против K2 и K3 на 2018–2021."""
    g, tt = t["GO"], t["TESTS"][(t["TESTS"]["window"] == "2018_2021") & t["TESTS"]["base"].isin(["K2", "K3"])]
    is_go, conf = bool(g.loc[g["cond"] == 0, "passed"].iloc[0]), bool((tt["p_holm_N"] < 0.05).all())
    return dict(cfg10=dict(go=is_go, p_holm_N_vs_K2_K3=[float(x) for x in tt["p_holm_N"]], status=REP.verdict(is_go, conf)),
                pairs_62={"%s|%s|%s|%s" % (r.hyp, r.x, r.y, r.window): [float(r.delta_q), float(r.p_one), int(r.years_neg), r.rule62, float(r.p_holm_N)] for r in t["PAIRS"].itertuples()})


def drift_frame() -> pd.DataFrame:
    from models_v2.s14 import v7_1e_pre as PRE
    pack, idx = PRE.pack_from_npz(STA.WINDOW, IO.OUT), DATA.load_tab().index
    return DX.drift_table(pack, idx, lambda y, s: STA.load(R4.enc_path(y, s)))


def write_eval() -> int:
    inp = load()
    t = compute(inp)
    R4.update_4("eval", dict(numbers(t, inp), written_at=RG.now()))                  # ДО записи и вывода p-значений
    for k, name in NAMES.items():
        IO.write_csv(t[k], name, NOTES[k], status=STATUS)
    R4.update_4("rules", rules(t))
    c11 = E4.cond11(t["PAIRS"])
    R4.update_4("cond11", dict(c11, written_at=RG.now()))                          # ДО любого счёта 11
    if not c11["cond11"]:
        RG.add_result("11", dict(stage="4a", reason="не считается: условие 4.cond11 ложно", cond11=False), suffix=RG.NOT_RUN11_SUFFIX)
    dr = drift_frame()
    IO.write_csv(dr, DRIFT, "4 G46: дрейф энкодера Stage B против Stage A (потери задачи-предлога, согласие z), год × seed; диагностика, частота не меняется", status=STATUS)
    R4.update_4("drift", DX.by_year(dr))
    cp = CR.read_cp()
    (IO.REP / "LEAF4_REPORT.md").write_text(REP.render(t, inp, cp["4"], dr), encoding="utf-8")
    print("оценка 10: %d дат; таблицы %s; отчёт LEAF4_REPORT.md; go = %s; cond11 = %s" % (len(inp["idx"]), ", ".join(NAMES.values()), cp["4"]["rules"]["cfg10"]["status"], c11["cond11"]))
    return 0


def drift_spot(dr: pd.DataFrame, year: int = 2019, seed: int = 0) -> list:
    """Пересчёт одной строки (год, seed) таблицы дрейфа из весов Stage A и `4/encB` и сверка с `4_DRIFT.csv` (допуск 1e-9)."""
    from models_v2.s14 import v7_1e_pre as PRE
    pack, idx = PRE.pack_from_npz(STA.WINDOW, IO.OUT), DATA.load_tab().index
    new = DX.row(year, seed, pack, idx, STA.load(STA.directory() / STA.name("tcn", year, seed)), STA.load(R4.enc_path(year, seed)))
    old = dr[(dr["year"] == year) & (dr["seed"] == seed)].iloc[0]
    return [] if np.allclose([new[c] for c in DX.COLS], [old[c] for c in DX.COLS], rtol=1e-9, atol=0) else ["%s: строка %d seed %d не равна пересчёту" % (DRIFT, year, seed)]


def _close(a: dict, b: dict, rtol: float = 1e-9) -> bool:
    """Годовые сводки дрейфа равны с допуском (CSV читается без побитного округления чисел)."""
    return a.keys() == b.keys() and all(np.allclose([a[y][k] for k in a[y]], [b[y][k] for k in a[y]], rtol=rtol, atol=0) and a[y].keys() == b[y].keys() for y in a)


def check_eval(cp: dict) -> list:
    """-> замечания (пусто = ок). Читает файлы, ничего не пишет."""
    cp4 = cp.get("4", {})
    if not all(k in cp4 for k in ("eval", "rules", "cond11", "drift")):
        return ["CHECKPOINT 4: нет eval/rules/cond11/drift"]
    inp = load()
    t, bad = compute(inp), []
    strip = lambda d: {k: v for k, v in d.items() if k != "written_at"}
    if numbers(t, inp) != strip(cp4["eval"]):
        bad.append("4.eval: пересчёт чисел ≠ CHECKPOINT")
    if rules(t) != cp4["rules"]:
        bad.append("4.rules: пересчёт ≠ CHECKPOINT")
    if E4.cond11(t["PAIRS"]) != strip(cp4["cond11"]):
        bad.append("4.cond11: пересчёт ≠ CHECKPOINT")
    if not (cp4["eval"]["written_at"] <= cp4["cond11"]["written_at"] and cp4["cfg10"]["written_at"] <= cp4["eval"]["written_at"]):
        bad.append("порядок записи нарушен: нужно cfg10 → eval → cond11")
    for k, name in NAMES.items():
        if not (IO.REP / name).exists() or not R1.same_frame(t[k], IO.read_csv(name)):
            bad.append("%s: нет или не совпал с пересчётом" % name)
    dr = IO.read_csv(DRIFT) if (IO.REP / DRIFT).exists() else None
    if dr is None or not _close(DX.by_year(dr), cp4["drift"]):
        bad.append("%s: нет или не равен 4.drift" % DRIFT)
    else:
        bad += drift_spot(dr)
    p = IO.REP / "LEAF4_REPORT.md"
    if dr is not None and (not p.exists() or p.read_text(encoding="utf-8") != REP.render(t, inp, cp4, dr)):
        bad.append("LEAF4_REPORT.md: нет или не равен тексту из таблиц")
    return bad


if __name__ == "__main__":
    sys.exit(write_eval())
