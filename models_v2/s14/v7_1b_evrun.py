"""V7-1b-C, G25: чтение рядов, запись CHECKPOINT `1b.eval` (числа QLIKE и Δ, условия «go» — БЕЗ p-значений) ДО записи и вывода p-значений, затем таблицы `1b_*.csv`, `1b.rules` (6.2, статусы; p-значения), отчёт.
`check_eval` ничего не пишет: пересчитывает таблицы и отчёт из сохранённых прогнозов и сверяет с файлами и CHECKPOINT. Реестр — строки результата 1–3 пишут `v7_1b_cfgrun` / `v7_1b_cfg3`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_ctrl as C, v7_0_io as IO, v7_1b_cfg3 as C3, v7_1b_cfgrun as CR, v7_1b_ev as EV, v7_1b_evrep as REP, v7_1m_data as DATA, v7_1m_stat as ST

STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-232, лист 1b-C (конфигурации 1–3; окна 2022+ — только оценка, выбора по ним нет)"
SRC = {"V4:NTP2": "data/regen/harj_hybrid_v4/NTP2.parquet", "V6:A2": "data/regen/harj_hybrid_v6/A2.parquet", "V6:A0": "data/regen/harj_hybrid_v6/A0.parquet", "V6:A0g": "data/regen/harj_hybrid_v6/A0g.parquet"}
TABLES = {"METRICS": "1b_METRICS.csv", "TESTS": "1b_TESTS.csv", "GO": "1b_GO.csv", "NEWINFO": "1b_NEWINFO.csv", "MCS": "1b_MCS.csv", "STRESS": "1b_STRESS.csv", "YEARS": "1b_YEARS.csv"}
NOTES = {"METRICS": "G25: QLIKE, RMSE, MAE, R², bias по годам и окнам, одинаковые даты", "TESTS": "G25: Δ в % уровня K1 (90 %, блок 20, 2000, seed 0), DM лаг 5, Холм N = 733 / 3166", "GO": "G25: «go» 6.1, шесть условий",
         "NEWINFO": "G25: правило 6.2, конфигурации 1–3 против A0 и A0g, 2018–2021", "MCS": "G25: MCS 90 %", "STRESS": "G25: стресс — без 1 % крайних дат, без 2018", "YEARS": "G25: годовой вклад в условия 4–5"}


def load() -> dict:
    """-> dict(idx, rv, tp, series (12 рядов на ОДИНАКОВЫХ датах), shas). Даты — даты конфигураций 1–3; любая база без значения на дате — ошибка."""
    tab, w = DATA.load_tab(), C.aligned()
    cfg = {"cfg1": pd.read_parquet(CR.path("tcn")), "cfg2": pd.read_parquet(CR.path("cnn")), "cfg3": pd.read_parquet(C3.PATH)}
    frames = {k: v.set_index("d")[["F"]] for k, v in cfg.items()}
    if not EV.same_dates(frames):
        raise SystemExit("даты конфигураций 1–3 не совпадают")
    idx, shas = frames["cfg1"].index, {}
    ext = {}
    for name, rel in SRC.items():
        ext[name] = C.forecast(name, IO.ROOT / rel, tab.index, w)
        shas[name] = IO.sha_file(IO.ROOT / rel)
    t = tab.loc[idx]
    refs = ST.refs(t["H"], t["K2"], t["K3"], pd.Series(ext["V4:NTP2"], tab.index).loc[idx], pd.Series(ext["V6:A2"], tab.index).loc[idx])
    refs.update({n: pd.Series(ext[n], tab.index).loc[idx].to_numpy(float) for n in ("V6:A0", "V6:A0g")})
    series = {**refs, **{k: frames[k]["F"].to_numpy(float) for k in EV.CFGS}}
    bad = [k for k, v in series.items() if not (np.isfinite(v).all() and (v > 0).all())]
    if bad or not np.isfinite(t["rv_next"]).all():
        raise SystemExit("нет значений на датах конфигураций: %s" % bad)
    fz = C.frozen()
    for name in ("V4:NTP2", "V6:A2"):
        if shas[name] != fz.loc[name, "sha256_file"]:
            raise SystemExit("%s: sha256 файла ≠ CMP_SERIES.csv" % name)
    return dict(idx=idx, rv=t["rv_next"].to_numpy(float), tp=t["t_plus"].to_numpy(), series=series, shas=shas)


def compute(inp: dict) -> dict:
    rv, tp, s = inp["rv"], inp["tp"], inp["series"]
    t = {"METRICS": EV.metrics_table(rv, s, tp), "MCS": EV.mcs_table(rv, s, tp)}
    refs = {k: s[k] for k in EV.REFS}
    parts = {k: dict(tests=EV.tests_table(rv, s[k], refs, tp), go=EV.go_table(rv, s[k], refs, tp), ni=EV.new_info_table(rv, s[k], refs, tp), yr=EV.years_table(rv, s[k], refs, tp)) for k in EV.CFGS}
    for key, part in (("TESTS", "tests"), ("GO", "go"), ("NEWINFO", "ni"), ("YEARS", "yr")):
        t[key] = pd.concat([p[part].assign(config=k) for k, p in parts.items()], ignore_index=True)
    t["STRESS"] = EV.stress_table(rv, {k: s[k] for k in ("K1", "K2", "K3", "M", "K8", "V4:NTP2", "V6:A2", "V6:A0", "V6:A0g") + EV.CFGS}, tp)
    return t


def numbers(t: dict, inp: dict) -> dict:
    """Что идёт в `1b.eval` до p-значений: QLIKE по окнам, Δ с интервалами, условия «go». Колонок p здесь нет."""
    m = t["METRICS"]
    ql = {s: {w: float(g.loc[g["window"] == w, "qlike"].iloc[0]) for w in g["window"]} for s, g in m.groupby("series")}
    dl = {c: {f"{r.window}|{r.base}": [float(r.delta_pct), float(r.lo90), float(r.hi90)] for r in g.itertuples()} for c, g in t["TESTS"].groupby("config")}
    go = {c: {str(int(r.cond)): [float(r.value), bool(r.passed)] for r in g.itertuples()} for c, g in t["GO"].groupby("config")}
    return dict(n_dates=len(inp["idx"]), first=str(inp["idx"][0].date()), last=str(inp["idx"][-1].date()), qlike=ql, delta_pct=dl, go=go, input_sha256=inp["shas"], status=STATUS, p_values="не входят: записаны в 1b.rules после этого ключа")


def rules(t: dict) -> dict:
    """6.2 (с p) и статус «go»: «подтверждено» — только при p Холма (N = 733) < 0,05 против K2 и K3 на 2018–2021."""
    out = {}
    for c in EV.CFGS:
        ni, go = t["NEWINFO"][t["NEWINFO"]["config"] == c], t["GO"][t["GO"]["config"] == c]
        tt = t["TESTS"][(t["TESTS"]["config"] == c) & (t["TESTS"]["window"] == "2018_2021") & t["TESTS"]["base"].isin(["K2", "K3"])]
        is_go, conf = bool(go.loc[go["cond"] == 0, "passed"].iloc[0]), bool((tt["p_holm_N"] < 0.05).all())
        out[c] = dict(new_information={r.base: [float(r.delta), float(r.p_dm_one), int(r.years_neg), bool(r.new_information)] for r in ni.itertuples()}, go=is_go, p_holm_N_vs_K2_K3=[float(x) for x in tt["p_holm_N"]],
                      status=REP.verdict(is_go, conf))
    return out


def write_eval() -> int:
    inp = load()
    t = compute(inp)
    CR.update_1b("eval", numbers(t, inp))                 # ДО записи и вывода p-значений
    for k, name in TABLES.items():
        IO.write_csv(t[k], name, NOTES[k], status=STATUS)
    CR.update_1b("rules", rules(t))
    (IO.REP / "LEAF1B_REPORT.md").write_text(REP.render(t, IO.read_csv("1b_E0.csv"), inp, CR.read_cp()["1b"]), encoding="utf-8")
    print("оценка 1–3: %d дат %s … %s; таблицы %s; отчёт LEAF1B_REPORT.md" % (len(inp["idx"]), inp["idx"][0].date(), inp["idx"][-1].date(), ", ".join(TABLES.values())))
    return 0


def same_frame(x: pd.DataFrame, y: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(x.reset_index(drop=True), y.reset_index(drop=True), check_dtype=False, rtol=1e-12, atol=1e-12)
        return True
    except AssertionError:
        return False


def check_eval(cp: dict) -> list:
    if "eval" not in cp or "rules" not in cp:
        return ["CHECKPOINT 1b: нет eval/rules"]
    inp = load()
    t, bad = compute(inp), []
    if numbers(t, inp) != cp["eval"]:
        bad.append("1b.eval: пересчёт чисел ≠ CHECKPOINT")
    if rules(t) != cp["rules"]:
        bad.append("1b.rules: пересчёт ≠ CHECKPOINT")
    for k, name in TABLES.items():
        if not (IO.REP / name).exists() or not same_frame(t[k], IO.read_csv(name)):
            bad.append("%s: нет или не совпал с пересчётом" % name)
    p = IO.REP / "LEAF1B_REPORT.md"
    if not p.exists() or p.read_text(encoding="utf-8") != REP.render(t, IO.read_csv("1b_E0.csv"), inp, cp):
        bad.append("LEAF1B_REPORT.md: нет или не равен тексту из таблиц")
    return bad
