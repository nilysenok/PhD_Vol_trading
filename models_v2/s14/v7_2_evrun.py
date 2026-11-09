"""V7-2b, G28–G30: чтение рядов, запись CHECKPOINT `2.eval` (числа QLIKE, Δ с интервалами, условия «go», специализация — БЕЗ p-значений) ДО таблиц с p, затем таблицы `2_*.csv`, `2.rules` (6.2, статусы; p), `2.cond9`, отчёт `LEAF2_REPORT.md`.
Запуск записи: `python3 -W ignore -m models_v2.s14.v7_2_evrun`. `check_eval` (из `run_v7 --stage 2 --check`) ничего не пишет: пересчитывает таблицы и отчёт из сохранённых прогнозов и сверяет с файлами и CHECKPOINT.
Конфигурации 1–7 — версия 2 (`1b/cfg1..3`, `2/cfg4..7`), версия 1 — `archive_v1/` (только рядом, для сравнения). Новых счётов нет."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_ctrl as C, v7_0_io as IO, v7_1b_cfg3 as C3, v7_1b_cfgrun as CR, v7_1b_ev as EV, v7_1b_evrep as REP1, v7_1b_evrun as R1, v7_1m_data as DATA, v7_1m_stat as ST
from models_v2.s14 import v7_2_ev as E2, v7_2_evrep as REP, v7_2_evspec as SP

STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-236, лист 2b (конфигурации 1–7 версии 2; окна 2022+ — только оценка, выбора по ним нет; версия 1 — только рядом)"
NAMES = {"METRICS": "2_METRICS.csv", "TESTS": "2_TESTS.csv", "PAIRS": "2_PAIRS.csv", "GO": "2_GO.csv", "YEARS": "2_YEARS.csv", "STRESS": "2_STRESS.csv", "MCS": "2_MCS.csv", "SPEC": "2_SPEC.csv", "WEIGHTS": "2_WEIGHTS.csv", "V1": "2_V1.csv"}
NOTES = {"METRICS": "2b G28: QLIKE, RMSE, MAE, R², bias, 16 рядов, одинаковые даты", "TESTS": "2b G28: Δ в % уровня K1 (90 %, блок 20, 2000, seed 0), DM лаг 5, Холм N = 733 / 3166",
         "PAIRS": "2b G29: гипотезы H1, H2, H3 и 4–7 против баз; правило 6.2; стресс", "GO": "2b G29: «go» 6.1, шесть условий, конфигурации 1–7", "YEARS": "2b: годовой вклад в условия 4–5",
         "STRESS": "2b: стресс — без 1 % крайних дат, без 2018", "MCS": "2b: MCS 90 %", "SPEC": "2b G30: специализация экспертов 5–7 (6.3), задним числом, описательно", "WEIGHTS": "2b G30: средние π по годам, до и после 2021 (6.5)", "V1": "2b: версия 1 рядом, QLIKE и Δ к K3"}
V1_DIR = IO.OUT / "archive_v1"


def paths_v2() -> dict:
    return {"cfg1": CR.path("tcn"), "cfg2": CR.path("cnn"), "cfg3": C3.PATH, **{"cfg%s" % c: IO.OUT / "2" / ("cfg%s.parquet" % c) for c in "4567"}}


def _cp_sha(cp: dict, name: str) -> str:
    return (cp["1b"] if name in ("cfg1", "cfg2", "cfg3") else cp["2"])[name]["sha256_file"]


def load() -> dict:
    """-> dict(idx, rv, tp, series (16), v1 (cfg → прогноз), frames (cfg5–7), shas). Файлы v2 — по sha256 CHECKPOINT; даты одинаковые; v1 — по `archive_sha256`."""
    tab, w, cp = DATA.load_tab(), C.aligned(), CR.read_cp()
    p2, shas = paths_v2(), {}
    raw = {k: pd.read_parquet(p) for k, p in p2.items()}
    for k, p in p2.items():
        shas[k] = IO.sha_file(p)
        if shas[k] != _cp_sha(cp, k):
            raise SystemExit("%s: sha256 файла ≠ CHECKPOINT" % k)
    frames = {k: v.set_index("d")[["F"]] for k, v in raw.items()}
    if not EV.same_dates(frames):
        raise SystemExit("даты конфигураций 1–7 не совпадают")
    idx = frames["cfg1"].index
    inp1 = _controls(tab, w, idx)
    v1 = {}
    for k in E2.CFGS:
        rel = ("1b/%s.parquet" if k in ("cfg1", "cfg2", "cfg3") else "2/%s.parquet") % k
        p = V1_DIR / rel
        v1[k], shas["v1:" + k] = pd.read_parquet(p).set_index("d")["F"].reindex(idx).to_numpy(float), IO.sha_file(p)
        if shas["v1:" + k] != (cp["1b_v1"] if k in ("cfg1", "cfg2", "cfg3") else cp["2_v1"])["archive_sha256"][rel]:
            raise SystemExit("v1 %s: sha256 архива ≠ CHECKPOINT" % k)
    t = tab.loc[idx]
    for k, v in raw.items():
        if not (np.allclose(v["rv"], t["rv_next"].to_numpy(float), rtol=0, atol=0) and np.allclose(v["H"], t["H"].to_numpy(float), rtol=0, atol=0)):
            raise SystemExit("%s: rv или H не равны таблице SBER" % k)
    series = {**inp1["refs"], **{k: raw[k]["F"].to_numpy(float) for k in E2.CFGS}}
    bad = [k for k, v in series.items() if not (np.isfinite(v).all() and (v > 0).all())]
    if bad:
        raise SystemExit("нет значений на датах конфигураций: %s" % bad)
    shas.update(inp1["shas"])
    return dict(idx=idx, rv=t["rv_next"].to_numpy(float), tp=t["t_plus"].to_numpy(), series=series, v1=v1, frames={k: raw[k] for k in SP.MOE}, shas=shas)


def _controls(tab, w, idx) -> dict:
    ext, shas = {}, {}
    for name, rel in R1.SRC.items():
        ext[name] = C.forecast(name, IO.ROOT / rel, tab.index, w)
        shas[name] = IO.sha_file(IO.ROOT / rel)
    fz = C.frozen()
    for name in ("V4:NTP2", "V6:A2"):
        if shas[name] != fz.loc[name, "sha256_file"]:
            raise SystemExit("%s: sha256 файла ≠ CMP_SERIES.csv" % name)
    t = tab.loc[idx]
    refs = ST.refs(t["H"], t["K2"], t["K3"], pd.Series(ext["V4:NTP2"], tab.index).loc[idx], pd.Series(ext["V6:A2"], tab.index).loc[idx])
    refs.update({n: pd.Series(ext[n], tab.index).loc[idx].to_numpy(float) for n in ("V6:A0", "V6:A0g")})
    return dict(refs=refs, shas=shas)


def compute(inp: dict) -> dict:
    rv, tp, s = inp["rv"], inp["tp"], inp["series"]
    t = E2.main_tables(rv, s, tp)
    t["V1"] = E2.v1_table(rv, s, inp["v1"], tp)
    t["SPEC"], t["WEIGHTS"] = SP.spec_table(inp["frames"]), SP.weights_table(inp["frames"])
    return t


def _n(x):
    return None if isinstance(x, float) and np.isnan(x) else x


def numbers(t: dict, inp: dict) -> dict:
    """`2.eval`: QLIKE по окнам, Δ с интервалами, пары (Δ, интервал, годы), условия «go», специализация. Колонок p здесь нет."""
    m, tt, pr, sp = t["METRICS"], t["TESTS"], t["PAIRS"], t["SPEC"]
    ql = {s: {w: float(g.loc[g["window"] == w, "qlike"].iloc[0]) for w in g["window"]} for s, g in m.groupby("series")}
    dl = {c: {"%s|%s" % (r.window, r.base): [float(r.delta_pct), float(r.lo90), float(r.hi90)] for r in g.itertuples()} for c, g in tt.groupby("config")}
    pair = {"%s|%s|%s|%s" % (r.hyp, r.x, r.y, r.window): [float(r.delta_pct), float(r.lo90), float(r.hi90), float(r.delta_q), int(r.years_neg), int(r.n_years), r.rule62] for r in pr.itertuples()}
    go = {c: {str(int(r.cond)): [float(r.value), bool(r.passed)] for r in g.itertuples()} for c, g in t["GO"].groupby("config")}
    cols = ["share_pi_gt03", "switch_freq", "mean_abs_dpi", "gain_on", "gain_rest", "diff_on_rest_lo90", "diff_on_rest_hi90", "spearman_pi_gain_hindsight", "approx_err"]
    spec = {"%s|%d|%s" % (r.config, r.k, r.window): [_n(float(getattr(r, c))) for c in cols] + [r.specialized] for r in sp.itertuples()}
    v1 = {"%s|%s" % (r.config, r.window): [float(r.qlike_v1), float(r.qlike_v2)] for r in t["V1"].itertuples()}
    return dict(n_dates=len(inp["idx"]), first=str(inp["idx"][0].date()), last=str(inp["idx"][-1].date()), qlike=ql, delta_pct=dl, pairs=pair, go=go, spec=spec, qlike_v1_v2=v1, input_sha256=inp["shas"], status=STATUS,
                p_values="не входят: записаны в 2.rules и 2.cond9 после этого ключа")


def rules(t: dict) -> dict:
    """6.2 (с p) по парам и статус «go» по конфигурациям 1–7: «подтверждено» — только при p Холма (N = 733) < 0,05 против K2 и K3 на 2018–2021."""
    out = {}
    for c in E2.CFGS:
        g, tt = t["GO"][t["GO"]["config"] == c], t["TESTS"][(t["TESTS"]["config"] == c) & (t["TESTS"]["window"] == "2018_2021") & t["TESTS"]["base"].isin(["K2", "K3"])]
        is_go, conf = bool(g.loc[g["cond"] == 0, "passed"].iloc[0]), bool((tt["p_holm_N"] < 0.05).all())
        out[c] = dict(go=is_go, p_holm_N_vs_K2_K3=[float(x) for x in tt["p_holm_N"]], status=REP1.verdict(is_go, conf))
    out["pairs_62"] = {"%s|%s|%s" % (r.hyp, r.x, r.y): [float(r.delta_q), float(r.p_one), int(r.years_neg), r.rule62] for r in t["PAIRS"][t["PAIRS"]["window"] == "2018_2021"].itertuples()}
    return out


def write_eval() -> int:
    inp = load()
    t = compute(inp)
    from models_v2.s14 import v7_2_run as R2
    R2.update_2("eval", numbers(t, inp))                      # ДО записи и вывода p-значений
    for k, name in NAMES.items():
        IO.write_csv(t[k], name, NOTES[k], status=STATUS)
    R2.update_2("rules", rules(t))
    R2.update_2("cond9", SP.cond9(inp["frames"]["cfg6"], t["PAIRS"]))
    (IO.REP / "LEAF2_REPORT.md").write_text(REP.render(t, inp, CR.read_cp()["2"]), encoding="utf-8")
    print("оценка 1–7: %d дат; таблицы %s; отчёт LEAF2_REPORT.md; cond9 = %s" % (len(inp["idx"]), ", ".join(NAMES.values()), CR.read_cp()["2"]["cond9"]["cond9"]))
    return 0


def check_eval(cp: dict) -> list:
    """-> замечания (пусто = ок). Читает файлы, ничего не пишет."""
    cp2 = cp.get("2", {})
    if not all(k in cp2 for k in ("eval", "rules", "cond9")):
        return ["CHECKPOINT 2: нет eval/rules/cond9"]
    inp = load()
    t, bad = compute(inp), []
    if numbers(t, inp) != cp2["eval"]:
        bad.append("2.eval: пересчёт чисел ≠ CHECKPOINT")
    if rules(t) != cp2["rules"]:
        bad.append("2.rules: пересчёт ≠ CHECKPOINT")
    if SP.cond9(inp["frames"]["cfg6"], t["PAIRS"]) != cp2["cond9"]:
        bad.append("2.cond9: пересчёт ≠ CHECKPOINT")
    for k, name in NAMES.items():
        if not (IO.REP / name).exists() or not R1.same_frame(t[k], IO.read_csv(name)):
            bad.append("%s: нет или не совпал с пересчётом" % name)
    p = IO.REP / "LEAF2_REPORT.md"
    if not p.exists() or p.read_text(encoding="utf-8") != REP.render(t, inp, cp2):
        bad.append("LEAF2_REPORT.md: нет или не равен тексту из таблиц")
    return bad


if __name__ == "__main__":
    sys.exit(write_eval())
