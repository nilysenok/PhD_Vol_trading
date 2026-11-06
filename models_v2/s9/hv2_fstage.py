"""Ж-214, лист 5: стадия `final` — статистика финалистов (V14, V15), лестница Л0–Л9 (V12), таблица п. 29, рейтинг архитектур, `FINAL_V2_CONFIG.json`. Новых trials нет.
Запуск: python3 -W ignore -m models_v2.s9.run_hv2 --stage final (run_in_background)."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from models_v2.s6.e import crit_e
from models_v2.s8 import hs_gstage as G, hs_store
from models_v2.s9 import hv2_fladder as FL, hv2_fpick as FP, hv2_fstats as FS, hv2_ledger as L, hv2_paths as P

TOP_N = 3


def sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_crit(df: pd.DataFrame, fin: pd.DataFrame, s: dict) -> float:
    """Сверка: crit_wf из таблиц направлений равен пересчёту по рядам (выбор финалиста не зависит от рассинхрона файлов)."""
    from models_v2 import metrics
    mx = 0.0
    for _, r in fin.iterrows():
        q = np.asarray(metrics.qlike(df["rv"].to_numpy(float), s["%s:%s" % (r["step"], r["model"])]), float)
        mx = max(mx, abs(crit_e.cw(q, df["target_date"]) - r["crit_wf"]))
    return mx


def reference(df: pd.DataFrame, fin: pd.DataFrame, n: int) -> tuple:
    """Справочно, вне семейства (без Холма): строки, устойчивые без 24.02.2022, + H9 (иное чтение V12 для Л2). Тесты A/B/C — только сырые p."""
    r = FP.robust_rows(set(fin["model"]))
    extra = pd.concat([r, FP.read_table(FP.HR).query("model == 'H9'").assign(file=FP.HR)[r.columns]]).drop_duplicates("model").reset_index(drop=True)
    ser = FP.load_series(list(extra["model"]), df)
    rows = []
    for m in extra["model"]:
        for x in G.V.tests_rows(df, m, ser[m], n):
            if x["period"] in ("all", "no_240222"):
                rows.append(x)
    t = pd.DataFrame(rows)
    t["note"] = "выбраны после просмотра (устойчивы без 24.02.2022 или иное чтение V12); вне семейства; p сырые, без Холма; потолок; HISTORICAL WALK-FORWARD, не OOS"
    return extra, t


def config(fin: pd.DataFrame, rk: pd.DataFrame, lv: pd.DataFrame, n: int) -> dict:
    """Зафиксированные спецификации для переноса (перенос не запускается): три первые по рейтингу и все финалисты."""
    def one(step, model):
        v = lv.set_index("model").loc["%s:%s" % (step, model)]
        return {"step": step, "model": model, "registry": FP.registry_spec(model), "level": v["verdict"], "ceiling": FS.CEIL}
    top = [one(r["step"], r["model"]) for _, r in rk.head(TOP_N).iterrows() if not r["excluded_reason"]]      # исключённые (R2T: V11) не входят; третья позиция — решение владельца
    return {"program": "Ж-214 HAR-J Hybrid V2, SBER h=1", "status": "DEVELOPMENT; " + FS.CEIL, "N_holm": n, "selection_rule": "минимум crit_wf среди кандидатов ступени (V12, У30)",
            "rating_rule": "шесть правил " + ",".join(FL.RATING) + "; ранг: число выполненных, затем ΔQLIKE без 24.02.2022 (У31)", "top3": top,
            "finalists": [one(r["step"], r["model"]) for _, r in fin.iterrows()], "controls": {"A0": "HAR-J(б)", "S_star": "XJ", "LF": "LGBM"},
            "transfer": "не запускался; предложение — в HARJ_HYBRID_V2_FINAL_RESEARCH.md; архитектура фиксирована этим файлом; панель — отдельный этап вне 80",
            "hashes": {"components": sha(P.COMP), "m10_daily": sha(P.M10), "ALL_TRIALS": sha(L.ALL), **{f.name: sha(f) for f in sorted(P.OUT.glob("*_series.parquet"))}}}


def stage_final() -> int:
    n = FS.declare_n()
    df = hs_store.frozen()
    fin = FP.finalists()
    ser = FP.load_series(list(fin["model"]) + ["T3", "E1"], df)
    s = FS.build_s(df, fin, ser)
    mx = check_crit(df, fin, s)
    if mx > 1e-9:
        raise SystemExit("стоп: crit_wf таблиц != пересчёт (%g)" % mx)
    w = FS.write
    w(fin, "FINALISTS.csv", "V12/У30: финалисты по минимуму crit_wf среди кандидатов ступени; crit_check_max_abs=%g" % mx)
    met = G.metric_rows(df, s)
    w(met, "FORECAST_COMPARISONS.csv", G.comparisons_note(s))
    w(G.annual_rows(df, s), "ANNUAL.csv", "QLIKE по годам и Δ к HAR-J / XGB / LGBM (минус — модель лучше)")
    w(G.regime_rows(df, s), "REGIME.csv", "QLIKE по режимам (терцили RV_w SBER и IMOEX — по прошлому, RVI, стресс Г4)")
    top, corr = G.errors_and_corr(df, s)
    w(top, "LARGEST_ERRORS.csv", "10 крупнейших по QLIKE дней каждой модели")
    w(corr, "ERROR_CORRELATION.csv", "корреляция ln-ошибок ln(RV/σ̂²) между моделями")
    t, lv = FS.stats_all(df, s, n)
    w(t, "STATISTICAL_TESTS.csv", "A (к HAR-J), B (к XJ), C (к LF): Δ=QLIKE(M)−QLIKE(опоры), DM НW лаг 5, бутстрап блок 20 / 5000 / seed 42; Холм (A и B, all) N=%d, финалистов %d" % (n, len(fin)))
    w(lv, "LEVELS.csv", "уровни 1–3 по V14; уровень 4 — не проверялся; ACCEPTED не присваивается")
    t5 = FS.top5_table(df, s)
    w(t5, "TOP5_DAYS.csv", "Δ без 5 дней с наибольшим вкладом в преимущество")
    w(FS.mcs_table(df, s), "MCS.csv", "MCS 90 %, блок 20, 5000, seed 42: вся история и без 24.02.2022")
    lad = FL.ladder(df, s, fin, {"T3": ser["T3"], "E1": ser["E1"]}, t5)
    P.write_csv(lad, "ABL_LADDER.csv", "V12: лестница Л0–Л9 (+V8), по одному механизму к A0; Δ<0 — ступень лучше; «почему» — только измеренные свойства")
    rk = FL.rating(lad, t, fin)
    w(rk, "ARCH_RANKING.csv", "У31: шесть правил устойчивости к A0, ранг по числу выполненных, затем ΔQLIKE без 24.02.2022")
    w(FL.p29(met, t, lv, fin, df), "FINAL_TABLE_P29.csv", "п. 29 блока: одинаковые %d дат цели, n и неопределённость; Panel Hybrid — не в этом пакете" % len(df))
    ref, rt = reference(df, fin, n)
    w(ref, "REFERENCE_ROBUST_EX_2402.csv", "справочно, вне семейства: выбраны после просмотра")
    w(rt, "REFERENCE_TESTS.csv", "справочно: тесты A/B/C для справочных строк, без Холма")
    (P.REP / "FINAL_V2_CONFIG.json").write_text(json.dumps(config(fin, rk, lv, n), ensure_ascii=False, indent=1, default=float))
    L.write_check(L.load(), decisions={**json.loads(L.CHECK.read_text()).get("decisions", {}), "final": {"finalists": {r["step"]: r["model"] for _, r in fin.iterrows()}, "top3": list(rk.head(TOP_N).query("excluded_reason == ''")["model"]), "N": n}})
    print(lv[["model", "qlike_all", "crit_wf", "L1", "L2", "L3", "verdict"]].round(5).to_string(index=False))
    print(rk.round(5).to_string(index=False))
    return 0
