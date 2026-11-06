"""Документы E5: таблица п. 21 (XGBoost против LightGBM), HYBRIDS.md, REPRODUCTION.md. Числа — из рядов и таблиц E3/E4."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s6 import paths6
from models_v2.s6.e import comp, crit_e, pe, ser
from models_v2.s6.xgb import c_calc


def train_secs() -> dict:
    tid, _ = comp.x2_moves()
    x = float("nan")   # время в реестре xgboost/ — не время всей ежедневной серии (6 с неправдоподобно): не используется
    p = comp.LDIR / "c" / "st_C0_inh_F4.json"
    l = float(json.loads(p.read_text()).get("secs", np.nan)) if p.exists() else float("nan")
    return {"XJ": x, "LF": l}


def table21(df, tests, sd) -> pd.DataFrame:
    cols = {"XJ": "XGBoost (x2_best_j; потолок 2017–21)", "XF": "XGBoost ctrl_f4 (чистая, справочно)", "LF": "LightGBM (C0_inh_F4)"}
    q_h, ts = ser.qloss(df, "HAR"), train_secs()
    yr, out = pd.Series(df["target_date"]).dt.year.to_numpy(), {}
    mk = lambda a, b: (yr >= a) & (yr <= b)  # noqa: E731
    for k in cols:
        d, q = crit_e.describe(k, df[k].to_numpy(float), df), ser.qloss(df, k)
        t = tests[(tests["candidate"] == k) & (tests["reference"] == "HAR") & (tests["period"] == "вся история")]
        s = sd[sd["id"] == k]
        out[cols[k]] = ["%.6f" % d["qlike_all"], "%.4f" % d["rmse_lvl"], "%.4f" % d["mae_lvl"], "%.4f" % d["r2_mz_lvl"], "%.4f" % q[mk(2017, 2021)].mean(), "%.4f" % q[mk(2022, 2022)].mean(),
                        "%.4f" % q[mk(2023, 2025)].mean(), "%.4f" % q[mk(2026, 2026)].mean(), "%d из 10" % sum(q[yr == y].mean() < q_h[yr == y].mean() for y in range(2017, 2027)),
                        "%.3f (%s)" % (q.max(), str(df["target_date"][int(q.argmax())])[:10]), ("%.0f с" % ts[k]) if k in ts and np.isfinite(ts[k]) else "н/д (в реестре нет времени)",
                        "HP: %s; признаков 33" % ("9 осей" if k[0] == "X" else "7 полей"), "sd QLIKE %.5f" % float(s["qlike_sd"].iloc[0]) if len(s) else "н/д",
                        "p=%.3g (Холм %.3g)" % (t["p_raw"].iloc[0], t["p_holm"].iloc[0]) if len(t) else "н/д", ""]
    t2 = tests[(tests["candidate"] == "XJ") & (tests["reference"] == "LF") & (tests["period"] == "вся история")]
    dmv = "Δ=%.5f, p=%.3g (XJ−LF)" % (t2["delta_qlike"].iloc[0], t2["p_raw"].iloc[0]) if len(t2) else "н/д"
    for k in out:
        out[k][-1] = dmv
    idx = ["Лучший QLIKE", "RMSE (уровень)", "MAE (уровень)", "R² (MZ)", "2017–2021", "2022", "2023–2025", "2026", "Годы с выигрышем у HAR", "Крупнейшая ошибка (QLIKE дня)", "Время обучения (ежедневно, сумма)", "Число параметров",
           "Стабильность seeds (42–46)", "DM против HAR", "DM между моделями"]
    return pd.DataFrame(out, index=idx).reset_index().rename(columns={"index": "Показатель"})


EXTRA_ZH204 = 106      # справочно: поиски Ж-204 (по Ж-204 D): N = N_Ж-206 + 106


def holm_md(tests: pd.DataFrame, extra: int = EXTRA_ZH204) -> str:
    """Оговорка о Холме и справочная таблица с N + extra из сохранённых сырых p (без пересчёта моделей)."""
    from models_v2.s5.d import stats_d
    p = tests[tests["period"] == "вся история"].reset_index(drop=True)
    n0 = pe.n_trials()
    p["h2"] = stats_d.holm_n(p["p_raw"].to_numpy(float), n0 + extra)
    rows = ["| %s к %s | %.6f | %.4f | %.4f |" % (r.candidate, r.reference, r.p_raw, r.p_holm, r.h2) for r in p.itertuples() if r.candidate in ("XJ", "XF", "LF") and r.reference in ("B2X", "B2L")]
    return ("## Статистика: оговорка о Холме\nN = %d (уникальные trials Ж-206) **не учитывает поиски Ж-204** (сетка И2 из 132 точек, Optuna 22 и др.); поправка занижена, p Холма — оценка снизу (потолок). "
            "Справочно, как в Ж-204 D: N = %d + %d = %d (из сохранённых сырых p, %d первичных сравнений «вся история», без пересчёта моделей).\n\n"
            "| Кандидат к baseline | p сырой | p Холма, N = %d | p Холма, N = %d (справочно) |\n|---|---|---|---|\n%s\n" % (n0, n0, extra, n0 + extra, len(p), n0, n0 + extra, "\n".join(rows)))


def rmse_md(df, ids=("B1q", "B2q")) -> str:
    """Что даёт RMSE уровня у residual-моделей: доля одного дня в SSE, прогноз, факт, g = ln(σ̂²/HAR), RMSE без этого дня."""
    rv, out = df["rv"].to_numpy(float), []
    for c in ids:
        e = (df[c].to_numpy(float) - rv) ** 2
        i = int(np.argmax(e))
        out.append("%s: RMSE %.1f; один прогноз от %s (цель %s, RV %.1f) даёт %.1f %% SSE, прогноз %.0f, g = ln(%s/HAR) = %+.2f; без него RMSE %.1f" % (
            c, np.sqrt(e.mean()), str(df["d"].iloc[i])[:10], str(df["target_date"].iloc[i])[:10], rv[i], 100 * e[i] / e.sum(), df[c].iloc[i], c,
            np.log(df[c].iloc[i] / df["HAR"].iloc[i]), np.sqrt(np.delete(e, i).mean())))
    return "**RMSE уровня у residual-моделей.** " + "; ".join(out) + ". У g нет верхней границы. Это отрицательный результат модели (необрезанная поправка в стрессовый день), не ошибка расчёта; QLIKE устойчив — один день весит в нём мало.\n"


def hybrids_md(sel: pd.DataFrame, ar: pd.DataFrame, t11_md: str, fin: dict, df, tests=None) -> str:
    sha = {p.name: paths6.sha_file(p)[:16] for p in (comp.FILE, ser.file("a"), ser.file("b"), ser.file("c"), pe.OUT / "seeds_e.parquet") if p.exists()}
    ab = pd.concat([pe.read(d / "RESIDUAL_RESULTS.csv") for d in (pe.REP / "har_xgboost", pe.REP / "har_lightgbm")])
    bd = pd.read_csv(pe.ENS / "HYBRID_BREAKDOWN.csv", comment="#")
    cols = ["id"] + ["qlike_y%d" % y for y in range(2017, 2027)] + ["qlike_calm_g4", "qlike_stress_g4", "qlike_low_rv", "qlike_mid_rv", "qlike_high_rv"]
    f = lambda d, c=None: d.to_markdown(index=False, floatfmt=".4f") if hasattr(d, "to_markdown") else d.to_string(index=False)  # noqa: E731
    secs = ab[["id", "crit_wf", "qlike_all"]].assign(k=ab.get("secs", 0))
    return "\n".join([
        "# HYBRIDS — гибриды HAR-J(б) × XGBoost × LightGBM (SBER, h=1, волна E, Ж-210)", "", "Статус: %s. Выбор — по внутреннему критерию `crit_wf` (причинный), p — сырые и Холм с N = %d (постселекционная оговорка)." % (
            "HISTORICAL WALK-FORWARD 2271 дата 2017–02.2026, просмотрено; INDEPENDENT FUTURE TEST не выполнялся", pe.n_trials()), "",
        "## Формулы", "- A (ансамбль прогнозов): σ̂² = Σ w_i σ̂²_i, w на симплексе (шаг 0,1); w(d) — функция закрытых прогнозов (target ≤ d); fixed_loyo — оптимум по целям < 01.01.Y, фиксирован в году Y;",
        "  inv_qlike_252, inv_rmse_252 — w ∝ 1/QLIKE (1/RMSE) за 252 закрытых; roll_W — оптимум по последним W; shrink_252 — (1−s)·w_roll + s·w_eq, s по последним 126.",
        "- B (residual): σ̂² = HAR·exp(g(X)); обучение — QLIKE (gamma, log-link, offset ln HAR + c0), XGBoost base_margin / LightGBM init_score; B·m — log-MSE на остатке; B1a — max(HAR + g, ε_PIT).",
        "- C (адаптивные): C2 вес ML по терцили RV_w, C3 по знаку ΔRV_w(5), C4 по стрессу Г4, C5 σ̂² = exp(ln HAR + c_t·g), C6 переключение HAR/ML по сумме потерь за 63.", "",
        "## Источники (sha256[:16])", "\n".join("- `%s`: %s" % (k, v) for k, v in sha.items()), "- компоненты: `data/regen/xgb_lgbm_full_v1/e/components.parquet` (пути исходных рядов — `xgb_lgbm_ensemble/EX0_REFERENCE.csv`)", "",
        "## Временная схема", "Для даты d: признаки на d; lb_t — PIT HAR-J(б); обучение на парах (t, t+1) с t+1 ≤ d; ежедневное переобучение, расширяющееся окно; веса — по закрытым; затем RV d+1 попадает в историю.", "",
        "## Итоговая таблица (п. 11 Ж-210)", t11_md, "## Выбор A–F (R9)", f(sel), "", "## ACCEPT/REJECT (R8)", f(ar[["id", "role", "reference", "c1_crit", "c2_2017_2021", "c3_years", "years_won", "c4_seeds", "verdict", "p_holm"]]), "",
        "## Веса по датам", "`har_xgboost/WEIGHTS_BY_DATE.csv`, `har_lightgbm/…`, `har_xgb_lgbm/…`, `xgb_lgbm_ensemble/WEIGHTS_BY_DATE.csv`, `xgb_lgbm_ensemble/ADAPTIVE_WEIGHTS_BY_DATE.csv`.", "",
        "## Результаты по годам и режимам", f(bd[bd["id"].isin(list(fin) + ["HAR", "XJ", "LF"])][cols]), "",
        "## Абляции и диагностика B", f(ab[["id", "qlike_all", "crit_wf", "added_value_vs_best_component"]]), "Диагностика «не воспроизводит ли бустинг HAR»: `har_*/RESIDUAL_DIAGNOSTICS.csv`.", "", rmse_md(df), holm_md(tests) if tests is not None else "",
        "## Контроль утечек", "Тесты: `tests/test_v2_xl_e_rules.py` (будущее не меняет вес; мутации «закрыто+1», «Y+1», «s по будущему», «Холм без N»), `tests/test_v2_xl_e_bfit.py` (градиент/гессиан = конечным разностям и встроенному gamma; будущие цели не меняют прогноз на d), `tests/test_v2_hyb_resid.py` (lb PIT), `tests/test_v2_xl_e_holm_pit.py` (Холм с N = 182 из реестров; offset `lb` = ln `s2_harj` до 1e-12), `tests/test_v2_xl_e_status.py` (статус гибрида без допуска R9 ≠ ACCEPT).", "",
        "## Вычислительная стоимость", "Правила A, C — секунды; B (2271 ежедневное переобучение, 4 процесса): `har_*/RESIDUAL_RESULTS.csv`, колонка secs в кэше `b_*.parquet`.", "",
        "## Журнал испытаний и конфигурации", "`xgb_lgbm_ensemble/OPTUNA_TRIALS.csv` (trial_id = хэш конфигурации), `CHECKPOINT.json`; прогнозы — `data/regen/xgb_lgbm_full_v1/e/series_{a,b,c}.parquet`, `seeds_e.parquet`.", ""])


def repro_md() -> str:
    cmd = "\n".join("python3 -W ignore models_v2/s6/e/run_e.py --stage %s" % s for s in ("comp", "ex0", "ex1", "a", "b", "c", "seeds", "e3", "e4", "e5"))
    return ("# REPRODUCTION — волна E (Ж-206 + Ж-210)\n\nПорядок (по одной, процессов ≤ 4): \n```\n%s\n```\nКэши: `data/regen/xgb_lgbm_full_v1/e/` (ключ = код + вход + seed). Проверка: `python3 -m pytest tests/test_v2_xl_e_*.py -q`.\n"
            "Ожидаемые значения (EX0): HAR-J(б) 0,2337; ctrl_f4 0,212538; x2_best_j 0,211295; C0_inh_F4 0,2117; Б-2 0,2205; LGBM-аналог 0,2223; c2_h3_roll_f4 0,2098 (допуск 5e-5). Детерминизм: seed 42–46, `deterministic=True`; B-ряд пересчитывается в `--stage e4` побитно.\n" % cmd)
