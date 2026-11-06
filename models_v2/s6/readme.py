"""README «Волна A» (docs/otchety/xgb_lgbm_full_v1/README.md): все числа читаются из CSV/JSON/PREDICTIONS; стадии readme (запись+проверка) и check-readme (A5, A15)."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from models_v2.s6 import paths6, readme_txt as T

R = paths6.REP
ROWS_A = [("fx_b2_g132_b", "Б-2"), ("b2_f4", "Б-2+F4"), ("c1_opt", "F4-Optuna"), ("c2_h3_roll_f4", "0,2098 (C)"), ("c3_f3calm_rv", "0,2101 (C)"), ("c3_f3calm_g4", "0,2101 (C)"), ("e3_roll", "e3_roll (D)")]


def f4(x) -> str:
    return ("%.4f" % x).replace(".", ",")


def table(head: list, rows: list) -> list:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def sec_inventory() -> list:
    df = paths6.read_csv(R / "INVENTORY.csv")
    cls = df["model_class"].str.split(" ").str[0].value_counts()
    nv = int(df["verified"].str.startswith("не подтверждено").sum())
    return ["## Волна A", "", "Статус: " + paths6.STATUS + ". Новых trials бюджета — 0 (`CHECKPOINT.json` обоих алгоритмов). Первая цифра — потолок: подбор и счёт на одних данных.", "",
            "### Инвентаризация", "", "Файл: `INVENTORY.csv` (источник: прогнозы `data/regen/v2_sber_h1*`, код `models_v2/`). Строк: %d; не подтверждено по коду/пересчёту: %d. Классы (по первому слову): %s."
            % (len(df), nv, ", ".join("%s %d" % (k, v) for k, v in cls.items())), ""]


def sec_answers() -> list:
    df = paths6.read_csv(R / "INVENTORY.csv").set_index("id")
    rows = [[lab, "`%s`" % i, df.loc[i, "model_class"], df.loc[i, "library"], df.loc[i, "origin_refs"].split(" | ")[0].split("; ")[0], f4(df.loc[i, "qlike_recomputed"])] for i, lab in ROWS_A]
    out = ["### Ответы (а)–(в)", "", "**(а) Алгоритм по коду (R1), не по названию.** Источник: `INVENTORY.csv` (`origin_refs` — файл:строка вызова обучения).", ""]
    out += table(["строка", "id", "алгоритм", "библиотека/версия", "файл:строка кода", "QLIKE (пересчёт)"], rows)
    out += ["", "Смеси: `c2_h3_roll_f4` = HAR-J(б) OLS + Б-2+F4 (XGBoost), `e3_roll` = HAR-J(б) OLS + `d_lgbm_f4` (LightGBM), `c3_f3calm_*` = Б-2+F4 + top2 (F4+F3), оба XGBoost. Версия xgboost/lightgbm при обучении не сохранена — указана текущая.", "", T.ANSWER_B, ""]
    a = paths6.read_csv(R / "ANSWERS_V.csv")
    out += ["**(в) Проверка «p = 0,038» и 0,045.** Источник: `ANSWERS_V.csv`, пересчёт `answers.py` из сохранённых прогнозов (расхождение с таблицами < 1e-16).", ""]
    out += table(["волна", "тест", "база", "N семейства", "фактических тестов", "единиц до N", "сырое p", "p Холма"],
                 [[r.wave, r.test, r.base, r.n_family, r.n_tests, r.padding_ones, "%.6f" % r.raw_p, "%.6f" % r.holm_p] for r in a.itertuples()])
    out += ["", "Строка во всех случаях `c2_h3_roll_f4` против Б-2 (не против HAR-J(б)); семейство C: 19 строк `C_ALL` + 71 единица = 90; семейство D: 6 финалистов (`c2_h3_roll_f4`, `c1_opt`, `c2_h3_roll_opt`, `e3_roll`, `b2_f4`, `c3_f3calm_rv`) + 100 единиц = 106 (состав — заголовок `ANSWERS_V.csv`). Множители Холма берутся с N, а не с числа фактических тестов, потому 0,038 и 0,045 — оценки при предрегистрированном N; потолок: отбор строки шёл на тех же данных.", ""]
    return out


def sec_baseline() -> list:
    xs, ls = paths6.read_csv(R / "xgboost/XGB_SEEDS.csv"), paths6.read_csv(R / "lightgbm/LGB_SEEDS.csv")
    bx, bl = (json.loads((R / a / "BASELINE.json").read_text()) for a in ("xgboost", "lightgbm"))
    row = lambda n, s, b: [n, f4(s.qlike_all.iloc[0]), f4(s.qlike_all.mean()), f4(s.qlike_all.std(ddof=1)), f4(s.qlike_all.min()), f4(s.qlike_all.max()), ("%.1f" % b["n_estimators"]["mean"]).replace(".", ",")]
    out = ["### Baseline", "", "Источник: `xgboost/BASELINE.json`, `lightgbm/BASELINE.json` (поля из `fit._params`, `lgbfit.params`, таблицы I2 по годам), `XGB_SEEDS.csv`, `LGB_SEEDS.csv` (5 seed 42–46: контроль R6, не выбор, не trials).", ""]
    out += table(["baseline", "QLIKE seed 42", "среднее 5 seed", "std", "min", "max", "деревьев (среднее по датам)"], [row("XGBoost Б-2 `fx_b2_g132_b`", xs, bx), row("LightGBM-аналог Б-2 (наследованные HP)", ls, bl)])
    d42 = ls.qlike_all.iloc[0] - xs.qlike_all.iloc[0]
    out += ["", "Пометки: LightGBM baseline — аналог Б-2 в том же конвейере с НАСЛЕДОВАННЫМИ HP (I2 из сетки 132 XGBoost); собственного подбора у LightGBM нет (ответ (б)). Для XGBoost распределение k по датам не сохранено — только среднее (не подтверждено).",
            "Разность LightGBM − XGBoost (seed 42) = %s QLIKE; разброс по seed у каждого ~%s (std) — разность сопоставима с шумом seed; различие алгоритмов этой цифрой не доказано." % (f4(d42), f4(max(xs.qlike_all.std(ddof=1), ls.qlike_all.std(ddof=1)))), ""]
    return out


def sec_diag() -> list:
    out = ["### Диагностика", "", "Источник: `<алг>/PREDICTIONS.parquet`, `ANNUAL_PERFORMANCE.csv`, `REGIME_PERFORMANCE.csv`; база сравнения — HAR-J(б); режимы PIT (по данным до d) отдельно от «по факту, не PIT».", ""]
    rows = []
    for alg in ("xgboost", "lightgbm"):
        an = paths6.read_csv(R / alg / "ANNUAL_PERFORMANCE.csv").set_index("slice")
        for s in ("2017–2021", "2022", "2023–2025", "2026", "весь период", "без 24.02.2022", "без топ-20 дней (по |ΔQLIKE к HAR|)"):
            rows.append([alg, s.replace("|", "/"), int(an.loc[s, "n"]), f4(an.loc[s, "qlike"]), f4(an.loc[s, "delta_vs_har"]), ("%.1f%%" % (100 * an.loc[s, "share_better_than_har"])).replace(".", ","), f4(an.loc[s, "bias_ln"])])
    out += table(["алгоритм", "срез", "n", "QLIKE", "Δ к HAR-J(б)", "доля лучше HAR", "bias ln"], rows)
    out += ["", "Годы 2017…2026 по каждому алгоритму — в `ANNUAL_PERFORMANCE.csv`. Режимы (PIT: квинтили RV_w, стресс Г4, переходы; по факту: терцили RV, скачки, завышение/занижение) — `REGIME_PERFORMANCE.csv`.", ""]
    return out + sec_reasons() + sec_lgb()


def sec_reasons() -> list:
    pr = pd.read_parquet(R / "xgboost/PREDICTIONS.parquet")
    m = (pr["date"].dt.year <= 2021).to_numpy()
    dl = (pr["QLIKE_XGB"] - pr["QLIKE_HAR"]).to_numpy()
    dm = dl[m]
    top = np.argsort(-np.abs(dm))[:20]                      # топ-20 внутри 2017–2021
    keep = np.ones(len(dm), bool)
    keep[top] = False
    e = lambda c: abs((np.log(pr["RV_actual"]) - np.log(pr[c]))[m].mean())
    hy = paths6.read_csv(paths6.OLD_REP / "HYPOTHESES_A.csv")
    ver = "; ".join("%s — %s" % (r.id, r.verdict) for r in hy.itertuples())
    return ["**Проигрыш XGBoost в 2017–2021 (пересчитано из PREDICTIONS, не скопировано).** Σ(QLIKE Б-2 − QLIKE HAR) на 2017–2021 = %.4f; без топ-20 дней по |Δ| = %.4f; доля топ-20 в Σ|Δ| = %.4f; |средний ln-bias| Б-2 минус HAR = %s (2017–2021). "
            "Вердикты гипотез прежней волны A (`docs/otchety/sber_h1_deep_v1/HYPOTHESES_A.csv`, прочитаны из файла): %s. Причины — оценки, не доказательства." % (
                dm.sum(), dm[keep].sum(), np.abs(dm[top]).sum() / np.abs(dm).sum(), f4(e("XGB_forecast") - e("HAR_forecast")), ver), ""]


def sec_lgb() -> list:
    ex = paths6.read_csv(R / "lightgbm/LGB_EXTRA.csv")
    lv = paths6.read_csv(R / "lightgbm/LGB_LEAVES.csv")
    v = ex[ex.kind.astype(str).str.startswith("leafwise_")].set_index("variant")
    sd = ex[ex.kind == "seeds_42_46"].set_index("metric").loc["qlike_all"]
    c99, d5 = (ex[ex.kind == k].ln_change.abs() for k in ("extreme_clip99", "extreme_drop5"))
    out = ["**LightGBM: специфика** (`LGB_LEAVES.csv`, `LGB_EXTRA.csv`, `LGB_SEEDS.csv`; 12 фиксированных дат: первая дата каждого года и 22.02.2022).", "",
           "- Листья: медиана числа наблюдений в листе по датам — %.0f (min по листьям %.0f); доля листьев < 2×min_data_in_leaf — %.2f (от %.2f до %.2f)." % (lv.leaf_median.median(), lv.leaf_min.min(), lv.share_lt_2min.mean(), lv.share_lt_2min.min(), lv.share_lt_2min.max()),
           "- Экстремальная RV: потолок цели 99-м процентилем меняет ln-прогноз в среднем на %s (max %s); без 5 наибольших целей — на %s (max %s) по |Δ|." % (f4(c99.mean()), f4(c99.max()), f4(d5.mean()), f4(d5.max())),
           "- Seed 42–46: QLIKE среднее %s, std %s, min %s, max %s." % (f4(sd["mean"]), f4(sd["std"]), f4(sd["min"]), f4(sd["max"])),
           "- Leaf-wise против ограниченных деревьев (контроли, не trials): base %s; S1 (`max_depth=-1`, `num_leaves=2^depth`) %s; S2 (`max_depth=-1`, `num_leaves=31`) %s; деревьев %s / %s / %s." % (
               f4(v.loc["base", "qlike_all"]), f4(v.loc["S1", "qlike_all"]), f4(v.loc["S2", "qlike_all"]), *(("%.1f" % v.loc[k, "trees_mean"]).replace(".", ",") for k in ("base", "S1", "S2"))), ""]
    return out


def sec_tail() -> list:
    out = ["### Предложения для волны B", "", "**Уже сделано для XGBoost — не повторять:**", ""] + ["- " + t for t in T.DONE_XGB]
    out += ["", "**Не делалось для LightGBM:**", ""] + ["- " + t for t in T.TODO_LGB]
    out += ["", "### Мутация -> тест (A4, A6, A8, A10, A12; каждая проверена в копии, падает на мутации и проходит на правке)", ""] + table(["гейт", "мутация", "тест"], [list(m) for m in T.MUTATIONS])
    return out + ["", "### Отступления", ""] + ["- " + t for t in T.DEVIATIONS] + [""]


def build() -> str:
    return "\n".join(sec_inventory() + sec_answers() + sec_baseline() + sec_diag() + sec_tail())


def problems(text: str) -> list:
    """A5/A15: разделы, формулировки, p Холма в тексте = ANSWERS_V."""
    need = ["## Волна A", "### Инвентаризация", "### Ответы (а)–(в)", "**(а)", "**(б)", "**(в)", "### Baseline", "### Диагностика", "### Предложения для волны B", "Уже сделано для XGBoost", "Не делалось для LightGBM",
            "### Мутация -> тест", "потолок", "НАСЛЕДОВАННЫМИ", "A4", "A6", "A8", "A10", "A12"]
    bad = ["нет: " + n for n in need if n not in text]
    bad += ["слово «%s» без пометки" % w for w in re.findall(r"видимо|наверное", text)]
    a = paths6.read_csv(R / "ANSWERS_V.csv")
    bad += ["нет p %.6f" % p for p in a.holm_p if "%.6f" % p not in text]
    bad += ["XGBoost seed-разброс не в тексте"] if "XGB_SEEDS" not in text else []
    return bad


def stage(a) -> int:
    p = R / "README.md"
    if a.stage == "readme":
        p.write_text(build(), encoding="utf-8")
    bad = problems(p.read_text(encoding="utf-8"))
    print("A5/A15 README: нарушений %d %s" % (len(bad), bad))
    return int(bool(bad))
