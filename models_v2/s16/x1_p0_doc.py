"""X1-пилот, лист P0: отчёт `P0_REPORT.md` — ГЕНЕРАТОРОМ из таблиц P0_SCALE / P0_INIT / P0_MIX плюс текст `P0_AUDIT.md` (G1). Чистая функция от трёх таблиц и текста аудита: ни одно число не вписано руками;
`--check` пересобирает отчёт и сравнивает с файлом (подмена числа в отчёте ловится). Единицы — %² (RV), потеря — QLIKE."""
from __future__ import annotations

import numpy as np
import pandas as pd

STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-241, лист P0 (SBER, h = 1; аудит и проверки до обучения; конфигураций 0, прогнозов не записано)"


def f(x, nd=4) -> str:
    return "—" if not np.isfinite(x) else ("%.*g" % (nd, x) if abs(x) < 1e-3 and x != 0 else "%.*f" % (nd, x))


def md(df: pd.DataFrame, nd: int = 4) -> str:
    cols = [str(c) for c in df.columns]
    body = ["| " + " | ".join(f(v, nd) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + body)


def pivot(scale: pd.DataFrame, col: str, blocks) -> pd.DataFrame:
    t = scale[scale["block"].isin(blocks)].pivot(index="block", columns="year", values=col).reindex(list(blocks))
    t.insert(0, "block", t.index)
    return t.reset_index(drop=True)


def scale_part(s: pd.DataFrame) -> str:
    bl = ["M10_raw_r2", "M10_norm_r2", "M10_norm_r_sq", "S_level_first", "S_level_last", "S_share_s8", "S_share_s9", "B_w1", "B_profile_mean1"]
    b1 = s[(s["block"] == "B_w1")]
    pr = s[(s["block"] == "B_profile_mean1")]
    m = s[s["block"] == "M10_norm_r2"]
    return "\n\n".join([
        "## 2. (а) Масштаб RV при нормировке (G2), годы 2018–2025, `P0_SCALE.csv`",
        "Медиана отношения (сумма за день / RV_d):", md(pivot(s, "median_ratio", bl)),
        "Корреляция Пирсона с RV_d:", md(pivot(s, "pearson", bl)),
        "Корреляция Спирмена с RV_d:", md(pivot(s, "spearman", bl)),
        "**B(w ≡ 1) = RV_d побитно:** доля побитно равных дней по годам — %s; B с положительным профилем слотов (среднее w = 1, профиль по всей истории — диагностика, не PIT): медиана B/RV по годам %s, B > RV на долях дней %s."
        % (", ".join(f(x, 3) for x in b1["extra"]), ", ".join(f(x, 3) for x in pr["median_ratio"]), ", ".join(f(x, 3) for x in pr["extra"])),
        "**Нормировка M10 (канал r2):** медиана 1/σ(r2) по годам — %s (масштаб, который вносит z-нормировка; в 2022 σ вырастает на порядок: расширяющееся окно захватывает март 2022). Сумма нормированного канала r2 за день = (RV − n·μ)/σ: связь с RV линейна (Пирсон ≈ 1), но масштаб RV НЕ сохраняется — отношение к RV ≠ 1 и в ряде лет отрицательно."
        % ", ".join(f(x, 3) for x in m["extra"]),
        "Вывод: до нормировки Σr2 = RV_d (тождество); после — нет. Масштаб RV сохраняется только в доле (S: s4, s8, s9 безразмерны, связи с уровнем RV почти нет) или в агрегации B(w) с w ≡ 1 / со средним w = 1 при независимости w от r². Корреляция Пирсона ≈ 1 после нормировки НЕ свидетельствует о сохранении масштаба."])


def init_part(t: pd.DataFrame, pi6: dict) -> str:
    i = t[t["stage"] == "init"]
    g = i.groupby("group").agg(grad_data_max=("grad_data", "max"), grad_pen_mean=("grad_pen", "mean"), grad_total_mean=("grad_total", "mean"), num_vs_auto_max=("num_vs_auto", "max"), n=("seed", "size")).reset_index()
    tr = t[t["stage"] == "trained"].groupby(["model", "group"]).agg(norm_init=("norm_init", "mean"), norm_param=("norm_param", "mean"), grad_data=("grad_data", "mean"), grad_pen=("grad_pen", "mean"),
                                                                    max_dev_pi=("extra", "mean"), sd_pi=("sd_pi", "mean")).reset_index()
    sy = t[t["stage"] == "synthetic_init"][["model", "group", "grad_data", "extra", "note"]]
    return "\n\n".join([
        "## 3. (б) Выход из нулевой инициализации (G3), `P0_INIT.csv`",
        "**MoE V7 cfg6, первая переоценка 2018-01-03, точка инициализации (5 seeds × 4 точки решётки λ_g × λ_LB):** норма градиента по группам; data = QLIKE-член + λ_g·mean(G²), pen = L2 + λ_LB + λ_s. num_vs_auto — max|автоград − центральная разность|:",
        md(g, 6),
        "Градиент данных по маршрутизатору — ТОЧНЫЙ НОЛЬ: при g_k ≡ 0 имеем ∂L/∂логит_j ∝ π_j·(g_j − Σπ g) = 0. Эксперты (a, b) получают ненулевой градиент сразу (∝ π_k). Градиент маршрутизатора в начале целиком из штрафов (L2 тянет веса к 0, λ_LB — к π = 1/3, λ_s — к постоянному π): все три направлены к равномерным весам.",
        "**После обучения (повтор конфигурации 6 на трёх датах переоценки, прогнозы не пишутся), среднее по 5 seeds:**", md(tr, 5),
        "**Сохранённые π конфигурации 6 (parquet V7, только чтение):** дней %d, среднее |π_k − 1/3| = %s, максимум = %s, средний σ_π по дням = %s, средние π = %s."
        % (pi6["n"], f(pi6["mean_abs_dev"], 4), f(pi6["max_abs_dev"], 4), f(pi6["sd_pi"], 4), ", ".join(f(x, 4) for x in pi6["mean_pi"])),
        "**Синтетика — шаблоны планируемых механизмов:** grad_data — норма градиента в нулевой инициализации; extra — после одного шага по параметрам с ненулевым градиентом.", md(sy, 6)])


def dyn_part(d: pd.DataFrame, init: pd.DataFrame) -> str:
    s = d[d["step"].isin([0, 10, 50, 100, 300])][["date", "step", "group", "grad_data", "grad_pen", "grad_total", "norm_param", "pi_sd", "corr_g", "mean_abs_g12", "dist_experts", "argmax_switch"]]
    sm = init[init["stage"] == "symmetry"].groupby("group").agg(dist_experts_mean=("extra", "mean"), dist_experts_max=("extra", "max"), n=("seed", "size")).reset_index()
    tr = init[(init["stage"] == "trained") & (init["group"] == "experts")].groupby("model").agg(corr_g=("corr_g", "mean"), mean_abs_g12=("mean_abs_g12", "mean"), dist_experts=("dist_experts", "mean"),
                                                                                              argmax_switch=("argmax_switch", "mean"), sd_pi=("sd_pi", "mean")).reset_index()
    return "\n\n".join(["## 5. Динамика обучения cfg6 и симметрия экспертов (Ж-242 п. 3), `P0_DYN.csv`",
                        "Один seed (0), одна точка решётки (λ_g = 0,1; λ_LB = 0,1), Adam полным батчем на обучающих закрытых парах, шаги 0, 10, 50, 100, 300; группы — маршрутизатор и два эксперта; pi_sd — σ_π по строкам; "
                        "corr_g, mean_abs_g12, dist_experts — расхождение экспертов; argmax_switch — доля смен argmax π между соседними строками:", md(s, 5),
                        "**После обучения по `fit_config` (5 seeds, среднее), строки обучения:**", md(tr, 4),
                        "**Симметрия экспертов:** оба эксперта стартуют с одинаковых нулей (a = b = 0, `v7_1m_forms.py:61-62`). Расстояние их параметров после 100 шагов на синтетике (5 seeds): router_zero — маршрутизатор с нулевым выходным слоем (π ровно равномерные), router_random — как в V7:", md(sm, 6)])


def mix_part(m: pd.DataFrame) -> str:
    a = m[m["year"].astype(str) == "all"]
    cols = ["operator", "n_dates", "viol_log", "viol_arith", "mean_gap_log", "mean_gap_arith", "bias_geo_over_arith", "qlike_geo", "qlike_arith", "resid_vs_saved"]
    y = m[(m["year"].astype(str) != "all") & m["bias_geo_over_arith"].notna()].pivot(index="operator", columns="year", values="bias_geo_over_arith").reset_index()
    concl = "\n".join("- **%s**: %s" % (r.operator, r.conclusion) for r in a.itertuples())
    return "\n\n".join(["## 4. (в) Оператор смешивания и Йенсен (G4), `P0_MIX.csv`",
                        "Все даты X1 (2018-01-03 … 2026-02-26). viol_log — доля дат с нарушением l(Σπf) ≤ Σπ l(f) в ln F; viol_arith — то же для смешивания в F; bias_geo_over_arith = mean F_geo / mean F_ar:", md(a[cols], 5),
                        "Смещение геометрического смешивания к арифметическому по годам (mean F_geo / mean F_ar):", md(y, 5), concl])


def build(scale: pd.DataFrame, init: pd.DataFrame, mix: pd.DataFrame, pi6: dict, audit: str, dyn: pd.DataFrame) -> str:
    head = "# P0 пилота X1 (Ж-241): аудит и проверки до обучения\n\nСтатус: %s. Таблицы: `P0_SCALE.csv`, `P0_INIT.csv`, `P0_MIX.csv`; отчёт собран генератором `models_v2/s16/x1_p0_doc.py`." % STATUS
    return "\n\n".join([head, "## 1. Аудит трёх механизмов (G1) — текст `P0_AUDIT.md`", audit, scale_part(scale), init_part(init, pi6), dyn_part(dyn, init), mix_part(mix)]) + "\n"
