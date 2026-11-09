"""V7-0, G2: аудит PIT/OOF сохранённых базовых прогнозов K1, K2, K3 (и ссылки на аудит NTP2, A2, E2). Выводы опираются на КОД, который строил ряды, и на файлы; числа считаются здесь из файлов (никакого обучения).
Код: K1 — `models_v2/har.py` (`pairs`, `forecast_at`, `eligible`); K2/K3 — `models_v2/s10/hp_k23_calc.py` (`forecast_frame`: ежедневное переобучение на расширяющемся окне, seed 42; HP года Y — `hp_k23_world.choose_i2`),
`models_v2/s6/lgbm/job.py::day_fc` (`pairs(F, sp, d)` — пары с t+1 ≤ d)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_ctrl as C, v7_0_io as IO


def k1_checks() -> dict:
    h = pd.read_parquet(IO.PAN / "har_SBER.parquet")
    d, t = pd.to_datetime(h["d"]), pd.to_datetime(h["target_date"])
    inc = np.diff(h["n_pairs"].to_numpy(float))
    return dict(n=len(h), first_pairs=int(h["n_pairs"].iloc[0]), last_pairs=int(h["n_pairs"].iloc[-1]), step_not_01=int(((inc != 0) & (inc != 1)).sum()), step_one=int((inc == 1).sum()), target_after_d=int((t > d).all()),
                min_gap_days=int((t - d).dt.days.min()))


def k23_checks() -> tuple:
    k = pd.read_parquet(IO.PAN / "k23_SBER.parquet")
    k = k[np.isfinite(k["sigma2_k2"]) & np.isfinite(k["sigma2_k3"])]       # последняя строка файла (t⁺ 2026-02-27) без прогноза K2/K3 — вне общих дат
    tp, dd = pd.to_datetime(k["t_plus"]), pd.to_datetime(k["date"])
    yr = tp.dt.year.to_numpy()
    same = int((k["hp_year"].to_numpy(float) == yr).sum())
    by = k.groupby("hp_year").agg(n=("date", "size"), first_d=("date", "min"), last_d=("date", "max"), k2=("hp_k2", "nunique"), k3=("hp_k3", "nunique")).reset_index()
    by["hp_cut"] = [pd.Timestamp(int(y), 1, 1).date().isoformat() for y in by["hp_year"]]
    by["cut_ok"] = [bool(pd.Timestamp(int(y), 1, 1) > pd.Timestamp(a)) for y, a in zip(by["hp_year"], by["first_d"])]
    return dict(n=len(k), hp_year_equals_year_tplus=same, design_rows=int(((yr >= 2017) & (yr <= 2021)).sum()), later_rows=int((yr >= 2022).sum()), t_after_d=bool((tp > dd).all())), by


def markdown() -> str:      # xtask: длинная — шаблон документа одним форматом
    a, (b, by) = k1_checks(), k23_checks()
    rows = "\n".join("| %d | %d | %s | %s | %d / %d | %s | %s |" % (r.hp_year, r.n, r.first_d.date(), r.last_d.date(), r.k2, r.k3, r.hp_cut, "да" if r.cut_ok else "НЕТ") for r in by.itertuples())
    return """# AUDIT0_V7 — PIT/OOF базовых прогнозов SBER (K1, K2, K3), лист 0 V7 (G2)

Статус: %s. Моделей не обучали, конфигураций 0; читались сохранённые прогнозы и код, который их строил. %s.

## Вывод (классы)
| Ряд | Вывод | Источник |
|---|---|---|
| K1 = HAR-J | **OOF.** Каждая дата d — отдельный OLS на расширяющемся окне, пары (x_t, y_{t+1}) только с t+1 ≤ d и внутри сегмента (без склейки через разрыв 2022); пересчёт `har.forecast_at` воспроизводит K1 побитно (V6-0). Форма HAR-J (ln RV d/w/m, ln(1+J)) задана литературой до данных, параметров, выбираемых по SBER, нет | `models_v2/har.py::pairs, forecast_at`; `docs/otchety/harj_hybrid_v6/AUDIT0_REPORT.md` п. 4 |
| K2 = XGBoost | **OOF с оговоркой.** Веса: ежедневное переобучение на расширяющемся окне, пары с t+1 ≤ d. Гиперпараметры года Y выбраны LOYO по целям < 01.01.Y (одни на весь год Y). **Оговорка: дизайн (набор признаков F4, сдвиги `x2_best_j` из X2_DECISION.json, ранняя остановка, обучатель) выбирался на SBER 2017–2021** (sber_h1_deep_v1; Ж-204/206/210) | `models_v2/s10/hp_k23_calc.py::k2_sigma, forecast_frame`; `hp_k23_world.py::choose_i2`; `docs/otchety/harj_panel_hybrid/K23_REPORT.md` («выбор видел 2017–2021»); `CMP_SERIES.csv` |
| K3 = LightGBM | **OOF с оговоркой**, как K2: HP наследуются от И2 XGB-года Y (`grid.from_i2`); дизайн F4 выбран на SBER 2017–2021 | `models_v2/s6/lgbm/job.py::day_fc`; `hp_k23_calc.py::k3_sigma` |
| V4:NTP2, V6:A2, V6:E2 | см. аудит V4/V6: параметры по закрытым парам (PIT-сопоставим); сами конфигурации выбраны после просмотра (потолок) | `docs/otchety/harj_hybrid_v4/` (V4-0 аудит), `docs/otchety/harj_hybrid_v6/AUDIT0_REPORT.md`, `CMP_SERIES.csv` |

## Проверки по файлам (пересчитаны при запуске)
- K1: %d строк; пар в обучении на первую дату %d, на последнюю %d; шаг n_pairs ∉ {0, 1}: %d раз (разрывы сегмента не дают скачков назад); target_date > d во всех строках: %s; минимальный зазор d → t⁺ = %d дн. Граница обучения на дату d: последняя пара (x_{d−1}, y_d), её цель = d ≤ d.
- K2/K3: %d строк; target_date > d во всех строках: %s; `hp_year` равен году t⁺ в %d из %d строк; HP года Y выбраны по целям до 01.01.Y, а d первого дня Y — не позже 31.12.Y−1 (колонка «cut_ok» ниже). Граница обучения на дату d: пары с t+1 ≤ d (`job.pairs(F, sp, d)` → `fit.pair_index(..., T=d)`); блок ранней остановки — внутри обучающих пар с эмбарго (`fit.MIN_ES`, `fit.EMBARGO`), будущего не видит (тест класса № 99/100: `tests/test_v2_fix_audit_train.py::test_final_model_matrix_has_pair_t_plus_1_eq_d_and_none_beyond`).

| hp_year | дат d | первая d | последняя d | разных HP (K2 / K3) | HP выбраны по целям до | cut_ok |
|---|---|---|---|---|---|---|
%s

## Что затронуто выбором дизайна на SBER 2017–2021
- Годы прогнозов t⁺ 2017–2021: **%d из %d дат** K2/K3 получены дизайном, подобранным на тех же годах → это потолок, не OOS. Годы 2022–2026 (%d дат): дизайн к ним не подбирался, но 2022+ просматривался программами V1–V6 (контроль гл. 3, «не независимый OOS»); честной остаётся только оговорка, не уровень.
- Следствие для V7: любое улучшение над K2/K3 на 2017–2021 нельзя читать как независимое; сравнение «против K2/K3» — по окну 2022–2026 и по разности при общем дизайне; новый корректор обязан использовать только пары t⁺ ≤ d (G5) и не подбирать дизайн по 2017–2021 вторично без записи в реестр.

## Ограничение аудита
Прогнозы не переобучались и не пересчитывались (бюджет V7 не утверждён): проверка границы — по коду, файлам, `n_pairs`, `hp_year` и тестам программ V4/V6. Побитное воспроизведение K2/K3 на SBER — `hp_k23.py` «после сверки SBER (побитно с components.XJ/LF)».
""" % (IO.STATUS, IO.CAVEAT, a["n"], a["first_pairs"], a["last_pairs"], a["step_not_01"], bool(a["target_after_d"]), a["min_gap_days"], b["n"], b["t_after_d"], b["hp_year_equals_year_tplus"], b["n"], rows,
       b["design_rows"], b["n"], b["later_rows"])
