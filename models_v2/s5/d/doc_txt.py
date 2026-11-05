"""D3: тексты — SBER_H1_BOOSTING_DEEP_IMPROVEMENT_FINAL.md, REPRODUCTION.md, блок волны D в README.md. Числа берутся из таблиц, вывод — по правилу."""
from __future__ import annotations

import importlib.metadata as im
import re

import numpy as np
import pandas as pd

from models_v2.s5 import common
from models_v2.s5.d import ctx_d, robust_d, tab_d

REP = common.REP
B2_Q = 0.220486502764   # Б-2, контроль (G1 проверяет равенство ≤ 1e-12)
FINAL, MARK = REP / "SBER_H1_BOOSTING_DEEP_IMPROVEMENT_FINAL.md", ("<!-- D:auto -->", "<!-- /D:auto -->")
ALPHA_CONV = 0.05   # ↺ объявлено при написании кода до просмотра D2: «убедительно» = p Холма (N=106) < 0,05 к Б-2 и по DM (весь период), и по бутстрапу


def f(x, n: int = 4, sign: bool = False) -> str:
    """Число с десятичной запятой; пусто/NaN — тире."""
    return "—" if x is None or not np.isfinite(x) else (("%+.*f" if sign else "%.*f") % (n, x)).replace(".", ",")


def verdict(top: dict, rk: pd.DataFrame) -> tuple:
    if not top["top3"]:
        return "Б-2 остаётся действующим контролем: ни одна строка не прошла правило D3 (perspective, seed-ok, plateau-ok, MCS).", False
    rows = [rk[rk.row_id == t].iloc[0] for t in top["top3"]]
    pas = [bool(r["p_holm_N106_dm_b2"] < ALPHA_CONV and r["p_holm_N106_boot_b2"] < ALPHA_CONV) for r in rows]
    s = "Лучшая устойчивая по порядку D3 — `%s` (QLIKE %s против %s у Б-2). p Холма (N=106) к Б-2 [DM / бутстрап]: %s. " % (
        rows[0]["row_id"], f(rows[0]["qlike_all"]), f(B2_Q), "; ".join("`%s` %s / %s" % (r["row_id"], f(r["p_holm_N106_dm_b2"], 3), f(r["p_holm_N106_boot_b2"], 3)) for r in rows))
    ok = pas[0]
    return s + ("Улучшение убедительно по объявленному порогу (p < 0,05 по обоим тестам) для первой строки тройки." if ok else
                "Первая строка тройки НЕ проходит объявленный порог (p < 0,05 по DM и бутстрапу): по правилу Б-2 остаётся контролем; порог проходят: %s." %
                (", ".join("`%s`" % r["row_id"] for r, p in zip(rows, pas) if p) or "никто")), ok


def diag_text(rids) -> str:
    t20 = pd.read_csv(REP / "D_TOP20.csv", comment="#")
    ly = pd.read_csv(REP / "D_LOYO.csv", comment="#")
    lw = pd.read_csv(REP / "D_LOWRV.csv", comment="#")
    rows = []
    for r in rids:
        a, y, w = t20[t20.row_id == r].iloc[0], ly[ly.row_id == r], lw[lw.model == r].iloc[0]
        y22 = y[y["year_excluded"] == 2022].iloc[0]
        rows.append("| %s | %s | %s | %s / %s | %d из %d / %d из %d | %d | %s | %s |" % (r, f(a["sum_abs_share"], 3), f(a["mean_without_top"], sign=True), f(y22["d_harj"], sign=True), f(y22["d_b2"], sign=True),
                                                                         y["sign_kept_harj"].sum(), len(y), y["sign_kept_b2"].sum(), len(y), w["n"], f(w["qlike"]), f(w["d_b2"], sign=True)))
    return "\n".join(["| Модель | доля топ-20 в Σ\\|Δ\\| (к HAR-J) | среднее Δ к HAR-J без топ-20 | Δ без 2022: HAR-J / Б-2 | знак Δ сохраняется при исключении года: HAR-J / Б-2 | нижний дециль: n | QLIKE | Δ к Б-2 |",
                      "|---|---|---|---|---|---|---|---|"] + rows)


# xtask: длинная — шаблон текста итогового отчёта (f-строка); логики в ней нет
def write_final(C, best, top, flags, rk, st, tabs, counts) -> None:
    v, ok = verdict(top, rk)
    nar = (REP / "D_NARRATIVE.md")
    nar = nar.read_text(encoding="utf-8") if nar.exists() else ""
    fin = robust_d.finalists(best)
    body = f"""# SBER h=1: глубокая доработка бустинга (Ж-204) — итоговый отчёт

Статус: разработка, до аудита; оценка 2017–02.2026 (2022+ — просмотр контроля гл. 3 по SBER), **не независимый OOS**; первая цифра — потолок (подбор и счёт на одних датах); множественный поиск: N = {tab_d.FAMILY} уникальных trials программы (справочно {tab_d.FAMILY_REF} с контролями). Числа — из csv этой папки.

## 0. Итог
{v}

Тройка лучших устойчивых (правило D3, `GATES_D.md` §4): {", ".join("`%s`" % r for r in top["top3"]) or "пусто"}. Допускаются строки: perspective И seed-ok И plateau-ok И MCS (p >= 0,10); строки `c3_f3calm_*` (калм-модель выбрана на оценочном периоде) показаны отдельно. Лучший ансамбль D1 по `Q_wf`: `{best}`; по `Q_wf_dev` (разработка 2017–2020): `{ctx_d.load_json("best.json", {})["best_dev"]}`.

{nar}

## 1. Таблица 1. Рейтинг (RMSE/MAE/R² — по логарифмам: ln RV − ln σ̂²; R² — регрессия Минцера–Заронвица в логарифмах)
{tabs["t1"]}

## 2. Таблица 2. Качество по периодам (QLIKE)
{tabs["t2"]}

## 3. Таблица 3. Вклад улучшений
{tabs["t3"]}

## 4. Таблица 4. По рыночным режимам (QLIKE; режимы PIT по 252 предыдущим, Г4 — стресс)
{tabs["t4"]}

## 5. Таблица 5. Стабильность
{tabs["t5"]}

## 6. Накопленная разность ежедневных потерь
`D_cum_vs_harj.png`, `D_cum_vs_b2.png` (данные — `D_CUMDIFF.csv`): ![к HAR-J(б)](D_cum_vs_harj.png) ![к Б-2](D_cum_vs_b2.png)

## 7. Диагностики (`D_TOP20.csv`, `D_LOYO.csv`, `D_LOWRV.csv`)
{diag_text(fin)}

## 8. Статистика и устойчивость
Полные таблицы: `STATISTICAL_TESTS.csv` (DM × 5 периодов × 2 базы, MCS, бутстрап, Холм N=106/116), `D_SEEDS.csv`, `D_PLATEAU.csv`, `D_WINDOW.csv`, `ENSEMBLE_RESULTS.csv`, `FINAL_RANKING.csv`, `MODEL_SPECIFICATIONS.json`. Счётчики trials: {"; ".join("%s %d" % kv for kv in sorted(counts.items()))} (всего {sum(counts.values())} из 240); D = 16 из 25.

## 9. Оговорки
1. 2017–02.2026 уже многократно просматривалось: это проверка исторической устойчивости, не новый OOS. 2. Лучший ансамбль выбран среди 13 по `Q_wf` на оценочном периоде — его 2017–2026 потолок; честная половина — выбор по `Q_wf_dev` и его результат на 2022–2026. 3. Для LGBM соседи HP не считались (HP наследуются от XGB-И2); охват — seeds. 4. Порог «убедительно» (p < 0,05 по DM и бутстрапу после Холма, к первой строке тройки) объявлен при написании кода D2 до просмотра его результатов. 5. Холм: семейства по 6 финалистов, недостающие до N — p = 1; подпериодные p сырые. 6. Торговые стратегии не пересчитывались.
"""
    FINAL.write_text(body, encoding="utf-8")


def write_repro(best: str) -> None:
    ver = {p: im.version(p) for p in ("numpy", "pandas", "xgboost", "lightgbm", "optuna", "matplotlib")}
    (REP / "REPRODUCTION.md").write_text(f"""# Воспроизведение SBER h=1 deep (Ж-204)

Окружение: Python 3.9; {", ".join("%s %s" % kv for kv in ver.items())}; `nthread/num_threads = 1`, seed 42 (seeds 43–46 — устойчивость). Кэш — с ключом = хэш входов и кода (`data/regen/v2_sber_h1_deep_v1/{{b1,b2,c,d}}`).

| Волна | Команда |
|---|---|
| A | `python3 models_v2/s5/run_a.py --stage all` |
| B1 | `python3 models_v2/s5/b1/run_b1.py --stage all` |
| B2 | `python3 models_v2/s5/b2/run_b2.py --stage all` |
| C | `python3 models_v2/s5/c/run_c.py --stage all` |
| D | `python3 models_v2/s5/d/run_d.py --stage all` (по стадиям: control, comp, d1, robust, d2, report) |
| Тесты | `python3 -m pytest tests/test_v2_*.py -q` |

Входы: `data/iss_check/{{bars_10m_full,index_10m}}`, `data/fastbt/s9/rvi_1850.parquet`, `data/regen/v2_sber_h1*` (хэши — `MODEL_SPECIFICATIONS.json`, `BASELINE_MANIFEST.json`). Контроль: Б-2 0,220486502764, HAR-J(б) 0,233662115694, 2271 дата. Лучший ансамбль D1: `{best}`. Перепроверка D: `python3 models_v2/s5/d/run_d.py --stage control` (код 1 при расхождении) и `pytest tests/test_v2_deep_d_*.py`.
""", encoding="utf-8")


def write_readme(best: str, top: dict, rk: pd.DataFrame, counts: dict) -> None:
    v, _ = verdict(top, rk)
    blk = f"""{MARK[0]}
## Волна D и итог программы (Ж-204)
**Итог.** {v} Тройка: {", ".join("`%s`" % r for r in top["top3"]) or "пусто"}. Лучший ансамбль D1 по `Q_wf`: `{best}`. Trials: {sum(counts.values())} из 240 (D: {len(tab_d.D_TRIALS)} из 25); Холм N={tab_d.FAMILY} (справочно {tab_d.FAMILY_REF}).
**Файлы D.** `SBER_H1_BOOSTING_DEEP_IMPROVEMENT_FINAL.md`, `FINAL_RANKING.csv`, `STATISTICAL_TESTS.csv`, `ENSEMBLE_RESULTS.csv`, `MODEL_SPECIFICATIONS.json`, `REPRODUCTION.md`, `D_SEEDS.csv`, `D_PLATEAU.csv`, `D_WINDOW.csv`, `D_TOP20.csv`, `D_LOYO.csv`, `D_LOWRV.csv`, `D_REGIMES.csv`, `D_CUMDIFF.csv`, `D_cum_vs_{{harj,b2}}.png`, `OPTUNA_TRIALS_D.csv`, `GATES_D.md` (EVIDENCE).
**↺ (до просмотра D2).** Порог «убедительно» — p Холма (N=106) < 0,05 к Б-2 по DM (весь период) и бутстрапу, применяется к первой строке тройки (порядок D3); остальные строки тройки показаны фактом; режимы таблицы 4 — квинтили RV_w по 252 предыдущим (низкая 0–1, средняя 2, высокая 3–4) и стресс Г4; RMSE/MAE/R² в таблице 1 — по логарифмам.
{MARK[1]}"""
    p = REP / "README.md"
    txt = p.read_text(encoding="utf-8")
    pat = re.compile(re.escape(MARK[0]) + r".*?" + re.escape(MARK[1]), re.S)
    p.write_text(pat.sub(lambda _: blk, txt) if pat.search(txt) else txt.rstrip("\n") + "\n\n" + blk + "\n", encoding="utf-8")
