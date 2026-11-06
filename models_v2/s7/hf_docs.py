"""Ж-211: документы SBER h=1 — таблица п. 17, HYBRID_ARCHITECTURE.md, REPRODUCTION.md, HYBRID_HAR_BOOSTING_FULL_RESEARCH_FINAL.md. Числа берутся из таблиц стадий, не набираются руками."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s7 import hf_data as D, hf_cmp as C, hf_ledger as L, hf_levels as V

ROWS17 = (("HAR-J", "HAR-J"), ("XGBoost", "XGBoost"), ("LightGBM", "LightGBM"), ("HAR + XGBoost", "H1"), ("HAR + LightGBM", "H2"), ("Residual XGBoost", "H3"),
          ("Residual XGBoost (причинное правило wf)", "H3_wf"), ("Residual LightGBM", "H4"), ("Residual LightGBM (причинное правило wf)", "H4_wf"),
          ("Adaptive Hybrid (режимы отклонены: C6/C5 Ж-210)", "H5"), ("HAR + XGBoost + LightGBM", "H7"))


STATUS_TXT = {"нет": "уровень 1 не достигнут; не принят", "L1 только сырой": "L1 только сырой (без Холма); EXPERIMENTAL, не заменяет принятую модель",
              "L1": "L1; EXPERIMENTAL, не заменяет принятую модель", "L2": "L2; кандидат для масштабирования"}


def h7_vs_h1(df, series, fin) -> str:
    """HF9: H7 допускается отдельным вариантом, только если A4/H7 лучше H1 при DM p < 0,10."""
    a, b = series["H7: " + fin["H7"]], series["H1: " + fin["H1"]]
    r = V.evalrows.vs(df["rv"].to_numpy(float), b, a, np.ones(len(df), bool))
    return "H7 против H1: Δ = %+.4f, DM p = %.3f → %s" % (r["d"], r["p"], "H7 допускается" if (r["d"] < 0 and r["p"] < 0.10) else "второй бустинг не добавляет (H7 не отдельный вариант)")


def md(df: pd.DataFrame, fmt="%.4f") -> str:
    h = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    f = lambda v: (fmt % v) if isinstance(v, (float, np.floating)) else str(v)
    return h + "\n".join("| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)) + "\n"


def table17(df, series, fin, t, L1) -> pd.DataFrame:
    rows = []
    for lab, key in ROWS17:
        name = key if key in series else next((k for k in series if k.startswith(key + ":")), None)
        if name is None:
            continue
        q = D.qs(df, series[name])
        r = {"Модель": lab + ("" if key in series else " [" + name.split(": ")[1] + "]"), "QLIKE": float(q.mean()), "crit_wf": C.crit_e.cw(q, df["target_date"])}
        for col, ref in (("DM против HAR (Δ; p сырой)", "HAR-J"), ("DM против XGB (Δ; p сырой)", "XGBoost")):
            dl = q - D.qs(df, series[ref])
            if name == ref:
                r[col] = "—"
                continue
            s = V.evalrows.vs(df["rv"].to_numpy(float), series[ref], series[name], np.ones(len(df), bool))
            r[col] = "%+.4f; p=%.3f" % (s["d"], s["p"])
        for col, ref in (("Годы с выигрышем (из 10) против HAR", "HAR-J"), ("против XGB", "XGBoost")):
            r[col] = "—" if name == ref else C.years_won(df, series[name], series[ref])
        r["Статус"] = "опора" if name in ("HAR-J", "XGBoost", "LightGBM") else STATUS_TXT[L1.set_index("model").loc[name, "verdict"]]
        r["Пометка"] = V.ceiling_note(name)
        rows.append(r)
    return pd.DataFrame(rows)


def why1718(w: pd.DataFrame) -> str:
    g = w.set_index("year")
    pick = lambda c: ", ".join("%d: %.2f" % (y, g.loc[y, c]) for y in g.index)
    return ("Отношение QLIKE(HAR)/QLIKE(XJ) по годам: %s. Вес HAR ex post (диагностика): %s. Корреляция ln-ошибок HAR/XJ: %s. "
            "Структура совместных ошибок не менялась (корреляция высока во все годы); менялась относительная точность: XJ учится с историей и в 2017–2018 мало пар."
            % (pick("ratio_HAR_over_XJ"), pick("w_HAR_expost_diag"), pick("corr_e_HAR_XJ")))


def write_all(df, series, fin, t, L1, cmp_, ann, tr, n_final, sk) -> None:
    t17 = table17(df, series, fin, t, L1)
    D.write(t17, "TABLE17_SBER_H1.csv", "п. 17 блока: SBER h=1; DM — двусторонний, НW лаг 5; Δ = QLIKE(модель) − QLIKE(опоры), минус — лучше")
    yearly = pd.read_csv(L.REP / "YEARLY_STRUCTURE.csv", comment="#")
    cp = L.load()
    used = L.used(cp)
    lvl = L1[["model", "L1_a", "L1_b", "L1_c", "L1_d", "L2_b", "L2_c", "L2_d", "L2_e", "verdict", "ceiling_note"]]
    tp = t[(t["period"] == "all") & (t["model"].isin(L1["model"]))][["model", "test", "delta", "p_raw", "boot_lo95", "boot_hi95", "p_holm"]]
    text = ["# Гибрид HAR-J + XGBoost / LightGBM — SBER h=1 (Ж-211)\n", "Статус: " + D.STATUS + "\n", "## 1. Таблица п. 17 (SBER h=1)\n", md(t17),
            "## 2. Уровни успеха (HF11)\n", md(lvl), "S* = %s. L1 — над HAR-J (QLIKE, crit_wf, 2 из 3 периодов, DM+бутстрап, Холм с N=%d); L2 — плюс над лучшим бустингом и не только 2017–2018.\n" % (sk, n_final),
            "## 3. Тесты A/B/C на всей истории\n", md(tp, "%.4g"), "## 4. Почему 2017–2018, а не после 2020\n", why1718(yearly) + "\n",
            "## 4а. H7 и корректор\n", h7_vs_h1(df, series, fin) + "\n\nНабор признаков лучших residual (`abl_S34_noF3`) — только F4 (квартичность, vov, range): абляция F3 из S34 улучшила crit_wf; наборы выбирались по EX1 (потолок).\n",
            "## 5. Бюджет и N\n", "Новых trials Ж-211: %s (всего %d ≤ 60). N_prior = %d, N для поправок = %d (верхняя оценка: пилот учтён по прогнозам INVENTORY).\n"
            % (used, sum(used.values()), L.counts(cp)["N_prior"], n_final), "## 6. Проверка обрезки\n", "Максимумы |Δ| при порче будущего: %s\n" % tr,
            "## 7. Оговорки\n", "Режимная конструкция отклонена (ABANDON HF-G8, REGIME_INTERACTION.csv). Финалист внутри семейства — min crit_wf среди кандидатов; фиксированные c и λ выбраны на всей "
            "истории (потолок), причинный вариант — `*_wf`. Данные 2017–02.2026 просмотрены; это не независимый OOS. Полная выборка MOEX (раздел 15 блока) — позже.\n"]
    (L.REP / "HYBRID_HAR_BOOSTING_FULL_RESEARCH_FINAL.md").write_text("\n".join(text), encoding="utf-8")
    arch(L.REP / "HYBRID_ARCHITECTURE.md", fin)
    repro(L.REP / "REPRODUCTION.md")


def arch(p, fin) -> None:
    p.write_text("""# HYBRID_ARCHITECTURE — механизмы A/B/C (Ж-211, SBER h=1)

Статус: %s

| | A. Комбинация прогнозов | B. Residual (корректор) | C. Условный (режимы) |
|---|---|---|---|
| Смысл | σ̂² = w·HAR + (1−w)·ML, вес по закрытым прогнозам | σ̂² = HAR·exp(g(X)); HAR — база, бустинг учит поправку | веса по режиму V1–V6 по прошлому |
| Формулы | w_t из `roll_126/252`; EMA: w_k = (1−α)w_{k−1}+αW_k, α = 1−2^(−1/h); лимит |Δw| ≤ r, нормировка на симплекс | R1: g = f(X), объектив QLIKE (градиент 1−RV·e^{−m}, гессиан RV·e^{−m}, m = ln σ̂²), offset ln HAR + c0; R2: clip(g, −c, c); R3: λ·g | `w_group`; тест взаимодействия Вальд (НW лаг 5) |
| Признаки | нет (только прогнозы) | F3 полувариации/доходности, F4 квартичность/vov/range, F5 IMOEX (наборы S345/S34/S35/S3) | V1–V6 на d, PIT-терцили по 252 предыдущим |
| Обучение | веса по закрытым прогнозам (target_date ≤ d) | ежедневный retrain, расширяющееся окно, пары t+1 ≤ d, HP годового выбора; c, λ — по среднему QLIKE закрытых прогнозов, до 252 закрытых — HAR-J | не строилось: взаимодействие не подтверждено |
| Источник информации | разнообразие ошибок HAR и ML | нелинейная зависимость остатка HAR от признаков | относительная точность по режиму |
| Сложность | 1–2 параметра правила | бустинг + 2 параметра ограничения | — |
| Риск | вес «подтягивается» к лучшему на выборке (2017–2018) | утечка через offset/признаки; редкие экстремальные поправки (смягчает R2/R3) | ложные режимы |

Финалисты по семействам (min crit_wf): %s. λ = 0 побитно даёт HAR-J (тест `test_lambda_zero_is_har_bitwise`).
""" % (D.STATUS, fin), encoding="utf-8")


def repro(p) -> None:
    p.write_text("""# REPRODUCTION — Ж-211, SBER h=1

Из корня репозитория, по одной стадии (процессов ≤ 6; retrain — spawn-пул 4):
```
python3 -W ignore -m models_v2.s7.run_hf --stage reg      # реестр, N_prior
python3 -W ignore -m models_v2.s7.run_hf --stage diag     # диагностика I
python3 -W ignore -m models_v2.s7.run_hf --stage bound    # R1/R2/R3
python3 -W ignore -m models_v2.s7.run_hf --stage inter    # тест режимов (ABANDON при отсутствии)
python3 -W ignore -m models_v2.s7.run_hf --stage weights  # EMA и лимит скорости
python3 -W ignore -m models_v2.s7.run_hf --stage feat     # retrain наборов признаков (≈40 с на ряд)
python3 -W ignore -m models_v2.s7.run_hf --stage final    # тесты, уровни, таблицы, документы
python3 -m pytest tests/test_v2_hf_rules.py tests/test_v2_hf_stages.py tests/test_v2_hyb_leak.py tests/test_v2_xl_e_bfit.py -q
```
Входы: `data/regen/xgb_lgbm_full_v1/e/` (components, b_B1q/B2q_s42, series_a/c), `feats_b2` (v2_sber_h1). Выходы: `docs/otchety/hybrid_final_v1/`, ряды — `data/regen/hybrid_final_v1/`.
Детерминизм: seed 42, блочный бутстрап (20, 5000, seed 42); кэш retrain — по хэшу кода и матрицы признаков.
""", encoding="utf-8")
