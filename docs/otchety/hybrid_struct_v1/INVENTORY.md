# INVENTORY — Ж-212, этап 0: что из пп. 3–8 блока уже посчитано (SBER h=1)

Статус: HISTORICAL WALK-FORWARD 2017–02.2026 (2271 дата, просмотрено; не независимый OOS); DEVELOPMENT. Без trials. Источники — файлы репозитория на 06.11.2026; «ключ» — ключ строки `docs/otchety/hybrid_final_v1/ALL_TRIALS.csv` (521 строка, N_prior Ж-211). Реестр в `docs/otchety/hybrid_struct_v1/ALL_TRIALS.csv` будет создан как побайтная копия первых 521 строк при старте этапа B (сейчас не копируется — новых trials нет).

## Контрольные значения (п. 2 блока) — сверены, `CONTROL_VALUES.csv`
HAR-J 0,233662 · XGB `x2_best_j` 0,211295 · LGBM `C0_inh_F4` 0,211671 · A1 HAR+XGB (`A1_roll_126`) 0,208140 — совпали с заявленными 0,2337 / 0,2113 / 0,2117 / 0,2081 до 4-го знака (n = 2271, 2016-12-30 … 2026-02-26 по d). Определение QLIKE — `models_v2/metrics.py::qlike`. Происхождение: `data/regen/xgb_lgbm_full_v1/e/components.parquet` (SHA-256 в `CONTROL_VALUES.csv`; файл в git не отслеживается — `git status` показывает `??`, поэтому коммит как таковой отсутствует; фиксируется SHA и конфигурации: `comp.NAMES` — HAR = HAR-J(б) PIT, XJ = `x2_best_j` (trial `f4e8e88d8a05`, признаки Б-2 + F4 = 33), LF = `C0_inh_F4`) и `series_a.parquet` (A1).

## Пункты 3–8: сделано / новое

| п. блока | Что требуется | Уже посчитано (файл) | Новое в Ж-212 |
|---|---|---|---|
| 3. ежедневная таблица | RV, три прогноза, потери, разность, компоненты HAR, J, полувариации, рыночная RV, состояние, признаки набора | `hybrid_final_v1/HAR_RESIDUAL_DIAGNOSTICS.csv` (2271 строка: rv, HAR, XJ, LF, e, q, dq, rv_d/w/m, доли скачка, rs±, 22 признака F1–F6, терциль RV_w) | `data/regen/hybrid_struct_v1/ERROR_STRUCTURE_DAILY.parquet` (105 столбцов): все 51 признак набора F1–F6, ln RV_d/w/m, ln(1+J), RS±, асимметрия, **уровень RV рынка (IMOEX)** d/w/m и PIT-терциль, режим RVI, стресс Ж-210, расхождения δ, **прошлые ошибки/потери/расхождение только по закрытым прогнозам** |
| 3.1–3.2 когда лучше HAR / бустинг | — | `HAR_ERROR_STRUCTURE.csv` (условия 1–7 по пунктам, ex post), `YEARLY_STRUCTURE.csv` | `Q1_Q2_WHEN.csv`: положение факта относительно прогнозов, знак и размер δ, вклад хвостовых дней |
| 3.3 признаки преимущества | — | Ж-210 EX1: AUC знака dq 0,44–0,62; ridge остатка HAR по F-группам (`HAR_RIDGE_BY_YEAR.csv`); Ж-211 `REGIME_INTERACTION.csv` (6 режимов × 2 ML, Вальд) — ABANDON | `Q3_PREDICTORS.csv`: 83 признака (в т.ч. рынок, δ, закрытые прошлые ошибки), HAC, Холм, устойчивость по периодам |
| 3.4 низкая/высокая RV | — | `HAR_ERROR_STRUCTURE.csv` п. 7 (терцили RV_w, рынок) | `Q4_Q6_SLICES.csv`: терцили SBER и IMOEX, RVI, стресс, знак доходности |
| 3.5 смещение | — | `HAR_ERROR_STRUCTURE.csv` (mean e), `ANNUAL_PERFORMANCE.csv` (bias_ln) | `Q4_Q6_SLICES.csv`: mean e с t (HAC) и mean(RV/σ̂²) по трём моделям и срезам |
| 3.6 корреляция ошибок | — | `YEARLY_STRUCTURE.csv` (corr_e_HAR_XJ, corr_e_XJ_LF по годам), Ж-210 `ERROR_CORRELATION.csv` | `Q4_Q6_SLICES.csv`: ещё корреляция потерь, по срезам, скользящая 252 |
| 3.7 разность прогнозов | — | не считалось | `Q7_DELTA_INFO.csv`, `Q8_BETA_PROFILE.csv` |
| 3.8 условная ценность HAR сверх бустинга | — | **не считалось**: A1–A4 Ж-210 — комбинации прогнозов, не энкомпассинг; `ALL_TRIALS` не содержит такого теста | `Q8_ENCOMPASSING.csv`, `Q8_OOS_BY_YEAR.csv` (QLIKE-согласованный HAC-тест и OOS по прошлому) |
| 4. HAR as Features (B) | HAR-прогноз/компоненты как признаки XGB/LGBM | **не встречается в реестре** (проверка дирижёра; подтверждено: Ж-204 `har_boosting` — остаток к HAR/offset, не признаки; Ж-206/210 `feature`, `obj_feat` — группы F1–F6 без HAR-прогноза) | весь этап |
| 5. Residual QLIKE (C) | f = HAR·exp(g), QLIKE-градиент/гессиан | Ж-204 H2-L4 (ключ `Ж-204:acd832df8c96fa27`, c = 1; `…:f14bdc11e221a8c9`, c = 0,5); Ж-210 `B1q` (`Ж-206:xgb_lgbm_ensemble:fec343a84fe5`), `B2q` (`…:6378404826d8`) — REJECT; Ж-211 R2 clip (`Ж-211:bounded:*`, 10), R3 λ, R23_wf, наборы признаков S3/S34/S35/S345 (`Ж-211:residual_xgb:*`, `residual_lgbm:*`, 24) — лучшие 0,2210 / 0,2193 (QLIKE), уровень 1 не достигнут | только конфигурации с записанным «архитектурным отличием» (см. `GATES_HS.md`, п. C); λ = 0 ≡ HAR и численная проверка производных — `tests/test_v2_hs_qfit.py` |
| 6. Component Hybrid (D) | абляция компонент HAR, нелинейные g_k | **не делалось**; единственное близкое — Ж-211 ridge по F-группам и `FEATURE_SELECTION.csv` | весь этап; предварительно Q8 (по компонентам) — в `DIAGNOSTICS_A.md` |
| 7. Conditional correction (E) | a_t по прошлому | Ж-210 адаптивные C2–C6 (`Ж-206:xgb_lgbm_ensemble:…` direction `adaptive`, база B2q + XJ); Ж-211 `R2_wf`/`R3_wf`/`R23_wf` (выбор c, λ по закрытым); режимы V1–V6 — ABANDON | E1–E3 с интерпретируемым показателем и контролем «самостоятельный бустинг» |
| 8. MoE (F) | веса HAR/XGB/LGBM | Ж-210 A1–A4 (arith., постоянные/скользящие веса), Ж-211 `DYNAMIC_WEIGHTS.csv` (EMA, ограничение скорости, 10 trials); корреляция потерь XGB/LGBM 0,998 (`Q4_Q6_SLICES.csv`: 0,9983) | только геометрические смеси/Hedge и тест условной информации третьего эксперта (не повторяет арифметические) |

## Ограничения
- Все числа этапа A — на той же просмотренной истории: p и «выигрыш в выборке» помечены потолком; OOS «по прошлому» — это walk-forward на тех же 2271 датах, не независимый OOS.
- Ж-211 `REGIME_INTERACTION.csv` не пересчитывался; `HAR_ERROR_STRUCTURE.csv` использует ex post условия (диагностика, не правило прогноза).
