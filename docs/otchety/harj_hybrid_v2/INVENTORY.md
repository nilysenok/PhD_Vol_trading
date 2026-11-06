# INVENTORY — Ж-214 HAR-J Hybrid V2, SBER h=1 (лист 1)

Что из блока владельца (`docs/zadaniya/2026-11-06_harj_hybrid_v2_blok.md`) уже сделано в проекте и где; что новое. Источники прочитаны: `docs/otchety/hybrid_struct_v1/` (Ж-212: `GATES_HS.md`, `ALL_TRIALS.csv` 541 строка, `CHECKPOINT.json`, `DIAGNOSTICS_A.md`, `COMPONENT_D.csv`), код `models_v2/{daily,har,metrics}.py`, `models_v2/s8/`, `models_v2/s5/b2/{feat,feat_day}.py`. Реестр по направлениям (`ALL_TRIALS.csv`): Ж-200/202 189 · Ж-204 106 · Ж-206/210 182 · Ж-211 44 · Ж-212 20 = **541**.

| № | Пункт блока | Статус | Где (файл / ключ реестра) | Новое в Ж-214 |
|---|---|---|---|---|
| 1 | Контроли HAR 0,2337 / XJ 0,2113 / LF 0,2117 / A1 0,2081 | сверено, заморожено | `hybrid_struct_v1/CONTROL_VALUES.csv`; `harj_qlike/CONTROL_VALUES.csv` (перемерено) | нет |
| 2 | Формула HAR-J(б), окна, переоценка (п. 4) | восстановлена, воспроизведена побитно | `models_v2/har.py`, `daily.py`; `harj_qlike/A0_SPEC.md`, `A0_FACTS.json` | нет (фиксация) |
| 3 | QLIKE-HAR-J с логарифмической связью (п. 5) | **сделано** в Ж-212 D0: `D0_full` QLIKE 0,2274, `D0_drop_a` 0,2287, формы `Dn_*`, `Dx_J` | `Ж-212:component:5d5e154cb6d9` (D0_full), `…:311098faa001` (D0_drop_a), `hybrid_struct_v1/COMPONENT_D.csv`, `models_v2/s8/hs_qfit.py`, `hs_dfit.py` | иные построения: линейные уровни с ограничениями (V5) |
| 4 | Остаток к HAR с QLIKE (g_ML, grad = 1 − RV/f, hess = RV/f) | **сделано** | Ж-204 `har_boosting` (20); Ж-206 `residual`; Ж-211 `residual_xgb/lgbm`, `bounded`; Ж-212 `residual_qlike` (Ca, Cc × XGB/LGBM) | другие признаки (M10-структура) и другая база (HARQ-J, V9) |
| 5 | HAR-признаки в бустинге | **сделано** | Ж-204 `har_boosting`; Ж-212 `har_features` (Bf, Bc × XGB/LGBM) | нет |
| 6 | Смеси, веса, режимы, ансамбли | **сделано** | Ж-206/210 `ensemble` (22), `ensemble_calibration`, `regime*`; Ж-211 `dyn_weights`, `bounded`; Ж-212 `moe` (F1, F2) | нет; согласование компонент (V11) — не смесь экспертов |
| 7 | Условная коррекция (E) | не запускалась (HS11/HS8) | Ж-212 `CONDITIONAL_E.csv` — строка-статус | нет |
| 8 | RQ (п. 7) как признак бустинга | **есть** (`f4_rq`, `f4_rq_rel` в F4 группе Б-2+F4) | `models_v2/s5/b2/feat.py`, `feat_day.py` (rq = n/3·Σr⁴, (%²)²) — M10-таблица Ж-214 совпадает (rtol 1e-9) | **реакция коэффициентов HAR-J на RQ (HARQ-J), V6** — не признак |
| 9 | Асимметрия RS−/RS+ | есть как признак (D0 `a`; Cc) | `hs_feat.py`, `hs_dfit.py` | RS± как **компоненты прогноза** (V7) |
| 10 | Разложение C + J, прогноз скачков | J — только регрессор HAR-J | `daily.build_daily` (J = max(RV − BV, 0)) | точное C = RV − J, прогноз C и J отдельно (V7) |
| 11 | OPEN / MIDDLE / CLOSE (п. 10) | **нет**. Есть только hhi долей RV по часам (`f4_hhi`) | `feat_day.py` | `m10_daily.parquet`: rv_open/mid/close по расписанию из фактических баров (V7, V8) |
| 12 | Семиковариации, реализованная корреляция (п. 16) | корреляция и бета SBER–IMOEX есть (`f5_corr_d`, `f5_corr22`, `f5_beta22`, `cross_stats`); семиковариаций нет | `feat.py` (F5), `feat_day.py::cross_stats` | cov_pp / cov_nn / cov_mix, семивариации рынка на синхронных барах (V10) |
| 13 | Сезонность, аукционы, шум (п. 15) | **нет** | — | `seasonality/DIAGNOSTICS_V.md` + CSV |
| 14 | Согласование прогнозов (п. 19) | **нет** | — | V11 |
| 15 | Статистика: DM (Ньюи–Уэст, лаг 5), блочный бутстрап (20/5000/42), MCS 90 %, Холм с N | готово | `models_v2/s5/b1/evalrows.py::vs`, `s5/d/stats_d.py::{boot_delta, mcs, holm_n}`; `s8/hs_levels.py` | N = 541 + новые |
| 16 | Реестр, лимиты, «архитектурное отличие» | готово для Ж-212 | `models_v2/s8/hs_ledger.py` | `models_v2/s9/hv2_ledger.py`: 541 + новые, лимиты 8/12/16/8/16/10/10 = 80 |

## Что считается новой информацией (не дубль Ж-204…Ж-212)
1. Коэффициенты HAR-J, зависящие от качества измерения (RQ, относительная ошибка RV).
2. QLIKE-оценивание **линейной в уровнях** параметризации с ограничениями; временные веса QLIKE-цели; шкала c по закрытым прогнозам вместо s²/2.
3. Прогноз компонент (RS±, C/J, OPEN/MIDDLE/CLOSE) и их согласование.
4. Сезонно скорректированный вход / отдельная версия цели (с прямым сравнением на исходной).
5. Семиковариации SBER–IMOEX; рыночная и идиосинкратическая части RV.

## Данные листа 1 (производные, `data/regen/harj_hybrid_v2/`)
- `A0_BETA.parquet` — коэффициенты и σ̂²_a, σ̂²_b A0 на 2271 дату.
- `m10_daily.parquet` — 2944 канонических дня × 35 столбцов; `grid_d` отмечает 2271 дату сетки; SHA-256 в `intraday_decomposition/M10_IDENTITIES.csv`.
- Входные файлы: `data/iss_check/bars_10m_full/SBER.parquet`, `data/iss_check/index_10m/IMOEX.parquet`, `data/regen/xgb_lgbm_full_v1/e/components.parquet` (SHA-256 в `A0_FACTS.json`).
