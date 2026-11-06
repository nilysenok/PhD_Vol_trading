# GATES_D_lgbm — волна D, поток LGBM (Ж-206), начато 06.11.2026

Задание: `docs/zadaniya/2026-11-05_xgb_lgbm_full.md` (волна D, поток LGBM); блок — `..._blok.md` §§ 16–17, 19–20, 30. Владение: `models_v2/s6/lgbm/d_*.py`, `runld.py`, `tests/test_v2_xl_lgbmd_*.py`, `docs/otchety/xgb_lgbm_full_v1/lightgbm/` (файлы D, реестр, CHECKPOINT), `data/regen/xgb_lgbm_full_v1/lightgbm/d/**`. Запуск: `python3 models_v2/s6/lgbm/runld.py --stage win|pick|freq|r1|r4|report` (процессов ≤ 4, без коммита, стадии по одной в фоне). Правила не двигаются после просмотра результата (изменение — строкой «↺» с датой и обеими версиями). Статус результатов: DEVELOPMENT / HISTORICAL WALK-FORWARD (2017–02.2026, уже просмотрено). Первая цифра — потолок.

## Правила, объявленные ДО счёта (06.11)
- **RD0 (база, критерий, бюджет).** База — `C0_inh_F4` (HP И2 по годам + Б-2 + F4; crit_pen 0,194400; QLIKE 0,2117) из каталога C, читается. Расширяющееся окно, ежедневно = эта же конфигурация (контроль, не trial, не пересчитывается). crit_pen как в B/C (годовой LOYO, pen_Y = mean + 0,25·(max − mean), среднее по Y = 2017…2026). Рядом `d_crit_pen_vs_base`; `d_qlike_vs_ref`, `d_crit_pen_vs_ref` — к baseline LGBM `lb0_inh` (0,2223). DM (Ньюи–Уэст лаг 5): к HAR-J(б), к базе C0_inh_F4 и к `lb0_inh`. Реестр: direction `temporal` (окна, частота) и `regime` (R1, R4); считаются в `temporal_regime` (лимит 60, потока LGBM ≤ 30); `hp`/`obj_feat` не трогаются (33/22). Контроли (`control`) не регистрируются.
- **RD1 (D1, окна).** Единица окна — пары (t, t+1) с t+1 ≤ d. Конфигурации (7 trials): rolling 252 / 504 / 756 / 1008 / 1260 последних пар; экспоненциальные веса (не окно) полураспад 756 строк (3 года) и 1260 строк (5 лет): w = 0,5^((d − t)/hl) (`s4/fit.weights`). 252 из блока оставлено, хотя Ж-204 B2 показал rolling 252–1008 хуже на XGBoost Б-2 — поток LGBM независим. Окна > 1260 не берутся (список блока). **Фолды LOYO для окна:** обучающие пары фолда (все пары с целями < 01.01.Y кроме года y) → W пар, ближайших по времени к центру года проверки (|t − центр|), а для распада w = 0,5^(|t − центр|/hl); в прогнозе на дату d центр = d, и это ровно «последние W пар» / `fit.weights`. При W ≥ числа пар фолд совпадает с базой (тест). Недостаток: фолд проверяет «локальность по времени в обе стороны», а прогноз — только прошлое (ограничение записывается в отчёт). «Окно по прошлому» (`D_pick_window`, 1 trial): для года Y — окно из 8 (расширяющееся + 7 выше) с min pen_Y (цели < 01.01.Y), прогноз года — из ряда этого окна; потолок (min по 8 кандидатам, выбор и оценка на тех же фолдах). Доступность пар по годам (`D_PAIRS_BY_YEAR.csv`): число пар к 01.01.Y и доля фолдов, где окно меньше числа пар.
- **RD2 (D2, частота).** «Лучшее окно» = конфигурация среди 8 (D1) с min crit_pen. Частота: ежедневно (= ряд из D1), еженедельно (блок = календарная неделя ISO сетки), ежемесячно (календарный месяц сетки): модель обучается на первой дате блока (пары t+1 ≤ якорь, HP года якоря) и прогнозирует все даты блока своими X[d]. Адаптивное обновление (`adaptive`): переобучение в начале каждого календарного месяца **и** в день смены режима стресс Г4 (stress[d] ≠ stress[d−1]). 3 trials. Фолды LOYO не чувствительны к частоте, поэтому критерий частоты — `crit_wf`: для Y = 2018…2026 pen_Y по годовым QLIKE ряда самой конфигурации за годы 2017…Y−1 (только реализованные цели < 01.01.Y), crit_wf = среднее pen_Y. **Порог (объявлен): ежедневное оправдано, только если crit_wf(ежедневно) < crit_wf(еженедельно) − 0,0005; еженедельное — если crit_wf(еженедельно) < crit_wf(ежемесячно) − 0,0005; иначе выбирается более редкая частота.** Время обучения (`secs`, число обучений) — колонки. crit_wf считается на тех же датах, что описание, поэтому как выбор — потолок.
- **RD3 (D3-R1, режимные признаки, 4 trials).** К F4 добавляются: R1a — stress Г4 (stress_p3, определён с 30.12.2016, раньше NaN); R1b — квинтиль 0…4 ранга rv_w[t] среди rv_w[≤ t] (расширяющийся, от 60 наблюдений, раньше NaN); R1c — «переход»: смена stress в последние 5 строк (t−4…t); R1d — все три. HP — И2 по годам (как база), расширяющееся окно, ежедневно; признак режима — только данные ≤ t (тесты). stress Г4 — готовый ряд `data/regen/v2_sber_h1_deep_v1/.../stress_p3.parquet` (общий для потоков, не пересчитывается; допущение «Г4 по данным ≤ d» унаследовано от Ж-204 и здесь не перепроверяется — в отчёт).
- **RD4 (D3-R4, переключение двух конфигураций по прошлому режиму, 2 trials).** R4a: стресс Г4 на дату d → ряд rolling 1008, спокойно → расширяющееся (пример блока); R4b: для года Y и режима r ∈ {спокойно, стресс} — окно из 8 (D1) с min средним по фолдам (Y, y) QLIKE на днях проверки режима r (цели < 01.01.Y), на дату d берётся ряд выбранного окна для режима стресс[d] (потолок: выбор и оценка на одних фолдах). crit_pen составной конфигурации — из фолдовых QLIKE по датам (хранятся) той же сборкой. Метрики переключения (`REGIME_PERFORMANCE_D.csv`): число смен режима и смен активной конфигурации; доля дней, когда выбранная конфигурация хуже невыбранной ex post (ошибки переключения — описательно, не критерий); задержка реакции = медиана по началам стресса (первый день серии stress) числа дней до первой даты, где σ̂²/RV ≥ 0,5 (кэп 20); QLIKE и доля занижения в первые 5 дней после входа в стресс и после выхода; QLIKE в спокойные/стрессовые дни. Дни без ряда стресса (до 30.12.2016, в фолдах — годы проверки 2015–2016) считаются «спокойными». Режимы, определяемые по будущей RV, не используются; R2/MoE не повторяются (Ж-204 B1).
- **RD5 (сравнения).** Каждая строка: crit_pen, crit_wf, QLIKE на 2271 дате и срезы (2017–21 «потолок: кандидаты видели 2017–2021», 2022, 2023–25, 2026, без 24.02.2022, без 2022, спокойные/стресс), DM, статус HISTORICAL WALK-FORWARD. «Перспективно»: crit_pen < crit_pen базы И Δ к HAR-J(б) без 24.02.2022 ≤ 0. Какие окна лучше в спокойном и в стрессовом режиме — `D_WINDOWS_CALM_STRESS.csv`. Старые прогнозы и таблицы не перезаписываются.
- **Итого trials D ≤ 7 + 1 + 3 + 4 + 2 = 17 ≤ 30.** Расширение после просмотра — только строкой «↺» с датой.

## Гейты
### D1 — код и тесты
CHECK: `python3 -m pytest tests/test_v2_xl_lgbmd_win.py tests/test_v2_xl_lgbmd_reg.py tests/test_v2_xl_lgbmd_rules.py -q`
EXPECT: passed; окно = последние W пар и без пар после d; распад — известный ответ; W ≥ числа пар = база побитно; прогноз блока не зависит от строк после якоря; режимные признаки не зависят от rv_w/stress после t; переключатель берёт флаг даты d; годовой выбор по pen своего года; порог частоты; реестр (31 > 30 и дубль — ошибка; контроли не регистрируются); главный тест обрезки на окна/распад/блок; файлы ≤ 200 строк, функции ≤ 40.
EVIDENCE: 06.11 до счёта: `pytest tests/test_v2_xl_lgbmd_{win,reg,rules}.py -q` -> 20 passed, rc 0; файлы d_*.py ≤ 174 строк (wc -l).
### D2 — окна и распад посчитаны (стадия win)
CHECK: `python3 models_v2/s6/lgbm/runld.py --stage win`
EXPECT: rc 0; 7 конфигураций `D_w*` в `data/.../d/`; `D_PAIRS_BY_YEAR.csv`; `D_WINDOWS_CALM_STRESS.csv`; +7 строк `temporal` в OPTUNA_TRIALS.
EVIDENCE: pending
### D3 — окно по прошлому (стадия pick)
CHECK: `python3 models_v2/s6/lgbm/runld.py --stage pick`
EXPECT: rc 0; `D_pick_window`; +1 `temporal`; `TEMPORAL_VALIDATION.csv` строки D1 (9 строк: база + 7 + pick).
EVIDENCE: pending
### D4 — частота (стадия freq)
CHECK: `python3 models_v2/s6/lgbm/runld.py --stage freq`
EXPECT: rc 0; 3 конфигурации (weekly, monthly, adaptive) на лучшем окне; `D_FREQUENCY.csv` с crit_wf, временем, числом обучений и вердиктом порога RD2; +3 `temporal`.
EVIDENCE: pending
### D5 — режимные признаки (стадия r1)
CHECK: `python3 models_v2/s6/lgbm/runld.py --stage r1`
EXPECT: rc 0; 4 конфигурации R1a–R1d; +4 `regime`.
EVIDENCE: pending
### D6 — переключение (стадия r4)
CHECK: `python3 models_v2/s6/lgbm/runld.py --stage r4`
EXPECT: rc 0; R4a, R4b; +2 `regime`.
EVIDENCE: pending
### D7 — отчёт-таблицы
CHECK: `python3 models_v2/s6/lgbm/runld.py --stage report`
EXPECT: rc 0; `TEMPORAL_VALIDATION.csv` (все строки D, статус), `REGIME_PERFORMANCE_D.csv`; `README_D.md` с таблицами и выводом; CHECKPOINT: used.temporal_regime ≤ 30, used.hp = 33, used.obj_feat = 22; trial_id уникальны.
EVIDENCE: pending
### D8 — прежнее не тронуто
CHECK: `shasum -a 256 docs/otchety/xgb_lgbm_full_v1/lightgbm/BASELINE.json data/regen/xgb_lgbm_full_v1/lightgbm/fc_lgbm_b2.parquet`; `git status --short models_v2 tests docs/otchety/xgb_lgbm_full_v1/lightgbm | head -50` (правки только в d_*, runld, test_v2_xl_lgbmd_*, файлах D, OPTUNA_TRIALS, CHECKPOINT)
EXPECT: sha BASELINE.json `62160aca…2488`, fc_lgbm_b2 `7cf07022…78ce`; без правок в `models_v2/s6/xgb`, `xgboost/`, C-файлах.
EVIDENCE: pending
### D9 — ответ дирижёру
CHECK: ручной гейт: ≤ 500 слов, цифры перемерены из CSV/JSON; окна (спокойно/стресс), частота и вердикт порога, режимные строки, 2022, расход trials; потолки помечены.
EVIDENCE: pending

## Мутации -> тест
| № | мутация | тест |
|---|---|---|
| MD1 | окно центрируется на d+1 / берёт пару с t+1 > d | `test_v2_xl_lgbmd_win.py::test_window_is_last_w_pairs_and_mutation_caught` |
| MD2 | веса распада перевёрнуты (старое тяжелее) | `...::test_decay_known_answer_and_mutation_caught` |
| MD3 | якорь блока позже дат блока (будущий) | `...::test_block_forecast_invariant_to_rows_after_anchor`; `::test_blocks_anchor_first_and_mutation_caught` |
| MD4 | признак режима с полной выборкой / сдвигом +1 | `test_v2_xl_lgbmd_reg.py::test_quintile_pit_and_mutation_caught`; `::test_transition_pit_and_mutation_caught` |
| MD5 | переключатель берёт флаг d+1 / выбор окна по pen другого года | `...::test_switch_uses_flag_of_own_day_and_mutation_caught`; `test_v2_xl_lgbmd_rules.py::test_pick_uses_only_pen_of_own_year` |
| MD6 | crit_wf включает год Y | `test_v2_xl_lgbmd_rules.py::test_crit_wf_uses_only_past_years_and_mutation_caught` |
| MD7 | порог частоты 0 (шум принимается); бюджет 31 / дубль / контроль в реестре | `...::test_frequency_threshold_*`; `::test_register_budget_and_controls` |
| MD8 | главный тест обрезки: окно/распад/блок на испорченных строках после d | `test_v2_xl_lgbmd_win.py::test_day_window_invariant_to_rows_after_d`; `::test_fold_window_invariant_to_rows_after_cut_and_mutation_caught` |
| MD9 | W ≥ числа пар не равно базе | `...::test_unbounded_window_equals_base_bitwise` |
