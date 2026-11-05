# GATES_B_lgbm — волна B, поток LGBM (Ж-206), написано 06.11.2026 00:45 MSK ДО счёта

Задание: `docs/zadaniya/2026-11-05_xgb_lgbm_full.md` (волна B, поток LGBM); блок — `..._blok.md` §§ 11–15, 30. Владение: `models_v2/s6/lgbm/**`, `tests/test_v2_xl_lgbm_*.py`, `docs/otchety/xgb_lgbm_full_v1/lightgbm/**`, `data/regen/xgb_lgbm_full_v1/lightgbm/**`. Статус результатов: DEVELOPMENT / HISTORICAL WALK-FORWARD (2017–02.2026, уже просмотрено; не независимый OOS). Первая цифра — потолок. Запуск: `python3 models_v2/s6/lgbm/run_b.py --stage lb0|loss|feat1|feat2|abl|arch` (процессов ≤ 4, без коммита). Правила и допуски ниже не двигаются после просмотра результата (изменение — строкой «↺» с датой и обеими версиями).

## Правила, объявленные до счёта
- **R1 (критерий).** Годовой LOYO как `s4/select.py`: для года выбора Y ∈ 2017…2026 фолды y ∈ [2015, Y−1], обучение — пары с целями < 01.01.Y кроме года y, HP года Y. `pen_Y = mean_y q + 0,25·(max_y q − mean_y q)` (q — QLIKE (б) фолда); **crit_pen = среднее pen_Y по Y**. Рядом `crit_mean` (без штрафа, как `b2/loyo`). Фолдов 65 на конфигурацию.
- **R2 (LB0).** Сетка LightGBM: num_leaves {7, 15, 31} × min_data_in_leaf {30, 100, 200} × lambda_l2 {2, 10, 30} × learning_rate {0,02; 0,05} × feature_fraction {0,5; 0,8} × bagging {0,8 (freq 1); 1,0 (freq 0)}, max_depth −1, ≤ 300 деревьев, ранняя остановка И1 (252 пары, эмбарго 1, терпение 100), seed 42 → 216 точек > 15 → случайная подвыборка 15 (`numpy.default_rng(42).choice(216, 15, replace=False)`, порядок по номеру) — **список (номер в itertools.product-порядке NL,MD,L2,LR,FF,BAG)**: №17 nl7·md30·l2=30·lr0,02·ff0,5·bag1; №18 nl7·md30·l2=30·lr0,02·ff0,8·bag0,8; №19 nl7·md30·l2=30·lr0,02·ff0,8·bag1; №42 nl7·md100·l2=30·lr0,02·ff0,8·bag0,8; №89 nl15·md30·l2=30·lr0,02·ff0,5·bag1; №111 nl15·md100·l2=10·lr0,05·ff0,8·bag1; №133 nl15·md200·l2=10·lr0,05·ff0,5·bag1; №145 nl31·md30·l2=2·lr0,02·ff0,5·bag1; №154 nl31·md30·l2=10·lr0,02·ff0,8·bag0,8; №157 nl31·md30·l2=10·lr0,05·ff0,5·bag1; №163 nl31·md30·l2=30·lr0,02·ff0,8·bag1; №177 nl31·md100·l2=10·lr0,02·ff0,5·bag1; №205 nl31·md200·l2=10·lr0,05·ff0,5·bag1; №207 nl31·md200·l2=10·lr0,05·ff0,8·bag1; №213 nl31·md200·l2=30·lr0,05·ff0,5·bag1. Выбор по каждому Y — минимум pen_Y среди 15 точек; ничья |Δ| ≤ 1e-12 — проще (меньше листьев, больше min_data, больше L2, меньше lr, меньше ff, bagging раньше). Признаки и цель как Б-2 (28 столбцов, ln RV_{t+1} − ln S, σ̂² = exp(off + f + s²/2)). Результат «LGBM baseline-own» (`lb0_own`) рядом с «baseline-inherited» (`lb0_inh`, пересчёт кодом потока наследованных HP I2). Оба — контроли; 15 точек — 15 trials направления `hp` (лимит потока 15 из 55).
- **R3 (потери, на LB0 own, признаки Б-2).** L1 — L2-регрессия по уровню y = RV/S, σ̂² = max(S·f, ε) (ε — PIT-граница `pit.eps_floor`; число срабатываний пишется); L2 — ln-цель = сам LB0 own (строка-ссылка, не trial); L3 — `objective=gamma` (log-link), σ̂² = exp(off + ln μ); L4 — custom QLIKE в лог-параметризации F = ln σ̂² (grad/hess `s5/b1/losses`, начальное значение ln mean y, ранняя остановка по QLIKE блока, веса применяются в objective); L5 — веса: L5a RV_w^−0,5, L5b RV_w^−1 (определения B1 Ж-204; L2 по ln-цели), L5t — полураспад 504 строки (`fit.weights`). Trials obj_feat: L1, L3, L4, L5a, L5b, L5t = 6. Устойчивость обучения (по каждой конфигурации): доля k = 1, доля k = потолок, число нечисловых ES-прогнозов, число срабатываний ε; σ̂² ≤ 0 или нечисловая — исключение (стоп).
- **R4 (признаки).** Группы F1–F6 и правило «дополнительной информации» (|corr| > 0,95 → столбец исключён) — `accepted.json` B2 Ж-204 без изменений (читаются, не пересчитываются): F1 9, F2 8, F3 8, F4 5, F5 8, F6 10 столбцов. Одиночные F1…F6 (6 trials) на LB0 own; затем `F_top2`, `F_top3` — две/три группы с лучшим crit_pen среди одиночных (механический выбор по внутреннему критерию, не по описанию); `F_all` (все 6 групп — чувствительность к числу признаков) → 3 trials; обратные абляции лучшего (по crit_pen) из {F1…F6, top2, top3, F_all}: DROP `no_imoex`, `no_market`, `no_semivar`, `no_gap` (4) + удаление каждой группы лучшего набора по одной (если групп > 1: до 6) → ≤ 10 trials. Ожидаемый итог obj_feat: 6 + 6 + 3 + ≤ 10 ≤ 25 ≤ 45. Дубль конфигурации (одинаковый trial_id) trial не считается. Вклад в спокойных/стрессовых днях: стресс Г4 по данным ≤ d (`stress_p3`), колонки `qlike_calm`, `qlike_stress`, `d_base_calm`, `d_base_stress`, `n_calm + n_stress = 2271`.
- **R5 («перспективно»).** crit_pen строки < crit_pen LB0 own **И** разность QLIKE к HAR-J(б) без 24.02.2022 ≤ 0. Ограничение: crit_pen LB0 own получен выбором лучшего из 15 точек на тех же фолдах → смещён вниз (потолок); строки F/L используют HP LB0 без нового подбора, поэтому сравнение «перспективно» к LB0 own честнее, чем LB0 own к inherited.
- **R6 (архитектура, контроли, не trials).** На лучшей конфигурации потока (min crit_pen среди LB0 own и всех строк F/абляций; потери — отдельный ряд, с признаками не скрещиваются): `A_depth3` (max_depth 3, 8 листьев), `A_depth4` (4, 16 листьев) — ограниченные деревья; `A_nl63` (63 листа, max_depth −1); `A_md_lo` (min_data ÷ 3, не меньше 5), `A_md_hi` (× 3); seed 43–46 (`arch_s43…46`, без фолдов); листья (наблюдений в листе, доля < 2·min_data) на 12 фиксированных датах: base / depth3 / nl63.
- **R7 (статистика).** DM (Ньюи–Уэст лаг 5, `s3/dm.dm`) к HAR-J(б) и к LB0 own, без поправки на множественность на этом шаге (описательные колонки; Холм с N — в финальной статистике). Колонки как Ж-204 волна B: весь, 2017–21, 2022, 2023–25, 2026, без 24.02.2022, без 2022, RV/σ̂² (`mean_r`), статус.
- **R8.** Старые прогнозы и таблицы не перезаписываются: `BASELINE.json`, `CHECKPOINT.json` (только пересчёт счётчиков), `OPTUNA_TRIALS.csv` (только дописывание) — sha до работы: BASELINE.json `62160aca…2488`, OPTUNA_TRIALS.csv (шапка) `4cb4fd45…e3d`, fc_lgbm_b2.parquet `7cf07022…78ce`.

## Гейты
### B1 — сетка и критерий (код)
CHECK: `python3 -m pytest tests/test_v2_xl_lgbm_grid.py -q`
EXPECT: passed; 216 точек; подвыборка = список R2 (номера выше); мутация «другой seed подвыборки» падает; `pen_year` известный ответ; штраф меняет выбор (мутация «штраф 0»); 65 фолдов, все y < Y; реестр без дублей; превышение бюджета поднимает ошибку.
EVIDENCE: pending

### B2 — функции потерь (код)
CHECK: `python3 -m pytest tests/test_v2_xl_lgbm_loss.py -q`
EXPECT: passed; градиент и гессиан L4 совпадают с конечными разностями (atol 1e-6); мутация «знак градиента» ловится; init_score входит в счёт custom objective (прогноз ≈ ln 3, не 2 ln 3); веса применяются в custom objective; `fit_l(kind='ln')` побитно равен `lgbfit.fit_lgbx` на HP, переведённых из I2; ε-граница считается; неизвестная потеря отвергается.
EVIDENCE: pending

### B3 — главный тест обрезки на реальном мире Б-2
CHECK: `python3 -m pytest tests/test_v2_xl_lgbm_leak.py -q`
EXPECT: passed; для kind ln / qlike и набора F-all прогноз на d (2018-06-01, 2022-06-01, 2025-06-02) побитно не меняется при порче X строк > d и y строк ≥ d; фолд LOYO (Y = 2020, y = 2018) не меняется при порче строк после среза; мутации «пары с целью d+1» и «срез на год позже» меняют результат (тест ловит).
EVIDENCE: pending

### B4 — пределы размера кода
CHECK: `python3 -m pytest tests/test_v2_xl_lgbm_limits.py -q`
EXPECT: passed (файл ≤ 200 строк, функция ≤ 40, нет `_test.`/`test_locked`/`.env` в коде потока).
EVIDENCE: pending

### B5 — LB0 посчитан
CHECK: `python3 models_v2/s6/lgbm/run_b.py --stage lb0`
EXPECT: rc 0; `data/regen/xgb_lgbm_full_v1/lightgbm/B/lb0_folds.parquet` 15·65 = 975 строк; `LB0_GRID.csv` 15 строк, `LB0_SELECTION.csv` 10 строк (Y 2017…2026); `fc_lb0_own.parquet`, `fc_lb0_inh.parquet` по 2271 строке; пересчёт inherited против сохранённого `fc_lgbm_b2.parquet` — max отн. разность < 1e-9 (иначе стоп и доклад: путь пересчёта не воспроизводит baseline A); QLIKE inherited = 0,2223 в пределах 5e-5; crit_pen own из `LB0_GRID` по выбранным точкам = crit_pen в `st_lb0_own.json` (допуск 1e-12); `OPTUNA_TRIALS.csv`: +15 строк направления `hp`; `CHECKPOINT.json` used.hp = 15.
EVIDENCE: pending

### B6 — функции потерь L1–L5 посчитаны
CHECK: `python3 models_v2/s6/lgbm/run_b.py --stage loss`
EXPECT: rc 0; `OBJECTIVE_RESULTS.csv` — 7 строк (lb0_own + L1, L3, L4, L5a, L5b, L5t), каждая 2271 дата, σ̂² > 0 и конечная, колонка статуса, колонки устойчивости (`share_k1`, `share_k_cap`, `n_nonfinite_es`, `n_eps_floor`); `OPTUNA_TRIALS.csv`: +6 `obj_feat`.
EVIDENCE: pending

### B7 — признаки: одиночные, комбинации, F-all, абляции
CHECK: `python3 models_v2/s6/lgbm/run_b.py --stage feat1 && python3 models_v2/s6/lgbm/run_b.py --stage feat2 && python3 models_v2/s6/lgbm/run_b.py --stage abl`
EXPECT: rc 0; `FEATURE_SETS_singles.csv` 7 строк (own + F1…F6), `FEATURE_SETS_combos.csv` 4 строки (own + top2, top3, F_all), `FEATURE_ABLATIONS.csv` ≥ 5 строк (own + 4 DROP [+ группы]); для каждой строки колонки `qlike_calm/stress`, `d_base_calm/stress` с `n_calm + n_stress = 2271`; `perspective` по R5; суммарно obj_feat ≤ 45.
EVIDENCE: pending

### B8 — диагностика архитектуры
CHECK: `python3 models_v2/s6/lgbm/run_b.py --stage arch`
EXPECT: rc 0; `ARCH_DIAGNOSTICS.csv` 10 строк (arch_base, A_depth3, A_depth4, A_nl63, A_md_lo, A_md_hi, arch_s43…46); `ARCH_LEAVES.csv` 36 строк (3 конфигурации × 12 дат); контроли в реестр trials не попадают (`direction=control`).
EVIDENCE: pending

### B9 — бюджет и реестр
CHECK: `python3 -m pytest tests/test_v2_xl_lgbm_grid.py -q -k "register or checkpoint"` и просмотр `docs/otchety/xgb_lgbm_full_v1/lightgbm/CHECKPOINT.json`
EXPECT: used.hp = 15 ≤ 15, used.obj_feat ≤ 45, trial_id уникальны в `OPTUNA_TRIALS.csv`.
EVIDENCE: pending

### B10 — прежнее не тронуто, владение
CHECK: `shasum -a 256 docs/otchety/xgb_lgbm_full_v1/lightgbm/BASELINE.json data/regen/xgb_lgbm_full_v1/lightgbm/fc_lgbm_b2.parquet; find models_v2 -newer docs/otchety/xgb_lgbm_full_v1/lightgbm/GATES_B_lgbm.md -type f -not -path 'models_v2/s6/lgbm/*' -not -path 'models_v2/s6/xgb/*' -not -name '*.pyc'`
EXPECT: sha BASELINE.json `62160aca…2488`, fc_lgbm_b2.parquet `7cf07022…78ce`; `find` пуст.
EVIDENCE: pending

### B11 — ответ дирижёру
CHECK: ручной гейт (команды нет): ответ ≤ 500 слов, цифры перемерены из CSV/JSON; QLIKE LB0 own, inherited, лучшие L и F, ARCH; «перспективно» по R5; потолок помечен; отступления отдельной строкой.
EVIDENCE: pending

## Мутации -> тест
| № | мутация | тест |
|---|---|---|
| M1 | знак градиента L4 перепутан | `test_v2_xl_lgbm_loss.py::test_mutation_wrong_sign_gradient_is_caught` |
| M2 | init_score выпал из счёта custom objective | `...::test_qlike_init_score_is_in_custom_objective_scores` |
| M3 | веса не применяются в custom objective | `...::test_qlike_weights_are_applied_in_custom_objective` |
| M4 | пары с целью d+1 в обучении (сдвиг, № 1) | `test_v2_xl_lgbm_leak.py::test_mutation_pairs_with_target_d_plus_1_is_caught` + `test_day_forecast_invariant_to_rows_after_d` |
| M5 | срез LOYO на год позже (текущий год в отборе, № 19/21) | `...::test_mutation_fold_with_year_Y_target_leaks`, `test_fold_invariant_to_rows_after_cut` |
| M6 | штраф за худший год снят | `test_v2_xl_lgbm_grid.py::test_mutation_penalty_changes_choice` |
| M7 | подвыборка LB0 с другим seed | `...::test_mutation_other_seed_changes_subsample` |
| M8 | ε-граница уровня не применена | `test_v2_xl_lgbm_loss.py::test_mutation_lvl_floor_counts_eps_hits` |
| M9 | ln-путь расходится с `lgbfit` (наследованные HP) | `...::test_ln_kind_equals_inherited_lgbfit_bitwise` |
| M10 | ключ веса (RV_w) использует строки после d | `test_v2_xl_lgbm_grid.py::test_weights_rvw_ignore_future_rows` |
| M11 | дубль trial считается дважды / бюджет превышен | `...::test_register_dedupes_and_checkpoint_limits` |
