# GATES_C_lgbm — волна C, поток LGBM (Ж-206), начато 06.11.2026

Задание: `docs/zadaniya/2026-11-05_xgb_lgbm_full.md` (волна C, поток LGBM); блок — `..._blok.md` §§ 12, 15, 26, 30. Владение: `models_v2/s6/lgbm/c_*.py`, `runlc.py`, `tests/test_v2_xl_lgbmc_*.py`, `docs/otchety/xgb_lgbm_full_v1/lightgbm/` (файлы C), `data/regen/xgb_lgbm_full_v1/lightgbm/c/**`. Запуск: `python3 models_v2/s6/lgbm/runlc.py --stage l0|l1|l2|nbr|leaves` (процессов ≤ 4, без коммита). Правила не двигаются после просмотра результата (изменение — строкой «↺» с датой и обеими версиями). Статус результатов: DEVELOPMENT / HISTORICAL WALK-FORWARD (2017–02.2026, уже просмотрено). Первая цифра — потолок.

## Правила, объявленные ДО счёта C-L0 (06.11)
- **RC0 (C-L0, контроль, не trial).** Четыре конфигурации на признаках Б-2 + набор: {HP LB0 own, HP И2 (наследованные)} × {`F_top3` (F4+F3+F5), `F_F4`}. Собственные HP — уже посчитаны в B (`F_top3`, `F_F4`, читаются); досчитываются `C0_inh_top3`, `C0_inh_F4` (HP И2 → LightGBM как `grid.from_i2`). **База развёрток = конфигурация с min crit_pen из четырёх**; ничья |Δ| ≤ 1e-12 — собственные HP, затем top3. Файл выбора `data/regen/xgb_lgbm_full_v1/lightgbm/c/C_BASE.json`. Сравнение «признаки или гиперпараметры» — таблица `C_L0.csv`.
- **RC1 (критерий).** crit_pen как в B (R1 GATES_B): годовой LOYO, pen_Y = mean + 0,25·(max − mean), среднее по Y = 2017…2026; рядом crit_mean. Смещение: crit_pen выбранной базы получен выбором лучшего из 4 (и LB0 — из 15) на тех же фолдах -> потолок.

## Результат C-L0 и ОБЪЯВЛЕННАЯ БАЗА развёрток (06.11, до счёта C-L1)
crit_pen (чем меньше, тем лучше): `own_top3` 0,207852; `own_F4` 0,211243; `inh_top3` 0,209347; **`inh_F4` 0,194400**. QLIKE на 2271 дате: lb0_inh 0,2223; lb0_own 0,2270; own_top3 0,2175; own_F4 0,2171; inh_top3 0,2183; **inh_F4 0,2117** (2017–21 0,1530; 2022 0,4733; 2023–25 0,2377; 2026 0,2136). Файл `C_L0.csv`, выбор `c/C_BASE.json`.
**База развёрток C-L1 — `C0_inh_F4`: наследованные HP И2 по годам (сетка 132 XGBoost -> LightGBM как `grid.from_i2`, глубинно-ограниченные деревья nl = 2^depth, depth 2…5, bagging 0,6/0,8, ff 0,5/1,0) + признаки Б-2 + F4.** Вывод «признаки или гиперпараметры»: для F4 наследованные HP лучше собственных LB0 (0,1944 против 0,2112), для top3 — почти равны (0,2093 против 0,2079, собственные чуть лучше); LB0 own (15 случайных точек) гиперпараметры не улучшил. crit_pen базы — min из 4 -> потолок (смещён вниз).

## Правила C-L1, C-L2, seed/соседи, листья (объявлены ДО их счёта)
- **RC2 (C-L1, ≤ 18 trials, направление `hp`).** Базa — `C0_inh_F4`. Девять осей × две точки, применяются к HP каждого года выбора: `num_leaves` ÷2 (≥ 3) / ×2 (при max_depth > 0 hi поднимает глубину на 1 — иначе листья не добавятся); `max_depth` (база > 0) d−1 (≥ 2) / d+1 при прежнем num_leaves; `min_data_in_leaf` ÷2 (≥ 5) / ×2; `learning_rate` ÷2 / ×2 (≤ 300 деревьев сохраняются, `share_k_cap` пишется); `feature_fraction` −0,2 (≥ 0,3) / +0,2 (≤ 1,0); `bagging_fraction` 0,6 / 1,0 (1,0 -> bagging_freq 0); `lambda_l1` 1 / 5; `lambda_l2` ÷3 / ×3; `min_gain_to_split` 1 / 10. Точка, совпавшая во всех годах с базой, trial не считается. **Ось «с эффектом»: min Δcrit_pen к базе ≤ −0,001** (≈ 0,5 % от 0,194; задание говорит «< 0», порог ↺ введён мною до просмотра L1: Δ между 0 и −0,001 — шум 15 LOYO-фолдов и сама по себе не повод запускать L2).
- **RC3 (C-L2, ≤ 20 trials Optuna).** Только по осям с эффектом из RC2. TPE, seed 42, `n_startup_trials` 8, минимизация crit_pen. Пространство: для осей num_leaves, max_depth, min_data, lr, ff, l2 — u ∈ [−1; 1] (0 — база года, −1/+1 — нижняя/верхняя точка RC2, линейно по значению, целые округляются; num_leaves двигает и глубину); bag ∈ [0,6; 1,0], l1 ∈ [0; 5], gain ∈ [0; 10] — абсолютно. Нет осей с эффектом — ABANDON (`C_L2_ABANDON.md`). Дубль конфигурации trial не считается. Итого C ≤ 18 + 20 = 38 ≤ 40.
- **RC4 (seed и соседи, контроли, не trials).** Лучшие 2 конфигурации среди {база, C1_*, C2_*} по crit_pen. Для каждой: seed 43–46 (без фолдов, как B) -> 5 seed с 42; соседи ±1 шаг (RC2) по ≤ 4 осям (сначала оси, отличающиеся от базы, затем оси с лучшим эффектом RC2) -> crit_pen и QLIKE. Устойчивость: размах QLIKE по 5 seed; соседи не хуже базы более чем на 0,002 по crit_pen — «плато» (порог объявлен сейчас). Годовой выбор из кандидатов C (`C_annual_pick`): для года Y — кандидат с min pen_Y (цели < 01.01.Y); потолок (min по ~30 кандидатам).
- **RC5 (листья, контроль).** `C_LEAVES_NL_MD_WIN.csv`: num_leaves {7, 15, 31} × min_data {30, 100, 200} × окно {504, 1008 последних пар, расширяющееся}; λ2 = 10, lr 0,05, ff 0,8, без bagging, признаки базы; 227 дат (каждая 10-я сетки); колонки: доля листьев с < 2·min_data, число листьев, QLIKE (прореженный), Δ к расширяющемуся окну.
- **RC6 (сравнения).** Таблицы C: DM (Ньюи–Уэст лаг 5) к HAR-J(б) и к базе C-L0; в колонках `d_qlike_vs_ref` и `d_crit_pen_vs_ref` — к baseline LGBM `lb0_inh` (0,2223). Колонки 2017–2021 — «потолок (кандидаты видели 2017–2021)»; чистые 2022–2026 помечены, но crit_pen по Y ≤ 2026 использует фолды с целями до 2025, и выбор среди кандидатов глобальный -> и 2022–2026 не независимый OOS (статус HISTORICAL WALK-FORWARD). «Перспективно» (R5): crit_pen < crit_pen базы И Δ к HAR-J(б) без 24.02.2022 ≤ 0. Старые прогнозы и таблицы не перезаписываются.

## Гейты
### C1 — код и тесты
CHECK: `python3 -m pytest tests/test_v2_xl_lgbmc_hp.py tests/test_v2_xl_lgbmc_rules.py tests/test_v2_xl_lgbmc_leak.py tests/test_v2_xl_lgbm_limits.py -q`
EXPECT: passed; λ1/min_gain доходят до LightGBM (мутация «не передан» ловится); шаги осей — известные ответы; порог эффекта RC2 (мутация EFFECT = 0 ловится); бюджет/реестр (41 > 40 и дубль поднимают ошибку; контроли не регистрируются); годовой выбор по pen своего года; главный тест обрезки на HP волны C; файлы ≤ 200 строк, функции ≤ 40.
EVIDENCE: pending
### C2 — C-L0 посчитан, база объявлена
CHECK: `python3 models_v2/s6/lgbm/runlc.py --stage l0` и `cat data/regen/xgb_lgbm_full_v1/lightgbm/c/C_BASE.json`
EXPECT: rc 0; `C_L0.csv` 6 строк; `C_BASE.json` name `C0_inh_F4`; в OPTUNA_TRIALS строк не добавлено (контроль).
EVIDENCE: pending
### C3 — C-L1 (9 осей × 2 точки)
CHECK: `python3 models_v2/s6/lgbm/runlc.py --stage l1`
EXPECT: rc 0; `C_L1_AXES.csv` 9 строк; `C_L1_RESULTS.csv` 19 строк (база + 18); +18 `hp` в OPTUNA_TRIALS (wave C); `C_AXES.json` effect — список осей с best Δ ≤ −0,001.
EVIDENCE: pending
### C4 — C-L2
CHECK: `python3 models_v2/s6/lgbm/runlc.py --stage l2`
EXPECT: rc 0; либо `C_L2_ABANDON.md` (осей с эффектом нет), либо `C_L2_RESULTS.csv` и ≤ 20 новых `hp`; CHECKPOINT used_stream_c.hp ≤ 40.
EVIDENCE: pending
### C5 — seed, соседи, годовой выбор
CHECK: `python3 models_v2/s6/lgbm/runlc.py --stage nbr`
EXPECT: rc 0; `C_SEEDS_NEIGHBORS.csv`: лучшие 2 + (4 seed + ≤ 8 соседей) на каждую + `C_annual_pick`; в реестр ничего не добавлено (контроли).
EVIDENCE: pending
### C6 — таблица листьев
CHECK: `python3 models_v2/s6/lgbm/runlc.py --stage leaves`
EXPECT: rc 0; `C_LEAVES_NL_MD_WIN.csv` 27 строк.
EVIDENCE: pending
### C7 — бюджет, прежнее не тронуто
CHECK: `shasum -a 256 docs/otchety/xgb_lgbm_full_v1/lightgbm/BASELINE.json data/regen/xgb_lgbm_full_v1/lightgbm/fc_lgbm_b2.parquet`; `find models_v2 tests -newer docs/otchety/xgb_lgbm_full_v1/lightgbm/GATES_B_lgbm.md -type f -not -path 'models_v2/s6/lgbm/*' -not -path 'models_v2/s6/xgb/*' -not -name '*.pyc' -not -name 'test_v2_xl_*'`; просмотр `CHECKPOINT.json`
EXPECT: sha BASELINE.json `62160aca…2488`, fc_lgbm_b2 `7cf07022…78ce`; `find` пуст; `used_stream_c.hp` ≤ 40, `used.hp` ≤ 55; trial_id уникальны.
EVIDENCE: pending
### C8 — ответ дирижёру
CHECK: ручной гейт: ≤ 500 слов, цифры перемерены из CSV/JSON; таблица осей, лучший по crit_pen, сравнение с baseline LGBM (0,2223) и HAR-J(б), seed/соседи, расход trials; потолок помечен.
EVIDENCE: pending

## Мутации -> тест
| № | мутация | тест |
|---|---|---|
| MC1 | lambda_l1 не передан в LightGBM | `test_v2_xl_lgbmc_hp.py::test_l1_changes_the_fitted_model` |
| MC2 | min_gain_to_split не передан | `...::test_huge_min_gain_gives_constant_prediction` |
| MC3 | порог эффекта оси снят | `test_v2_xl_lgbmc_rules.py::test_mutation_effect_threshold_zero_would_accept_noise` |
| MC4 | пары с целью d+1 в обучении (сдвиг, № 1) | `test_v2_xl_lgbmc_leak.py::test_mutation_pairs_with_target_d_plus_1_is_caught` |
| MC5 | срез LOYO на год позже (№ 19/21) | `...::test_fold_invariant_to_rows_after_cut_and_mutation_caught` |
| MC6 | контроли попадают в реестр / бюджет C превышен / дубль trial | `test_v2_xl_lgbmc_rules.py::test_register_only_hp_direction_controls_are_not_trials`, `::test_register_only_hp_and_budget` |
| MC7 | годовой выбор использует pen другого года (№ 21 класс) | `test_v2_xl_lgbmc_rules.py::test_annual_pick_uses_only_pen_of_own_year` |

## EVIDENCE (06.11.2026; заменяет `pending` выше, исходные строки не удалялись)
- C1: `pytest tests/test_v2_xl_lgbmc_{hp,rules,leak}.py tests/test_v2_xl_lgbm_limits.py -q` -> 23 passed, rc 0 (до начала L1).
- C2: `runlc.py --stage l0` rc 0 (повтор после правки c_hp: ключи кэша обновлены); `C_L0.csv` 6 строк данных; `C_BASE.json` name `C0_inh_F4`; OPTUNA_TRIALS без изменений (37 строк). Закрыт. Результат — в разделе «Результат C-L0».
- C7 (частично): sha BASELINE.json `62160acadb0e`, fc_lgbm_b2 `7cf07022ce1b` — не изменились.
- C3…C6, C8: pending (счёт после «коммит есть»).
