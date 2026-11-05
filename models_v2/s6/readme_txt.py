"""Статические тексты README «Волна A»: ответ (б), предложения для волны B, мутации -> тесты, отступления. Числа сюда не пишутся (они берутся из файлов в readme.py)."""

ANSWER_B = """\
**(б) Подбор гиперпараметров LightGBM: собственный или наследованный.**
- Единственный собственный выбор LightGBM — пилотный `models_v2/s2/tune.py:41` (`choose`, Ж-197): для каждой пары (библиотека, потеря) минимум QLIKE (б) на целях 2016 года при обучении на целях до 2015-12-31, сетка глубина {3; 4} × δ Хьюбера {0,5; 1,0}. Он же считает XGBoost; общие константы (lr, min_leaf, l2, subsample) у обеих библиотек одни.
- Годового LOYO-выбора HP (`models_v2/s4/select.py`) и пространства Optuna (`models_v2/s5/c/space.py`) для LightGBM нет: в обоих файлах нет ни одного обращения к `lgb`/`lgbm` (поиск по файлам, 06.11.2026).
- Во всех LightGBM-строках конвейера Б-2/C/D гиперпараметры наследованы от XGBoost: `models_v2/s5/d/lgbfit.py:11` (`params`) берёт поля `HP` из годового выбора I2 (сетка 132 для XGBoost) как есть; новый подбор в `lgbfit.py` не выполняется (docstring файла).
- Следствие: LightGBM-аналог Б-2 (baseline LightGBM) — строка без собственного подбора; сравнение с XGBoost без этой оговорки недопустимо."""

DONE_XGB = [
    "Объективы L1/L3/L4/L5 (семейство `OBJECTIVE_EXPERIMENTS`) и бустинг на остатке HAR-J(б), H1 (`HAR_BOOSTING_RESULTS`).",
    "Признаки F1–F6 одиночные и top2/top3 (`FEATURE_ABLATIONS`; строки `b2_f1`…`b2_f5`, `b2_f4`).",
    "Окна и частота переобучения (`WINDOW_RESEARCH`: скользящие 252/504/756/1008, веса, месяц/день).",
    "Optuna TPE на Б-2+F4: 22 trials, seed 42, волна C (шапка `C_TPE.csv`; `C_ALL`); смеси с HAR-J(б) и режимные правила (`REGIME_MODEL_RESULTS`, `C_ALL`).",
    "Seeds и соседи плато (`C_SEEDS.csv`, `C_PLATEAU.csv`, `D_SEEDS.csv`, `D_PLATEAU.csv`); ансамбли E1–E7 (`ENSEMBLE_RESULTS`).",
]
TODO_LGB = [
    "Собственные объективы LightGBM (Huber, L1, gamma/Tweedie, quantile; `objective`/`alpha`): из `OBJECTIVE_EXPERIMENTS` для LightGBM есть одна справочная строка `obj_l3l` (gamma), остальные объективы (L1, L3x, L4, L5a–c) считались только для XGBoost.",
    "Признаки F1–F6 для LightGBM по отдельности (в D есть только `d_lgbm_f4` и ансамбли с ним).",
    "Собственные HP: `num_leaves`, `min_data_in_leaf`, `feature_fraction`, `lambda_l1/l2`, `min_gain_to_split`, `max_bin`; leaf-wise против глубины (S1/S2 — лишь два контроля).",
    "Окна и частота переобучения для LightGBM (в D — только XGBoost R1008/HL504).",
    "Режимные модели LightGBM и нормализация цели; DART/GOSS — не пробовались.",
    "Любой Optuna на LightGBM: бюджет в `CHECKPOINT.json` не потрачен (0 из 45/55), учёт N — единый для двух алгоритмов.",
]
MUTATIONS = [
    ("A1", "убран обязательный id; дубль id; нет колонки; нет лучших 3 строк этапа 2; нет префикса `fx_`", "tests/test_v2_xl_inventory.py::test_mutation_*"),
    ("A4", "в fitter `xgb.train` -> `lgb.train`; класс по имени строки", "tests/test_v2_xl_origin.py::test_mutation_m1_*, test_mutation_m2_*"),
    ("A6", "подмена `max_depth` одного года в BASELINE.json / в коде выбора; поле удалено", "tests/test_v2_xl_baseline.py::test_mutation_max_depth_by_year_caught"),
    ("A8", "признаки +1 день (№ 1); пары с t+1 = d+1; режим с днём d в квантиле (№ 22)", "tests/test_v2_xl_leak.py::test_mutation_features_plus_one_day_caught, test_mutation_pairs_with_target_d_plus_1_caught, test_predictions_regime_invariant_to_future_and_mutation_caught"),
    ("A10", "режим по факту RV цели; QLIKE с σ вместо σ²; сумма n режимов != 2271", "tests/test_v2_xl_diag.py::test_mutation_*, test_builder_reproduces_stored_predictions"),
    ("A12", "S1 без `max_depth=-1`; clip99/drop5 без эффекта; fit_var(base) != fit_lgbx", "tests/test_v2_xl_lgbextra.py::test_var_params_variants, test_variants_and_ymods_change_forecast, test_apply_ymod_known_answer, test_fit_var_base_equals_fit_lgbx_bitwise"),
]
DEVIATIONS = [
    "↺ 23.02.2022 — нерабочий день (нет в ряду): в LGB_LEAVES/EXTRA взята d = 22.02.2022 (последняя d перед целью 24.02.2022). Откат: заменить дату в `lgb_extra.fixed_dates`.",
    "↺ «после отрицательной доходности» — прокси (rs− > rs+ на d): дневной доходности в разрешённых таблицах нет. Откат: подключить доходность.",
    "↺ INVENTORY: 288 строк (а не ~200); поля, не разобранные по коду, помечены «не подтверждено» (18 строк, в т. ч. `c2_h2l4_f4_c05` — прогноз не сохранён по имени).",
    "↺ GATES_A A1: семейства Б в `forecasts_pb` — b1…b5 и b7; `b6_` нет ни в прогнозах, ни в коде пилота (в тексте гейта сказано b1…b7). Проверка состава требует b1–b5, b7.",
    "↺ A3/A13: буквально «новее GATES_A» — 30 файлов данных D и 5 файлов `models_v2/s5/d`; их изменил коммит a697209a «Zh-204 wave D fixes after review» (06.11 00:13, другая сессия). Критерий в коде уточнён: новее коммита a697209a — 0; обе цифры выводятся командами.",
    "Распределение k по датам для Б-2 в прежних файлах не сохранено (только среднее) — в XGBoost BASELINE.json пометка «не подтверждено».",
    "XGBoost-seed 43–46 пересчитаны тем же кодом (`xgb_seeds.py`, контроль R6; seed 42 — сохранённый Б-2, побитно подтверждён A9).",
]
