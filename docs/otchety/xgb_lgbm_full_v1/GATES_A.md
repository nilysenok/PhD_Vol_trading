# GATES_A — волна A программы «полная независимая доработка XGBoost и LightGBM» (Ж-206)

Задание: `docs/zadaniya/2026-11-05_xgb_lgbm_full.md` (волна A); блок: `docs/zadaniya/2026-11-05_xgb_lgbm_full_blok.md` (при расхождении действует блок). Статус результатов волны: DEVELOPMENT / HISTORICAL WALK-FORWARD на уже просмотренной истории 2017–02.2026 (2271 дата), не независимый OOS. Новых trials бюджета — 0 (счётчики в `CHECKPOINT.json` подкаталогов = 0; LightGBM-аналог Б-2, 5 seed и 2 строки leaf-wise — контроли, не trials). Код: `models_v2/s6/`, запуск `python3 models_v2/s6/run_a.py --stage <inventory|origin|baseline|lgb_b2|diag|lgb_extra|check>`; тесты `tests/test_v2_xl_*.py`; данные `data/regen/xgb_lgbm_full_v1/`. Гейты написаны до счёта; допуски и правила ниже не двигаются после просмотра результата (изменение — только отдельной строкой «↺» с датой, обеими версиями результата и без удаления гейта). Формат: CHECK / EXPECT / EVIDENCE (`pending` до запуска).

## Правила, объявленные до счёта
- **R1 (происхождение).** Алгоритм строки определяется по коду: от функции обучения, которой строка получена, статический разбор исходника (`ast`: импорты и вызовы `xgb.train` / `lgb.train` / `fit_i1`, `fit_i1x`, `fit_lgbx`) плюс сохранённые параметры (`MODEL_SPECIFICATIONS.json`, реестры trials, `st_*.json`). Название строки доказательством не служит. Строка-смесь (H3, переключатель, ансамбль) получает класс «смесь», состав — по членам (какой член XGBoost / LightGBM / OLS HAR-J(б)). Не подтверждено → слово «не подтверждено» и причина.
- **R2 (baseline XGBoost).** Б-2 = `fx_b2_g132_b` при подтверждении по R1, что обучение — `xgb.train`, цель — ln RV_{t+1} − ln RV_m, HP — по году из сетки 132 (LOYO по целям < 01.01.Y), без HAR в признаках и без смеси. Не подтверждено — стоп и доклад.
- **R3 (baseline LightGBM; правило до счёта).** Кандидаты — самостоятельные LightGBM-строки (по R1: `lgb.train`, нет HAR-признака, смеси, остатка) на сетке 2271. «Собственный» подбор LightGBM = выбор HP по его собственному критерию в его собственном пространстве, до оценочного периода. Проверка по коду (`s2/tune.py`, `s4/select.py`, `s5/c/space.py`): единственный собственный выбор LightGBM — Ж-197 `tune.choose` (глубина {3,4}, δ {0,5; 1,0}, цели 2016 года; остальное — общие константы XGBoost: lr 0,03, min_leaf 50, l2 5, subsample 0,8); аналога годового LOYO-выбора HP или Optuna-пространства LightGBM нет. Поэтому: LGBM baseline = LightGBM-аналог Б-2 в том же конвейере (I1 ранняя остановка по 252 последним парам + годовой I2 из сетки 132, признаки Б-2, цель та же), параметры наследованы (`s5/d/lgbfit.params`), пометка «наследованные HP» обязательна; пилотные LightGBM-строки (`b2_m_lgbm_b__daily` и др.) и `d_lgbm_f4`, `obj_l3l` — справочные в INVENTORY, не baseline, потому что работают в другом конвейере/бюджете. Если по R1 окажется, что такая строка с собственным LOYO-выбором существует, правило пересматривается явной строкой ↺ до использования.
- **R4 (QLIKE).** Только `models_v2/metrics.qlike(rv, s2)`, RV и σ̂² в %², сетка 2271 дат (без 23.02.2026); допуск пересчёта к заявленным в п. 2 блока значениям: |Δ| < 5e-5 (округление 4 знаков); к значениям в таблицах прежних волн: < 1e-12.
- **R5 (режим на дату d).** Только по данным ≤ d: квинтили/терцили RV_w по 252 предыдущим (`pit.past_bucket`), стресс Г4 (`stress_p3.parquet`). Режимы по факту RV цели помечаются «по факту, не PIT» и в `regime` не входят.
- **R6 (seed-контроль).** 5 seed = 42–46; число деревьев, k и ES-блок — как в `fit_lgbx`; результат — разброс QLIKE, не выбор.
- **R7 (leaf-wise).** Две контрольные строки на baseline-параметрах: S1 — `max_depth=-1`, `num_leaves=2^depth` (без ограничения глубины); S2 — `max_depth=-1`, `num_leaves=31` (умолчание библиотеки). Остальное — baseline.

## Гейты

### A1 — INVENTORY.csv: состав
CHECK: `python3 -m pytest tests/test_v2_xl_inventory.py -q`
EXPECT: passed. Присутствуют все id: `harj_b`, `fx_b2_g132_b`, `b2_f4`, `c1_opt`, `c2_h3_roll_f4`, `c3_f3calm_rv`, `c3_f3calm_g4`, `e3_roll`; лучшие 3 строки этапа 2 для каждой библиотеки (по QLIKE `_b`, ежедневное); все строки `fx_*` (И1–И6, сетки 24 и 132); все семейства Б (`b1_*`…`b7_*` из `forecasts_pb`); все строки B1 (`OBJECTIVE_EXPERIMENTS`, `HAR_BOOSTING_RESULTS`, `REGIME_MODEL_RESULTS`), B2 (`FEATURE_ABLATIONS`, `WINDOW_RESEARCH`), C (`C_ALL`), D (`ENSEMBLE_RESULTS`). Колонки ровно по п. 1 задания (id, класс модели, библиотека+версия, objective, цель, признаки группы и число, гиперпараметры+источник, окно, частота переобучения, ранняя остановка, seed, путь прогноза+sha256, путь кода+sha256, trial_id, статус, выбор на оценочном периоде, QLIKE пересчитанный). Нет дублей id.
EVIDENCE: pending

### A2 — INVENTORY.csv: числа пересчитаны из прогнозов
CHECK: `python3 models_v2/s6/run_a.py --stage inventory --check`
EXPECT: для каждой строки с сохранённым прогнозом QLIKE пересчитан `metrics.qlike` на 2271 датах; для смесей — пересборка кодом из сохранённых членов (`c/mix.build`, `d/ens.build`); расхождение с таблицами прежних волн < 1e-12; контрольные значения: HAR-J(б) 0,2337, Б-2 0,2205, `b2_f4` 0,2125, `c1_opt` 0,2136, `c2_h3_roll_f4` 0,2098, `c3_f3calm_rv` 0,2101, `e3_roll` 0,2095 — в пределах R4. Строка без пересчёта помечена «не подтверждено» с причиной.
EVIDENCE: pending

### A3 — sha256 прогнозов и кода
CHECK: `python3 models_v2/s6/run_a.py --stage inventory --check-sha`
EXPECT: для каждого пути в INVENTORY файл существует, sha256 в таблице = sha256 файла на момент проверки; прежние файлы `data/regen/v2_sber_h1*` и `docs/otchety/sber_h1_deep_v1/*` не изменены (`find … -newer docs/otchety/xgb_lgbm_full_v1/GATES_A.md` вне новых каталогов пусто).
EVIDENCE: pending

### A4 — происхождение по коду (R1) и мутация
CHECK: `python3 -m pytest tests/test_v2_xl_origin.py -q`
EXPECT: passed; на синтетических исходниках: функция с `xgb.train` → «XGBoost», с `lgb.train` → «LightGBM», смесь/OLS → без бустера; мутации: (м1) в fitter подменён вызов `xgb.train`→`lgb.train` — класс меняется (тест падает на неверной классификации); (м2) класс по имени строки (`lgbm` в названии) вместо кода — тест на строке с обманчивым названием падает. На реальных строках: `fx_b2_g132_b`, `b2_f4`, `c1_opt` → XGBoost; `d_lgbm_f4` → LightGBM; `c2_h3_roll_f4` → смесь (HAR-J(б) OLS + XGBoost); `e3_roll` → смесь (HAR-J(б) OLS + LightGBM).
EVIDENCE: pending

### A5 — ответы (а)–(в) в README
CHECK: `python3 models_v2/s6/run_a.py --stage check-readme`
EXPECT: в README раздел «Ответы (а)–(в)» содержит (а) таблицу «строка → алгоритм → файл:строка кода → библиотека/версия» для Б-2, Б-2+F4, F4-Optuna, `c2_h3_roll_f4` (0,2098), `c3_f3calm_*` (0,2101), `e3_roll`; (б) вывод о собственном и наследованном подборе LightGBM со ссылками на `s2/tune.py`, `s4/select.py`, `s5/d/lgbfit.py`; (в) для `c2_h3_roll_f4` (волна C, N = 90) и 0,045 (волна D, N = 106): какой тест (DM Ньюи–Уэст лаг 5 / блочный бутстрап), против какой базы (Б-2 или HAR-J(б)), сырое p, число фактических тестов, дополнение до N единицами, состав семейства; p Холма пересчитан из сохранённых прогнозов и совпадает с таблицей (0,038 и 0,045) до 1e-9. Каждое утверждение — со ссылкой на файл и строку; в тексте нет слов «видимо/наверное» без пометки «не подтверждено».
EVIDENCE: pending

### A6 — XGBoost BASELINE.json
CHECK: `python3 -m pytest tests/test_v2_xl_baseline.py -q`
EXPECT: passed; `xgboost/BASELINE.json` содержит все поля п. 6 блока (признаки, цель, objective, max_depth, min_child_weight, learning_rate, n_estimators, subsample, colsample_bytree, reg_alpha, reg_lambda, gamma, окно, частота переобучения, early stopping, seed) — значения читаются кодом из `fit.HP`, `fit._params`, таблицы I2 по годам, не набраны вручную; n_estimators — как распределение k по датам (ES), с пометкой; версия xgboost и дата прогноза; тест падает при подмене одного поля в JSON (мутация: max_depth по году).
EVIDENCE: pending

### A7 — LightGBM BASELINE.json и LightGBM-аналог Б-2 (контроль)
CHECK: `python3 models_v2/s6/run_a.py --stage lgb_b2` затем `python3 -m pytest tests/test_v2_xl_baseline.py -q`
EXPECT: прогноз `data/regen/xgb_lgbm_full_v1/lightgbm/fc_lgbm_b2.parquet`: 2271 строк, сетка = сетке Б-2, σ̂² > 0 и конечна; `lightgbm/BASELINE.json` содержит все поля п. 11 блока (objective, boosting_type, num_leaves, max_depth, min_data_in_leaf, learning_rate, n_estimators, feature_fraction, bagging_fraction, bagging_freq, lambda_l1, lambda_l2, min_gain_to_split, окно, частота переобучения, seed) из `lgbfit.params` по году, пометка «наследованные HP (I2 из сетки 132 XGBoost)», справочные строки пилота и их QLIKE; QLIKE и sha записаны; в отчёте нет сравнения «XGBoost лучше/хуже LightGBM» без раскрытия, что LightGBM-строка — единственная в этом конвейере без собственного подбора.
EVIDENCE: pending

### A8 — главный тест обрезки на новых строках
CHECK: `python3 -m pytest tests/test_v2_xl_leak.py -q`
EXPECT: passed; для LightGBM-аналога Б-2, S1, S2 (и для сборки PREDICTIONS): прогноз на d побитно одинаков при порче/обрезке всех баров после 18:50 дня d (helper `damage/invariant` из `tests/test_v2_boost_leak.py`; синтетические бары); мутации: признаки +1 день (№ 1) падает; обучающая пара с t+1 = d+1 падает; режим на d с квантилем, включающим день d (№ 22), падает.
EVIDENCE: pending

### A9 — воспроизводимость и контроль конвейера
CHECK: `python3 models_v2/s6/run_a.py --stage lgb_b2 --control`
EXPECT: тем же кодом (`engine_d`, `lib=xgb`) Б-2 воспроизводится побитно на 3 датах (01.06.2018, 01.06.2022, 02.06.2025) к `fix_forecasts_c.parquet`; LightGBM на этих датах — два запуска побитно одинаковы (`deterministic=True`, `num_threads=1`).
EVIDENCE: pending

### A10 — PREDICTIONS.parquet по алгоритму
CHECK: `python3 -m pytest tests/test_v2_xl_diag.py -q`
EXPECT: passed; `xgboost/PREDICTIONS.parquet` и `lightgbm/PREDICTIONS.parquet`: 2271 строка, колонки ровно `date, RV_actual, <ALG>_forecast, HAR_forecast, QLIKE_<ALG>, QLIKE_HAR, regime, features` (+ `target_date` допускается второй колонкой не вместо `date`); `date` — день цели; QLIKE пересчитан из прогнозов (< 1e-12); `regime` — по R5; `features` — метка набора и число признаков. Мутация: режим по факту RV цели → тест падает; QLIKE с σ вместо σ² → падает.
EVIDENCE: pending

### A11 — ANNUAL_PERFORMANCE.csv и REGIME_PERFORMANCE.csv
CHECK: `python3 models_v2/s6/run_a.py --stage diag --check`
EXPECT: по каждому алгоритму: годы 2017…2026 (QLIKE, RMSE, MAE, bias ln, R², n, доля лучше HAR-J(б), Δ к HAR-J(б)); режимы: низкая/высокая RV по факту (помечено) и по прошлому, скачки (дни и ±5), после отрицательной доходности (по дневной доходности d), спокойный/стресс, переходы режима, завышение/занижение (доля, средний ln-bias); сумма n по взаимоисключающим срезам = 2271; взвешенная по n средняя QLIKE годов = общий QLIKE (< 1e-12); отдельно строки «2017–2021, 2022, 2023–2025, без 24.02.2022, без топ-20 дней» и текст причин проигрыша в 2017–2021 (для XGBoost — по `HYPOTHESES_A.csv` волны A прежней программы, пересчитано, не скопировано).
EVIDENCE: pending

### A12 — LightGBM: специфическая диагностика
CHECK: `python3 models_v2/s6/run_a.py --stage lgb_extra` затем `python3 -m pytest tests/test_v2_xl_lgbextra.py -q`
EXPECT: `lightgbm/LGB_EXTRA.csv` (+ `LGB_LEAVES.csv`, `LGB_SEEDS.csv`): (1) число наблюдений в листьях (min, медиана, доля листьев < min_data_in_leaf×2) на фиксированных датах — первая сетка-дата каждого года и 23.02.2022 как d; (2) чувствительность к экстремальной RV: пересчёт прогноза на тех же датах при обучающих целях, ограниченных сверху 99-м процентилем прошлых, и без 5 наибольших — ln-изменение прогноза; (3) разброс по 5 seed 42–46: QLIKE, mean, std, min/max, среднее число деревьев; (4) строки S1 и S2 из R7: QLIKE по всем периодам. Все строки — контроли (trials 0).
EVIDENCE: pending

### A13 — сохранность старого и бюджет
CHECK: `python3 models_v2/s6/run_a.py --stage check`
EXPECT: `xgboost/OPTUNA_TRIALS.csv` и `lightgbm/OPTUNA_TRIALS.csv` — только шапка (0 строк); `CHECKPOINT.json` обоих = 0 использовано из лимитов 45/55 (xgb), 45/55 (lgbm), 60, 40; ни один прежний прогноз/таблица не перезаписан (см. A3); `models_v2/` вне `s6/` не изменён (`git status` вне s6 — только то, что было до начала); запрещённые файлы не читались (`.env`, `test_locked/`, `*_test.*`).
EVIDENCE: pending

### A14 — код и тесты
CHECK: `python3 -m pytest tests/test_v2_xl_*.py -q` и `python3 models_v2/s6/run_a.py --stage audit`
EXPECT: все passed; аудит: файлов > 200 строк — 0, функций > 40 строк — 0 (или `// xtask: длинная — почему` аналог в docstring); одновременно процессов ≤ 6; каждая мутация каждого гейта A4, A6, A8, A10, A12 падает на подложенной ошибке и проходит на правке (список «мутация → тест» в README).
EVIDENCE: pending

### A15 — README «Волна A» и предложения для волны B
CHECK: `python3 models_v2/s6/run_a.py --stage check-readme`
EXPECT: `README.md` раздел «Волна A»: инвентаризация (ссылка), ответы (а)–(в), baseline обоих алгоритмов с пометками, диагностика с цифрами из таблиц (источник — файл), предложения для волны B в двух списках: «уже сделано для XGBoost, не повторять» (из прежних B1/B2/C: L1/L3/L4/L5, H1, F1–F6 одиночные и top2/top3, окна, частота, Optuna 22, режимы, seeds/соседи) и «не делалось для LightGBM» (собственные объективы, признаки F1–F6, собственные HP, окна, режимы); каждое число в README — из файла или команды; первая цифра помечена «потолок» там, где подбор и счёт на одних данных.
EVIDENCE: pending

### A16 — ограничения чтения и правила данных
CHECK: `python3 models_v2/s6/run_a.py --stage check` (раздел «чтение»)
EXPECT: журнал чтения кода волны содержит только разрешённые пути (`data/iss_check/{bars_10m_full,index_10m}`, `data/fastbt/s9/rvi_1850.parquet`, `data/regen/v2_sber_h1*`, `data/regen/xgb_lgbm_full_v1/`); `grep -rn "_test\.\|test_locked\|\.env" models_v2/s6` — пусто.
EVIDENCE: pending

## Реестр
Закрыто 0 из 16 (до счёта). Незакрытые гейты в отчёте — «передача».
