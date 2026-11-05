# GATES — пилот SBER h=1 (Ж-197), models_v2

Задание: `docs/zadaniya/2026-11-05_v2_sber_h1.md` + блок `…_blok.md`. Гейты этапа 1 написаны до кода (05.11.2026). Этапы 2 и 3–4 дописываются в этот файл до их кода.
Запуск этапа 1: `python3 models_v2/run_p1.py` (пишет все таблицы и данные ниже). Тесты: `python3 -m pytest tests/test_v2_*.py -q`.

## Объявлено до счёта (допуски тестов утечки, О10)
- (i) цель назад: QLIKE_oracle <= 0.5 * QLIKE_work (для каждой из HAR-J а/б, HAR а/б).
- (ii) обучение на всей выборке 2014–2026: QLIKE_full <= QLIKE_work + 1e-12 (на сетке оценки).
- (iii) перемешанные признаки (перестановка строк X внутри окна, seed 42+номер даты): для моделей (б) |QLIKE_perm / QLIKE_RV_m − 1| <= 0.30; для моделей (а) проверяется только QLIKE_perm >= QLIKE_work (у (а) нет поправки на смещение, «около RV_m» к ним не применяется). Если (iii) для (б) не выполнено — это красный результат, допуск после счёта не двигается (только строкой «↺» с обеими версиями).
- Главный тест: прогноз на d побитно одинаков (tobytes float64) при обрезке и при порче всех баров после дня d (и баров дня d с begin >= 18:50) — 5 дат на реальных данных + синтетика.
- Мутации (обязаны делать главный тест красным): m1 признаки сдвинуты на +1 день (строка d видит признаки d+1); m2 окно обучения включает пару (x_d, y_{d+1}).
- Допуск сверки с arch: 1e-8 (абсолютно по ln σ̂² и по коэффициентам; если коэффициент при ln(1+J) (масштаб ~1e4) не укладывается абсолютно — смотрится также относительно, оба числа в таблице).

## G1 дневной ряд
CHECK: `python3 -m pytest tests/test_v2_daily.py -q`
EXPECT: все тесты passed (ручной пример RV/BV/J; окно 10:00–18:40; порог 30 баров; разрыв 28.02–24.03.2022 без склейки: RV_w NaN первые 4 дня после разрыва, RV_m — 21; мутация «склейка» ловится)
EVIDENCE: пересчёт после решений дирижёра (шкала %², выходные с 01.03.2025 исключены): `tests/test_v2_daily.py` → 10 passed. ЗАКРЫТ.

## G2 HAR-J (а)/(б), прогнозы 2017–02.2026
CHECK: `python3 models_v2/run_p1.py` затем `python3 -m pytest tests/test_v2_tables.py -q`
EXPECT: `data/regen/v2_sber_h1/forecasts_p1.parquet` (d, target_date, model, sigma2, rv); первая d = 2016-12-30, target 2017-01-03; последняя target 2026-02-27; одинаковое число строк у всех 7 моделей; ни одной пары (d, target) через разрыв 2022
EVIDENCE: run_p1.py → «дней в ряду 2945 исключено 80 сетка 2272»; `tests/test_v2_tables.py` → 8 passed. ЗАКРЫТ.

## G3 эталоны RW, RV_w, RV_m, HAR (а)/(б) на той же сетке
CHECK: `python3 -m pytest tests/test_v2_tables.py -q`
EXPECT: passed; в forecasts_p1 модели rw, rv_w, rv_m, har_a, har_b, harj_a, harj_b с одним и тем же множеством d
EVIDENCE: `tests/test_v2_tables.py::test_forecasts_one_grid_seven_models` passed (8 passed). ЗАКРЫТ.

## G4 сверка с arch на 10 датах
CHECK: `python3 -m pytest tests/test_v2_tables.py -q` (читает `docs/tables/v2_sber_h1/p1_arch_check.csv`)
EXPECT: 10 строк (равномерно по 2017–2026), max |Δ ln σ̂²| < 1e-8 и max |Δ коэффициент| < 1e-8 (или относительная; см. выше)
EVIDENCE: p1_arch_check.csv: max |Δ ln σ̂²| 8.9e-16, max |Δ коэф| 8.7e-16 (отн. 9.8e-13); `test_arch_check_ten_dates_under_1e8` passed. ЗАКРЫТ.

## G5 метрики по всем срезам
CHECK: `python3 -m pytest tests/test_v2_tables.py -q`
EXPECT: `docs/tables/v2_sber_h1/p1_metrics.csv`: 7 моделей × (all, годы 2017–2026, rvi_high, rvi_low, mar_jun_2020, y2022), столбцы n, qlike, mean_r, rmse_ln, mae_ln; шапка О11
EVIDENCE: `test_metrics_table_matches_independent_recompute`, `test_headers_carry_current_code_sha` passed (8 passed). ЗАКРЫТ.

## G6 коэффициенты на последний день каждого года
CHECK: `python3 -m pytest tests/test_v2_tables.py -q`
EXPECT: `p1_coef.csv`: harj и har, по строке на последнюю дату d каждого года 2016…2026, b0, b_d, b_w, b_m, b_j, s2, n
EVIDENCE: `test_coef_table_last_day_each_year` passed (8 passed). ЗАКРЫТ.

## G7 тесты утечки и мутации
CHECK: `python3 -m pytest tests/test_v2_leak.py -q`; таблица `docs/tables/v2_sber_h1/p1_leak.csv`
EXPECT: passed; допуски (i)–(iii) и главный тест — как объявлено выше; мутации m1, m2 ловятся
EVIDENCE: `tests/test_v2_leak.py` → 17 passed, 4 xfailed (strict). Главный тест и мутации m1/m2 зелёные; (i) выполнено у 4 моделей. После пересчёта (%²) КРАСНЫЕ по объявленному допуску: (ii) har_a 0.2553>0.2509, harj_a 0.2559>0.2507; (iii) har_b 1.154 и harj_b 1.113 против RV_m 0.381 (прежние 14.66 были выбросом ln(1+J) в долях). Допуски не двигались. НЕ ЗАКРЫТ.

## G8 синтетика с известным ответом
CHECK: `python3 -m pytest tests/test_v2_daily.py tests/test_v2_har.py -q`
EXPECT: passed: RV/BV/J на ручном примере; OLS восстанавливает известные β; s² известного шума; пустое/короткое окно — отказ
EVIDENCE: `tests/test_v2_daily.py tests/test_v2_har.py` → 19 passed (после пересчёта; общий прогон 44 passed, 4 xfailed). ЗАКРЫТ.

## G9 раздел «п. 1» отчёта
CHECK: `grep -c "^## п. 1" docs/otchety/2026-11-05_v2_sber_h1.md`
EXPECT: 1; в разделе — сделано/не сделано/проверено/отступления/решения, реестр «закрыто N из M», числа из файлов
EVIDENCE: grep -c → 1. ЗАКРЫТ (отчёт — «передача»: G7 не закрыт).

---
# Этап 2 (п. 2 блока): XGBoost и LightGBM. Гейты написаны до кода (05.11.2026)
Запуск: `python3 models_v2/run_p2.py` (подбор → счёт → таблицы). Тесты: `python3 -m pytest tests/test_v2_boost*.py -q`. Процессов <= 6, `nthread=1` в каждом.

## Объявлено до подбора и счёта
- Строки: {xgb, lgbm} × {mse, hub} × {nohar, har} × {free, mono} = 16; для каждой (а) exp(f), (б) exp(f+s²/2) (основная), чувствительность (btr) — s² по остаткам обучения. Имя: `<lib>_<loss>_<nohar|har>_<free|mono>_<a|b|btr>`.
- Признаки (все в %², только день d и раньше): ln RV_d, ln RV_w, ln RV_m, ln(1+J), ln BV, ln RV первого часа (бары 10:00–10:50), ln RV последнего часа (17:50–18:40), Σr²·1[r>0], Σr²·1[r<0], ночной гэп²/RV_m, день недели; с HAR — ещё ln σ̂² HAR-J(б) (PIT: OLS этапа 1 на данных <= t, мин. 252 пары). Пол для логарифма часового RV: 1e-6.
- Моно (+): ln RV_d, ln RV_w, ln RV_m.
- Ранняя остановка: последние 252 пары окна, эмбарго 1 пара (пара на стыке выбрасывается), терпение 100, потолок 2000 деревьев; метрика остановки: rmse/l2 для mse, mphe/huber для hub.
- Гиперпараметры: lr 0.03, min_child_weight=min_data_in_leaf=50, subsample 0.8 (bagging_freq=1), L2=5, seed 42, deterministic. LightGBM: max_depth=d, num_leaves=2^d. Hub: xgb huber_slope=δ, lgbm alpha=δ.
- **Сетка подбора (до подбора):** глубина ∈ {3, 4}; δ Хьюбера ∈ {0.5, 1.0} (в единицах ln RV; остатки HAR ~0.5). Для mse — только глубина. Всего 2 либ × (2 + 4) = 12 запусков. Критерий — QLIKE (б) на целях 2016 года; обучение — пары с целью <= 2015-12-31, один раз, без переобучения. Подбор на варианте nohar/free; параметры того же (lib, loss) переносятся на har и mono (у har до середины 2015 нет признака). Ничья — меньшая глубина, затем меньшее δ. После подбора заморожено.
- Переобучение — первый годный день месяца, обучение на парах t+1 <= этого дня; прогноз на d использует модель последнего переобучения <= d. Ежедневное — не делается (замер времени в отчёте).
- Допуски утечек (как этап 1, не двигаются): (i) QLIKE_oracle <= 0.5·QLIKE_work; (ii) QLIKE_full <= QLIKE_work + 1e-12; (iii) для (б) |QLIKE_perm/QLIKE_RV_m − 1| <= 0.30. Строки, где объявленный допуск не выполнен, фиксируются в отчёте красными, допуск не двигается.
- Главный тест: прогноз на d (модель + признаки) побитно одинаков при порче/обрезке данных после 18:50 d, несколько дат, детерминизм (два запуска побитно). Мутации: признак +1 день; HAR-признак из подгонки на всей выборке (Б2); блок остановки без эмбарго (структурный тест).

## G10 признаки
CHECK: `python3 -m pytest tests/test_v2_boost_feat.py -q`
EXPECT: passed: ручные RV часов/полудисперсии/гэп²/RV_m, NaN гэпа после разрыва, день недели, нет дальнейших данных
EVIDENCE: `tests/test_v2_boost_feat.py` → 4 passed. ЗАКРЫТ. (NaN гэпа в первый день после разрыва в тесте обеспечен RV_m=NaN; отдельной мутации правила `same` нет.)

## G11 HAR-признак PIT
CHECK: `python3 -m pytest tests/test_v2_boost_feat.py -q`
EXPECT: passed: признак в строке t побитно равен ln прогноза HAR-J(б) этапа 1 из данных <= t; до 252 пар — NaN; мутация «подгонка на всей выборке» ловится
EVIDENCE: `test_har_feature_equals_stage1_forecast_and_is_pit` passed (4 passed); мутация — `tests/test_v2_boost_leak.py::test_mutation_har_feature_full_sample_caught` (+ реальные данные) passed. ЗАКРЫТ.

## G12 подбор
CHECK: `python3 models_v2/run_p2.py` затем `python3 -m pytest tests/test_v2_boost_tables.py -q`
EXPECT: `docs/tables/v2_sber_h1/p2_tuning.csv`: 12 строк по заявленной сетке, выбор по правилу; в `forecasts_p2` одни и те же параметры во всех месяцах
EVIDENCE: run_p2 → выбор lgbm-hub (3; 0,5), lgbm-mse (3), xgb-hub (3; 1,0), xgb-mse (3); `test_tuning_grid_as_declared_and_rule` passed (`tests/test_v2_boost_tables.py` → 39 passed, 16 xfailed). ЗАКРЫТ формально; содержательно подбор вырожден (n_train=124, глубина не различима, xgb-hub — 1 дерево), см. отчёт.

## G13 прогнозы и общая сетка
CHECK: `python3 -m pytest tests/test_v2_boost_tables.py -q`
EXPECT: `forecasts_p2.parquet`: 16 строк × (a,b,btr) = 48 моделей, у каждой те же 2272 даты d, что у `forecasts_p1`; все sigma2 > 0 и конечны
EVIDENCE: `test_forecasts_48_models_one_grid_as_stage1`, `test_retrain_daily_trees_bounded_and_monthly_sensitivity_exists` passed. ↺ переобучение ежедневное (замер 0,03–0,07 с на fit ⇒ ~5 мин), месячный — чувствительность. ЗАКРЫТ.

## G14 метрики
CHECK: `python3 -m pytest tests/test_v2_boost_tables.py -q`
EXPECT: `p2_metrics.csv` (модуль metrics.py этапа 1, срезы О7) + строка harj_b для сравнения; независимый пересчёт QLIKE совпадает
EVIDENCE: `test_metrics_match_independent_recompute` passed; harj_b_p1 all = 0,2348. ЗАКРЫТ.

## G15 важность
CHECK: `python3 -m pytest tests/test_v2_boost_tables.py -q`
EXPECT: `p2_importance.csv`: 16 строк × признаки, gain нормирован (сумма по строке = 1), среднее по переобучениям
EVIDENCE: `test_importance_normalised_and_columns` passed (усреднение без моделей без разбиений; их 28 у xgb_hub_har_*). ЗАКРЫТ.

## G16 утечки
CHECK: `python3 -m pytest tests/test_v2_boost_tables.py -q` (читает `p2_leak.csv`)
EXPECT: для 16 строк (б): work/oracle/full/perm; (i) выполнено; (ii),(iii) — по объявленным допускам, красные названы в отчёте
EVIDENCE: (i) 16/16 выполнено; (ii) 16/16 выполнено; (iii) КРАСНЫЙ 16/16 (перм. 1,064–1,133 против RV_m 0,381; перм. не лучше рабочего — выполнено); 16 xfailed с числами. НЕ ЗАКРЫТ.

## G17 главный тест и мутации
CHECK: `python3 -m pytest tests/test_v2_boost_leak.py -q`
EXPECT: passed: побитная инвариантность на нескольких датах (реальные и синтетика), детерминизм, мутации (+1 день; HAR на всей выборке; без эмбарго) ловятся
EVIDENCE: `tests/test_v2_boost_leak.py` → 20 passed (синтетика ежедневный/месячный режимы, 2 даты × 2 строки на реальных данных, детерминизм, 3 мутации + граница 18:50). ЗАКРЫТ.

## G18 этап 1 не затронут
CHECK: `python3 -m pytest tests/test_v2_daily.py tests/test_v2_har.py tests/test_v2_leak.py tests/test_v2_tables.py -q`; `shasum -a 256 models_v2/{daily,har,metrics,leak,prov,arch_check}.py data/regen/v2_sber_h1/forecasts_p1.parquet`
EXPECT: 44 passed, 4 xfailed; префиксы sha256 до этапа 2: daily e421ca15, har cf6da1f8, metrics fddfd122, leak a6750244, prov 9b576e69, arch_check b59f8e8d, forecasts_p1 4a0109a6 (код этапа 2 лежит в подпакете `models_v2/s2/`, чтобы не менять хэш кода этапа 1 в шапках его таблиц)
EVIDENCE: этап 1 → 45 passed, 4 xfailed (были 44+4 до правок по замечаниям проверяющего: +1 тест 18:50). sha: daily e421ca15, har cf6da1f8, metrics fddfd122, leak a6750244, arch_check b59f8e8d, forecasts_p1 4a0109a6 — без изменений; prov.py изменён осознанно (статус «после аудита этапа 1 проверяющим» в шапке; было 9b576e69, стало c7cb16e6), run_p1 перезапущен, forecasts_p1 побитно тот же. ЗАКРЫТ.

## G19 раздел «п. 2» отчёта
CHECK: `grep -c "^## п. 2" docs/otchety/2026-11-05_v2_sber_h1.md`
EXPECT: 1; реестр «закрыто N из M»; замер времени ежедневного переобучения; умолчания Б1–Б9
EVIDENCE: grep -c → 1. ЗАКРЫТ (отчёт — «передача»: G16 не закрыт).

# Этап 3, Б — доработка бустинга Б-1…Б-7 (Ж-198), первый шаг; гейты написаны 05.11.2026 до кода и до счёта
Задание: `docs/zadaniya/2026-11-05_v2_sber_h1_etap3.md`, раздел «Ж-198». Код — `models_v2/s3/` (`resid.py` — Б-1; остальное — `scale.py`, `pb.py`), запуск `models_v2/run_pb.py`; таблицы `pb_*.csv`, прогнозы `forecasts_pb.parquet`. Этапы 1–2 не меняются. Счёт — одним прогоном после «коммит есть».

## Объявлено до счёта (Б)
- Общее: потеря MSE в ln; гиперпараметры из `p2_tuning.csv` (chosen); библиотеки xgb и lgbm (Б-3 — только lgbm); строка = (б) основная + (а) рядом; сетка 2272 даты; ES 252 + эмбарго 1; seed 42; расписание: ежедневное и ежемесячное (колонка `schedule`, Б-6).
- Б-1: e = ln RV_{t+1} − ln HAR-J(б)_t (PIT, >= 252 пар); признаки этапа 2 без HAR-признака, без монотонности; σ̂²(б) = HAR-J(б)_d·exp(c·f + s²/2), s² = средний квадрат остатка остаточной модели на ES-блоке; c ∈ {0,5; 0,75; 1}; (а) — то же без s²/2; чувствительность: c = 1 без s²/2 в (б)-колонке (одна строка).
- Б-2: S ∈ {RV_m; RV_252 = среднее уровней RV за min(252, доступно в сегменте) годных дней с d, минимум 22, без склейки через 2022}; цель ln RV_{t+1} − ln S_t; ln RV_d/w/m, ln BV, ln RV первого/последнего часа, полудисперсии (ln) — минус ln S_d; ln(1+J) → ln(1+J/S); гэп²/RV_m, dow — как есть; σ̂² = S_d·exp(f + s²/2) (s² по ES-блоку); без HAR-признака, без монотонности.
- Б-3: LightGBM linear_tree=True, признаки/цель этапа 2, 2 строки (без HAR-признака / с ним), без монотонности; несовместимость — записать в отчёт.
- Б-4: Б-2 (оба масштаба) + монотонность (+) по нормированным ln RV_d/w/m.
- Б-5: признаки ln RV_d, ln RV_w, ln RV_m, ln(1+J), HAR-J(б)-прогноз; цель — уровень; без монотонности и с ней.
- Б-7: σ̂² = HAR-J(б), если RV_w,d > 90-й процентиль RV_w по 252 предыдущим датам сетки (без d; до 252 — защита выключена); иначе — `xgb_mse_nohar_mono_b` этапа 2 (читается из forecasts_p2). Вторая строка — защита поверх лучшей из Б-1…Б-5 по QLIKE(all) (выбор на оценочном периоде, пометка). Доля дней защиты по годам — `pb_defense.csv`.
- Утечки: главный тест обрезки на каждой строке Б (Б-1/2/4/5 — через бары; Б-7 — подмена rv/rv_w в строках после d); (i)–(iii) как в этапе 2 (oracle <= 0,5·work; full <= work + 1e-12; perm: у (б)-строк около RV_m ±30 %, красное — красным, допуски не двигаются; для Б-1 дополнительно perm >= work). Мутации: остаток Б-1 по HAR-J(б), подогнанному на всей выборке; масштаб S с днём d+1 (или окно RV_252 с будущим); RV_252 склеен через разрыв 2022; q90 Б-7 с днём d; слитые (а)/(б).
- Критерий вывода (до счёта): «закрывает 2022» = QLIKE(y2022) <= QLIKE harj_b(y2022). «Без потери» = для каждого среза из {mar_jun_2020, y2023, y2024, y2025} QLIKE строки <= min(QLIKE harj_b, QLIKE лучшей строки (б) этапа 2 на этом срезе). Допуск 0. Слабый вариант (<= только harj_b на этих срезах) показывается отдельной колонкой и так и называется. Выбор «лучших 3» для DM — по QLIKE(all) среди Б. Колонки «2022» и «без 2022» в `pb_all.csv`.
- ↺ Б-3: linear_lambda = L2 = 5 (в задании не назван) · откат: константа `LINEAR_LAMBDA` в `gen.py` (до счёта, 05.11).
- ↺ Утечки (i)–(iii) и full считаются только при monthly (как этап 2); daily — только work · откат: добавить варианты в `run_pb.run_fits` (до счёта, 05.11).
- Уточнения: ничья Г3 — разница QLIKE <= 1e-12 ⇒ меньший w; логистика Г4 — не менее 30 конечных z среди closed, иначе жёсткая смесь.

## GB1 Б-1 остаток
CHECK: `python3 -m pytest tests/test_v2_hyb_resid.py -q`
EXPECT: PIT-остаток к HAR-J(б) не использует данные после t (мутация «подгонка на всей выборке» ловится); s² берётся с ES-блока; c масштабирует f; синтетика с известным остатком — f его восстанавливает
EVIDENCE: pending

## GB2 Б-2/Б-4 масштаб
CHECK: `python3 -m pytest tests/test_v2_hyb_scale.py -q`
EXPECT: RV_252 — min(252, доступно) дней, минимум 22, не склеен через разрыв; S_d не содержит d+1; синтетика: форма учится при смене уровня ×10 (прогноз следует уровню); монотонность на нормированных признаках
EVIDENCE: pending

## GB3 Б-3, Б-5
CHECK: `python3 -m pytest tests/test_v2_hyb_pb.py -q`
EXPECT: linear_tree строится на синтетике с линейным трендом и экстраполирует; набор признаков Б-5 ровно из 5 столбцов
EVIDENCE: pending

## GB4 Б-7
CHECK: `python3 -m pytest tests/test_v2_hyb_post.py -q`
EXPECT: q90 без дня d; до 252 защита выключена; доля по годам; мутация ловится
EVIDENCE: pending

## GB5 счёт Б, главный тест, утечки
CHECK: `python3 models_v2/run_pb.py`; `python3 -m pytest tests/test_v2_hyb_leak.py tests/test_v2_hyb_tables.py -q`
EXPECT: `forecasts_pb.parquet` на 2272 датах для всех строк × (а, б) × расписание; sigma2 > 0, конечны; главный тест passed; `pb_leak.csv` с (i)–(iii)
EVIDENCE: pending

## GB6 отчёт Б и вывод
CHECK: `python3 -m pytest tests/test_v2_hyb_tables.py -q`; `grep -c "^## п. 2, доработка" docs/otchety/2026-11-05_v2_sber_h1.md`
EXPECT: `pb_all.csv` с «2022» и «без 2022»; `pb_dm.csv` (3 лучших, Ньюи–Уэст лаг 5, Холм); критерий применён механически, сильный и слабый варианты; вывод одной строкой; grep = 1
EVIDENCE: pending

# Этап 3 — гибриды Г2–Г5 (Г1 = Б-1, см. выше), итоговая таблица, DM (п. 3–4 блока Ж-197)
Примечание (Ж-198): Г1 заменён на Б-1; пункты Г1/G20 ниже по HAR-J(а) утратили силу и заменены GB1. Гейты Г2–Г5 можно перечитать и уточнить отдельным коммитом до их счёта.
Задание: `docs/zadaniya/2026-11-05_v2_sber_h1_etap3.md`. Гейты написаны 05.11.2026 до кода и до счёта; порядок фиксирует коммит дирижёра. Код — `models_v2/s3/` (этапы 1–2 не меняются), запуск `models_v2/run_p3.py`, `run_p4.py`. Счёт — только после «коммит есть».

## Объявлено до счёта (этап 3)
- Сетка дат — та же 2272 даты; closed(d) = прогнозы с target_date <= d (прогноз с d в закрытые не входит).
- Лучший бустинг (Г0): argmin QLIKE(all) среди 16 строк (б) этапа 2; ожидается `xgb_mse_nohar_mono_b`; не совпало — стоп и доклад (выбор на оценочном периоде, пометка в таблицах).
- Г1: e_t = ln RV_{t+1} − ln HAR-J(а)_t (PIT, >= 252 пар; без HAR-признака, монотонность выключена; библиотека/потеря/глубина/δ как у лучшего; ежедневное переобучение, ES 252 + эмбарго 1). σ̂² = HAR-J(б)_d·exp(c·f), c ∈ {0,5; 0,75; 1} → g1_c050, g1_c075, g1_c100; чувствительность g1_c100_mono (монотонность по ln RV_d/w/m).
- Г2: g2 = лучшая по QLIKE(all) из 8 строк (б) этапа 2 с HAR-признаком (копия).
- Г3: сетка w ∈ {0; 0,1; …; 1}; w_d = argmin среднего QLIKE для ln σ̂² = w ln B + (1−w) ln H на последних 252 closed; ничья — меньший w; до 252 closed w = 0. B — лучший бустинг (б), H — harj_b.
- Г4: стресс_d = (RVI_d > медианы RVI по датам < d, >= 20 дат; после 2021 — ложь) ИЛИ (RV_w,d > квантиль 2/3 RV_w по 252 предыдущим датам сетки без d; до 252 — ложь). Спокойный эксперт C = argmin QLIKE(all) среди g1_c050/c075/c100, g2, g3 (пометка «выбор на оценочном периоде»). g4_hard: стресс → H, иначе C. g4_logit: ln σ̂² = π ln H + (1−π) ln C, π = σ(a + b z), z = ln RV_w,d − ln q67_d; (a, b) = argmin среднего QLIKE по closed (L-BFGS-B, границы [−5; 5], старт (0; 0)), переоценка в первый день сетки каждого месяца; до 252 closed или z не определён — как g4_hard.
- Г5: k_d = clip(EWMA по closed от RV/σ̂², 0,7; 1,5), веса 0,5^(возраст/h), возраст в шагах closed, h ∈ {10; 20; 60}; нет closed — k = 1. Основа = argmin QLIKE(all) среди Г1–Г4 (пометка). g5_always_h*, g5_stress_h* (k только при стрессе Г4; EWMA по всем closed) — 6 строк.
- Утечки (i)–(iii) на каждую новую строку, допуски не двигать. Г1: (i) цель e как признак: oracle <= 0,5·work; (ii) модель на всей выборке: full <= work + 1e-12; (iii) перемешанные X: perm >= work («около RV_m» к Г1 не применяется — основа Г1 это HAR-J(б)). Г3/Г4: (i) oracle = выбор w / эксперта по QLIKE в точке d+1: oracle <= 0,5·work; Г5: oracle = k по RV_{d+1}/σ̂² с клипом: oracle < work строго (клип ограничивает выигрыш, порог 0,5 не применяется). Г3–Г5 (ii): правило на окне [d−251; d+252] включая будущее: full <= work + 1e-12. Г3–Г5 (iii): ряд RV циклически сдвинут на 100 шагов внутри оценки параметров: perm >= work. Красное — красным, допуск не двигается.
- Главный тест: прогноз каждой строки на d побитно одинаков при порче/обрезке данных после дня d (Г1 — бары после 18:50; Г3–Г5 — замена rv/sigma2/RVI в строках с target_date > d).
- Мутации: m3 Г3/Г5 берут прогноз с целью d+1 (closed включает d); m4 остаток Г1 по HAR-J, подогнанному на всей выборке; m5 q67 Г4 с днём d; m6 EWMA с неверным весом (возраст 0 у старейшего); m7 клип k не применён.
- DM: d_t = QLIKE_harj_b − QLIKE_модель (плюс = модель лучше), Ньюи–Уэст, Бартлетт, лаг 5, N(0;1), двусторонний p; множество: 5 лучших по QLIKE(all) среди всех строк кроме harj_b + лучшая каждого класса (бустинг (a,b), Г1, Г2, Г3, Г4, Г5); выборки: all, цели <= 2021, цели 2022–02.2026; Холм по числу сравнений внутри выборки — колонка p_holm.
- p4: 53 строки = 7 эталонов/HAR + 32 бустинга (a,b) + 14 гибридов (g1×4, g2, g3, g4×2, g5×6); срезы этапа 1 + stress/calm (по Г4); колонки metrics.summary.

## G20 Г1 остаток
CHECK: `python3 -m pytest tests/test_v2_hyb_resid.py -q`
EXPECT: PIT-остаток e_t не использует данные после t (мутация m4 ловится); ln HAR-J(б) из harj_ab совпадает с признаком этапа 2 (1e-12); синтетика с известным остатком: f восстанавливает сдвиг; c масштабирует f
EVIDENCE: pending

## G21 Г2 и Г0
CHECK: `python3 models_v2/run_p3.py` (после «коммит есть»), `python3 -m pytest tests/test_v2_hyb_tables.py -q`
EXPECT: Г0 совпал с `xgb_mse_nohar_mono_b`; g2 = argmin из 8 строк (б) с HAR-признаком
EVIDENCE: pending

## G22 Г3
CHECK: `python3 -m pytest tests/test_v2_hyb_post.py -q`
EXPECT: на синтетике, где B лучше H, w → 1, где хуже — 0; до 252 closed w = 0; ничья → меньший w; ряд w в `p3_w.csv`
EVIDENCE: pending

## G23 Г4
CHECK: `python3 -m pytest tests/test_v2_hyb_post.py -q`
EXPECT: q67 без дня d (m5 ловится); RVI после 2021 не используется; логистика находит известные (a, b) на синтетике; доля стресса по годам в `p3_stress.csv`
EVIDENCE: pending

## G24 Г5
CHECK: `python3 -m pytest tests/test_v2_hyb_post.py -q`
EXPECT: EWMA на ступени с известным ответом; клип 0,7/1,5 (m7); возраст весов (m6); «только в стресс»: k = 1 вне стресса
EVIDENCE: pending

## G25 прогнозы и сетка
CHECK: `python3 models_v2/run_p3.py`; `python3 -m pytest tests/test_v2_hyb_tables.py -q`
EXPECT: `forecasts_p3.parquet`: 14 моделей × те же 2272 даты, sigma2 > 0 и конечны; таблицы p3_* с шапкой «до аудита», sha кода s3 и входов
EVIDENCE: pending

## G26 утечки и главный тест
CHECK: `python3 -m pytest tests/test_v2_hyb_leak.py -q`; `p3_leak.csv`
EXPECT: главный тест passed для Г1 (бары) и Г3–Г5; (i)–(iii) по допускам выше, красные названы
EVIDENCE: pending

## G27 мутации
CHECK: `python3 -m pytest tests/test_v2_hyb_post.py tests/test_v2_hyb_resid.py tests/test_v2_hyb_leak.py -q -k mut`
EXPECT: m3…m7 ловятся (тест красный на мутации, зелёный на правке)
EVIDENCE: pending

## G28 итоговая таблица
CHECK: `python3 models_v2/run_p4.py`; `python3 -m pytest tests/test_v2_hyb_tables.py -q`
EXPECT: `p4_all.csv`: 53 модели × срезы, одна сетка (одинаковое n на all), независимый пересчёт QLIKE совпал
EVIDENCE: pending

## G29 DM
CHECK: `python3 -m pytest tests/test_v2_hyb_dm.py -q`; `p4_dm.csv`
EXPECT: на синтетике постоянная разность → знак и p≈0; шум → p в ожидаемых границах; ручной Ньюи–Уэст лаг 5 совпал; p_holm >= p
EVIDENCE: pending

## G30 этапы 1–2 не затронуты
CHECK: `python3 -m pytest tests/test_v2_daily.py tests/test_v2_har.py tests/test_v2_leak.py tests/test_v2_tables.py tests/test_v2_boost_leak.py tests/test_v2_boost_tables.py -q`; `shasum -a 256 models_v2/*.py models_v2/s2/*.py`
EXPECT: 108 passed, 20 xfailed (как до этапа 3); sha файлов этапов 1–2 без изменений
EVIDENCE: pending

## G31 разделы «п. 3», «п. 4», «Итог»
CHECK: `grep -c "^## п. 3\|^## п. 4\|^## Итог" docs/otchety/2026-11-05_v2_sber_h1.md`
EXPECT: 3; «Итог» одной строкой (что даёт бустинг над HAR-J и где); коэффициенты HAR-J, важность (этап 2 + Г1), тесты утечки, реестр «закрыто N из M»
EVIDENCE: pending
