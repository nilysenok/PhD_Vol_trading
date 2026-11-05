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
EVIDENCE: pending

## G11 HAR-признак PIT
CHECK: `python3 -m pytest tests/test_v2_boost_feat.py -q`
EXPECT: passed: признак в строке t побитно равен ln прогноза HAR-J(б) этапа 1 из данных <= t; до 252 пар — NaN; мутация «подгонка на всей выборке» ловится
EVIDENCE: pending

## G12 подбор
CHECK: `python3 models_v2/run_p2.py` затем `python3 -m pytest tests/test_v2_boost_tables.py -q`
EXPECT: `docs/tables/v2_sber_h1/p2_tuning.csv`: 12 строк по заявленной сетке, выбор по правилу; в `forecasts_p2` одни и те же параметры во всех месяцах
EVIDENCE: pending

## G13 прогнозы и общая сетка
CHECK: `python3 -m pytest tests/test_v2_boost_tables.py -q`
EXPECT: `forecasts_p2.parquet`: 16 строк × (a,b,btr) = 48 моделей, у каждой те же 2272 даты d, что у `forecasts_p1`; все sigma2 > 0 и конечны
EVIDENCE: pending

## G14 метрики
CHECK: `python3 -m pytest tests/test_v2_boost_tables.py -q`
EXPECT: `p2_metrics.csv` (модуль metrics.py этапа 1, срезы О7) + строка harj_b для сравнения; независимый пересчёт QLIKE совпадает
EVIDENCE: pending

## G15 важность
CHECK: `python3 -m pytest tests/test_v2_boost_tables.py -q`
EXPECT: `p2_importance.csv`: 16 строк × признаки, gain нормирован (сумма по строке = 1), среднее по переобучениям
EVIDENCE: pending

## G16 утечки
CHECK: `python3 -m pytest tests/test_v2_boost_tables.py -q` (читает `p2_leak.csv`)
EXPECT: для 16 строк (б): work/oracle/full/perm; (i) выполнено; (ii),(iii) — по объявленным допускам, красные названы в отчёте
EVIDENCE: pending

## G17 главный тест и мутации
CHECK: `python3 -m pytest tests/test_v2_boost_leak.py -q`
EXPECT: passed: побитная инвариантность на нескольких датах (реальные и синтетика), детерминизм, мутации (+1 день; HAR на всей выборке; без эмбарго) ловятся
EVIDENCE: pending

## G18 этап 1 не затронут
CHECK: `python3 -m pytest tests/test_v2_daily.py tests/test_v2_har.py tests/test_v2_leak.py tests/test_v2_tables.py -q`; `shasum -a 256 models_v2/{daily,har,metrics,leak,prov,arch_check}.py data/regen/v2_sber_h1/forecasts_p1.parquet`
EXPECT: 44 passed, 4 xfailed; префиксы sha256 до этапа 2: daily e421ca15, har cf6da1f8, metrics fddfd122, leak a6750244, prov 9b576e69, arch_check b59f8e8d, forecasts_p1 4a0109a6 (код этапа 2 лежит в подпакете `models_v2/s2/`, чтобы не менять хэш кода этапа 1 в шапках его таблиц)
EVIDENCE: pending

## G19 раздел «п. 2» отчёта
CHECK: `grep -c "^## п. 2" docs/otchety/2026-11-05_v2_sber_h1.md`
EXPECT: 1; реестр «закрыто N из M»; замер времени ежедневного переобучения; умолчания Б1–Б9
EVIDENCE: pending
