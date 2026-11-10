# GATES — пилот на базе X1 (Ж-241), только SBER, h = 1

Статус: исследовательский, DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; первая цифра — потолок. Основание: журнал, Ж-241 (дословно и толкование).

## Договорённости (объявлены до счёта, 10.11.2026 01:00)
- **База X1 заморожена:** `data/regen/harj_deep/E1/x1.parquet`, sha256 — в CHECKPOINT DE (`E1.x1`). Каждая новая поправка при отключении даёт F = X1 побитно.
- **Контроли только для чтения:** K1, K2, K3, M, K8, NTP2, A2, A0g, V7 cfg8, X1 и их хэши.
- **Даты:** одинаковые, 2018-01-03 … 2026-02-26. Окно разработки — 2018–2021; 2022–2025 — историческая проверка, не OOS; 2026 — отдельно.
- **Бюджет — 0 до утверждения.** Бюджеты V7, V7+ и DE не используются.

## Лист P0 — аудит трёх механизмов и проверки до обучения (задание `docs/zadaniya/2026-11-10_harj_x1_P0.md`), 0 конфигураций

- **G1 — аудит.** CHECK: `docs/otchety/harj_x1/P0_AUDIT.md`. EXPECT: по каждому из трёх механизмов Ж-241 — что из него уже проверено в V1–V7 и DE (программа, id, файл:строка кода, отчёт, результат) и что нет. Вывод — «проверено / частично / не проверено». Ручной гейт с источниками; дирижёр сверяет выборочно. EVIDENCE: `P0_AUDIT.md` написан (10.11.2026): М1 — частично (доля последнего часа и доли трети дня — признаки; динамика затухания состояния закрытия не проверялась); М2 — частично (сквозное обучение энкодера cfg10 — к K1, не к X1; агрегация B(w) не проверялась); М3 — частично (V2: части дня в сумме; V4 I: C + J раздельно; совместного обучения нет); ссылки файл:строка в таблицах файла.
- **G2 — масштаб RV при нормировке.**
  - CHECK: `python3 -W ignore -m models_v2.s16.run_x1 --stage P0 --check`; pytest `tests/test_v2_x1_p0_*.py`.
  - EXPECT: числа по годам 2018–2025:
    - для построителя M10 V7 (`v7_0_seq`) — корреляция и отношение суммы нормированного канала r² за день к RV_d, а также ранговая корреляция;
    - то же для признаков V6 S (части дня);
    - для предполагаемой агрегации B = Σ_k w_k r²_k при w ≡ 1 — B = RV_d побитно (тест);
    - вывод, сохраняется ли масштаб.
  - EVIDENCE: `python3 -W ignore -m models_v2.s16.run_x1 --stage P0 --check` rc 0 («OK: P0_SCALE/INIT/MIX пересчитаны и равны файлам, отчёт = генератор…»); pytest tests/test_v2_x1_p0_*.py — 15 passed. P0_SCALE.csv, 2018–2025: B(w≡1) = RV_d побитно на 100 % дней всех лет (Пирсон 1); сырой Σr2 = RV (1.0); после нормировки V7 канал r2: Пирсон 0,856–0,9998, но медиана Σz/RV от −5,4 до −0,07 (масштаб не сохраняется; 1/σ: 2,7–3,2 в 2018–21 и 0,21–0,25 в 2022–25); нормированный r² даёт отношение 9,9–18,5; доли S (s8, s9) безразмерны (Пирсон с RV от −0,22 до 0,18); B с положительным профилем среднего 1: медиана B/RV 1,06–1,37 (масштаб сдвигается при связи w с r²). Вывод: масштаб RV сохраняется только при w ≡ 1; нормировка V7 его не сохраняет. — выход из нулевой инициализации.**
  - CHECK: та же.
  - EXPECT:
    - для MoE V7 (`v7_1m_forms`, `v7_1m_fit`) — аналитический и численный градиент по параметрам маршрутизатора и экспертов в точке инициализации на реальной первой переоценке конфигурации 6;
    - нормы параметров маршрутизатора после обучения на 3 датах переоценки (повтор зарегистрированной конфигурации без записи прогнозов);
    - объяснение равномерных π ≈ 1/3 из V7 числом;
    - то же для планируемых поправок нового пилота на синтетике.
  - EVIDENCE: та же команда rc 0; P0_INIT.csv, P0_DYN.csv. cfg6, первая переоценка 2018-01-03, 5 seeds × 4 точки решётки: градиент данных по маршрутизатору = 0 точно, по экспертам до 0,084; автоград против центральной разности ≤ 2,7e-10; градиент маршрутизатора в начале — только штрафы (≈ 0,010). Динамика (seed 0, шаги 0…300, 4 даты): маршрутизатор покидает 1/3 (σ_π по строкам 0,13–0,21), эксперты расходятся уже к шагу 10 (расстояние 0,16–0,34, к шагу 300 — 1,5–2,1; corr(g1,g2) отрицательная); средние π ≈ 1/3 при σ_π по строкам 0,10–0,11 после fit_config; сохранённые π cfg6: среднее |π−1/3| 0,053, σ_π 0,071 (усреднение по 5 seeds сглаживает). Эксперты-копии только при нулевом выходном слое маршрутизатора (расстояние 0 против 0,45 при случайном). Синтетика: М2 при θ = 0 — мёртвая точка (γ и θ градиент 0 и после шага); при малом θ γ оживает (4,8e-5), θ — после γ; М1 κ — после a; М3 — все параметры сразу. Дефект симметрии не подтверждён.
- **G4 — оператор смешивания.**
  - CHECK: та же.
  - EXPECT: для каждого оператора V4–DE (X1 геометрический M^{1−π}C^π; MoE V7 Σπ_k g_k в лог-пространстве; X2 арифметический (1 − p)c_n + p·c_s; NTP; K8):
    - смысл экспертов (условное среднее или поправка в логе);
    - проверка Йенсена для QLIKE: l(Σπ f_k) ≤ Σπ l(f_k) в лог-пространстве (f = ln F) на всех датах X1 — доля нарушений;
    - то же для арифметического смешивания в F;
    - вывод, соответствует ли оператор смыслу экспертов.
  - EVIDENCE: та же команда rc 0; P0_MIX.csv, все 2018 дат X1. Доля нарушений Йенсена в ln F — 0 у X1, NTP2, K8, V7 MoE, X2 (QLIKE выпукла по ln F); при смешивании в F — 17,9–23,1 % дат (X1 18,0 %, X2 23,1 %). Смещение mean F_geo/mean F_ar: X1 0,9983, NTP2 0,9994, K8 0,9910, V7 MoE 0,9755 (2022: 0,954), X2 0,9897. Вывод: геометрический оператор X1/NTP2/K8 корректен как лог-ансамбль под QLIKE, но эксперты — условные средние, смещение вниз ≤ 1 % (K8) — не дефект; MoE V7 соответствует лог-поправкам; X2 арифметический по множителям.
- **G5 — проверки только читают.** CHECK: pytest; хэш выходов V6, V7, DE до и после. EXPECT: rc 0; выходы прежние; P0 не пишет прогнозов. EVIDENCE: `--check` rc 0, хэш выходов V6/V7/DE до и после 07fdd942c58b79b4 (совпал и с замером до записи таблиц); тест test_p0_computations_write_nothing_to_v6_v7_de и test_p0_modules_have_no_write_calls_to_forecast_dirs — passed; прогнозов 0, конфигураций 0.

## Лист P — план ≤ 6 конфигураций (дирижёр, после P0)
- **G6.** `docs/otchety/HARJ_HYBRID_SBER_X1_PILOT_PLAN.md`. Для каждой конфигурации:
  - формула и вариант, дающий X1 при отключении;
  - отличие от V1–V7 и DE;
  - обучаемые параметры;
  - источник ожидаемого улучшения;
  - абляция;
  - даты сравнения;
  - правила «go» до счёта.
  
  Утверждение — владелец. EVIDENCE: pending

## Лист Y-A — Y1 и Y2 (Ж-243; задание `docs/zadaniya/2026-11-10_harj_x1_YA.md`), 2 конфигурации

- **YA1 — реестр до счёта.**
  - CHECK: `python3 -W ignore -m pytest tests/test_v2_x1_ya_*.py -q`; чтение `docs/otchety/harj_x1/ALL_TRIALS.csv`.
  - EXPECT: строки Y1–Y6 со статусом и временем записи; Y1, Y2 записаны раньше `computed_at` прогнозов; Y5, Y6 — «условная, не запускается без разрешения владельца (Ж-243)»; бегун отказывает Y5/Y6 и незарегистрированным. EVIDENCE (10.11.2026 04:32): `python3 -W ignore -m pytest tests/test_v2_x1_ya_*.py -q` — 39 passed (`test_v2_x1_ya_reg.py` 3: отказ незарегистрированной, записанной позже прогноза, Y5/Y6, повтор не переписывает). Реестр записан отдельной командой `python3 -W ignore -m models_v2.s16.x1_reg` 10.11.2026 04:21:03 (Y1–Y4 «зарегистрирована», Y5, Y6 «условная, не запускается без отдельного разрешения владельца (Ж-243)»); `computed_at` прогнозов — CHECKPOINT `YA` (после 04:21:03); `--check` перепроверяет порядок.
- **YA2 — отключение даёт X1.**
  - CHECK: та же.
  - EXPECT: a = 0 ⇒ F = X1 побитно; до 2018-07-02 F = X1 побитно на боевом ряду; закрытых пар на 2018-07-02 = 124 (≥ 120), бегун отказывает при < 120. EVIDENCE (10.11.2026 04:32): `test_v2_x1_ya_walk.py` 8 passed — `test_zero_amplitudes_give_x1_bitwise`, `test_activation_and_bitwise_before_inclusion_on_synthetic`, `test_refuses_with_fewer_than_120_closed_pairs`, `test_real_series_before_inclusion_bitwise_and_124_pairs` (боевой ряд: закрытых пар на 2018-07-02 = 124, 124 дня active = 0, F = X1 побитно).
- **YA3 — PIT и признаки.**
  - CHECK: та же.
  - EXPECT: мутации ловятся — бар дня t⁺ в s_d; пара «закрыто + 1» в переоценке; λ выбран по парам вне закрытых; e_d ≠ колонке `s` X1; Δ_d не по календарю t⁺. Аналитический градиент = численному (≤ 1e-6); κ оживает после a (синтетика). EVIDENCE (10.11.2026 04:32): `test_v2_x1_ya_feat.py` 6 passed (бар дня t⁺ в s_d: `test_pit_day_d_only_and_t_plus_bar_mutation_is_caught`; Δ_d: `test_delta_known_answer_and_calendar_mutation`; e_d: `test_e_d_is_x1_column_equal_to_de_states_and_mutation_is_caught`; карта слотов 46…51 против паспорта V7: `test_slot_map_matches_v7_passport`), `test_v2_x1_ya_fit.py` 8 passed (градиент ≤ 1e-6, κ оживает после a, восстановление a_n, a_e, κ на синтетике, λ по закрытым парам — мутация «после D» ловится), `test_v2_x1_ya_walk.py` («закрыто + 1» на боевом пути walk, Y2 с κ ≠ 0, e_d не из X1).
- **YA4 — прогнозы и оценка.**
  - CHECK: `python3 -W ignore -m models_v2.s16.run_x1 --stage YA --check`.
  - EXPECT: rc 0; пересчёт = файлам `data/regen/harj_x1/YA/y1.parquet`, `y2.parquet`; таблицы `YA_*.csv` = генератору; хэш выходов V6/V7/DE/P0/T0 до и после равен; `--check` ничего не пишет. EVIDENCE (10.11.2026 04:32): `python3 -W ignore -m models_v2.s16.run_x1 --stage YA` (запись, rc 0), затем `--stage YA --check` rc 0: «OK: Y1/Y2 пересчитаны и равны файлам, YA_*.csv и отчёт = генератор, реестр до прогноза, ключ YA.eval раньше таблиц с p; выходы V6/V7/DE/P0/T0 и YA не изменены (хэш 4e5732da8d99ab8e)». `x1_t0 --check` rc 0. Файлы: `data/regen/harj_x1/YA/y1.parquet`, `y2.parquet` (2018 строк), `docs/otchety/harj_x1/YA_*.csv`; CHECKPOINT ключи `YA`, `YA.eval`, `YA.rules`. ОТСТУПЛЕНИЕ/КРАСНОЕ: `run_x1 --stage P0 --check` теперь rc 1 («код s16 ≠ CHECKPOINT»): хэш кода P0 включает `run_x1.py`, который по заданию правлен (добавлена стадия YA); таблицы P0 и CHECKPOINT P0 не менялись, `pytest tests/test_v2_x1_p0_*.py` — 15 passed.
- **YA5 — отчёт.**
  - CHECK: та же; тест «подменённое число».
  - EXPECT: `YA_REPORT.md` содержит по Y1 и Y2: Δ к X1, K2, K3, NTP2, cfg8, A2, cfg6, K1 (в % QLIKE базы, 90 % ДИ, p DM, Холм 744); правило 6.2 к X1; Y1 против Y2; по годам; окно «после включения»; клетки остатка X1; go 6.1; MCS; стресс. «Снижение ошибки» и «подтверждено» — раздельно. EVIDENCE (10.11.2026 04:32): `YA_REPORT.md` — генератор `x1_ya_doc.build` (разделы 2–13), `--check` rc 0 сверяет текст с генератором; `test_v2_x1_ya_eval.py` 6 passed (знак Δ и база процентов, Холм N = 744 / 3193 против числа строк, «go» = V7 при тех же базах, слово «подтверждено» — только при p Холма < 0,05) и `test_v2_x1_ya_stage.py` 8 passed (подменённое число в таблице и в отчёте ловится, p в `YA.eval` и порядок ключей ловятся, проверка с записью ловится). Числа: QLIKE все даты Y1 0,221268, Y2 0,220969, X1 0,221248; правило 6.2 к X1 — «нет» у обеих; p Холма N = 1,000 везде в основных окнах (исключение — NTP2, 2026: 0,644).

## Лист Y-B — Y3 и Y4 (Ж-243; задание `docs/zadaniya/2026-11-10_harj_x1_YB.md`), 2 конфигурации

- **YB1 — реестр.**
  - CHECK: `python3 -W ignore -m pytest tests/test_v2_x1_yb_*.py -q`.
  - EXPECT: Y3, Y4 зарегистрированы раньше `computed_at`; реестр не переписан (строки Y-A прежние); Y5/Y6 — отказ. EVIDENCE (10.11.2026): `pytest tests/test_v2_x1_yb_*.py` 20 passed; Y3, Y4 в `ALL_TRIALS.csv` с 10.11.2026 04:21:03 (лист Y-A), прогноз Y-B позже; `--check` (rc 0) проверяет `guard(computed_at)`; `test_y5_y6_always_refused` — отказ `x1_reg.guard` и `x1_yb_stage --stage Y5/Y6`.
- **YB2 — отключение и масштаб.**
  - CHECK: та же.
  - EXPECT: θ = 0 или γ = 0 ⇒ F = X1 побитно; w ≡ 1 ⇒ B = RV побитно; среднее w = 1; до 2018-07-02 F = X1. EVIDENCE (10.11.2026): `test_w_one_gives_rv_bitwise_and_mean_one`, `test_theta_zero_or_gamma_zero_gives_x1_bitwise`, `test_y4_weights_mean_one_and_fallback`, `test_walk_off_before_activation_bitwise` — passed; на боевом ряду 124 дня active = 0 и F = X1 (в отчёте YB, раздел 1); Y4: среднее w − 1 = 0,00; дней RV_d = 0: 0.
- **YB3 — PIT.**
  - CHECK: та же.
  - EXPECT: мутации ловятся — бар t⁺ в B_d; медиана слота с днём d; профиль Y4 по всей выборке; «закрыто + 1»; softmax по отсутствующим барам. Градиент аналитический = численному. EVIDENCE (10.11.2026): `test_bar_of_t_plus_in_B_mutation_caught`, `test_median_slot_with_day_d_mutation_caught_in_pair_data`, `test_slot_median_excludes_current_day_and_mutation_caught`, `test_profile_y4_pit_and_whole_sample_mutation_caught`, `test_closed_plus_one_mutation_caught`, `test_softmax_over_absent_bars_mutation_breaks_scale`, `test_analytic_gradient_equals_numeric_and_dead_point` (мёртвая точка θ = 0, γ = 0: градиент 0; старт плана: все компоненты ≠ 0; макс. расхождение с численным 5e-9 на боевых данных) — 20 passed.
- **YB4 — прогнозы и оценка.**
  - CHECK: `python3 -W ignore -m models_v2.s16.x1_yb_stage --check`; также `run_x1 --stage P0 --check` и `run_x1 --stage YA --check` — rc 0.
  - EXPECT: rc 0; пересчёт = `data/regen/harj_x1/YB/y3.parquet`, `y4.parquet`, таблицам `YB_*.csv`; хэш выходов V6/V7/DE/P0/T0/YA до и после равен. EVIDENCE (10.11.2026): `x1_yb_stage --check` rc 0 (хэш выходов 10842b39127889da), `run_x1 --stage P0 --check` rc 0, `run_x1 --stage YA --check` rc 0.
- **YB5 — отчёт и условие Y5/Y6.**
  - CHECK: та же; тесты условия.
  - EXPECT: `YB_REPORT.md` — как YA5 для Y3, Y4, плюс Y3 против Y4 и против Y1/Y2; `YB.condition` в CHECKPOINT = «правило 6.2 против X1 = да у Y1 или Y3»; Y5/Y6 не посчитаны. EVIDENCE (10.11.2026): `YB_REPORT.md` = генератор (`--check` rc 0); `YB.condition` = «нет» (Y1: нет, Y3: нет); `test_v2_x1_yb_cond.py` — известный ответ (4 сочетания), мутация «окно 2022+» ловится, подменённое число в отчёте ловится, проверка с записью ловится по хэшу; Y5/Y6 не запускались.

## Лист T0 — единая таблица QLIKE (Ж-242 пп. 1–2), 0 конфигураций
- **T0.** CHECK: `python3 -W ignore -m models_v2.s16.x1_t0 --check`; `python3 -W ignore -m pytest tests/test_v2_x1_t0.py -q`. EXPECT: rc 0, расхождений нет; 8 passed (мутации «среднее показателей периодов», «не общие даты», «подменённое число» ловятся). EVIDENCE: 10.11.2026 — `--check`: «расхождений нет», rc 0, хэши P0_TABLE.*, P0_STATS.csv и CHECKPOINT V7/DE до и после совпали; pytest: 8 passed. Общий период: d 2018-01-03…2026-02-26, t⁺ 2018-01-04…2026-02-27, n = 2018, без 2017; X1 численно лучший (QLIKE 0,221248 против NTP2 0,222556), с поправкой Холма N = 738 / 3181 не подтверждён (p Холма 1,0000 против K2, K3, NTP2).

## Лист Z — ZA, ZB, ZB0, ZC, ZCp (Ж-246; план `docs/otchety/HARJ_HYBRID_SBER_Z_PLAN.md`; задание `docs/zadaniya/2026-11-10_harj_z.md`), 5 конфигураций

- **Z1 — реестр.**
  - CHECK: `python3 -W ignore -m pytest tests/test_v2_x1_z_*.py -q`.
  - EXPECT: 5 строк Z (с пометкой «после просмотра Y1–Y4» у ZB, ZC) записаны раньше любого прогноза Z; строки Y1–Y6 не изменены; незарегистрированная конфигурация — отказ. EVIDENCE (10.11.2026): `pytest tests/test_v2_x1_z_*.py` 54 passed; ZA, ZB, ZB0, ZC, ZCp дописаны в `ALL_TRIALS.csv` отдельной командой `x1_z_reg` 10.11.2026 16:12:23 (ZB, ZC — «предложена после просмотра Y1–Y4 (Ж-246)»), прогноз Z записан 16:21 (CHECKPOINT `Z`.computed_at); строки Y1–Y6 — как в `x1_reg.ROWS` (`test_registry_has_five_z_rows_with_post_view_marks_and_y_rows_unchanged`); отказ Y5/Y6/Z6/ZX (`test_guard_refuses_unregistered_extra_late_and_bad_status`). Хэш ALL_TRIALS.csv в сверку P0/YA/YB не входит. КРАСНОЕ вне владения: `tests/test_v2_x1_ya_reg.py::test_real_registry_written_before_forecasts` (лист Y-A) требует ровно 6 строк реестра и теперь падает — нужна правка одного утверждения (`list(t.index)[:6]`) владельцем листа Y-A.
- **Z2 — отключение.**
  - CHECK: та же.
  - EXPECT: g ≡ 0 ⇒ F = X1 побитно у всех пяти. ZB, ZB0 вне e_d = 1 — F = X1 побитно на боевом ряду. До включения F = X1. ZCp ≠ ZC только сдвигом z. EVIDENCE (10.11.2026): `pytest tests/test_v2_x1_z_walk.py` — `test_g_zero_gives_x1_bitwise[ZA|ZB|ZB0|ZC|ZCp]` (u ≡ 1 ⇒ F = X1 побитно), `test_zb_zb0_x1_outside_area_and_before_activation_and_before_inclusion`, `test_za_active_from_activation_and_zb_waits_for_30_extremes`, `test_zcp_is_zc_on_shifted_z_and_no_shift_mutation_is_caught` — passed; на боевом ряду 124 дня active = 0 у всех пяти (отчёт Z, раздел «Реестр и протокол»), ZB/ZB0 вне e_d = 1: Δ = 0,000 % (Z_STATES, состояние N), включение ZB с 2018-07-02 (33 закрытые экстремальные пары ≥ 30).
- **Z3 — PIT и обучение.**
  - CHECK: та же.
  - EXPECT: мутации ловятся:
    - «закрыто + 1»;
    - λ/ω по парам после D;
    - стандартизация z по всей выборке;
    - v_d с днём d + 1;
    - x_d с баром t⁺;
    - валидация ZB по всем парам вместо экстремальных;
    - ZCp без сдвига.

    Аналитический градиент ZC = численному. Из старта плана градиент по b, a ≠ 0, по γ = 0. EVIDENCE (10.11.2026): `test_pit_future_targets_do_not_change_forecasts_and_closed_plus_one_is_caught[ZA|ZB|ZC]`, `test_lambda_omega_by_future_pairs_mutation_is_caught[ZA|ZC]`, `test_zc_standardization_by_whole_sample_mutation_is_caught`, `test_zb_validation_uses_extreme_pairs_and_all_pairs_mutation_is_caught`, `test_future_bars_and_days_do_not_change_features_and_mutations_are_caught` (бар t⁺ в x_d; день d + 1 в v_d), `test_standardize_uses_only_pool_and_whole_sample_mutation_is_caught`, `test_shift_is_cyclic_252_and_no_shift_mutation_is_caught`, `test_gate_gradient_analytic_equals_numeric_and_start_point` (расхождение ≤ 1e-7; в старте градиент по γ0, γ ровно 0, по b, a ≠ 0), `test_known_answer_zb_recovers_b_a_when_g_only_on_area` — passed (54 passed всего).
- **Z4 — прогнозы и оценка.**
  - CHECK: `python3 -W ignore -m models_v2.s16.x1_z_stage --check`; `x1_yb_stage --check`; `run_x1 --stage P0 --check`; `run_x1 --stage YA --check`; `x1_t0 --check`.
  - EXPECT: все rc 0; пересчёт = `data/regen/harj_x1/Z/*.parquet` и `Z_*.csv`; хэш чужих выходов до и после равен. EVIDENCE (10.11.2026): `x1_z_stage --check` rc 0 (хэш выходов 9328dddf33219bcf), `x1_yb_stage --check` rc 0 (1d9a5f243be7959f), `run_x1 --stage P0 --check` rc 0 (07fdd942c58b79b4), `run_x1 --stage YA --check` rc 0 (5c1a590da69a8f0c), `x1_t0 --check` rc 0 («расхождений нет»). Хэш чужих выходов Z-стадии исключает `ZD_*` и ключи `ZD*` CHECKPOINT (параллельный лист ZD).
- **Z5 — отчёт и критерии.**
  - CHECK: та же; тесты отчёта.
  - EXPECT: `Z_REPORT.md` — генератором. Содержит:
    - QLIKE по окнам и годам;
    - Δ к X1, K2, K3, NTP2, cfg8, Y2, Y3;
    - состояния E/N по годам; клетки остатка;
    - контрасты ZB/ZB0, ZC/ZCp, ZA/ZB/ZC;
    - K1–K5 по каждой гипотезе с «продолжить / закрыть»;
    - 6.2 и Холм 749 — «подтверждено» только при p Холма < 0,05.

    Подменённое число ловится. EVIDENCE (10.11.2026): `Z_REPORT.md` = генератор (`--check` rc 0); `test_check_catches_substituted_table_number_and_substituted_crit` (Z_PAIRS, Z_CRIT), `test_check_catches_substituted_number_in_report`, `test_check_catches_p_in_eval_and_wrong_key_order`, `test_check_that_writes_a_file_is_caught_by_hash`; K1–K5 — `test_each_condition_alone_closes_the_hypothesis` (7 случаев), `test_boundaries_are_inclusive_...`; Холм N = 749 / N_wide = 3203 — `test_main_tables_shape_holm_n_749_and_states`, `test_holm_n_equals_row_count_mutation_is_caught`. Итог по Z_CRIT.csv: ZA — закрыть (1 из 5), ZB — продолжить (5 из 5), ZC — закрыть (0 из 5); правило 6.2 против X1 — нет у всех пяти, p Холма N = 749 = 1,000 — «не доказано» у всех.

## Лист Z2 — H0/HD и условно H1 + Hc (Ж-246 п. 6; план `docs/otchety/HARJ_HYBRID_SBER_Z2_HAWKES_DRAFT.md`; задание `docs/zadaniya/2026-11-10_harj_z2.md`), ≤ 2 конфигурации

- **Z2.1 — реестр.**
  - CHECK: `python3 -W ignore -m pytest tests/test_v2_x1_z2_*.py -q`.
  - EXPECT: H1, Hc, H2 записаны до любого счёта; H2 — «не считается без решения владельца»; строки Y и Z не изменены. EVIDENCE (10.11.2026 17:03): `pytest tests/test_v2_x1_z2_*.py -q` rc 0, 35 passed; H1, Hc, H2 дописаны в `ALL_TRIALS.csv` отдельной командой `x1_z2_reg` 10.11.2026 16:47:35 (H1, Hc — «зарегистрирована; условная (L1–L3); после просмотра Y1–Y4, ZD», H2 — «не считается без решения владельца»), счёт H0 — позже (Z2.condition записан после 16:47); `test_registry_rows_before_any_count_and_h2_always_refused` (H2 — отказ при любом `computed_at`; H3 и реестр позже прогноза — отказ; строки Y1–Y6 и ZA–ZCp = `x1_reg.ROWS`/`x1_z_reg.ROWS`), `test_register_appends_only_missing_rows_bytes_intact`. КРАСНОЕ вне владения: `tests/test_v2_x1_z_reg.py::test_registry_has_five_z_rows_with_post_view_marks_and_y_rows_unchanged` (лист Z) требует `reg.index[6:] == ZR.IDS` и падает от дописанных строк Z2 — нужна правка одного утверждения (`reg.index[6:11]`) владельцем листа Z.
- **Z2.2 — H0 и HD.**
  - CHECK: та же; `python3 -W ignore -m models_v2.s16.x1_z2_stage --check`.
  - EXPECT: тождество e_d = 1[ΣN ≥ 1] побитно; синтетический Hawkes восстановлен в допуске; Пуассон → n̂ ≈ 0; `Z2_H0_*`, `Z2_HD.csv` = пересчёт. EVIDENCE (10.11.2026 17:03): `pytest tests/test_v2_x1_z2_*.py` 35 passed; `test_identity_on_real_store_all_days` (e_d = 1[ΣN ≥ 1] побитно на 2924 днях склада, `de_e1_state.states`), `test_known_answer_hawkes_recovered_and_poisson_gives_n_zero` (μ, n, β, ρ в допуске; Пуассон n̂ < 0,12, p χ²₄ > 0,01), `test_analytic_gradient_equals_numeric_...` и `test_kernel_equals_plan_reference_...` (медленный эталон по формулам плана); `python3 -W ignore -m models_v2.s16.x1_z2_stage --check` rc 0 («OK: H0/HD/условие пересчитаны и равны файлам … хэш 7f2efe0d390047e9»). H0: 2015–2021 n̂ = 0,651, полураспад 257 мин, η̂ = 2 (граница), ρ̂ = 1 (граница), LR = 1258,9, p χ²₄ = 2,7e-271; помесячно 91 окно, минимум событий 40 (17 окон < 100, с 2024-10-01); дубль: |corr| ξ_d с N_d = 0,931 > 0,8.
- **Z2.3 — условие.**
  - CHECK: та же.
  - EXPECT: `Z2.condition` в CHECKPOINT раньше любого прогноза H1. Известный ответ на 4 сочетаниях. При ложном условии H1 и Hc не считаются; H2 — отказ всегда. EVIDENCE (10.11.2026 17:03): `Z2.condition` записан (value = false; L1 истинно, L2(а) истинно, L2(б) ложно, L3 ложно: окно < 100 событий (минимум 40) и дубль |corr| = 0,931 с N_d), ключей `Z2`, `Z2.eval`, `Z2.rules` и файлов `data/regen/harj_x1/Z2/` нет (`check` их отсутствие проверяет); `test_condition_known_answer_on_four_combinations` (4 сочетания), `test_condition_boundaries_and_l2b_alternative` (100 / 99 событий, p = 0,01, |corr| = 0,8 / 0,81), `test_runner_refuses_h1_on_false_condition_never_computes_and_h2_always`, `test_key_order_condition_forecast_eval_rules_and_no_p_in_eval`, `test_check_flags_forecasts_or_eval_keys_when_condition_false`.
- **Z2.4 — PIT и прогнозы (если условие истинно).**
  - CHECK: та же; `x1_z_stage --check`, `x1_zd_stage --check`, `x1_yb_stage --check`, `run_x1 --stage P0|YA --check`, `x1_t0 --check`.
  - EXPECT: все 11 мутаций раздела 5 плана Z2 ловятся; a = 0 ⇒ F = X1 побитно; все проверки rc 0; хэш чужих выходов до и после равен. EVIDENCE (10.11.2026 17:03; условие ложно — H1/Hc на боевых данных не считались, PIT и ход проверены на синтетике): 11 мутаций — `test_threshold_uses_only_past_bars_and_day_d_mutation_caught` (1), `test_mutation_self_excitation_bar_k_excites_bar_k_caught` (2), `test_xi_of_day_d_ignores_bars_of_t_plus_and_mutation_caught` (3), `test_stage1_window_excludes_session_D_and_mutation_caught` (4), `test_pool_xi_uses_theta_D_for_all_rows_and_own_month_mutation_caught` (5, № 146), `test_closed_plus_one_mutation_caught` (6), `test_hc_permutation_invariants_and_mutations_caught` (7), `test_identity_e_equals_any_event_and_mutation_caught` (8), `test_a_zero_gives_x1_bitwise_and_stray_constant_mutation_caught` (9), `test_start_gradient_nonzero_and_dead_point_mutation_caught` (10), `test_branching_n_capped_at_09_and_mutation_caught` (11) + `test_mutation_no_overnight_decay_caught`; проверки rc 0: `x1_z2_stage --check`, `x1_z_stage --check`, `x1_zd_stage --check`, `x1_yb_stage --check`, `run_x1 --stage P0 --check`, `run_x1 --stage YA --check`, `x1_t0 --check`; хэш чужих выходов до и после равен внутри `x1_z2_stage --check` (7f2efe0d390047e9) и в каждой стадии.
- **Z2.5 — отчёт.**
  - CHECK: та же.
  - EXPECT: `Z2_REPORT.md` генератором. H0, HD, условие; при счёте — K1–K5, 6.2, Холм 751, H1 против ZB. Подменённое число ловится. EVIDENCE (10.11.2026 17:03): `Z2_REPORT.md` = генератор (`--check` rc 0), содержит H0, HD, условие (первая строка — статус «разработка; историческая; после просмотра Y1–Y4, ZD, Z»); H1 не считалась → K1–K5, 6.2, Холм 751, H1 против ZB в отчёт не входят (раздел 4 фиксирует «не запускались»); `test_check_clean_and_substituted_report_or_table_or_foreign_write_caught` (отчёт, `Z2_H0_LR.csv`, запись чужого), `test_substituted_number_in_z2_crit_csv_caught` (`Z2_CRIT.csv` на синтетике), `test_k1_k5_known_answer_boundaries_and_each_condition_alone_closes`. Ветка оценки H1/Hc (`x1_z2_walk`, `x1_z2_eval`, `x1_z2_run`, запись в `x1_z2_stage.run_h1`) на боевых рядах НЕ исполнялась: walk — на синтетике, критерии — на таблицах известного ответа, целиком боевой прогон — не проверялся.

## Лист ZD — предсказуемость полезности поправки (Ж-246; задание `docs/zadaniya/2026-11-10_harj_zd.md`), 0 конфигураций

- **ZD1 — PIT.**
  - CHECK: `python3 -W ignore -m pytest tests/test_v2_x1_zd_*.py -q`.
  - EXPECT: модель D на дату d учится только на парах t⁺ ≤ D — мутация «закрыто + 1» ловится; стандартизация по прошлому — мутация «вся выборка» ловится; контроль сдвига 252 ≠ исходному. EVIDENCE: `python3 -W ignore -m pytest tests/test_v2_x1_zd_*.py -q -p no:warnings` → rc 0, 17 passed (10.11.2026): `test_future_pair_does_not_change_month` (мутация «закрыто + 1» меняет прогноз месяца), `test_standardization_pool_only_and_whole_sample_mutation`, `test_features_use_only_day_d_and_past` (мутация: бары дня t⁺), `test_shift_control_differs_from_z_on_real_data` (AUC (г) ≠ (а), 2018 строк). Закрыт.
- **ZD2 — таблицы.**
  - CHECK: `python3 -W ignore -m models_v2.s16.x1_zd_stage --check`.
  - EXPECT: rc 0; `ZD_STATES.csv` (D по E/N × год для Y1, Y2, Y3), `ZD_PRED.csv` (AUC, ранговая корреляция по годам: z, только e_d, сдвиг 252), `ZD_CALIB.csv`; Σ D по состояниям = Σ D по годам (тождество). EVIDENCE: `python3 -W ignore -m models_v2.s16.x1_zd_stage --check` → rc 0, «OK: таблицы, отчёт, манифест = пересчёт; тождество Σ D выполнено; чужие выходы не изменены»; ZD_STATES.csv, ZD_PRED.csv, ZD_CALIB.csv, ZD_CAUSES.csv записаны запуском без `--check` (rc 0); тесты тождества (`test_v2_x1_zd_tab.py`: выброшенная дата и подмена суммы ломают его) — в 17 passed. Закрыт.
- **ZD3 — изоляция.**
  - CHECK: та же.
  - EXPECT: ZD не пишет прогнозов, `CHECKPOINT.json`, `ALL_TRIALS.csv`; хэш чужих выходов до и после равен; `--check` ничего не пишет. EVIDENCE: та же команда `--check` → rc 0, хэш выходов YA/YB/P0/T0, `data/regen/harj_deep`, `harj_hybrid_v7`, `harj_x1/YA|YB` до и после равен (проверка внутри `check()`); `tests/test_v2_x1_zd_stage.py`: AST — модули ZD не содержат записи parquet, CHECKPOINT.json, ALL_TRIALS.csv, импорта `x1_reg`; детектор ловит такую запись; `--check` с записью в чужой выход ловится (`test_check_that_writes_is_caught`). CHECKPOINT.json и ALL_TRIALS.csv исключены из хэша осознанно (их пишет параллельно лист Z), вместо этого — AST-запрет записи. Закрыт.
- **ZD4 — отчёт.**
  - CHECK: та же.
  - EXPECT: `ZD_REPORT.md` генератором; по каждой из 6 причин плана — число ZD или ссылка на готовую таблицу; подменённое число ловится. EVIDENCE: `ZD_REPORT.md` записан генератором `x1_zd_doc.build`, первая строка «диагностика, не отбор; историческая; без конфигураций»; раздел «Причины 1–6» — 6 причин (4 и 5 — «ZD не проверяет» + путь к готовой таблице); `--check` rc 0 (отчёт = генератор); `test_substituted_number_in_report_table_and_manifest_is_caught` и `test_report_first_line_zero_configs_and_identity` — в 17 passed. Закрыт.

## Лист ZE — частичная подстройка к уровню дня d (Ж-247 п. 4; план `docs/otchety/HARJ_HYBRID_SBER_ZE_PLAN.md`; задание `docs/zadaniya/2026-11-11_harj_ze.md`), 2 конфигурации (ZE, ZEp); Y5 не запускается (план, разд. 1)

- **ZE0 — предусловие дубля.**
  - CHECK: `python3 -W ignore -m models_v2.s16.x1_ze_stage --precondition`.
  - EXPECT: на 179 парах e_d = 1 с t⁺ ≤ 2021-12-31 max |corr(G, ·)| по v_d, s_d ≤ 0,8 (дирижёр 11.11 00:05: 0,693 / −0,172); результат записан в CHECKPOINT `ZE.precondition` раньше реестра. EVIDENCE: `--precondition` 11.11.2026 00:10:06–00:10:08 rc 0: «n = 179, corr(G, v_d) = 0.6931, corr(G, s_d) = -0.1724, corr(G, x_d) = 0.1490, max |corr| = 0.6931 → условие выполнено»; ключ `ZE.precondition` (written_at 11.11.2026 00:10:08) раньше реестра (00:10:15); `test_real_precondition_known_answer_conductor_numbers` passed. Закрыт.
- **ZE1 — реестр.**
  - CHECK: `python3 -W ignore -m pytest tests/test_v2_x1_ze_*.py -q`.
  - EXPECT: ZE, ZEp дописаны в `ALL_TRIALS.csv` (пометка «после просмотра Y1–Y4, Z, ZD; Ж-247») раньше любого прогноза ZE; прежние строки побайтно не изменены; незарегистрированная — отказ; Y5 — отказ. EVIDENCE: `python3 -W ignore -m models_v2.s16.x1_ze_reg` rc 0 «добавлено строк 2», registered_at 11.11.2026 00:10:15; прогноз `ZE.computed_at` позже (00:12:56+); прежние 4983 байта `ALL_TRIALS.csv` — префикс нового файла (`cmp`: «EOF on» копии до записи, расхождений до конца нет); `tests/test_v2_x1_ze_reg.py` — 5 тестов (1 skip→passed после записи ZE), `test_guard_refuses_unregistered_extra_late_and_bad_status` (Y5, Y6, ZA, ZB, H1, ZE2…), `test_register_appends_only_missing_rows_and_keeps_existing_bytes`, порядок строк не проверяется. Закрыт.
- **ZE2 — отключение и PIT.**
  - CHECK: та же.
  - EXPECT: g ≡ 0 ⇒ F = X1 побитно; вне e_d = 1 и до включения F = X1; c = 0 ⇒ ZE = ZB-формула побитно; мутации ловятся: «закрыто + 1»; G_d с RV дня t⁺; G_d из таблицы пар (пропуск 2022-04-29); X1_d со строки t⁺; ZEp без сдвига; c вне [0, 1]; валидация по всем парам. EVIDENCE: `pytest tests/test_v2_x1_ze_*.py` 44 passed, rc 0. Побитно: `test_g_zero_gives_x1_bitwise`, `test_outside_area_and_before_activation_and_before_inclusion_f_equals_x1`, `test_c_zero_loss_and_gradient_equal_zb_bitwise_…` (значение и градиент потери при c = 0 = `x1_z_fit.lin_loss_grad` побитно). ОГОВОРКА: сам L-BFGS-B при c ≡ 0 (3 переменные, одна закреплена) даёт (b, a) ZB лишь с точностью ~1e-7, не побитно: `test_walk_with_c_fixed_zero_equals_zb_walk` — F совпадает с rtol 1e-5. Мутации ловятся: `test_pit_closed_plus_one_mutation_is_caught`, `test_future_rows_do_not_change_g_and_mutation_rv_of_t_plus_is_caught`, `test_real_day_2022_04_29_g_from_store_and_pair_table_mutation_is_caught`, `test_mutation_x1_from_row_t_plus_is_caught`, `test_zep_uses_g_shifted_252_and_mutation_without_shift_is_caught`, `test_c_bound_zero_one_and_mutation_without_bound_is_caught[-0.6-0.0|2.0-1.0]`, `test_validation_block_is_last_60_extreme_pairs_and_mutation_all_pairs_is_caught`; известный ответ — `test_known_answer_recovers_b_a_c_on_area`. Закрыт (с оговоркой о c ≡ 0: потеря и формула побитно, оптимизатор — 1e-7).
- **ZE3 — прогнозы, оценка, критерии.**
  - CHECK: `python3 -W ignore -m models_v2.s16.x1_ze_stage --check`; `x1_z_stage --check`; `x1_zd_stage --check`; `x1_z2_stage --check`; `x1_yb_stage --check`; `run_x1 --stage P0 --check`; `run_x1 --stage YA --check`; `x1_t0 --check`.
  - EXPECT: все rc 0; пересчёт = файлы ZE; хэш чужих выходов до и после равен; K1–K6 по ZE с «продолжить / закрыть»; 6.2 и Холм N = 751; «без 2022» и 2023–2025 — диагностикой. EVIDENCE: все rc 0 (11.11 ≈ 00:15–00:20): `x1_ze_stage --check` «OK: ZE/ZEp пересчитаны и равны файлам … хэш 9c01ac01de5acddb»; `x1_z_stage --check` OK; `x1_zd_stage --check` OK; `x1_z2_stage --check` OK (хэш 39212fcffee42dfa); `x1_yb_stage --check` OK; `run_x1 --stage P0 --check` OK; `run_x1 --stage YA --check` OK; `x1_t0 --check` «расхождений нет». Хэш чужих выходов до и после `--check` равен (внутри `check`; `test_real_check_reads_only_and_mutation_write_in_check_is_caught`). K1–K6 по ZE: K1 да (−1,750 %), K2 да (0,000 %), K3 да (5 из 8), K4 да (−0,528 %), K5 НЕТ (+0,374 % > +0,10 %), K6 НЕТ (+2,228 %) → 4 из 6 → «закрыть»; 6.2 против X1 — нет (Δ QLIKE +0,000595, DM p 0,818, 2 из 4 лет), p Холма N = 751 равно 1,000. Окна «без 2022» и 2023–2025 — `ZE_DIAG.csv`, раздел 7 отчёта. Красное вне владения: `tests/test_v2_x1_z2_cond.py::test_register_appends_only_missing_rows_bytes_intact` (берёт `lines[:-3]` реестра — требует, чтобы строки Z2 были последними, класс № 164; после дозаписи ZE даёт 1 ≠ 3). Закрыт.
- **ZE4 — отчёт.**
  - CHECK: та же; тесты отчёта.
  - EXPECT: `ZE_REPORT.md` генератором; первая строка — статус «разработка, историческая, после просмотра»; подменённое число ловится. EVIDENCE: `ZE_REPORT.md` записан генератором `x1_ze_doc.build`, первая строка «разработка, историческая пошаговая проверка, не независимая; ZE — после просмотра Y1–Y4, Z, ZD; первая цифра — потолок.»; `--check` rc 0 (отчёт = генератор); `test_report_first_line_is_status_and_contains_all_sections`, `test_tamper_report_crit_csv_and_parquet_are_caught_and_honest_is_clean` (подменены число в отчёте, в `ZE_CRIT.csv`, прогнозы) — в 44 passed. Закрыт.
