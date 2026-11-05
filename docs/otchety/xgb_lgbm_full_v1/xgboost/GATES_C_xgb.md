# GATES_C_xgb — волна C (гиперпараметры), поток XGBoost, программа Ж-206

Задание: `docs/zadaniya/2026-11-05_xgb_lgbm_full.md` (волна C, поток XGB); блок: `docs/zadaniya/2026-11-05_xgb_lgbm_full_blok.md` (разделы 10, 26, 30). Статус: DEVELOPMENT (выбор — внутренний критерий LOYO) и HISTORICAL WALK-FORWARD (ряды прогнозов 2271 дата, 2017–02.2026, уже просмотрено; не независимый OOS). Первая цифра — потолок, p сырые. Код: `models_v2/s6/xgb/c_*.py`, `runxc.py`; тесты `tests/test_v2_xl_xgbc_*.py`; данные `data/regen/xgb_lgbm_full_v1/xgboost/c/`; ничего из `s6/xgb` вне `c_*` не меняется, прежние файлы только читаются.

## Правила, объявленные до счёта
- **Q1 (критерий).** Как B (P1): `crit_pen = mean_Y [mean_y q + 0,25·(max_y q − mean_y q)]`, y ∈ [2015, Y−1], Y = 2017…2026; HP года Y — годовой выбор И2 (сетка 132) со сдвигом. Опора: `ctrl_f4` (Б-2+F4, l2, И2) crit_pen 0,20114, весь 0,2125; для X3 — опора `x3_gamma_f4` (crit_pen 0,2157, весь 0,2112). Δcrit_pen = crit_pen(точка) − crit_pen(опора своей семьи); минус — лучше.
- **Q2 (оси и точки, C-X1).** Сдвиг применяется к HP каждого года Y отдельно (вокруг годового выбора И2, остальные оси и `max_trees` как у И2), результат зажимается в границы. Единица сдвига u: `u=±1` = один шаг.
  | ось (xgboost) | шаг | точки u | границы |
  |---|---|---|---|
  | max_depth | ±1 | −1, +1 | 1…6 |
  | min_child_weight | ×2 | −1, +1 | 2…200 |
  | learning_rate | ×1,5 | −1, +1 | 0,01…0,2 |
  | subsample | ±0,1 | −1, +1 | 0,4…1,0 |
  | colsample_bytree | ±0,2 | −1, −2 (база 1,0 в 8 годах из 10: «плюс» пуст) | 0,3…1,0 |
  | reg_alpha | +1 | +1, +3 (база 0: односторонняя) | 0…5 |
  | reg_lambda | ×3 | −1, +1 | 0,05…200 |
  | gamma | +0,05 | +1, +4 (база 0) | 0…1 |
  | max_delta_step | +0,1 | +1, +3 (база 0 = выкл.) | 0…5 |
  Итого 9 осей × 2 = 18 trials. Точка, у которой после зажима HP не изменился ни в один год, — дубль опоры: не trial, помечается. Число лет, где точка зажата, публикуется в `HP_AXES_C.csv`.
- **Q3 (эффект, C-X2).** Ось «с эффектом»: min по двум точкам Δcrit_pen < −1e-12. Если осей с эффектом 0 — X2 = `ABANDON` с причиной. Осей с эффектом больше 4 — берутся 4 с наименьшим Δ (чтобы источник улучшения был понятен, п. 10 блока). Совместный Optuna: TPE seed 42, n_startup 8, ≤ 20 trials; параметр — u по каждой оси эффекта в диапазоне `u ∈ [−2; +2]` (одно-сторонние: colsample [−3; 0], alpha [0; 5], gamma [0; 6], mds [0; 5]); цель Optuna `J` = среднее `pen_Y` по Y = 2017…2021 (цели < 01.01.2021; как Ж-204 C). Колонки 2017–2021 такой строки — «потолок (кандидаты видели 2017–2021)».
- **Q4 (годовой PIT-выбор X2).** Для каждого Y выбирается argmin `pen_Y` по {И2-база} ∪ {20 кандидатов}, где `pen_Y` — только по фолдам y < Y; ничья (|Δ| ≤ 1e-12) — в пользу И2. Строка `x2_annual` — процедура (не trial). Строка `x2_best_j` — кандидат с наименьшим J, постоянный сдвиг во все годы (крит. по всем Y — потолок).
- **Q5 (C-X3).** База `x3_gamma_f4` (reg:gamma; custom QLIKE даёт те же прогнозы — B3): 4 trials: learning_rate u = −1, +1; min_child_weight u = −1, +1 (≤ 6 разрешено; 2 запас не используются без причины). Вопрос: снимает ли регуляризация проигрыш 2017–21 (QLIKE 2017–21 базы 0,1600 против 0,1548 у `ctrl_f4`).
- **Q6 (seed и соседи, не trials).** Лучшие 2 по `crit_pen` среди постоянных сдвигов (X1-точки, X2 best_j, X3-точки; `x2_annual` — не берётся). Для каждой: seed 43–46 (ряд прогнозов; метрики из ряда; разброс); соседи ±1 шаг по каждой изменённой оси (u±1, в пределах диапазона; u=0 — опора). Для опоры `ctrl_f4` те же seed 43–46 (честное сравнение).
- **Q7 (перспективно).** Как P2 B: `crit_pen` лучше `ctrl_f4` на > 1e-12 И разность QLIKE к HAR-J(б) без 24.02.2022 ≤ 0 И ряд устойчив. Дополнительно `beats_base` — лучше своей опоры по crit_pen.
- **Q8 (учёт).** Бюджет HP ≤ 55: X1 ≤ 18 + X2 ≤ 20 + X3 ≤ 6 = 44. Реестр: `OPTUNA_TRIALS.csv` (дописывание, wave `C-xgb`), `CHECKPOINT.json` `used.hp`. Seed/соседи — контроли, `used` не растёт. Процессов ≤ 4 (spawn-пул).
- **Q9 (кэш).** Ключ кэша = trial_id + код (`s6/xgb`, `b1/{designs}`, `s4/fit.py`, `s5/c/fitx.py`) + хэш входов; кэш — только в `data/regen/xgb_lgbm_full_v1/xgboost/c/`.

## Гейты

### C1 — воспроизводство опоры новым конвейером
CHECK: `python3 -W ignore models_v2/s6/xgb/runxc.py --stage control`
EXPECT: «C1: OK» — при нулевом сдвиге (gamma = alpha = mds = 0, seed 42) прогнозы и фолды нового конвейера побитно равны кэшу B (`raw_/loyo_ctrl_f4` и `raw_/loyo_x3_gamma_f4`, чтение), `crit_pen` ctrl_f4 = 0,20114.
EVIDENCE: pending

### C2 — обучение с осями gamma / alpha / max_delta_step / seed
CHECK: `python3 -m pytest tests/test_v2_xl_xgbc_fit.py -q`
EXPECT: passed. Нулевой сдвиг побитно = `fit.fit_i1` (l2) и `fitobj._xgb` (gamma); каждая ось доходит до бустера и меняет прогноз; seed меняет прогноз при subsample < 1; мутации: ось игнорируется, max_delta_step=0 как «выкл.» перепутано, seed не передан — ловятся.
EVIDENCE: pending

### C3 — главный тест обрезки на каждой новой строке
CHECK: `python3 -m pytest tests/test_v2_xl_xgbc_leak.py -q`
EXPECT: passed. Для l2 и gamma с каждой из 9 осей прогноз на d побитно не меняется при порче баров после 18:50 дня d (синтетика); мутация (признаки +1 день) делает тест красным.
EVIDENCE: pending

### C4 — пространство, trial_id, дубли, бюджет
CHECK: `python3 -m pytest tests/test_v2_xl_xgbc_space.py -q`
EXPECT: passed. 18 точек X1; u=0 и зажатые во все годы — дубль; trial_id уникальны/детерминированы и зависят от оси, u и семьи; реестр только дописывается; `used.hp` ≤ 55; мутации: зажим снят, ось и u не в id, дубль учтён как trial — ловятся.
EVIDENCE: pending

### C5 — критерий, эффект, годовой выбор (PIT)
CHECK: `python3 -m pytest tests/test_v2_xl_xgbc_sel.py -q`
EXPECT: passed. `J` — только Y 2017–2021; годовой выбор `pen_Y` использует фолды y < Y и ничьи отдаёт И2; правило эффекта и cap 4; соседи u±1; мутации: фолды y ≥ Y, ничья не в пользу И2, J по всем годам, отсутствие cap — ловятся.
EVIDENCE: pending

### C6 — C-X1: 18 одно-осевых trials
CHECK: `python3 -W ignore models_v2/s6/xgb/runxc.py --stage x1`
EXPECT: `HP_AXES_C.csv` (ось, точка u, HP-пример, Δcrit_pen, QLIKE весь/2017–21/2022/2023–25/без 24.02.2022, годы зажима, статус); 18 строк (дубли помечены, не trial); реестр +≤18.
EVIDENCE: pending

### C7 — C-X2: решение и совместный Optuna или ABANDON
CHECK: `python3 -W ignore models_v2/s6/xgb/runxc.py --stage x2`
EXPECT: `X2_DECISION.json` (оси с эффектом, cap, решение); при ≥ 1 оси — 20 trials, строки `x2_best_j`, `x2_annual`; иначе запись `ABANDON: C-X2 <причина>` в README_C.md и в отчёт первой строкой.
EVIDENCE: pending

### C8 — C-X3: Gamma/QLIKE + learning_rate, min_child
CHECK: `python3 -W ignore models_v2/s6/xgb/runxc.py --stage x3`
EXPECT: 4 строки, Δ к `x3_gamma_f4` по crit_pen и по QLIKE 2017–21; вывод «снимает/не снимает».
EVIDENCE: pending

### C9 — seed и соседи лучших 2 (контроли)
CHECK: `python3 -W ignore models_v2/s6/xgb/runxc.py --stage seeds`
EXPECT: `HP_SEEDS_C.csv` (2 лучших + опора × seed 42–46, среднее, sd, min/max QLIKE), `HP_NEIGHBORS_C.csv` (соседи ±1 шаг); trials не растут.
EVIDENCE: pending

### C10 — таблицы, реестр, отчёт
CHECK: `python3 -W ignore models_v2/s6/xgb/runxc.py --stage tables` и `python3 -m pytest tests/test_v2_xl_xgbc_*.py -q`
EXPECT: `HP_RESULTS_C.csv` (все строки C + опоры, колонка status, crit_pen, DM к HAR-J(б)/ctrl_f4 с пометкой «сырые»), `README_C.md`, `CHECKPOINT.json` `used.hp` = число строк C-реестра ≤ 55; все тесты C passed.
EVIDENCE: pending

### C11 — гигиена
CHECK: `python3 -W ignore models_v2/s6/xgb/runx.py --stage check`
EXPECT: «B9: нарушений 0» (размеры файлов ≤ 200, функций ≤ 40, запрещённых обращений нет, прежнее не тронуто).
EVIDENCE: pending
