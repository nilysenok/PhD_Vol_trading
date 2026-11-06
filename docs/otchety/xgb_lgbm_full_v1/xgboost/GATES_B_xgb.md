# GATES_B_xgb — волна B (функции потерь и признаки), поток XGBoost, программа Ж-206

Задание: `docs/zadaniya/2026-11-05_xgb_lgbm_full.md` (волна B, поток XGB); блок: `docs/zadaniya/2026-11-05_xgb_lgbm_full_blok.md` (при расхождении действует блок). Статус результатов: DEVELOPMENT (выбор — внутренний критерий LOYO) и HISTORICAL WALK-FORWARD (ряды прогнозов на уже просмотренной истории 2017–02.2026, 2271 дата; не независимый OOS). Первая цифра — потолок. Код: `models_v2/s6/xgb/`, запуск `python3 models_v2/s6/xgb/runx.py --stage <control|stab|trials|cond|tables|check>`; тесты `tests/test_v2_xl_xgb_*.py`; данные `data/regen/xgb_lgbm_full_v1/xgboost/`; отчёты `docs/otchety/xgb_lgbm_full_v1/xgboost/`. Формат: CHECK / EXPECT / EVIDENCE (`pending` до запуска). Правила и список конфигураций ниже объявлены до счёта и не двигаются после просмотра результата (изменение — только строкой «↺» с датой и обеими версиями).

## Правила, объявленные до счёта
- **P1 (внутренний критерий).** Годовой LOYO по прошлым целям (< 01.01.Y, HP года Y из I2, как `models_v2/s4/select.py` и `s5/b1/jobs.fold_job`): для года Y (2017…2026) и фолдов y ∈ [2015, Y−1] берётся QLIKE(б) q(Y, y). `pen_Y = mean_y q + 0,25 · (max_y q − mean_y q)`; критерий `crit_pen = mean_Y pen_Y` (контроль худших периодов: худший ПРОШЛЫЙ год относительно Y). Рядом `crit_mean = mean_Y mean_y q` (то же, что `q_loyo` Ж-204, для сопоставления). NaN-фолды (год < 50 пар) пропускаются.
- **P2 (перспективно).** Строка перспективна, если `crit_pen` строки строго лучше `crit_pen` baseline XGBoost (Б-2, `fx_b2_g132_b`; фолды — `data/regen/v2_sber_h1_deep_v1/b1/loyo_ctrl_b2.parquet`, только чтение) на > 1e-12 И разность QLIKE к HAR-J(б) без 24.02.2022 ≤ 0. Так как Б-2+F4 заведомо лучше Б-2, прирост НАД опорной строкой Б-2+F4 (`ctrl_f4`) показывается отдельной колонкой `beats_f4_ref`: `crit_pen` строки < `crit_pen` `ctrl_f4`. Решения «брать в следующие шаги» опираются на `beats_f4_ref`.
- **P3 (дубли и перенос).** trial_id = первые 12 hex sha256 канонической JSON-конфигурации (алгоритм, потеря, веса γ, группы признаков, drop, HP-источник, seed, окно, частота). Конфигурация, совпадающая с Ж-197…Ж-204, не считается trial: ссылка на прежний trial_id, счётчик не растёт, прогноз не пересчитывается. Перенос — из `docs/otchety/sber_h1_deep_v1/{OBJECTIVE_EXPERIMENTS,FEATURE_ABLATIONS,C_ALL}.csv` и реестров trials; строка с пометкой `transferred=true`.
- **P4 (воспроизводство опорной строки).** `ctrl_f4` (Б-2+F4, l2, I2, seed 42) в новом конвейере = `data/regen/v2_sber_h1_deep_v1/b2/fc_b2_f4.parquet` побитно на 3 датах (2018-06-01, 2022-06-01, 2025-06-02) и `crit_mean` = 0,1546074… (`st_b2_f4.json`) с допуском 1e-9. Это контроль, не trial.
- **P5 (устойчивость кастомного QLIKE).** (а) аналитические g/h = конечные разности (центральные, h = 1e-4) на сетке y ∈ [1e-3; 1e3], F ∈ [−6; 6], относительная ошибка ≤ 1e-6; (б) на фиксированных датах (первая d каждого года сетки и 22.02.2022; их число по сетке — публикуется в `STABILITY_QLIKE.csv`): fit custom QLIKE и `reg:gamma` на одном окне — все прогнозы конечны и > 0; |Δ ln σ̂²| между ними и разность k деревьев публикуются без порога (диагностика эквивалентности потерь: при log-link gamma-deviance совпадает с QLIKE, отличие — ранняя остановка и минимум гессиана); (в) строка «нестабильна», если хотя бы один σ̂² не конечен, ≤ 0 или σ̂²/σ̂²(ctrl_f4) вне [0,1; 10]; нестабильная строка не получает «перспективно».
- **P6 (метрики строки).** Только из ряда прогнозов (2271 дата, `metrics.qlike`): весь, 2017–2021, 2022, 2023–2025, 2026, без 24.02.2022, без 2022, RV/σ̂² (`mean_r`), разность и DM (Ньюи–Уэст, `s3/dm.py`, как Ж-204) к HAR-J(б) и к Б-2 и к `ctrl_f4`; колонка `status`. p-значения сырые: множественный поиск не учтён (Холм — в отдельной волне), потолок.
- **P7 (процессы).** ≤ 4 процессов (`jobs.run` — spawn-пул ≤ 4). Старые прогнозы и таблицы не перезаписываются; кэш — только в `data/regen/xgb_lgbm_full_v1/xgboost/`, ключ = конфигурация + код (`s6/xgb`, `s5/b1/{jobs,fitobj,losses,designs}.py`, `s4/fit.py`) + хэш входов.

## Объявленные конфигурации (бюджет obj/feat ≤ 45; опорная строка Б-2+F4, HP I2, seed 42, расширяющееся окно, ежедневное переобучение)
Новые (9): `x3_gamma_f4` (X3, reg:gamma, log-link, цель RV/S); `x4_qlike_f4` (X4, custom QLIKE, цель RV/S, base_margin ln среднего); `x5_l2_g05_f4`, `x5_l2_g10_f4` (X5, L2 по ln-цели, веса w ∝ RV_w^(−γ), γ = 0,5; 1,0); `x5_qlike_g05_f4` (X5 на X4, γ = 0,5); `f4_f2` (Б-2+F4+F2); `f4_abl_no_imoex`, `f4_abl_no_market`, `f4_abl_no_semivar` (обратная абляция на Б-2+F4 по `engine.DROP`; no_market снимает и прокси, и IMOEX).
Условные (объявлены сейчас, исполняются только по правилу): `cond_best_obj_f4_f2` — лучший по `crit_pen` объектив из X3/X4/X5 на признаках Б-2+F4+F2, только если и этот объектив, и `f4_f2` имеют `beats_f4_ref`; `f4_f1_f2` — Б-2+F4+F1+F2, только если `f4_f2` имеет `beats_f4_ref`. Итого новых trials ≤ 11 из 45.
Переносятся без пересчёта (ссылкой на trial_id): X1/X2 (L2 ln-цель = Б-2, `obj_l1n`/`obj_l1raw` — L2 по уровню, катастрофически хуже), L3/L4/L5 на Б-2 (`obj_l3x`, `obj_l3l`, `obj_l4`, `obj_l5a/b/c`), F1–F6 одиночные, top2/top3, абляции на Б-2 (`no_gap`, `no_imoex`, `no_market`, `no_semivar`), `c1_f4_range`, `c1_f4_park`, Б-2+F4+F1 (`c1_f4_f1`), F3/F4 в калме (`c3_f3calm_*`, смесь двух XGBoost — защита F3 в стрессе уже сделана), окна/частота, Optuna.

## Гейты

### B1 — перенос Ж-204 без пересчёта
CHECK: `python3 -m pytest tests/test_v2_xl_xgb_transfer.py -q`
EXPECT: passed. `FEATURE_SETS.csv` и `OBJECTIVE_RESULTS.csv` содержат строки Ж-204 с `transferred=true`, trial_id из реестров Ж-204, числа равны исходным CSV (сверка по trial_id); колонка `status`; мутации: подмена trial_id, число из другой строки, `transferred=false` на переносе — тест падает.
EVIDENCE (06.11.2026): `pytest tests/test_v2_xl_xgb_transfer.py -q` -> 9 passed, rc 0. Перенесено 8 строк OBJECTIVE_RESULTS и 17 FEATURE_SETS с transferred=true. ЗАКРЫТ.

### B2 — воспроизводство опорной строки (P4)
CHECK: `python3 models_v2/s6/xgb/runx.py --stage control`
EXPECT: «B2: OK», побитное равенство σ̂² ctrl_f4 и `fc_b2_f4.parquet` на 3 датах; `crit_mean` ctrl_f4 в 1e-9 от 0,1546074…; `crit_pen` Б-2 и ctrl_f4 записаны в `CONTROL_B.json`.
EVIDENCE (06.11.2026): `runx.py --stage control` (выполнена ранее в этой передаче) -> «B2: OK»; CONTROL_B.json: 2018-06-01 / 2022-06-01 / 2025-06-02 bitwise=true, full_grid_max_rel_diff=0.0, crit_mean ctrl_f4 = 0.1546074143317876 = st_b2_f4.json; crit_pen Б-2 = 0.21473, ctrl_f4 = 0.20114. ЗАКРЫТ.

### B3 — численная проверка кастомного QLIKE (P5)
CHECK: `python3 -m pytest tests/test_v2_xl_xgb_stab.py -q` и `python3 models_v2/s6/xgb/runx.py --stage stab`
EXPECT: passed; `STABILITY_QLIKE.csv` (фиксированные даты × {qlike, gamma}: k, σ̂², |Δ ln|, конечность); (а) относительная ошибка g/h ≤ 1e-6; мутации: знак g, h без y, пол гессиана 1e-16 → 1.0 — ловятся.
EVIDENCE (06.11.2026): `pytest tests/test_v2_xl_xgb_stab.py -q` -> 8 passed; `runx.py --stage stab` -> fd_error = 1.898e-07 (допуск 1e-06), 12 дат × {qlike, gamma} все конечны; STABILITY_QLIKE_PAIRS.csv: |Δ ln σ̂²| между QLIKE и gamma ≤ 5,8e-08, Δk = 0 на всех 12 датах (потери эквивалентны при log-link, как и ожидалось). ЗАКРЫТ.

### B4 — главный тест обрезки на каждой новой строке
CHECK: `python3 -m pytest tests/test_v2_xl_xgb_leak.py -q`
EXPECT: passed. Для каждого типа новой строки (gamma, qlike, l2+γ, qlike+γ, +F2, три абляции, условные) прогноз на d побитно не меняется при порче/обрезке баров после 18:50 дня d (синтетика); мутации: признаки +1 день (№ 1), пары с t+1 = d+1, веса γ по RV_w с днём d — ловятся.
EVIDENCE (06.11.2026): `pytest tests/test_v2_xl_xgb_leak.py -q` -> 15 passed, rc 0. ЗАКРЫТ.

### B5 — внутренний критерий (P1)
CHECK: `python3 -m pytest tests/test_v2_xl_xgb_crit.py -q`
EXPECT: passed. `crit_pen` на известном ответе (фолды с заданными q) = ручной расчёт; штраф за худший ПРОШЛЫЙ год: фолд y ≥ Y не входит; мутации: mean вместо max, 0,25→1, фолды года Y в pen_Y, перепутанный знак — ловятся.
EVIDENCE (06.11.2026): `pytest tests/test_v2_xl_xgb_crit.py -q` -> 7 passed, rc 0. ЗАКРЫТ.

### B6 — конфигурации, дубли, реестр, бюджет
CHECK: `python3 -m pytest tests/test_v2_xl_xgb_cfg.py -q`
EXPECT: passed. 9 объявленных + ≤ 2 условных; trial_id уникальны и детерминированы; конфигурация, совпадающая с Ж-204 (Б-2+F4, Б-2+F4+F1), не получает нового trial_id и не увеличивает счётчик; `OPTUNA_TRIALS.csv` только дописывается (повторный запуск не дублирует), `CHECKPOINT.json` `used.obj_feat` = число новых строк реестра. Мутации: дубль учтён как trial, повторная запись удваивает.
EVIDENCE (06.11.2026): `pytest tests/test_v2_xl_xgb_cfg.py -q` -> 6 passed; все 55 тестов `tests/test_v2_xl_xgb_*.py` -> 55 passed. ЗАКРЫТ.

### B7 — счёт: ряды прогнозов новых trials
CHECK: `python3 models_v2/s6/xgb/runx.py --stage trials` (затем `--stage cond`)
EXPECT: для каждой объявленной конфигурации `fc_<id>.parquet` (2271 дата, σ̂² > 0 и конечна, сетка = Б-2), `loyo_<id>.parquet`, запись в `OPTUNA_TRIALS.csv`; `CHECKPOINT.json` `used.obj_feat` = 9 (+ условные). Условные запущены или не запущены строго по правилу (причина — в `COND_DECISION.json`).
EVIDENCE (06.11.2026): `--stage trials` выполнена ранее; `--stage cond` -> rc 0, COND_DECISION.json: best_obj=null, f4_f2_beats=false, run_cond_obj=false, run_f1_f2=false (ни один объектив и ни `f4_f2` не бьют ctrl_f4 по crit_pen; условные НЕ запущены строго по правилу). CHECKPOINT.json used.obj_feat = 9; OPTUNA_TRIALS.csv = 9 строк. Ряды прогнозов: raw_<id>.parquet и loyo_<id>.parquet по 9 trials + ctrl_f4, по 2271 строке, unstable=0 у всех (σ̂² конечны, > 0, отношение к ctrl_f4 в [0,1; 10]). ОТСТУПЛЕНИЕ: имя файла `raw_<id>.parquet` (кэш calcx), а не `fc_<id>.parquet` из EXPECT; содержание то же. ЗАКРЫТ с отступлением по имени.

### B8 — таблицы
CHECK: `python3 models_v2/s6/xgb/runx.py --stage tables`
EXPECT: `OBJECTIVE_RESULTS.csv` (перенос + X3/X4/X5), `FEATURE_SETS.csv` (перенос + F4+F2, абляции, условные), колонки: id, trial_id, transferred, objective, features, crit_mean, crit_pen, perspective, beats_f4_ref, unstable, qlike по периодам, без 24.02.2022, без 2022, mean_r, d/DM/p к HAR-J(б), к Б-2, к ctrl_f4, status; шапка со статусом; ни одна строка не получает результат другой.
EVIDENCE (06.11.2026): `runx.py --stage tables` -> «B8: OBJECTIVE_RESULTS 14 строк, FEATURE_SETS 22 строк; дубли id/trial_id: нет», rc 0. ЗАКРЫТ.

### B9 — гигиена
CHECK: `python3 models_v2/s6/xgb/runx.py --stage check`
EXPECT: «B9: нарушений 0»: файлы `models_v2/s6/xgb/*.py` ≤ 200 строк, функции ≤ 40 (иначе пометка `# xtask: длинная`); нет `_test.`, `test_locked`, `.env` в коде; `git status --short models_v2` вне `s6/` и вне чужих изменений — без новых строк от этой работы; прежние файлы `data/regen/v2_sber_h1*` и `docs/otchety/sber_h1_deep_v1` не новее коммита GATES (пересчёт командой).
EVIDENCE (06.11.2026): `runx.py --stage check` -> «B9: нарушений 0 {}», rc 0. ЗАКРЫТ.

### B10 — отчёт
CHECK: ручной (выражен частично командой `runx.py --stage tables`): ответ ≤ 500 слов; числа из CSV; отрицательные результаты сохранены; перенесённое отделено от нового.
EXPECT: реестр «закрыто N из 10»; ABANDON-гейтов нет либо названы первой строкой.
EVIDENCE (06.11.2026): `xgboost/README_B.md` написан; числа из OBJECTIVE_RESULTS.csv / FEATURE_SETS.csv; перенесённое отделено от нового; отрицательные результаты сохранены. Реестр: закрыто 10 из 10 (B7 — с отступлением по имени файла); ABANDON нет. Ручной гейт: проверка ответа ≤ 500 слов — за дирижёром.
