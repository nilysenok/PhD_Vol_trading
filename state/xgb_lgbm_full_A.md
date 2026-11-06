# Передача №2 — волна A (Ж-206), kodir, 06.11.2026 (~250k токенов). Статус: ПЕРЕДАЧА, не «готово»

Не коммитить. Процессов <= 6. Скрипты — Write (heredoc/`python3 -c` запрещены). Хук post-edit гоняет несвязанные `tests/test_*diag*` (2 падения в `tests/test_s6v9_diag.py` — не мои, существовали до правок).

## Сделано (код `models_v2/s6/`, запуск `python3 models_v2/s6/run_a.py --stage ...`)
paths6, engine6 (fit_var: base/S1/S2, clip99/drop5, статистика листьев), lgb_b2 (аналог Б-2 + seeds 43–46 + S1/S2 + контроль A9), origin (ast R1), inv_meta/inv/inv_cd (INVENTORY), diag (PREDICTIONS/ANNUAL/REGIME + `--check`), lgb_extra, baseline (BASELINE.json, OPTUNA_TRIALS.csv шапка, CHECKPOINT.json нули), answers (пересчёт Холма), stages (inventory/--check/--check-sha), run_a.
Данные: `data/regen/xgb_lgbm_full_v1/lightgbm/*` (fc_lgbm_b2[_s43..46,_S1,_S2].parquet, st_*.json, CONTROL_A9.json), `docs/otchety/xgb_lgbm_full_v1/{INVENTORY.csv, ANSWERS_V.csv, xgboost/*, lightgbm/*}`.
Подтверждено прогонами: A9 (Б-2 побитно к fix_forecasts_c на 3 датах, LGBM два запуска побитно), `inventory --check` (288 строк, 0 нарушений: таблицы < 1e-12, контрольные значения < 5e-5), `diag --check` (A11, 0 нарушений), ANSWERS_V: p Холма 0,037949 (C, N=90: 19+71 единиц), 0,045280 (D, DM, 6 финалистов+100 единиц), 0,041992 (D, бутстрап) — совпали с таблицами < 1e-16.

## НЕ сделано (по порядку)
1. Стадии `check` / `audit` / `check-readme` (модули `checks.py`, `readme.py`; `stages.dispatch` уже ссылается на них: checks.stage(a), readme.stage(a)): A13 (OPTUNA_TRIALS пустые, CHECKPOINT=0, `git status models_v2` вне s6, find -newer), A14 (файлы <=200 строк, функции <=40: проверить diag.py, inv_cd.py `rows` >40 строк, baseline.stage), A16 (grep `_test\.|test_locked|\.env` в models_v2/s6), A15/A5 (README).
2. Тесты `tests/test_v2_xl_{inventory,origin,baseline,leak,diag,lgbextra}.py` с мутациями (A1, A4, A6, A8, A10, A12). Образец утечки — `tests/test_v2_boost_leak.py` (damage/invariant, synth бары); для новых строк: ft.build(bars) -> design.make(F,"m") -> runner.pairs_upto -> engine6.fit_var (base, S1, S2) на d: прогноз побитно равен при порче баров после 18:50 d; мутации: X сдвинут на +1 день; пары с t+1 = d+1; `diag.regime_pit(..., include_current=True)` (№ 22). fit_var(base) == lgbfit.fit_lgbx побитно. Мутации A4: подмена xgb.train->lgb.train в исходнике меняет `origin.classify_src`; классификация по имени (lgbm в названии) на обманчивой строке. A6: подмена max_depth по году в BASELINE.json. A10: режим по факту RV; QLIKE с σ вместо σ². A12: S1 (max_depth=-1) отличается от base; clip99 меняет прогноз.
3. README «Волна A» (скрипт readme.py из CSV/JSON) с разделами: инвентаризация (ссылка), ответы (а)–(в), baseline, диагностика с цифрами, предложения для B (два списка), список «мутация -> тест».
4. EVIDENCE в `GATES_A.md` (выполнить CHECK каждого гейта, вывод в EVIDENCE; сейчас все pending). Реестр: закрыто 0 из 16 формально.
5. `--check-sha` (A3) написан, НЕ запускался. Причина «не подтверждено» в INVENTORY: 18 строк (в т. ч. c2_h2l4_f4_c05 — прогноз не сохранён по имени).

## Отступления / находки
- ↺ 23.02.2022 — нерабочий день (нет в ряду); в LGB_LEAVES/EXTRA взят d = 22.02.2022 (последний d перед целью 24.02.2022).
- ↺ «после отрицательной доходности» — прокси (rs− > rs+ на d), дневной доходности в разрешённых таблицах нет; скачок по факту: j/rv цели > 0,3 (объявлено до счёта, в diag.JUMP_SHARE).
- INVENTORY: 288 строк (а не ~200): пилот p1/p2/p3/pb/fix + B1/B2 + C + D; поля, не разобранные по коду, помечены «не подтверждено».
- Распределение k по датам для Б-2 в прежних файлах не сохранено (только среднее st_BASE.json) — в BASELINE.json пометка.
- Хук post-edit запускает несвязанные тесты `*diag*`; их падения (tests/test_s6v9_diag.py) не относятся к волне.

## Результаты (источник — CSV в каталогах алгоритмов)
QLIKE: Б-2 0,2205; LGBM-аналог (seed 42) 0,2223; seeds 42–46: 0,2223/0,2206/0,2213/0,2201/0,2214; S1 0,2229; S2 0,2299 (деревья 114,6/109,7/78,2). Первая цифра — потолок.
