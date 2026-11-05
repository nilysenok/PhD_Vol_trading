# Передача — волна D (Ж-204), kodir, 05.11.2026 (запись до счёта, ~240k токенов)

Задание: `docs/zadaniya/2026-11-05_sber_h1_deep.md` (Волна D); гейты `docs/otchety/sber_h1_deep_v1/GATES_D.md` (написаны, EVIDENCE G0 заполнен, остальные pending). Не коммитить; процессов ≤ 6. Ждём «коммит есть» от дирижёра, на реальных данных ничего не считалось.

## Сделано
- D0: 6 тестов починены (tests/test_v2_deep_a_registry.py, _a_scope.py, _b1_tables.py; новый _c_tables.py). `pytest tests/test_v2_*.py` — 517 passed, 23 xfailed, 0 failed; `tests/test_v2_deep_*` — 209 passed (до тестов D).
- Код `models_v2/s5/d/`: `lgbfit.py` (LightGBM-пара fit_lgbx), `engine_d.py` (day_fit/daily: lib xgb|lgbm, win, hl; пул ≤ 6), `ens.py` (SPEC 13 ансамблей + 10 контролей, simplex, w_window/w_regime/w_inverse, build), `stats_d.py` (block_idx, mcs, boot_delta, holm_n, dm_periods), `select_d.py` (best_ensemble, top3, q_wf_dev), `diag_d.py` (loyo_years, top20, low_rv_*), `tab_d.py` (бюджет, реестр D, чекпойнт).
- Тесты `tests/test_v2_deep_d_{ens,fit,leak,stats,select}.py` — 45 passed вместе с `test_v2_deep_a_scope.py`.

## Не сделано (порядок после «коммит есть»)
1. `ctx_d.py` (кэш с ключом = ctx_c.code_sha + sha файлов d + C key + hp/seed/lib/win/hl; каталог `data/regen/v2_sber_h1_deep_v1/d/`), `stages_d.py`, `robust_d.py`, `report_d.py`, `run_d.py --stage control|comp|d1|robust|d2|report|all`. Использовать: `ctx_c.load()` (C: w, ctrl, i2, cols), `stages.daily(C, None, "f4", C["i2"])` (кэш C, без пула), `stages.chosen_hp()`, `mix.context`, `stages.boosters`, `tab_c.build_row(rid, fc, ctrl, f4, "wf", q_wf, 0.0, False, DIRECTION, tid, extra)`, `evalrows.q_wf`. Два пула последовательно (engine_c.pool для LOYO-фолдов top2-соседей, engine_d.pool для дневных прогонов).
2. Финалисты C пересобрать без обучения: `mix.build` над кэшем бустеров (c2_h3_roll_f4, c2_h3_roll_opt, c3_f3calm_rv, c1_opt = дневной кэш `opt`); сверка с `C_ALL.csv` ≤ 1e-12.
3. Стадии счёта: comp (3 прогона), d1 (13 ансамблей + 10 контролей + ссылки E2 из C_ALL; best по q_wf и по q_wf_dev), robust (seeds 43–46: top2, LGBM и нужные члены; соседи top2 через `robust.plateau_table`; соседи правила ансамбля win 126/504, step 0.05; окно 1008: opt, top2, LGBM), d2 (DM×5 периодов×2 базы, MCS 8 моделей, бутстрап Δ, Холм N=106/116, топ-20, LOYO по годам, нижний дециль, PNG кумулятивных разностей — `s5/cum.py`), report (реестр D, чекпойнт, FINAL_RANKING, STATISTICAL_TESTS, ENSEMBLE_RESULTS, MODEL_SPECIFICATIONS.json — признаки из BASELINE_MANIFEST/b2 feat, HP по годам из C_CHOICE/I2, sha; REPRODUCTION.md; SBER_H1_BOOSTING_DEEP_IMPROVEMENT_FINAL.md с таблицами 1–5; README.md).
4. EVIDENCE G1–G11 в GATES_D.md, ответ ≤ 500 слов.

## Ловушки
- Таблица `FEATURE_ABLATIONS.csv` не содержит колонки `row_id` (ключ `model`).
- Mask нижнего дециля при «≤» не чувствительна к включению текущего дня — мутант проверяется на пороге (`low_rv_thr`).
- K=2 веса совпадают с `rules.h3_weights` до 1e-15, но не побитно (сетка 1−w).
- Heredoc / `python3 -c`, `sed -i` запрещены; скрипты — файлами.
- Не менять `models_v2/` вне `s5/d/`; optuna в d не нужна.
