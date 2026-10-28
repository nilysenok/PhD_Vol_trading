# GATES — Э1-в, режимы итераций (factorial / morris / optuna) поверх `sweep_drivers` владельца, T1 кодер 2, 27.10

Ветка `iter-e1-rezhimy` от `oat-e1-v1` (#315). Код: `fastbt/sweep_run_iter.py` (новый, режимы) + правки `fastbt/sweep_run.py`
(`--mode`, постоянный ledger `<out>/_ledger/<S>/ledger.json`, диспетчер). `sweep_drivers.py` не правится. Всё считает модуль
владельца; здесь — сетка уровней, замыкание по комиссии, таблицы, DSR по ledger. Проверочный скрипт `gates_v4.py` разовый,
вне репозитория (синтетический `sweep_synth.SynthEvaluator`, сетка формата #305 с условным блоком); тесты пишет тестировщик.
Запуск: `FASTBT_MAIN=$PWD python3 gates_v4.py`. Pytest: только `tests/test_sweep_*.py` (запрет 19:15 на остальные).

## G1 — факторный: 3 параметра × 3 уровня = 27 различных точек; уровни = центр ± шаг сетки вокруг базы; база на краю → три ближайших узла; условие блока подставлено
CHECK: gates_v4.py, блок G1
EXPECT: 27 точек; центр [i-1,i,i+1]; край [0,1,2] / [n-3,n-2,n-1]; сетка < 3 узлов или база вне сетки → ValueError; условный + его условие среди выбранных → ValueError
EVIDENCE: «G1 ok 27» (синтетика); на S3: min_strength (база 0,0 на краю) → [0,0; 0,1; 0,2]

## G2 — Моррис: `sd.morris_screen`, seed 20261027, paths по флагу (10); повтор тот же
CHECK: gates_v4.py, блок G2
EXPECT: вызов модуля на каждую комиссию (2); n_evaluations = 10·(1+3)=40; mu_star = прямому вызову (< 1e-12); повтор побайтно
EVIDENCE: «G2 ok»

## G3 — Optuna: `sd.optuna_search`, seed 20261027, пространство = сетка #305, n_jobs ≤ 6
CHECK: gates_v4.py, блок G3
EXPECT: аргументы модуля (20, 20261027); n_trials_modul = 20; повтор при n_jobs=1 — те же точки; ключи = сетка; n_jobs=7 → ValueError
EVIDENCE: «G3 ok 12» (12 уникальных из 20 проб)

## G4 — выбор: факторный — `sd.pick_by_plateau` (radius 1, min_neighbors 3); Optuna — плато по соседям сетки среди посчитанных точек; топ-3 в `S{k}_top3.csv`
CHECK: gates_v4.py, блок G4
EXPECT: top3 факторного = первые три pick_by_plateau; плато Optuna = независимый пересчёт (< 1e-12); колонки plateau, neighbors, edge, mode, rank
EVIDENCE: «G4 ok»

## G5 — цель = настройка 2018, контроль 2019 отдельными колонками; validation и ≥ 2022 не читаются
CHECK: gates_v4.py, блок G5
EXPECT: objective == tune_sharpe == tune_sharpe_2018; ctrl_sharpe_2019 есть; max(end) ≤ 2020-01-01
EVIDENCE: «G5 ok 2020-01-01 00:00:00»

## G6 — N = уникальные пробы ledger при 0,04, накопленные между запусками (OAT + factorial + optuna); 0,035 N не растит; DSR = `sd.dsr` по всему ledger
CHECK: gates_v4.py, блок G6
EXPECT: ledger после factorial = OAT ∪ 27 точек; после optuna ⊇ предыдущего, прирост ≤ 20; колонка N = len(ledger) в обеих комиссиях; dsr = прямому sd.dsr (< 1e-12)
EVIDENCE: «G6 G7 ok: OAT 16 -> factorial 36 -> optuna 47»

## G7 — условный блок в Optuna: неактивный сводится к базе, дубль не увеличивает N
CHECK: gates_v4.py, блок G7
EXPECT: две пробы с разным неактивным условным параметром = 1 запись ledger; активные = 2
EVIDENCE: «G6 G7 ok» (ledger 1, затем 3)

## G8 — выход: `S{k}_{mode}_{comm}.csv` на 0,04 и 0,035, `S{k}_top3.csv`, шапка статуса; повторный режим дополняет top3
CHECK: gates_v4.py, блок G8 (+ G4: top3 с обоими mode, 6 строк)
EXPECT: файлы есть; первая строка «# статус: разработка, train 2018 (tune) / 2019 (ctrl)»
EVIDENCE: «G8 ok»

## G9 — OAT не сломан, `sweep_drivers.py` без правок, ≤ 200 строк, компиляция
CHECK: `python3 -m pytest tests/test_sweep_*.py -q -W ignore` ; `git diff origin/oat-e1-v1 --stat -- fastbt/sweep_drivers.py` ; `wc -l fastbt/sweep_run*.py` ; `python3 -m py_compile ...`
EXPECT: 0 failed; пустой diff; ≤ 200; код 0
EVIDENCE: 97 passed, 2 xfailed; diff пуст; sweep_run.py 179, sweep_run_iter.py 185; compiled

## G10 — дымовой прогон S3 на реальных данных (таблицы НЕ коммитятся, черновик в /private/tmp): OAT → factorial по топ-3 торнадо → optuna 20 проб
CHECK: `python3 -m fastbt.sweep_run [--mode factorial --params tornado:3 | --mode optuna --trials 20] S3_setka_e1b.yaml --out <черновик> --jobs 1`
EXPECT: три режима отработали, время и N записаны; validation не открывалась
EVIDENCE: OAT 4 с (ledger 66); factorial (min_strength, cooldown_bars, rm_parabolic.parabolic_step) 1 с, ledger 66→90; optuna 20 проб 1 с, 90→108; N накопленное S3 = 108 (после ещё Морриса по тем же 3: 129, вне запрошенной цепочки). Время малое: сигналы S3 кэшируются, расчёт PnL на 2018–2019 быстрый.
