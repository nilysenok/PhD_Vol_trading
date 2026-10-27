# GATES — Э1-а′ драйверы перебора (v1: OAT + sweep_run.py; v1.1: факторный, Моррис, Optuna, PBO), кодер 2, 27.10

Тестов в репозиторий не пишу (их пишет тестировщик в `tests/test_sweep_*.py`). Проверочные скрипты G1–G7 — разовые, вне репозитория
(`gates_v1.py`, SynthEvaluator; независимый пересчёт плато и размаха вручную из той же таблицы). Запуск: `FASTBT_MAIN=$PWD python3 gates_v1.py` (хук требует).

## v1

## G1 — OAT даёт ровно Σ(|сетка_i|−1)+1 точек; база вне сетки отвергается
CHECK: FASTBT_MAIN=$PWD python3 gates_v1.py   # блок G1
EXPECT: «G1 ok 6 6» (сетки 4 и 3 узла: 3+2+1=6); ValueError при base.a=7
EVIDENCE: G1 ok 6 6

## G2 — повторный запуск попадает в кэш сигналов, результат тот же
CHECK: тот же скрипт, блок G2
EXPECT: вызовов signals: 1-й запуск > 0, 2-й = 0; таблицы равны (assert_frame_equal)
EVIDENCE: G2 ok 4 0

## G3 — validation (2020–2021) не читается при period=train
CHECK: тот же скрипт, блок G3 (Spy-evaluator пишет все `end`)
EXPECT: max(end) ≤ 2020-01-01; period=validation/test отвергается ValueError
EVIDENCE: G3 ok 2020-01-01 00:00:00

## G4 — обе комиссии, по файлу на комиссию
CHECK: тот же скрипт, блок G4
EXPECT: S9_oat_0.035.csv, S9_oat_0.04.csv, S9_oat_tornado.csv; в файле комиссии одно значение comm_pct
EVIDENCE: G4 G5 ok ['S9_oat_0.035.csv', 'S9_oat_0.04.csv', 'S9_oat_tornado.csv']

## G5 — CSV с шапкой статуса (статус, код, хэш входов), читается pandas comment='#'
CHECK: тот же скрипт, блок G5
EXPECT: строки «# статус: разработка, train 2018–2019», «# код:», «# вход:»; read_csv даёт 6 строк
EVIDENCE: G4 G5 ok (см. G4)

## G6 — плато по параметру = среднее Шарпа по соседям ±1 узла этого параметра; торнадо = размах Шарпа
CHECK: тот же скрипт, блок G6
EXPECT: расхождение с независимым пересчётом < 1e-12
EVIDENCE: G6 ok

## G7 — процессов ≤ 8
CHECK: тот же скрипт, блок G7 (`sweep_run.main(["x.yaml","--jobs","9"])`)
EXPECT: ValueError; пул = min(jobs, число сеток) в коде
EVIDENCE: G7 ok

## G8 — время S{k} на train, одна комиссия, реальные данные
CHECK: run_oat на S3 (dc_window×3 узла, initial_sl_mult×3), comm 0,035, сетки архитектора ещё нет
EXPECT: время выведено
EVIDENCE: 5 точек S3 (холодный кэш сигналов) — 1,7 с, т.е. ≈0,3 с/точку; цифра по сеткам архитектора — потолок оценки, пересчитать на настоящей сетке

## G9 — компиляция и размер
CHECK: python3 -m py_compile fastbt/sweep_run.py; wc -l
EXPECT: код 0; ≤ 200 строк
EVIDENCE: ok; 162 строки

## v1.1 (вторым PR)

## G10 — факторный план, Моррис, Optuna, PBO (pbo_cscv в sweep_pbo.py)
CHECK: FASTBT_MAIN=$PWD python3 -m pytest tests/test_sweep_modes.py tests/test_sweep_stat.py -q   # тесты кодера 1/тестировщика, в т.ч. PBO ≈ 0,5 / лидер < 0,1 / крайности; Моррис; Optuna seed, pruning, n_jobs>8
EXPECT: все passed
EVIDENCE: 13 passed in 0.64s. Прочее: `ss.pbo_cscv` остался доступен реэкспортом; факторный план > 20000 точек → ValueError. Мимо моих файлов: tests/test_sweep.py::test_param_changes_result (RecursionError, лямбда в monkeypatch вызывает сама себя) и test_param_pnl_side_changes_result падают — не мои, до v1.1 тоже.

## Реестр
v1: закрыто 9 из 9; v1.1: закрыто 1 из 1; ABANDON: 0
