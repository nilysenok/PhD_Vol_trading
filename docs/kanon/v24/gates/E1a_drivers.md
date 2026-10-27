# GATES — Э1-а′ (Ж-035): адаптер оценщика к `sweep_drivers.oat_sweep` владельца + запуск OAT, кодер 2, 27.10

Заменяет первую редакцию (свои драйверы; отменена Ж-035, v1.1 закрыт без слияния). Проверочные скрипты G1–G7 разовые, вне репозитория
(`gates_v2.py`, SynthEvaluator; независимый пересчёт плато и размаха). Запуск: `FASTBT_MAIN=$PWD python3 gates_v2.py` (хук требует).
`sweep_drivers.py` не правился.

## G1 — число расчётов OAT = Σ(|сетка|−1)+1; проб (n_trials) = 1+Σ|сетка| и это N в DSR
CHECK: gates_v2.py, блок G1
EXPECT: расчётов 6, проб 8 для сеток 4 и 3 узла; колонка N = 8; база вне сетки → ValueError
EVIDENCE: G1 ok: расчётов 6 проб 8

## G2 — сигналы считаются один раз на параметры сигнала, общие для двух комиссий
CHECK: gates_v2.py, блок G2
EXPECT: signals() вызван 4 раза (4 узла сигнального параметра) на обе комиссии
EVIDENCE: G2 ok: signals calls 4

## G3 — validation и ≥ 2022 не читаются
CHECK: gates_v2.py, блок G3 (Spy пишет все end в evaluator.pnl)
EXPECT: max(end) ≤ 2020-01-01; evaluate(period="validation"/"test") → PermissionError
EVIDENCE: G3 ok 2020-01-01 00:00:00

## G4 — обе комиссии, CSV и торнадо на каждую, шапка статуса
CHECK: gates_v2.py, блок G4/G5
EXPECT: S9_oat_{0.035,0.04}.csv и S9_oat_tornado_{0.035,0.04}.csv; шапка «# статус: разработка, train 2018–2019 / # код / # вход»; read_csv(comment='#') читает
EVIDENCE: G4 G5 ok [4 файла]

## G5 — плато (±1 узел этого параметра, Шарп) и размах Шарпа в торнадо = независимый пересчёт; DSR ∈ [0,1]
CHECK: gates_v2.py, блок G6
EXPECT: расхождение < 1e-12
EVIDENCE: G6 ok

## G6 — процессов ≤ 8
CHECK: gates_v2.py, блок G7
EXPECT: `--jobs 9` → ValueError; пул = min(jobs, число сеток)
EVIDENCE: G7 ok

## G7 — самопроверка модуля владельца
CHECK: python3 fastbt/sweep_drivers.py; echo $?
EXPECT: 0
EVIDENCE: «sweep_drivers: самопроверка пройдена; OAT n = 13 …», exit=0

## G8 — sweep_modes.py удалён; тесты тестировщика
CHECK: git ls-files fastbt/sweep_modes.py; pytest tests/test_sweep_stat.py
EXPECT: файла нет; stat проходит
EVIDENCE: удалён. ЧАСТИЧНО: tests/test_sweep.py и tests/test_sweep_modes.py не собираются (ImportError sweep_modes; `fastbt/sweep.py` кодера 1 импортирует sm) — нужна правка у их владельцев; test_sweep_stat.py — см. отчёт

## G9 — время S{k} на реальных данных train, одна комиссия
CHECK: S3 (dc_window×3, initial_sl_mult×3), comm 0,035
EXPECT: время выведено
EVIDENCE: 5 расчётов, 7 проб — 0,4 с (≈0,08 с/расчёт; сигналы S3 лёгкие; потолок оценки, настоящие сетки считать отдельно)

## G10 — размер, компиляция
CHECK: py_compile; wc -l fastbt/sweep_run.py
EXPECT: 0; ≤ 200 строк
EVIDENCE: ok; 150 строк

## Реестр
закрыто 9 из 10; частично G8 (тесты кодера 1 / тестировщика ждут их правки); ABANDON: 0
