# GATES — Э1-а′ (Ж-035, правки по вердикту #302): запуск OAT поверх `sweep_drivers` владельца, кодер 2, 27.10

Ветка `sweep-drivers` пересобрана от `sweep-engine-2` (#308) + `kanon-v22-pereschet` (#305 слит). Сетки читаются через `sweep_setka.expand`.
Проверочные скрипты G1–G8 разовые, вне репозитория (`gates_v3.py`, SynthEvaluator с сеткой формата #305 вкл. условный блок);
запуск: `FASTBT_MAIN=$PWD python3 gates_v3.py` (хук требует). `sweep_drivers.py` не правился.

## G1 — уникальных проб в ledger = множеству точек OAT (все группы, база условного блока со своим условием); N в CSV = len(ledger); n_trials_modul — отдельно (больше N)
CHECK: gates_v3.py, блок G1
EXPECT: ledger 9 = независимо построенное множество 9; n_trials_modul 12 > N
EVIDENCE: G1 ok: проб 9 n_trials_modul 12 (гейт поймал баг: ключ ledger без условия блока — исправлено до коммита)

## G2 — сигналы общие на обе комиссии
CHECK: gates_v3.py, блок G2
EXPECT: signals() = 4 (узлы сигнального параметра)
EVIDENCE: G2 ok signals calls 4

## G3 — validation и ≥ 2022 не читаются; период сетки сопоставлен словарём tune2018→setup, ctrl2019→control
CHECK: gates_v3.py, блок G3
EXPECT: max(end) ≤ 2020-01-01; tsel=validation или validation≠zakryto → ValueError
EVIDENCE: G3 ok 2020-01-01 00:00:00

## G4 — обе комиссии (0,04 отбор, 0,035 пересчёт), CSV и торнадо на каждую, шапка статуса; обе группы колонок tune_* и ctrl_*; цель = Шарп 2018
CHECK: gates_v3.py, блок G4/G5
EXPECT: 4 файла; шапка «# статус: разработка, цель tune 2018, контроль ctrl 2019 …»; objective == tune_sharpe; ctrl_sharpe_2019 есть
EVIDENCE: G4 G5 ok [4 файла]

## G5 — плато (±1 узел этого параметра, Шарп tune) = независимый пересчёт
CHECK: gates_v3.py, блок G6
EXPECT: расхождение < 1e-12
EVIDENCE: G6 ok

## G6 — процессов ≤ 6
CHECK: gates_v3.py, блок G7
EXPECT: MAX_JOBS == 6; --jobs 7 → ValueError
EVIDENCE: G7 ok

## G7 — skew/kurt из строки оценщика, своего расчёта нет
CHECK: gates_v3.py, блок G8 (в sweep_run.py нет .skew( / .kurt()
EXPECT: пройдено
EVIDENCE: G8 ok

## G8 — модуль владельца и тесты
CHECK: python3 fastbt/sweep_drivers.py; pytest tests/test_sweep_stat.py tests/test_sweep.py
EXPECT: exit 0; passed
EVIDENCE: drivers=0; 5 passed (других tests/test_sweep_*.py на ветке нет)

## G9 — время S3 на реальных данных, обе… одна комиссия 0,04, настоящая сетка S3 (19 параметров, 66 проб, tune+ctrl)
CHECK: run_setka("S3", S3_setka_e1b.yaml с pereschet=[]) 
EXPECT: время выведено
EVIDENCE: 1,6–2,0 с на 66 проб (с контролем 2019). Потолок оценки: S1/S5/S6 тяжелее, не мерено

## G10 — размер
CHECK: wc -l fastbt/sweep_run.py
EXPECT: ≤ 200
EVIDENCE: 165

## Реестр
закрыто 10 из 10; ABANDON: 0; pending: 0
