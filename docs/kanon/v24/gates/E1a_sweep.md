# GATES — Ж-031 Э1-а, оценщик движка перебора (fastbt/sweep*.py), kodir, 27.10

Состав сужен Ж-034/Ж-035: режимы, цель, плато, DSR, PBO — в `fastbt/sweep_drivers.py` (владелец); тесты пишет тестировщик
(`tests/test_sweep_*.py`). Мои черновые тесты в PR удалены. Гейты ниже сохранены как были; закрытые — с выводом, переданные — ABANDON с причиной.

## G1 — параметр изменён → результат изменился
CHECK: тест тестировщика tests/test_sweep_*.py -k param_changes
EXPECT: passed
EVIDENCE: ABANDON G1 (тесты — зона тестировщика, Ж-034). Косвенно: sweep_smoke проверяет len(set(sharpe)) > 1 по 5 значениям ma_window на реальном 2018 — OK.

## G2 — validation не влияет на выбор
CHECK: тест тестировщика -k validation
EXPECT: passed
EVIDENCE: ABANDON G2 (зона тестировщика). Код: Sweep._df бросает PermissionError на годах 2020–2021 без open_validation(); Periods не допускает пересечения.

## G3 — границы настройка/контроль/validation не пересекаются
CHECK: тест тестировщика -k periods
EXPECT: passed
EVIDENCE: ABANDON G3 (зона тестировщика). Код: sweep_stat.Periods.__post_init__.

## G4 — комиссия списывается с обеих сторон
CHECK: тест тестировщика -k commission
EXPECT: passed
EVIDENCE: ABANDON G4 (зона тестировщика). Счёт — боевой s9.Engine.port через sweep_sk._Mini; смоук проверяет Шарп(0,04) ≤ Шарп(0,035) — OK.

## G5 — PBO на случайных данных ≈ 0,5
CHECK: тест -k pbo
EXPECT: passed
EVIDENCE: ABANDON G5 (pbo_cscv — в sweep_drivers, Д7; зона драйверов/тестировщика).

## G6 — DSR при N=1 растёт по сравнению с N>1
CHECK: тест -k dsr
EXPECT: passed
EVIDENCE: ABANDON G6 (dsr — в sweep_drivers; оценщик отдаёт T, skew, kurt, sr_d, trials_sr()).

## G7 — Моррис на аналитической функции даёт известные μ*
CHECK: тест -k morris
EXPECT: passed
EVIDENCE: ABANDON G7 (morris_screen — в sweep_drivers).

## G8 — Optuna с фиксированным seed воспроизводится; n_jobs ≤ 8
CHECK: тест -k optuna
EXPECT: passed
EVIDENCE: ABANDON G8 (optuna_search — в sweep_drivers).

## G9 — кэш сигналов: ключ меняется от параметров, ряда и кода; PnL отдельно
CHECK: тест тестировщика -k cache
EXPECT: passed
EVIDENCE: ABANDON G9 (зона тестировщика). Код: SignalCache.key в sweep_spec.py.

## G10 — режим validation: флаг, повторное открытие без записи запрещено, N удваивается
CHECK: тест тестировщика -k reopen
EXPECT: passed
EVIDENCE: ABANDON G10 (зона тестировщика). Код: Sweep.open_validation / n_mult.

## G11 — дымовой прогон OAT, один параметр, S1, реальный 2018: OK и время
CHECK: FASTBT_MAIN=<своя копия> FASTBT_CACHE=<папка> /usr/bin/python3 -m fastbt.sweep_smoke
EXPECT: строка OK и время, числа не печатаются
EVIDENCE: OK: 5 точек, 0.5 с (тёплый кэш; холодный — 1.6 с); код возврата 0

## G12 — файлы ≤ 200 строк, py_compile
CHECK: wc -l fastbt/sweep.py fastbt/sweep_spec.py fastbt/sweep_stat.py fastbt/sweep_sk.py fastbt/sweep_synth.py fastbt/sweep_smoke.py; /usr/bin/python3 -m py_compile на них
EXPECT: каждый ≤ 200, код возврата 0
EVIDENCE: 96 / 64 / 66 / 97 / 44 / 37 строк; compiled, код 0 (sweep_drivers.py — файл владельца, 275 строк)

## G13 — evaluate отдаёт ключи контракта драйверов
CHECK: sweep_smoke (assert all(k in row for k in sharpe, by_year, msr, sortino, mdd, T, skew, kurt, N))
EXPECT: OK
EVIDENCE: пройден в составе G11 (OK, код 0)

## ABANDON (только с причиной; гейт не удалять и не переписывать)
ABANDON: G1–G10 тесты и режимы переданы тестировщику и драйверам (Ж-034, Ж-035); мои черновые тесты удалены из PR.

## Реестр
закрыто 3 из 13 (G11, G12, G13); ABANDON: 10; pending: 0 → «передача», не «готово»
