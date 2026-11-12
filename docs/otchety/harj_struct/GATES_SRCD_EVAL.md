# GATES — S-RCD-EVAL (лист оценки и отчёта; Ж-259; `S_RCD_SPEC.md` разд. 6, 7, 8.4)

Владение: `models_v2/s18/rd_eval*.py`, `rd_doc*.py`, `S_RCD_*.csv`, `S_RCD_REPORT.md`, ключ `S.RCD.eval`. Образец — `z_eval*`, `z_doc*`.
Объявленная диагностика (02:47, до QLIKE): нулевое распределение полной процедуры S-RCD (10 шумов, `rd_walk` режим R, форма CD) — `S_RCD_NULL.csv`.

| id | CHECK | EXPECT | EVIDENCE |
|---|---|---|---|
| E1 | `python3 -W ignore -m models_v2.s18.rd_eval_null --run`, затем `--check` | 10 строк seeds 0–9, сводка = пересчёту; файл до таблиц QLIKE | run 02:48–02:49 (6–7 с на шум, 75 активных месяцев); `--check`: «rd_eval_null --check: ok»; файл 02:49:01, ключ `S.RCD.eval` 02:54:53 — позже файла |
| E2 | `python3 -W ignore -m models_v2.s18.rd_eval --write` | ключ `S.RCD.eval` слиянием раньше таблиц; 12 таблиц | rc 0; RCD-DEC 2; проверка порядка ключа — в `--check` (p_key) |
| E3 | `python3 -W ignore -m models_v2.s18.rd_eval --check` | ok; выходы до и после равны | «S-RCD eval --check: ok»; N 858 / 3421, m = 6; S-CD, S-CK, S-RCK, S-RCKf = таблицам S_C / S_D (≤ 1e-10 из-за 12 знаков CSV) |
| E4 | `python3 -W ignore -m models_v2.s18.rd_doc --write`, затем `--check` | отчёт = таблицам; порядок разделов; без «эквивалентно» | «S-RCD doc --check: ok» |
| E5 | `python3 -m py_compile models_v2/s18/rd_eval*.py models_v2/s18/rd_doc*.py`; файлы ≤ 200 строк | rc 0 | rc 0; максимум 149 строк (`rd_eval_chk.py`) |

Примечания: `--write` выполнен дважды (второй раз — после правки `rd_eval_chk.py` и текста правила в `rd_eval_null.py`, чтобы хэши кода в ключе были итоговыми); числа таблиц не менялись.
