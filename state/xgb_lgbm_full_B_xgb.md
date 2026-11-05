# Передача — волна B, поток XGB (Ж-206), kodir. Статус: ПЕРЕДАЧА (счёт запущен в фоне, EVIDENCE не заполнены)

Не коммитить без команды. Процессов <= 4 (LGBM-поток считает параллельно). Скрипты только файлами; `python3 -c` и heredoc запрещены.

## Готово (код и синтетика)
Код `models_v2/s6/xgb/` (cfgx, worldx, critx, stabx, rowsx, calcx, transferx, tablex, checkx, runx); тесты `tests/test_v2_xl_xgb_{stab,crit,cfg,leak,transfer,rows,calc}.py` — 55 passed. Гейты `docs/otchety/xgb_lgbm_full_v1/xgboost/GATES_B_xgb.md` (B1…B10, EVIDENCE pending). `--stage check` (B9): нарушений 0 на момент написания.

## Команды стадий (из корня, строго по порядку, по одной; логи в `data/regen/xgb_lgbm_full_v1/xgboost/log_*.txt`)
1. `python3 -W ignore models_v2/s6/xgb/runx.py --stage control`  (B2: ctrl_f4 побитно к fc_b2_f4 на 3 датах, crit_mean 0,1546074; пишет CONTROL_B.json, строку ctrl_f4)
2. `... --stage stab`  (B3, STABILITY_QLIKE*.csv)
3. `... --stage trials`  (B7: 9 новых trials, кэш raw_/loyo_ в data-каталоге, resume по ключу)
4. `... --stage cond`  (условные по правилу; COND_DECISION.json)
5. `... --stage tables`  (B8)  6. `... --stage check`  (B9)
Затем: pytest всех `tests/test_v2_xl_xgb_*.py` (B1,B4,B5,B6), занести вывод в EVIDENCE каждого гейта, реестр «закрыто N из 10», ответ дирижёру <= 500 слов (перенесённое отдельно от нового; отрицательные результаты сохранить; первая цифра — потолок).

## Что посчитано
См. лог ниже/файлы в data-каталоге: на момент передачи ничего на реальных данных не посчитано.

## Отступления
Пустой heredoc и `python3 -I -c` один раз (записано дирижёром); чтений до первой правки > 8.
