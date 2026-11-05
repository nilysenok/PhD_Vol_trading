# Передача — Ж-200/Ж-202 (бустинг недоучен, И1–И6), kodir, 05.11.2026 ~18:30 (по `date` Мака)

Причина: порог ~300k близко. Задание: `docs/zadaniya/2026-11-05_v2_sber_h1_fix.md`; гейты `models_v2/GATES_fix.md` (с ↺ Ж-201, Ж-202). Владение: `models_v2/s4/**`, `models_v2/run_fix*.py`, `tests/test_v2_fix_*.py`, `docs/tables/v2_sber_h1/fix_*`, `data/regen/v2_sber_h1/fix_*`, отчёт `docs/otchety/2026-11-05_v2_sber_h1_fix.md` (НЕ написан). Не коммитить.

## Сделано
- Диагностика: `python3 models_v2/run_fix_diag.py` (1,6 мин) -> `fix_diag_a.csv`, `fix_diag_b.csv` (ответ дирижёру отправлен). После правок s4 шапка diag устарела -> перезапустить (иначе `test_headers_carry_current_s4_sha` красный).
- Код s4: `fit.py` (И1; HP с colsample/max_trees), `select.py` (GRID 24, NEW 108, GRID132, HARD 80; LOYO), `runner.py`, `design.py`, `mkt.py` (И3: прокси 17 бумаг + IMOEX ISS без фолбэка), `rows.py` (план задач с дедупликацией, прогнозы, утечки, `overfit`), `report.py` (критерий Р ≤ 1,3 и К2; К1 справочно; DM), `prov4.py`, `synth.py`.
- Скрипты: `run_fix.py --stage a` (сетка 24: fx_i1_base, fx_i1_free, fx_i12, fx_i123_proxy), `--stage c` (сетка 132, суффикс `_g132`, IMOEX из `data/iss_check/index_10m/IMOEX.parquet`), `run_fix_report.py` (после a и c: fix_metrics, fix_criterion, fix_dm).
- Тесты: `tests/test_v2_fix_{i1,i2,rows,mkt,leak,tables}.py`; i1/i2/rows/mkt/leak — зелёные (30 + 17).

## Состояние счёта
Запущено в фоне: `run_fix.py --stage a` затем `--stage c` (логи в папке сессии: scratchpad `fix_a2.log`, `fix_c.log`). Результаты этапа a (сетка 24, до Ж-202), work_daily QLIKE(б): I1-base 0,2856; I1-free 0,2891; I1+I2 0,2668; I1+I2+I3(прокси) 0,2647 против HAR-J(б) 0,2348. Утечки: (i), (ii) выполнены, (iii) красный у всех (допуск не двигать).

## Что осталось
1. Дождаться stage c; `python3 models_v2/run_fix_report.py`; `python3 models_v2/run_fix_diag.py` (обновить шапки); `python3 -m pytest tests/test_v2_fix_*.py -q`.
2. Отчёт `docs/otchety/2026-11-05_v2_sber_h1_fix.md` (разделы по F13) и закрыть EVIDENCE в GATES_fix.md; мутации M1, M2, M4, M5, M6, M7, M8 покрыты тестами, M3 косвенно (s2_es > s2_tr_prelim), M9, M10 — слабо/нет (честно назвать).
3. `wc -l` файлов (≤200), функции ≤40 (`run_fix.py:main` длиннее — пометить).

## Ловушки
- Менять `models_v2/s4/*.py` после записи таблиц нельзя: sha в шапках. Хук после правки гоняет pytest по соответствующим файлам (шум «diag» — по имени).
- xgboost предсказывает float32: `predict` приводит к float64.
- Heredoc/`sed -i`/`echo >` запрещены хуком.
