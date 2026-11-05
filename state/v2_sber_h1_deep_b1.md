# Передача — поток B1 (Ж-204, волна B), kodir, 05.11.2026

Задание: `docs/zadaniya/2026-11-05_sber_h1_deep.md` (Волна B, поток B1). Гейты: `docs/otchety/sber_h1_deep_v1/GATES_B1.md` (закоммичены дирижёром). Не коммитить; процессов <= 4.
Владение: `models_v2/s5/b1/**`, `tests/test_v2_deep_b1_*.py`, `docs/otchety/sber_h1_deep_v1/{GATES_B1.md,OBJECTIVE_EXPERIMENTS.csv,HAR_BOOSTING_RESULTS.csv,REGIME_MODEL_RESULTS.csv,OPTUNA_TRIALS_B1.csv,README_B1.md}`, `data/regen/v2_sber_h1_deep_v1/b1/**`.

## Сделано (до счёта)
- Код `models_v2/s5/b1/` (13 файлов, audit: 0 нарушений): losses (L4), fitobj (gamma/qlike/lgb), designs (Spec, режимы, веса L5), jobs (пул spawn <= 4), world (кэш мира с ключом), streams, rules (H3/R3/R4), rowsfc, evalrows, tables, configs (28 trials), report, run_b1.
- Тесты зелёные: `tests/test_v2_deep_b1_{losses,floor,registry,rules,leak}.py`. `test_v2_deep_b1_tables.py` написан, без выходов счёта пропускается.
- `--stage controls`: OK (Б-2 0,220486502764; HAR-J(б) 0,233662115694; n=2271; 3 даты и 2 фолда побитно).

## Не сделано
Счёт первого прохода, таблицы, реестр, README_B1.md, EVIDENCE гейтов B1-1…B1-10 (все pending).

## Команды
1. `python3 models_v2/s5/b1/run_b1.py --stage all` (~50 мин; resume по ключам: raw_*/loyo_*.parquet в `data/regen/v2_sber_h1_deep_v1/b1/`). Ошибки логики в rowsfc/tables ещё не обкатаны — чинить код, критерии из GATES не менять.
2. `python3 -m pytest tests/test_v2_deep_b1_*.py -q`
3. `python3 models_v2/s5/b1/run_b1.py --stage readme` (после ручного текста README_B1.md: сделано/не сделано/проверено/отступления), затем `--stage check-readme`, `--stage audit`.
4. Заполнить EVIDENCE в GATES_B1.md, ответ дирижёру <= 500 слов.

## Ловушки
Скрипты только файлами; `python3 -c`/heredoc запрещены. Accelerate: матричные произведения через einsum/sum. Менять код b1 после счёта нельзя без пересчёта (ключ кэша включает sha кода). CHECKPOINT.json не трогать.
