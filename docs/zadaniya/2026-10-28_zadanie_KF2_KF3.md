# Задание kodir — К-Ф2 и К-Ф3: подключение ФДР внутри дня и выход лимиткой (Ж-065 п. 3)

## Источники

**Контракт.** `docs/otchety/2026-10-28_kontrakt_fdr_iter16_20.md`, разделы:
- §1.1–1.7;
- таблица §1.4 (строки К-Ф2 и К-Ф3);
- мутации §1.5 (М-Ф1…М-Ф6, М-В1…М-В6).

**Модуль.** `fastbt/fdr.py` (#377, слит).

**Толкования** (обязательны) — в `docs/resheniya/tolkovaniya.md`:
- [Ж-065 п. 3];
- «Факт из контракта»: у A все сделки по close, T+0 = база побайтно;
- «Умолчания контракта #376», п. 1: умолчание `fdr` в ledger не переключать.

**Блок владельца.** Ж-065 п. 3 в `docs/resheniya/resheniya.md`.

**База.** `origin/kanon-v22-pereschet` не раньше `5a815f7`. В ней уже слиты #370 (дивиденды) и #387 (периоды В1).

## Цель
1. **К-Ф2.** Параметр `fdr=None | "t0" | "t1"`. При `None` всё побайтно как сейчас. Где пробросить:
   - `Engine.port_rows`, `_finish(noch)` в `fastbt/s9.py`;
   - `run_e1(fdr)` в `etap1_run.py`;
   - `Evaluator(fdr)` в `etap1_eval.py`;
   - `etap1_ansambl_eval.py`, `etap1_plus_eval.py`;
   - ledger по режиму в `zh051_plato.py` — отдельный файл `ledger_it{i}_fdr-{rezhim}.pkl`, режимы не смешиваются (Д7).
2. **К-Ф3.** Выход лимиткой, §1.3 В:
   - `Params.vyhod_lim`, по умолчанию `False`;
   - исполнение — только при проходе цены на тик сквозь уровень;
   - стоп проверяется первым;
   - не исполнилось — выход по close d+1;
   - счётчики `x_orders` и `x_fills`;
   - ось `vyhod` в `params_of`.

   Файлы: `fastbt/exec_e_types.py`, `exec_e.py`, `exec_e_run.py`, `params_of` в `etap1_eval.py`.

Это одна ветка, но два логических коммита: сначала К-Ф2, потом К-Ф3. Если не укладываешься примерно в 60 минут, режь на два PR.

## Владение
- код: `fastbt/s9.py` (только `port_rows` и `_finish`), `fastbt/etap1_run.py`, `fastbt/etap1_eval.py`, `fastbt/etap1_ansambl_eval.py`, `fastbt/etap1_plus_eval.py`, `fastbt/zh051_plato.py`, `fastbt/exec_e_types.py`, `fastbt/exec_e.py`, `fastbt/exec_e_run.py`;
- тесты: `tests/test_fdr_vnutri_dnya.py`, `tests/test_vyhod_limit.py`.

Ветка `kf2-kf3-fdr`. **Счёт не запускать** — паспорт A (К-Ф4) будет отдельной задачей.

## Тесты
- **Golden.** `fdr=None` и `vyhod_lim=False` дают результат побайтно как сейчас:
  - якоря `tests/test_etap1_run_yakor.py`, `tests/test_dividendy_yakor.py`;
  - `test_s9_scope::test_dev_reproduces_final3_tables` — с `--tb=no`, значения не выводить.
- **У A при T+0 итог совпадает с базой побайтно** — Г3 контракта.
- **Мутации М-Ф1…М-Ф6 и М-В1…М-В6 из §1.5** — каждая роняет тест; вывод pytest с мутацией приложи в тело PR.
- **Добавочная мутация** (замечание проверяющего к #377): `noch_iz_pos` не покрыт мутациями — добавь хотя бы одну.

## Гейты
- **G1.** Новые тесты зелёные, все мутации §1.5 плюс `noch_iz_pos` ловятся.
- **G2.** Прогон зелёный, без новых красных против базы:
  ```
  pytest tests/test_dividendy*.py tests/audit_fastbt/test_23_cache_keys.py tests/test_etap1_run_yakor.py tests/test_zh051_*.py tests/test_period_v1.py tests/test_fdr.py tests/test_t_*.py tests/test_diag_zh06*.py -q --tb=no
  ```
  С `FASTBT_BARS1M=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/D25/bars1m`.
- **G3.** `test_s9_scope.py` — 0 failed, `--tb=no`.
- **G4.** `git diff --stat` — только файлы владения.

## Запрещено
- данные ≥ 2022, `test_locked/`, `.env`, метрики 2020–2021, счёт стратегий;
- `fastbt/sweep_drivers.py`;
- `pytest tests` целиком, `fastbt/tests/test_core.py`, `tests/test_a_weights_future.py`;
- `ln`, `scratchpad/main/`, `scripts/finish.sh` (git/gh — вручную), heredoc, `sed -i`, `python -c`;
- запись вне владения.

## Отчёт
- не длиннее 10 строк: реестр «закрыто N из M», отступления, расход;
- полный разбор — в тело PR.

## Лимит
- не больше 3 процессов Python;
- около 60 мин;
- порог передачи — 300k.

Правила: `.claude/rules/polnota.md`, `testy.md`, `mutations.md`, `dannye.md`, `pravki.md`.
