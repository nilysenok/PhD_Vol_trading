## R-д. Сигналы S1–S6 против к2 и плацебо (реверс раунд 1, 2022–2024, справочно, без выбора)

Статус 29.10: **СТОП, не посчитано** (таблицы `d_*` не созданы). Проверка сдвига (Д1) не выполнена: входов Z2 нет. `python3 scripts/revers/r1_d.py schet S1 --period test1` падает: `ValueError: negative dimensions are not allowed` в `fastbt/etap1_run.py:74` (`run_e1`, стек: `r1_d.schet` → `evaluate_v2` → `_rows2` → `_ladder`).
Причины (код не менялся): (1) `r1_d.schet` не выставляет `progon_sobol.PERIOD = "test1"` (там зашито `"train4"`, строка 33), и `fabrika_real` строит входы на train4, а `evaluate_v2` затем считает окно test1 — окно пустое. (2) Даже при исправлении (1) файлов зоны Z2 нет: не заданы `FASTBT_Z2_DAILY`, `FASTBT_Z2_DIV`, `FASTBT_Z2_VPRED` (σ̂ 2022–2024 лежит только в `test_locked/`) — `ABANDON-pending: ждёт нарезки`. (3) Кубы прохода 1 (`scripts/revers/kub_prohod1`) — выбор точек на train4, на test1 не перебираются — это по заданию.
Контроль train4 (18 таблиц, #467) — как в абзаце ниже, на этом запуске не перезапускался.

Проверено на train4: скрипт воспроизводит образец #467 (`docs/tables/progon_e1/diag_{tochki,kontrol,terciles}_S1…S6`) по всем 18 таблицам, `assert_frame_equal` rtol 1e-9 (тест `tests/test_revers_r1_d.py`, сверочные таблицы в `docs/tables/revers_r1/_sverka_train4/`, не коммитятся).

Отступления:
- Точки — топ-20 каждой Sk по ledger прохода 1 (выбор на train4), на test1 те же, перебора нет. Кубы прохода 1 взяты из коммита 63906a32 (`scripts/revers/kub_prohod1/`): кубы прохода 2 в репозитории изменены (новая ось vhod_tip), ledger по ним не восстанавливается.
- Плацебо: train4 — 10 (как в образце, для побайтной сверки), test1 — 20 (по заданию); лаги к2 — 20.
- `sdelok_v_god` и `sharpe_rasch` на test1 пересчитаны на 3 года (в `metriki` зашито 4).

Запуск после снятия замка и слияния R-P: `python3 scripts/revers/r1_d.py schet S<k> --period test1` (k = 1…6), затем `svod --period test1`.
