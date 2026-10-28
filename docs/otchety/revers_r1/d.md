## R-д. Сигналы S1–S6 против к2 и плацебо (реверс раунд 1, 2022–2024, справочно, без выбора)

Статус: `--period test1` НЕ запускался (замок Z2 закрыт, период `test1` в Evaluator — лист R-P). Готов `scripts/revers/r1_d.py`.

Проверено на train4: скрипт воспроизводит образец #467 (`docs/tables/progon_e1/diag_{tochki,kontrol,terciles}_S1…S6`) по всем 18 таблицам, `assert_frame_equal` rtol 1e-9 (тест `tests/test_revers_r1_d.py`, сверочные таблицы в `docs/tables/revers_r1/_sverka_train4/`, не коммитятся).

Отступления:
- Точки — топ-20 каждой Sk по ledger прохода 1 (выбор на train4), на test1 те же, перебора нет. Кубы прохода 1 взяты из коммита 63906a32 (`scripts/revers/kub_prohod1/`): кубы прохода 2 в репозитории изменены (новая ось vhod_tip), ledger по ним не восстанавливается.
- Плацебо: train4 — 10 (как в образце, для побайтной сверки), test1 — 20 (по заданию); лаги к2 — 20.
- `sdelok_v_god` и `sharpe_rasch` на test1 пересчитаны на 3 года (в `metriki` зашито 4).

Запуск после снятия замка и слияния R-P: `python3 scripts/revers/r1_d.py schet S<k> --period test1` (k = 1…6), затем `svod --period test1`.
