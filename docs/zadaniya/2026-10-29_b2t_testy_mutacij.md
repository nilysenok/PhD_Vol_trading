# Б2-т: тесты мутаций гэпа под ValueError (к PR #617) — kodir (тестировщик), 29.10 16:35
Бюджет ~25м · ≤ 80т. Общая часть: `docs/zadaniya/2026-10-29_obshchee_16.md`. Работать в ветке PR #617: `git fetch origin && git checkout -b b2-testy origin/fix-vne-oshibka` (имя ветки PR проверить: `gh pr view 617 --json headRefName`), push обычный в ветку PR.

## Цель
PR #617 сделал нарушение «цена вне [low; high]» в дневном автомате `ValueError`. Шесть старых мутационных тестов досчитывали мутанта до конца и сравнивали числа; теперь мутант падает с `ValueError` — мутация ловится сильнее, но тест красный:
- `tests/test_ispolnenie_v_bare.py::test_mutaciya_stop_vhoda_po_urovnyu_lovitsya_svojstvom`
- `tests/test_exec_e_parabolic.py::test_mutatsiya_4_gep_po_urovnyu_lovitsya`
- `tests/test_exec_e_rules.py::test_stop_gap_at_level_mutation_caught`
- `tests/test_exec_e_v5_ed.py::test_mutatsii_2_3_4_lovyatsya` (случай `_scen_gep-_mut_stop_hit_level`)
- `tests/test_fast_etap1_equiv.py::test_mutation_caught` (S1-gap, S4-gap)
Сделать: мутация считается пойманной, если мутант даёт другой результат ИЛИ `ValueError` про исполнение вне бара (сообщение проверять). Немутированный путь — без изменений. Ничего не ослаблять: тест, который на мутанте не падает и не даёт отличия, — красный.

## Владение
Только эти пять файлов тестов, только перечисленные тесты. Код `fastbt/*` не трогать.

## Гейты
G1 все шесть зелёные на коде #617. G2 на коде базы ДО #617 (`git stash`-не использовать; `git show origin/kanon-v22-pereschet:fastbt/exec_e_run.py` во временный файл и подмена через monkeypatch не нужна — достаточно прогнать эти тесты на `origin/kanon-v22-pereschet` в отдельном checkout `git worktree`-не создавать: `git checkout origin/kanon-v22-pereschet -- fastbt/exec_e_run.py`, прогнать, `git checkout HEAD -- fastbt/exec_e_run.py`) — тоже зелёные. G3 подложить «мутант не отличается и не падает» (тождественная мутация) — тест красный. G4 `NUMBA_NUM_THREADS=1 python3 -m pytest -q tests/test_vne_oshibka.py tests/test_ispolnenie_v_bare.py tests/test_rassl_tf_s1.py tests/test_exec_e*.py tests/test_fast_v5_ekviv.py tests/test_fast_etap1_equiv.py` — 0 failed. Комментарий в PR #617 с итогом.
