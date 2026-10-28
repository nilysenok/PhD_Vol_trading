# Регрессия final3 (test_s9_scope), 28.10.2026

Статус: расследование, код и тесты не менялись.

**Виновник: слияние #394 (c44ed02, ветка div2015-v1-dannye).** Его родитель 78884bf (#399) и 46ac001 (#388): 2 passed. На c44ed02: 2 failed
(`test_dev_reproduces_final3_tables`, `test_test_path_on_dev_years`). #397 (a808b6f) лежит ниже #399 и зелёный, значит не виноват. #398 не проверялся: красное уже на #394.

**Причина:** `fastbt/dividendy.py` — `_tables()` теперь читает вторую таблицу `TABLE_2015` (дивиденды 2015–2016, потолок: dohod.ru, первичные не сверены).
`events()` добавляет эти события в `lr` и ряд сигналов s9 (прогрев до 2018), и эталоны `docs/tables/strat_dev/s9_*_final3.csv` расходятся
(в тесте перечислены s9_B_targetirovanie, s9_C3_tochnost_v_rezultat, s9_C_rezhim, s9_D_zatvor, s9_D_zatvor_harj, s9_P1_vesa_tikerov, …; значения не выводились).
Это не устаревший кэш: `build_cache` пересобирает всегда, прогон с `FASTBT_NO_CACHE=1`; ключ s9 включает `dividendy.digest()`.

**Воспроизведение:** `git checkout --detach c44ed02`, затем
`FASTBT_BARS1M=<основная>/data/regen/D25/bars1m FASTBT_NO_CACHE=1 python3 -m pytest tests/test_s9_scope.py -k "dev_reproduces_final3 or test_path_on_dev_years" -q --tb=no -p no:cacheprovider` → 2 failed.
Контроль: на c44ed02 временно `_tables()` → `[Path(TABLE)]` (без TABLE_2015) → 2 passed; правка откатана `git checkout`. (Тесты dividendy_2015 при этом контроле падают — ожидаемо.)

**Оговорка:** s9 пишет кэш в `MAIN/data/fastbt/s9` (общий; путь задаётся только `FASTBT_MAIN`, отдельной переменной нет). Прогоны шли последовательно.

**Не сделано (починка):** эталоны final3 пересчитать на коде #394 как новые (старые в архив с хэшами, `pravki.md` п. 11) либо исключить TABLE_2015 из dev-окна s9. Это решение класса A (эталон).
