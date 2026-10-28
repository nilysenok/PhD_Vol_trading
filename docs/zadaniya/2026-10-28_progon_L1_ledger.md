# Задание kodir — Л-1 «ledger v2» (прогон этапа 1, Ж-078/080/082)

## Источники
- Контракт: `docs/otchety/2026-10-28_progon_etap1.md` §1 целиком (коммит `df6284b`, ветка `progon-postanovka`) — ключ §1.1, значение §1.2, `canon_v2` §1.3, ряды §1.4, API §1.5, проверки читателя §1.6, мутации §1.7, быстрый слой и счёт на ветке §1.8.
- Образец: `fastbt/reestr_otbora_v1.py` (`_chitat_ledger`, сверка SHA256SUMS), `fastbt/etap1_poisk_vorota.py` (`canon`, `pkey`), `fastbt/metriki_e.py:32` (`sr_dnevnoy`, ddof = 1).
- База: `origin/kanon-v22-pereschet` не раньше `45a93e6`. Ветка `progon-L1-ledger`.

## Цель
`fastbt/ledger_v2.py` строго по API §1.5:
- `tok`, `canon_v2`, `klyuch`, `klyuch_sha`, `kvartaly`, `tselevaya`;
- `Pisatel` — шарды по 256 строк, `.part` → переименование → строки → SHA; `done()` для резюма;
- `Chitatel` — проверки §1.6, `matrica`, `tik` (memmap), `n_unique`.

Данных модуль не читает. `docs/tables/progon_e1/_ledger_v2/.gitignore`: всё, кроме `.gitignore`, `*/manifest.json`, `*/SHA256SUMS.txt`.

## Владение
`fastbt/ledger_v2.py`, `tests/test_ledger_v2.py`, `docs/tables/progon_e1/_ledger_v2/.gitignore`.

## Тесты и мутации
- Синтетика с известным ответом:
  - 16 кварталов на датах 2016–2019 (граница 31.03/01.04);
  - Шарп с ddof = 1;
  - nan-квартал не положительный;
  - медиана с nan → nan.
- Резюм: обрыв после `add` без `close` → `Chitatel` не видит строк незакрытого шарда, `done()` их не содержит.
- Дубли: две строки с одним `klyuch_sha` → `n_unique` = 1.
- Мутации М-L1…М-L9 из §1.7. Каждая роняет тест; вывод pytest с мутацией — в тело PR. Мутации подкладывать в отдельной копии.
- Мутация, которую также вносит дирижёр в `mutations.md`: ключ без комиссии (тема № 7).

## Гейты
- G1: `python3 -m pytest tests/test_ledger_v2.py -q` → 0 failed.
- G2: все 9 мутаций §1.7 пойманы (вывод в PR).
- G3: `git diff --stat origin/kanon-v22-pereschet` — только файлы владения.

## Запрещено
- данные ≥ 2022, `test_locked/`, `.env`, метрики 2020–2021, счёт стратегий;
- `fastbt/sweep_drivers.py`, `fastbt/etap1_eval.py`, `fastbt/exec_e_run.py`;
- `pytest tests` целиком, `fastbt/tests/test_core.py`, `tests/test_a_weights_future.py`;
- `skipif` / `xfail` (Ж-081);
- `ln`, `scripts/finish.sh`, heredoc в файл, `sed -i`, `python -c`, `git stash`;
- запись вне владения;
- менять формат §1 — нашёл дыру → `❓ … (для arhitektor)` в PR, не правка.

## Отчёт
До 10 строк: реестр «закрыто N из M», отступления, расход. Разбор — в тело PR.

## Лимит
- до 2 процессов Python;
- ~30 мин (p90 50);
- порог передачи — 500k (Ж-078);
- коммит в свою копию не реже раза в 30 минут (Ж-077), scratchpad — только временное.

Правила: `.claude/rules/polnota.md`, `testy.md`, `mutations.md`, `dannye.md`, `pravki.md`.
