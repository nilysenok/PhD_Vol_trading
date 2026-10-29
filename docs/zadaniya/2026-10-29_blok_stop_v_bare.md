# БЛОКЕР #596/#604: исполнение стопа в баре входа — kodir, 29.10 15:17

Бюджет: ~40м · ≤ 150т. Жёсткий порог ~300k — передача (`templates/handoff.md`).
Блок владельца: Ж-129 п. 8, Ж-130 (ночной Соболь v5 не стартует, пока блокер не закрыт; старт медленным путём — не позже 17:30).

## Цель
PR ветки `blok-stop-v-bare`: выход при стопе в баре входа — не по уровню S вне [low; high]; инвариант «цена исполнения ∈ [low; high] бара» в медленном и быстром слое; разработка ≤ 2021.

## Рабочая копия (работать только в ней)
`/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/.claude/worktrees/agent-a06d3b8db1d8ccfba` (ветка `blok-stop-v-bare`). Гейты — `GATES_blok_stop.md` там же (G1–G6, все pending): закрыть каждый с EVIDENCE.

## Владение
`fastbt/exec_e_run.py`, `fastbt/exec_e_stop.py`, `fastbt/exec_e_zayavki.py`, `fastbt/exec_e_rules.py`, `fastbt/exec_e_v6.py`, `fastbt/tf_ispolnenie_v5.py`, `fastbt/fast_exec.py`, `tests/test_ispolnenie_v_bare.py`, `tests/test_fast_v5_ekviv.py` (только добавление случаев нового инварианта; существующие проверки не ослаблять), `tests/test_rassl_tf_s1.py` (только снятие xfail), `GATES_blok_stop.md`.

## Входы
`docs/otchety/2026-10-29_rassl_tf_s1.md` (разбор причины, #604).

## Гейты
Из `GATES_blok_stop.md`. G6 (масштаб до правки) — число исполнений вне [low; high] на опубликованных конфигурациях train4, цифрой со ссылкой на команду. В теле PR — «что изменилось в числах» (`pravki.md` п. 12) и мутация «стоп в баре входа исполнен по уровню S» (G3).

## Обязательные правила
Прочитать целиком: `.claude/rules/polnota.md`, `.claude/rules/orkestratsiya.md`, `.claude/rules/proverka.md`, `.claude/rules/mutations.md`, `.claude/rules/dannye.md`, `.claude/rules/testy.md`, `.claude/rules/pravki.md`.
Блоки владельца принимаются только в main; если агент получил блок напрямую, он пересылает его дирижёру и не действует до квитанции (Ж-115).
Это ПРОДОЛЖЕНИЕ: прежний агент оборвался ~13:20 (прервалась сессия дирижёра). Сначала прочитать свои коммиты (`git log kanon-v22-pereschet..HEAD`), незакоммиченные правки (`git diff`: `fastbt/fast_exec.py`, `tests/test_fast_v5_ekviv.py`) и файл гейтов; незакоммиченное — продолжить, не выбрасывать. Перед PR — слить свежую базу `origin/kanon-v22-pereschet`.

## Запрещено читать
Данные ≥ 2022, `test_locked/`, `.env`, `docs/tables/` следующих шагов. Хуки не обходить.

## Формат отчёта
`templates/otchet_agenta.md`, ≤ 500 слов: сделано / не сделано / проверено (команды) / отступления / решения; «закрыто N из M». PR в `kanon-v22-pereschet`, номер — в отчёт. Не сливать.

## Лимит процессов
≤ 3 одновременно.
