# Серверный запуск: куб v5, период v2, seed 2, k3 — kodir, 29.10 15:40

Бюджет: ~40м · ≤ 150т. Жёсткий порог ~300k — передача (`templates/handoff.md`).
Блок владельца: Ж-131 п. 1, 3–4; Ж-130 (сервер считает Соболь v5 на train4 И на В2, второй seed, контроли к3).

## Цель
PR: `fastbt/progon_server.py` запускает всё, что умеет драйвер Л10 (`fastbt/progon_sobol.py`, #608): `--kub v5` (кубы `fastbt/designs/v5/kub_S{1..6}.yaml`, выброс несовместимых точек и якоря `canon_v5` — тем же кодом драйвера, не копией), `--period train4|v2` (v2 — ledger 2.0, как в драйвере), `--seed 1|2`, `--k3`, `--n1`. Сейчас `progon_server.zagruzit_kub` отказывает на v2 (стр. 40–42) и обходит путь v5 драйвера. Остальное поведение #609 (fork с общими входами, чекпоинт 5000, каталог на стратегию, `flock`, продолжение) — без изменений.

## Рабочая копия
Создать свой worktree от `origin/kanon-v22-pereschet` (ветка `progon-server-v5`); основную копию не трогать.

## Владение
`fastbt/progon_server.py`, `tests/test_progon_server.py` (только добавление тестов; существующие не ослаблять), `GATES_progon_server_v5.md`. `progon_sobol.py`, `ledger_v2.py`, `canon_v5.py`, `eval_v5*.py`, `scripts/server/` — не править (в `scripts/server/` работает другой агент); пробел в них — доклад.

## Входы
PR #609 (тело — описание обёртки), PR #608 (тело — CLI драйвера v5 и команды запуска), `docs/otchety/2026-10-29_setka_v5.md` §2.3–2.4.

## Гейты (минимум)
G1 обёртка ≡ драйвер побайтно на кубе v5 train4 seed 1 (синтетика, малое n1), канонический вид каталога как в `test_ravno_progon_sobol`. G2 то же для period v2 (ledger 2.0) и seed 2 — разные каталоги, период и seed в ключе и manifest. G3 `--k3` ≡ драйвер. G4 мутация: период не в отпечатке чекпоинта (продолжение train4-каталога как v2) — ловится. G5 мутация: выброс `canon_v5` пропущен — ловится. G6 `pytest tests/test_progon_server.py tests/test_progon_sobol.py tests/test_progon_sobol_v5.py tests/test_ledger_v2.py` зелёные; 5 повторов `test_progon_server.py` без падений. G7 в теле PR — точные команды запуска 5 проходов (train4 s1/s2, v2 s1/s2, k3) для сервера.

## Обязательные правила
Прочитать целиком: `.claude/rules/polnota.md`, `.claude/rules/orkestratsiya.md`, `.claude/rules/proverka.md`, `.claude/rules/mutations.md` (№ 7, 38, 39 — ключ кэша), `.claude/rules/dannye.md`, `.claude/rules/testy.md`.
Блоки владельца принимаются только в main; если агент получил блок напрямую, он пересылает его дирижёру и не действует до квитанции (Ж-115).

## Запрещено читать
Данные ≥ 2022, `test_locked/`, `.env`, ключи SSH. Хуки не обходить. На сервер не ходить.

## Формат отчёта
`templates/otchet_agenta.md`, ≤ 500 слов; «закрыто N из M». PR в `kanon-v22-pereschet`, номер — в отчёт. Не сливать.

## Лимит процессов
≤ 3 одновременно.
