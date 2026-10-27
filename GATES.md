# GATES — канон v23: хуки и deny-списки (не включать до «да»), arhitektor, 27.10.2026; оговорки #292 (Ж-029)

PROGON = python3 -I .claude/hooks/progon.py --papka <scratchpad>   (самодостаточен: копия из заглушек, удаляется после прогона)
TST = /private/tmp/claude-501/-Users-nilysenok-Desktop--------PhD-Vol-trading/b58320a0-0dac-4690-a8f8-b32419175dd9/scratchpad/tst (скрипты проверяющего; в `pg/m` — этот же sdf_hook.py и zamki.json ветки)

## G1 — хуки компилируются
CHECK: python3 -m py_compile .claude/hooks/sdf_hook.py .claude/hooks/progon.py && echo OK
EXPECT: OK
EVIDENCE: OK (код 0)

## G2 — законные не блокируются, запрещённые блокируются со ссылкой «Правило:»
CHECK: $PROGON | tail -1
EXPECT: законных проходят: 62 из 62; запрещённых блокируются: 59 из 59; ложных срабатываний: 0; … (код 0)
EVIDENCE: законных проходят: 62 из 62; запрещённых блокируются: 59 из 59; ложных срабатываний: 0; max pre/post: 46 мс; max stop: 133 мс (код 0)

## G3 — время Pre/Post < 2 с
CHECK: поле max pre/post в G2
EXPECT: < 2000 мс
EVIDENCE: 46 мс

## G4 — Stop при EVIDENCE: pending → код 2; ABANDON снимает; у дирижёра без GATES.md — 0
CHECK: $PROGON | grep -E '^S0'
EXPECT: S01, S02 — 2; S03, S04, S06 — 0; S05 — 2,2,2,0; все OK
EVIDENCE: S01 2 OK · S02 2 OK · S03 0 OK · S04 0 OK · S05 2,2,2,0 OK · S06 0 OK

## G5 — fail-closed: опасное PreToolUse при сбое хука — 2 «не смог проверить»; чтение/прочее — 0 и запись в аудит
CHECK: $PROGON | grep -E '^(Z|D07)'
EXPECT: Z01, Z02, Z06, Z08, Z09, D07 — 2; Z03–Z05, Z07, Z10–Z12 — 0; Z13 — ≥ 10 записей sboy_huka; все OK
EVIDENCE: все 14 OK; Z13 = 13 записей

## G6 — resheniya.md только дописывание через proverit(old, new)
CHECK: $PROGON | grep -E '^D0'
EXPECT: D01, D02 (дописывание) — 0; D03 (правка старой строки), D04, D05, D06 — 2 со ссылкой Ж-025; все OK
EVIDENCE: D01 0 OK · D02 0 OK · D03 2 OK · D04 2 OK · D05 2 OK · D06 2 OK · D07 2 OK

## G7 — обход через python -c / bash -c / eval / find -exec / $(…) закрыт без ложных срабатываний
CHECK: $PROGON | grep -E '^B'
EXPECT: B01–B10, B16 — 2; B11–B15 — 0; все OK
EVIDENCE: все 16 OK

## G8 — скрипты проверяющего: законные проходят, запрещённые блокируются
CHECK: cd $TST/.. && /usr/bin/python3 -I tst/t.py > o; sed -n '/--- legal/,/^Write STATUS/p' o | grep -c '^ok'; … grep -c 'PASS!!' o; /usr/bin/python3 -I tst/t3.py | grep -c '^BLOCK'
EXPECT: законных t.py 40 из 40 (BLOCK 0); PASS!! 0 (запрещённых 23 из 23 BLOCK); t3.py BLOCK 0 из 21
EVIDENCE: t legal ok=40 block=0 forb_pass=0 forb_block=23; t3 ok=21 block=0

## G9 — settings.v23.json: валидный JSON, ни одно действующее правило не потеряно, 6 событий
CHECK: comm старых и новых правил (jq), jq -r '.hooks|keys|join(",")' .claude/settings.v23.json
EXPECT: пропало 0; добавлено 25; PostToolUse,PreCompact,PreToolUse,SessionStart,Stop,SubagentStop
EVIDENCE: пропало 0; добавлено 25; PostToolUse,PreCompact,PreToolUse,SessionStart,Stop,SubagentStop

## G10 — в основной копии ничего не включено, settings.json не изменён
CHECK: shasum -a 256 <основная>/.claude/settings.json | cut -c1-16; ls <основная>/.claude
EXPECT: 88590758509bda10; нет hooks
EVIDENCE: 88590758509bda10; agents rules settings.json worktrees

## G11 — отчёт: таблица хуков, оговорки, включить/текстом, 3 шага, откат
CHECK: grep -cE '^## (Хук → что ловит|Дополнения по оговоркам|Что включить|Включение \(после|Отступления)|^\*\*Откат одним шагом' docs/otchety/2026-10-27_huki_v23.md
EXPECT: 6
EVIDENCE: 6

## G12 — слой прав (permissions.deny) в живой сессии
CHECK: РУЧНОЙ — нужен запуск агента Claude Code, поручение запрещает
EXPECT: Read test_locked/ отказан правом
EVIDENCE: не проверено (см. ABANDON)

## ABANDON
ABANDON: G12 запуск агентов запрещён поручением; права проверены синтаксически (G9); проверить — шаг 3 (б) инструкции включения

## Реестр
закрыто 11 из 12; ABANDON: 1 (G12); pending: 0 после G11 → передача (G12 — после включения)
