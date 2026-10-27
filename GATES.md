# GATES — канон v23: хуки и deny-списки (не включать), arhitektor, 27.10.2026

KOP = /private/tmp/claude-501/-Users-nilysenok-Desktop--------PhD-Vol-trading/b58320a0-0dac-4690-a8f8-b32419175dd9/scratchpad/kop
(облегчённая копия: `git archive` только кода, 15 МБ; рабочая копия агента — `$KOP/.claude/worktrees/agent-test`; фиктивные закрытые файлы — пустые).

## G1 — хуки компилируются
CHECK: python3 -m py_compile .claude/hooks/sdf_hook.py .claude/hooks/progon.py && echo OK
EXPECT: OK
EVIDENCE: OK (код 0)

## G2 — законные действия не блокируются, запрещённые блокируются с понятным сообщением (ссылка «Правило:»)
CHECK: python3 -I .claude/hooks/progon.py $KOP | tail -1
EXPECT: законных проходят: 47 из 47; запрещённых блокируются: 38 из 38; ложных срабатываний: 0; … (код 0)
EVIDENCE: законных проходят: 47 из 47; запрещённых блокируются: 38 из 38; ложных срабатываний: 0; max pre/post: 46 мс; max stop: 136 мс (код 0)

## G3 — время Pre/Post < 2 с
CHECK: та же строка G2, поле max pre/post
EXPECT: < 2000 мс
EVIDENCE: 46 мс

## G4 — Stop при EVIDENCE: pending → код 2; ABANDON с причиной снимает гейт; у дирижёра без GATES.md — 0
CHECK: python3 -I .claude/hooks/progon.py $KOP | grep -E '^S0'
EXPECT: S01, S02 — код 2; S03, S04, S06 — 0; S05 — 2,2,2,0; все OK
EVIDENCE: S01 2 OK · S02 2 OK · S03 0 OK · S04 0 OK · S05 2,2,2,0 OK · S06 0 OK

## G5 — в копии прогонялся тот же хук, что в ветке
CHECK: shasum -a 256 .claude/hooks/sdf_hook.py $KOP/.claude/hooks/sdf_hook.py | cut -c1-16
EXPECT: две одинаковые строки
EVIDENCE: 1d66ea66faf36eb5 / 1d66ea66faf36eb5

## G6 — settings.v23.json: валидный JSON, ни одно действующее правило не потеряно, 6 событий хуков
CHECK: jq -r '.permissions.deny[], .permissions.allow[]' <основная>/.claude/settings.json | sort > old; jq -r '…' .claude/settings.v23.json | sort > new; comm -23 old new | wc -l; comm -13 old new | wc -l; jq -r '.hooks|keys|join(",")' .claude/settings.v23.json
EXPECT: 0; 25; PostToolUse,PreCompact,PreToolUse,SessionStart,Stop,SubagentStop
EVIDENCE: пропало 0; добавлено 25; PostToolUse,PreCompact,PreToolUse,SessionStart,Stop,SubagentStop

## G7 — в основной копии ничего не включено, текущий settings.json не изменён
CHECK: shasum -a 256 <основная>/.claude/settings.json | cut -c1-16; ls <основная>/.claude
EXPECT: 88590758509bda10; agents rules settings.json worktrees (нет hooks)
EVIDENCE: 88590758509bda10; agents rules settings.json worktrees

## G8 — копия маленькая
CHECK: du -sh $KOP
EXPECT: < 50 МБ
EVIDENCE: 15M

## G9 — отчёт: таблица хуков, включить/текстом, включение в 3 шага, откат одним шагом
CHECK: grep -cE '^## (Хук → что ловит|Что включить|Включение \(после|Отступления)|^\*\*Откат одним шагом' docs/otchety/2026-10-27_huki_v23.md
EXPECT: 5
EVIDENCE: 5

## G10 — слой прав (permissions.deny) работает в живой сессии
CHECK: РУЧНОЙ — нужен запуск агента Claude Code в копии, а поручение запрещает запускать агентов
EXPECT: Read test_locked/ отказан правом
EVIDENCE: не проверено (см. ABANDON)

## ABANDON
ABANDON: G10 запуск агентов запрещён поручением; права проверены только синтаксически (G6); проверить — шаг 3 (б) инструкции включения

## Реестр
закрыто 9 из 10; ABANDON: 1 (G10); pending: 0 → передача, не «готово» (ждёт «да» владельца и G10 после включения)
