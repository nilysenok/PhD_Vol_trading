# Проверка возможностей Claude Code по официальной документации (для канона v24, Ж-025, Ж-026)

Дата просмотра всех страниц: 2026-10-27. Источник: code.claude.com/docs/en (официальная документация). Неофициальных источников не использовано.
Страницы: [sub-agents](https://code.claude.com/docs/en/sub-agents), [hooks](https://code.claude.com/docs/en/hooks), [settings](https://code.claude.com/docs/en/settings), [settings-reference](https://code.claude.com/docs/en/settings-reference), [cli-reference](https://code.claude.com/docs/en/cli-reference), [commands](https://code.claude.com/docs/en/commands), [skills](https://code.claude.com/docs/en/skills), [memory](https://code.claude.com/docs/en/memory), [agent-teams](https://code.claude.com/docs/en/agent-teams).

Не прочитано до конца (страница длиннее лимита выборки, читались первые ~100 000 знаков): sub-agents (хвост 16 тыс.), hooks (хвост 182 тыс.), settings-reference (хвост 342 тыс.), skills (хвост 18 тыс.). Что не подтверждено — помечено «не подтверждено».

## 1. Поля субагента (`.claude/agents/*.md`, frontmatter)

Ссылка: [sub-agents, Frontmatter fields](https://code.claude.com/docs/en/sub-agents). Обязательны только `name`, `description`. Неизвестные поля молча игнорируются (опечатка не даст ошибки).

| возможность | есть ли | точный синтаксис | ссылка | примечание |
| :- | :- | :- | :- | :- |
| `name` | да | строка, до 256 знаков, без `:` | sub-agents | обязательное |
| `description` | да | строка | sub-agents | обязательное; по ней выбирается делегирование |
| `tools` | да | `tools: Read, Grep, Bash` или YAML-список | sub-agents | |
| `disallowedTools` | да | тот же формат | sub-agents | |
| `model` | да | `sonnet`, `opus`, `haiku`, `fable`, полный ID, `inherit` | sub-agents | |
| `effort` | да | `low`, `medium`, `high`, `xhigh`, `max` | sub-agents | |
| `maxTurns` | да | число | sub-agents | |
| `memory` | да | `user`, `project` или `local` | sub-agents | пути: user `~/.claude/agent-memory/<имя>/`, project `.claude/agent-memory/<имя>/`, local `.claude/agent-memory-local/<имя>/` |
| `isolation` | да | `isolation: worktree` | sub-agents | единственное значение |
| `skills` | да | список скиллов для предзагрузки | sub-agents | для тиммейта из agent-teams НЕ применяется |
| `hooks` | да | хуки в рамках этого агента | sub-agents | `Stop` в frontmatter превращается в `SubagentStop` |
| `omitClaudeMd` | да | `true` — не грузить user/project/local CLAUDE.md | sub-agents | |
| `autoCompactWindow` | да | целое от 100000 до 1000000 | sub-agents | у тиммейта нужна версия 2.1.296+ |
| `permissionMode` | да | `default`, `acceptEdits`, `auto`, `dontAsk`, `bypassPermissions`, `plan`, `manual` (= `default`) | sub-agents | |
| `color` | да | `red blue green yellow purple orange pink cyan` | sub-agents | |
| прочие поля | есть сверх списка | `mcpServers`, `background`, `initialPrompt`, `experimental` | sub-agents | поля `prompt` в frontmatter нет: это ключ JSON для `--agents` |

## 2. Ограничение типов агентов для главной сессии

| возможность | есть ли | синтаксис | ссылка | примечание |
| :- | :- | :- | :- | :- |
| allowlist типов | да, но только для главного потока `claude --agent <имя>` | `tools: Agent(worker, researcher), Read, Bash` | sub-agents | в определении обычного субагента список в скобках игнорируется; без `Agent` в `tools` порождать субагентов нельзя; `Task(...)` — принятый псевдоним |
| запрет конкретных типов | да | `permissions.deny: ["Agent(Explore)"]` | sub-agents | подходит для обычной главной сессии без `--agent` |

## 3. Настройки

| возможность | есть ли | синтаксис | ссылка | примечание |
| :- | :- | :- | :- | :- |
| `subagentPromptCacheTtl` | да | `"subagentPromptCacheTtl": "1h"` (значения `"5m"` или `"1h"`) | [settings-reference](https://code.claude.com/docs/en/settings-reference) | действует на субагентов, workflows и фоновые запросы (сжатие, названия сессий); по умолчанию не задан; 1-часовая запись кэша тарифицируется дороже (agent-teams) |
| `permissions.allow` / `deny` | да | `"permissions": {"allow": ["Bash(npm run test *)"], "deny": ["Read(./.env)", "Read(./.env.*)"]}` | [settings](https://code.claude.com/docs/en/settings) | списки из файлов сливаются, а не перекрываются; `deny` и `ask` из репозитория действуют сразу, `allow` из общего файла ждёт доверия к папке |
| `hooks` в settings | да | ключ `hooks` в любом settings-файле | settings-reference (индекс) | подробная схема — на странице hooks |
| `.claude/settings.local.json` | да | личные переопределения проекта, вне git | settings | Claude Code сам добавляет в глобальный git-exclude; созданный вручную — в `.gitignore` самому |
| приоритет файлов | да | managed > командная строка (`--settings`) > `.claude/settings.local.json` > `.claude/settings.json` > `~/.claude/settings.json` | settings, Settings precedence | переменные окружения — не уровень стека; `defaultMode` `auto`/`bypassPermissions` из проектных файлов не действуют |
| `autoMemoryEnabled`, `autoMemoryDirectory` | да (только строка индекса) | ключи в settings | settings-reference | полные записи не прочитаны |

## 4. CLI

Ссылка: [cli-reference](https://code.claude.com/docs/en/cli-reference).

| возможность | есть ли | синтаксис | примечание |
| :- | :- | :- | :- |
| `--max-budget-usd` | да, только print-режим | `--max-budget-usd <сумма>` | оценка на стороне клиента, может отличаться от счёта |
| `--max-turns` | да, только print-режим | `claude -p --max-turns 3 "query"` | по достижении лимита выход с ошибкой; по умолчанию без лимита |
| `--continue` / `-c` | да | `claude -c` | последний разговор в текущей папке |
| `--resume` / `-r` | да | `claude -r "<ID или имя>" "query"` | без аргумента — интерактивный выбор |
| `--debug` | да | `--debug='mcp,startup'` | фильтр работает только в форме с `=`; есть ещё `--debug-file <путь>` |

## 5. Slash-команды

Ссылка: [commands](https://code.claude.com/docs/en/commands).

| команда | есть ли | синтаксис | примечание |
| :- | :- | :- | :- |
| `/usage` | да | `/usage` | расход сессии, лимиты плана; `/cost` и `/stats` — псевдонимы |
| `/context` | да | `/context [all]` | сетка использования контекста, подсказки |
| `/permissions` | да | `/permissions` | диалог правил allow/ask/deny |
| `/doctor` | да (встроенный скилл) | `/doctor [prompt-audit [path]]` | `prompt-audit` есть, аудит CLAUDE.md, rules, skills, subagents; можно указать путь, напр. `.claude/skills/deploy` |
| `/skill-doctor` | да | `/skill-doctor` | стоимость скиллов в контексте и частота использования; нужна версия 2.1.252+ и загрузка feature-флагов |
| `/compact` | да | `/compact [instructions]` | |
| `/memory` | да | `/memory` | правка CLAUDE.md, включение и выключение auto memory, просмотр записей |

## 6. События хуков

Ссылка: [hooks](https://code.claude.com/docs/en/hooks). Все 12 событий существуют. Коды возврата: 0 — успех (JSON на stdout для структурного управления; stdout становится контекстом для Claude только у `UserPromptSubmit`, `UserPromptExpansion`, `SessionStart`, `PostModelSwitch`); 2 — блокировка действия (смысл ниже); любой другой, включая 1 — неблокирующая ошибка, действие идёт дальше («Exit code 1 ... non-blocking error ... use exit 2» для политик). Входные поля подтверждены не для всех событий: «не подтверждено» там, где в прочитанной части страницы нет.

| событие | вход (подтверждено) | код 2 |
| :- | :- | :- |
| `PreToolUse` | `tool_name`, `tool_input`, `tool_use_id` | блокирует вызов инструмента |
| `PostToolUse` | вход не подтверждён | stderr показывается Claude; инструмент уже отработал |
| `Stop` | `last_assistant_message` | не даёт остановиться, разговор продолжается |
| `SubagentStop` | `agent_id`, `agent_type`, `last_assistant_message` | не даёт субагенту остановиться |
| `SubagentStart` | `agent_id`, `agent_type`; matcher — тип агента (в sub-agents: имя) | stderr только пользователю |
| `PreCompact` | matcher `manual` / `auto` | блокирует сжатие |
| `SessionStart` | matcher `startup`, `resume`, `clear`, `compact`, `fork`; поле `source` | stderr только пользователю |
| `UserPromptSubmit` | вход не подтверждён; matcher не поддерживает | блокирует промпт |
| `Notification` | matcher — тип уведомления (`permission_prompt`, `idle_prompt`) | код и stderr игнорируются |
| `TaskCreated` | вход не подтверждён | откатывает создание задачи (agent-teams: «exit 2 — не создавать, отправить обратную связь») |
| `TaskCompleted` | вход не подтверждён | не даёт отметить задачу выполненной |
| `TeammateIdle` | вход не подтверждён | не даёт тиммейту простаивать, он продолжает работу |

Также есть (не из запрошенных): `Setup`, `UserPromptExpansion`, `PermissionRequest`, `PermissionDenied`, `PostToolUseFailure`, `PostToolBatch`, `MessageDisplay`, `StopFailure`, `InstructionsLoaded`, `ConfigChange`, `CwdChanged`, `DirectoryAdded`, `FileChanged`, `WorktreeCreate`, `WorktreeRemove`, `PostCompact`, `PreModelSwitch`, `PostModelSwitch`, `Elicitation`, `ElicitationResult`, `SessionEnd`. Неизвестное имя события в settings — «Settings Warning», запись пропускается, остальной файл действует.

## 7. Скиллы

Ссылка: [skills](https://code.claude.com/docs/en/skills).

| возможность | есть ли | синтаксис | примечание |
| :- | :- | :- | :- |
| расположение | да | проект `.claude/skills/<имя>/SKILL.md`; личные `~/.claude/skills/<имя>/SKILL.md` | есть ещё enterprise, вложенные, плагинные |
| `disable-model-invocation` | да | `disable-model-invocation: true` | запретить автозагрузку моделью; вызов вручную `/имя` |
| `allowed-tools` | да | строка через пробел или запятую, либо YAML-список | разрешает инструменты без запроса на время хода; доступные инструменты НЕ ограничивает |
| `context: fork` | да | `context: fork` | запуск в форкнутом субагенте, по умолчанию в фоне; `background: false` — ждать результата |
| `paths` | да | glob-шаблоны строкой через запятую или списком | авто-загрузка только при работе с подходящими файлами |
| прочие поля | есть | `name`, `description`, `when_to_use`, `argument-hint`, `arguments`, `user-invocable`, `disallowed-tools`, `model`, `effort`, `agent`, `background`, `hooks`, `shell`, `metadata`, `license`, `compatibility` | |
| `claude plugin eval` | да, для скиллов в плагине | синтаксис и опции не подтверждены | гоняет промпты с плагином и без, оценивает graders, завершается с ненулевым кодом ниже порога (для CI); `skillOverrides` на плагинные скиллы не действует |
| `skill-creator` | да, плагин | `/plugin install skill-creator@claude-plugins-official` | тест-кейсы в `evals/evals.json` внутри папки скилла, результаты в `grading.json`; формат несовместим с `claude plugin eval` |

## 8. Память

Ссылки: [memory](https://code.claude.com/docs/en/memory), [sub-agents](https://code.claude.com/docs/en/sub-agents).

| возможность | есть ли | синтаксис | примечание |
| :- | :- | :- | :- |
| auto memory, `MEMORY.md` | да | каталог с индексом `MEMORY.md` и топик-файлами | при старте грузятся первые 200 строк или 25 КБ, что раньше; лимит только для `MEMORY.md`; для репозитория, общая между worktree; выключается через `/memory` или `autoMemoryEnabled` |
| `memory: project` у агента | да | `memory: project` → `.claude/agent-memory/<имя>/` | есть ещё `user` и `local` (`.claude/agent-memory-local/<имя>/`) |
| `CLAUDE.local.md` | да | `./CLAUDE.local.md` в корне проекта | грузится вместе с CLAUDE.md; в `.gitignore` добавлять самому; в другом worktree файла не будет; при наличии скрывает `AGENTS.md` |
| `CLAUDE.md` размер | рекомендация | до 200 строк на файл | жёсткий предел 4 MiB, больше файл не грузится |

## 9. Команды агентов (agent teams)

Ссылка: [agent-teams](https://code.claude.com/docs/en/agent-teams). Есть, экспериментально, выключено по умолчанию. Включение: `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` в окружении или `{"env": {"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"}}` в settings.json. Один лид, тиммейты — отдельные сессии с общим списком задач и почтой (`~/.claude/teams/...`). Режимы: `teammateMode` / `--teammate-mode` (`in-process`, `auto`, `tmux`, `iterm2`; флаг экспериментальный, в `--help` не виден). Предупреждение: при включённых командах именованный субагент запускается как тиммейт, а не как субагент; вернуть — значение `0`. Ограничения: нет возобновления in-process тиммейтов, одна команда на сессию, нет вложенных команд, права на старте общие с лидом. В `-p` режиме тиммейты не создаются. Хуки качества: `TeammateIdle`, `TaskCreated`, `TaskCompleted` (код 2 — вернуть замечание).

## Итог для канона v24

Нет в документации (кандидаты в ABANDON): из запрошенных пунктов — ничего. Все поля, настройки, флаги, команды и события из списка найдены. Не подтверждено документацией (в прочитанной части): входные поля `PostToolUse`, `UserPromptSubmit`, `TaskCreated`, `TaskCompleted`, `TeammateIdle`; синтаксис `claude plugin eval`.

Есть, но иначе или с оговоркой:
- `tools: Agent(a, b)` работает только для агента, запущенного как главный поток (`claude --agent`); для обычной главной сессии — `permissions.deny: ["Agent(Тип)"]`.
- `--max-budget-usd` и `--max-turns` — только в print-режиме (`-p`), в интерактивной сессии не действуют.
- `claude plugin eval` — только для скиллов, поставляемых в плагине; для проектных скиллов в `.claude/skills/` — `skill-creator` и `evals/evals.json`.
- `skills` в определении агента не применяется к тиммейту из команд агентов; `autoCompactWindow` не применяется к split-pane тиммейту.
- `allowed-tools` скилла не ограничивает инструменты, а лишь снимает запрос разрешения на ход.
- `/skill-doctor` требует версию 2.1.252+ и feature-флаги; `/doctor prompt-audit` — скилл внутри `/doctor`, не отдельная команда.
- Опечатки во frontmatter агента и скилла не дают ошибки (поле просто игнорируется): нужна проверка валидатором.
- Код возврата хука 1 не блокирует; для политики нужен 2.
