# План этапов канона v24 (Ж-025, «довести все 21 ось до 95+»)

Blok: Ж-025 · arhitektor, 27.10.2026 · этап 0 (проектирование). Гейты — `GATES.md`, пути — `podstanovki.md`.

Общие правила плана:
- Хуки — скрипты в `scripts/kanon/huki/` с тестами на JSON-событиях. Подключение в `.claude/settings.json` делает только владелец: агент свои права не меняет. Каждый этап с хуками заканчивается файлом-предложением `docs/kanon/v24/settings_<этап>.json` и ручным шагом владельца.
- Сливать между прогонами агентов семьи E, не во время них: хуки начинают действовать в уже запущенных сессиях, а Э5 трогает загрузчики, которые эти агенты читают. Перед слиянием дирижёр смотрит, нет ли живых агентов E (`git worktree list`, фоновые pid на `sdf`).
- Код fastbt в v24 не переписывается. Нужные проверки — новые файлы рядом (`scripts/kanon/`, `tests/kanon/`) и тесты, которые вызывают старые функции.

## Этапы

| Э | Что делается | Владение файлами | Исполняет | Зависит от | Когда сливать | Оценка |
|---|---|---|---|---|---|---|
| Э1 | Журнал решений: `resheniya.md` (блоки дословно с квитанциями, в том числе из `docs/kanon_v23/zhurnal_resheniy.md` и `docs/resheniya/zhurnal.md` из #291), `ispolnenie.md`, `tolkovaniya.md`; хук «только дописывание»; хук и CI-проверка `Blok: Ж-NNN`; `sverka_blokov.py` | `docs/resheniya/*`, `scripts/kanon/sverka_blokov.py`, `scripts/kanon/proverka_pr.py`, `scripts/kanon/huki/{resheniya,pr_blok}.py`, `tests/kanon/test_huki.py` (часть Э1), `.github/workflows/kanon.yml` (новый), `.github/pull_request_template.md` (строка Blok) | pisar — тексты и квитанции (2 задачи); kodir — хуки, сверка, CI (2 задачи) | — (после слияния #291) | сразу, в работе E это не мешает (только новые файлы) | 1,5 ч |
| Э2 | Шаблон задания из 10 полей, `validate_zadanie.py`, 5 примеров, хук запуска агента (PreToolUse `Agent`), пробное задание kodir | `templates/zadanie_agentu.md`, `templates/examples/*` (5), `scripts/kanon/validate_zadanie.py`, `scripts/kanon/huki/zapusk_agenta.py`, `tests/kanon/test_validate_zadanie.py`, `tests/kanon/zadaniya_plohie/*` | arhitektor — шаблон, поля и примеры; kodir — валидатор, хук, тесты; kodir (другой экземпляр) — пробное задание для G8 | Э1 (строка Blok) | между прогонами E: хук запуска затрагивает все новые агенты | 2 ч |
| Э3 | Модель угроз и запреты: deny по `[ТЕСТОВЫЕ_ПУТИ]`, роль проверяющего, PII-фильтр (`git add` и pre-commit), `logs/audit.jsonl`, линтер чтения, список 20 законных команд E; красная команда | `scripts/kanon/huki/{zapret_chteniya,rol,pii,audit}.py`, `scripts/kanon/lint_chteniya.py`, `scripts/kanon/git-hooks/pre-commit`, `scripts/kanon/krasnaya_komanda.py`, `tests/kanon/{test_huki,test_pre_commit}.py`, `tests/kanon/zakonnye_komandy.jsonl`, `docs/kanon/v24/model_ugroz.md`, `docs/kanon/v24/settings_e3.json`, `.gitignore` (`logs/`) | arhitektor — модель угроз и список deny; kodir — хуки (3 задачи); proveryayushchiy — красная команда в отдельной копии; владелец — подключение settings, просмотр `/permissions` | Э1 (audit, Blok), Э2 (хук запуска использует валидатор) | только между прогонами E; подключение settings — когда агентов нет | 3 ч |
| Э4 | Жизненный цикл: Stop (тесты, ≤ 3 повтора), PreCompact → `handoff.md`, проверка места (SessionStart предупреждает, PreToolUse `Agent` запрещает), SubagentStop (≤ 500 слов), пульс и watchdog | `scripts/kanon/huki/{stop,precompact,mesto,subagentstop,puls}.py`, `scripts/kanon/watchdog.sh`, `scripts/kanon/testy_dlya.sh`, `tests/kanon/test_watchdog.py`, `docs/kanon/v24/settings_e4.json` | kodir (3 задачи); владелец — подключение settings и запуск watchdog | Э3 (общий разборщик событий и audit) | между прогонами E | 2,5 ч |
| Э5 | Реестр рядов с отметкой сдвига, загрузчик-обёртка, тест сдвига +1 на каждом ряду, тест ключей кэша, константные отрезки, `.meta`-шапки и воспроизведение трёх таблиц | `docs/kanon/v24/reestr_ryadov.yaml`, `docs/kanon/v24/reestr_klyuchey.md`, `scripts/kanon/reestr.py`, `scripts/kanon/vosproizvesti.sh`, `tests/kanon/{test_reestr,test_klyuch_kesha}.py`, `docs/tables/{semya_e_proverka_sdvig,tabl_4_marzha_2018_2021,tabl_4_sharpe_vs_stavka_2018_2021}.csv.meta` | arhitektor — контракт реестра и ключей; kodir (3 задачи; счёт — на `sdf` под `nice`); proveryayushchiy — подложенные сдвиги и мутации ключей | Э1; Э4 не обязателен | строго между прогонами E (читаются те же ряды; кэши не трогаются) | 3 ч |
| Э6 | FINAL-A (реконструкция с таблицей отступлений 26–27.10), теги `final-a-*`/`final-b-*`, хук заморозки, `proverka_tablits.py` и приложения контролей, навык `/otkrytie` без вызова моделью, реестры отбора | `docs/final/FINAL_A.md`, `docs/otbor/*_reestr.csv`, `scripts/kanon/{proverka_tablits,sverka_otbora}.py`, `scripts/kanon/huki/zamorozka.py`, `.claude/skills/otkrytie/SKILL.md`, `docs/tables/**/*.kontrol.csv`, `tests/kanon/test_proverka_tablits.py` | pisar — FINAL-A и отступления по журналу и `dev_log` (grep, не целиком); kodir — проверка, сверка, хук (3 задачи); владелец — теги и навык (`.claude/`) | Э1, Э3, Э5 (команды из `.meta`) | после закрытия текущего раунда E, до следующего открытия | 3 ч |
| Э7 | `mutations.md` по всем темам, прогон мутантов по `[КРИТИЧНЫЕ_МОДУЛИ]`, проверки слияния: отчёт мутаций и владение | `.claude/rules/mutations.md` (из #291), `scripts/kanon/{pokrytie_mutatsiy,mutanty,proverka_vladeniya}.py`, `tests/kanon/test_sliyanie.py`, `scripts/sliyanie.sh` (проверки), `.github/workflows/kanon.yml` (шаги), `docs/otchety/mutatsii/*` | kodir (2 задачи); proveryayushchiy — прогон мутантов на `sdf`; владелец — обязательная проверка в настройках GitHub | Э2 (владение в задании), Э5 (тесты, которые убивают мутантов) | между прогонами E | 2,5 ч |
| Э8 | Сжатие канона: `CLAUDE.md` ≤ 150 строк, правила по путям, таблица «сохранён / заменён / удалён», `/context`, `/doctor prompt-audit` | `CLAUDE.md`, `.claude/rules/*`, `docs/kanon/v24/izmeneniya_v23_v24.md`, `scripts/kanon/sverka_kanona.py` | pisar — тексты; arhitektor — таблица и решения по каждому пункту; kodir — сверка; владелец — `/context`, `/doctor` | Э1–Э7 (сжимается уже новый канон) | между прогонами E (меняет то, что читают агенты при старте) | 1,5 ч |
| Э9 | 11 метрик и 21 ось: формулы — функциями в коде, ретроспектива 26–27.10, радар | `scripts/kanon/metriki.py`, `tests/kanon/test_metriki.py`, `docs/otchety/kanon_v24/{osi_2026-10-26_27.md,radar_2026-10-26_27.png}` | arhitektor — формулы осей (из текста Ж-025); kodir (2 задачи) | Э1 (дословный текст осей), Э3 (audit для будущих замеров) | в любой момент, только новые файлы | 2 ч |
| Э10 | Финальный аудит, красная команда в живой сессии, таблица «инцидент → механизм» (14 строк) | `docs/otchety/kanon_v24/{audit_final,e10_krasnaya,intsidenty}.md`, `scripts/kanon/{reestr_geitov,sverka_intsidentov}.py` | proveryayushchiy (sonnet) — аудит и красная команда; arhitektor — таблица инцидентов; kodir — две сверки | Э1–Э9 | последним | 2 ч |

Итого ≈ 23 ч работы агентов, ≈ 45 задач по 15–20 мин. При потолке 3 сборочных агента и окнах между прогонами E — 2–3 календарных дня. Критический путь: Э1 → Э2 → Э3 → Э4 → Э10. Параллельно ему: Э5 → Э6 → Э7, затем Э8. Э9 — в любой момент после Э1.

## Конфликты и ABANDON-кандидаты

Проверено 27.10 в этом окружении: Claude Code 2.1.311 (строки и описания событий в самом бинаре), `crontab -l`, `which mutmut`, `/usr/bin/python3 --version`.

**(i) События хуков и команды.**
- `SubagentStart` есть, но **заблокировать запуск не может**: выход 2 только показывает stderr пользователю. Замена для Э2(б) и Э4(в) — PreToolUse с matcher `Agent`: он умеет отвечать deny. SubagentStart оставить на подсказку (additionalContext).
- `SubagentStop` есть, выход 2 возвращает ответ субагенту: Э4(д) делается штатно.
- `SessionStart` **отказать не может** (выход 2 — только stderr пользователю). Э4(в) пишется как «SessionStart предупреждает, PreToolUse `Agent` запрещает»; формулировку пункта — в `tolkovaniya.md`. Не ABANDON.
- `Stop` (выход 2 продолжает разговор) и `PreCompact`/`PostCompact` есть: Э4(а, б) делаются штатно. Против бесконечного цикла — `stop_hook_active` и ≤ 3 повтора.
- `/context` есть; доля видна только в живой сессии, поэтому G34 ручной.
- `/permissions` есть, это интерактивный экран, поэтому G13 ручной, со сверкой по `jq`.
- `/doctor prompt-audit` есть, но передаёт аудит встроенному навыку claude-api: это LLM-оценка, она недетерминирована и выключается настройками `skillOverrides`/`disableBundledSkills`. **ABANDON-кандидат G35** в буквальном виде «без замечаний». Замена: «каждое замечание закрыто или отклонено с причиной».
- Есть `disable-model-invocation`, а в событиях хука — `agent_type` и `agent_id`. Значит, Э6(в) и Э3(б) выполнимы: роль проверяющего берётся из `agent_type`, а `tools` в описании агента — второй слой.

**(ii) mutmut.** Не установлен (`which mutmut` пусто). Системный Python — 3.9.6. Поставить можно только из сети, а агенту в сеть нельзя. Совместимость mutmut 3.x с Python 3.9 и с macOS — ❓ для issledovatel. **ABANDON-кандидат — только слово «mutmut» в Э7(б), не сама метрика.** Замена: свой прогон мутантов `scripts/kanon/mutanty.py`. Операторы — из `mutations.md` (`shift(1)→shift(0)`, `ddof 1↔0`, `<=↔<`, выброс составляющей ключа кэша, `bfill`). Тот же отчёт «выжило k из M». Образец уже есть: `tests/test_s9_shag_db_mut.py`. Если владелец поставит mutmut — G30 считается им.

**(iii) watchdog.** Crontab пуст, cron в macOS требует прав доступа к диску. Есть launchd (`~/Library/LaunchAgents`) и в Claude Code — CronCreate/ScheduleWakeup, но они живут только в сессии. Хуки срабатывают на события, а у зависшего агента событий нет. Решение: PostToolUse пишет пульс `state/puls/<agent_id>`, `scripts/kanon/watchdog.sh` проверяет его каждые 5 мин с порогом 20 мин (≤ 25 мин). Запуск: (а) launchd-агент — изменение системы, только владелец; (б) умолчание без изменения системы — `nohup`-петля, которую поднимает SessionStart главной сессии (pid-файл, одна копия). ❓ владельцу: launchd или петля; умолчание — петля, срок 28.10.

**(iv) «Ничего в данных, расчётах и результатах не менять» против Э5 и Э6.** Развожу так, чтобы ни одно число не сдвинулось:
- Э5 (а, б): реестр и загрузчик `scripts/kanon/reestr.py` — **новые файлы рядом**. Старые загрузчики (`data.py`, `bars5m.py`, `sigma_e.py`, `msp.py`) не переписываются. Тест сверяет кадр старого загрузчика с кадром реестра, равенство — побайтно. Перевод вызовов на реестр — после FINAL v2, отдельным решением владельца.
- Э5 (в): единый `cache_key` **не вводится**. Тест мутирует каждую составляющую каждой функции ключа из `[КРИТИЧНЫЕ_МОДУЛИ]` и требует смены ключа. Если ключ не реагирует — это находка (❓ владельцу с PR-исправлением), а не молчаливый рефакторинг: исправление ключа инвалидирует кэши и меняет время пересчёта.
- Э5 (д): шапка таблицы — сопроводительный `<таблица>.meta`, сам CSV не меняется.
- Э6 (б): столбцы контролей — файл-приложение `<таблица>.kontrol.csv` с ключом строки. `proverka_tablits.py` соединяет таблицу с приложением. Сами таблицы разработки не меняются, `SHA256.txt` архивов не ломается.
- Э6 (д): FINAL-A реконструируется как документ. Результаты не пересчитываются.
- Если владелец под «добавить столбцы» имел в виду именно правку CSV — это его решение. ↺ — столбцы контролей в приложении, а не в самой таблице · откат: перенести столбцы в CSV одной командой `proverka_tablits.py --vlit` после FINAL v2 (27.10).

**(v) Пересечение с ветками v23.**
- `kanon-v23-tekst` (PR #291, открыт): `templates/` (GATES, handoff, zadanie_agentu, otchet_agenta, FINAL_DESIGN), `.claude/rules/mutations.md`, `docs/resheniya/zhurnal.md`, метрики §8 (6 чисел). Предложение: **слить #291 в `kanon-v22-pereschet` до Э1** и перебазировать `kanon-v24`. Дальше v24 поглощает так: `zhurnal.md` → `resheniya.md` (Э1); `zadanie_agentu.md` → шаблон из 10 полей (Э2); `mutations.md` дополняется темами (Э7); 6 чисел §8 входят в 11 метрик (Э9); `FINAL_DESIGN.md` — основа `FINAL_A.md` (Э6). Если #291 не сольют — v24 переносит эти файлы сам, `GATES.md` остаётся в запасном формате.
- `kanon-v23-huki`: ветка есть только локально (рабочая копия `agent-a9841f902f8161382`, locked, HEAD = a8562c2, своих коммитов нет). На origin её нет, чернового PR в `gh pr list` 27.10 нет. Предложение: её архитектор становится исполнителем проектирования хуков Э3–Э4, гейты G9–G18 — её приёмка; второй набор хуков не заводится. Работающего агента не прерываю. ❓ для дирижёра — подтвердить слияние ролей.

**(vi) Права.** Э3, Э4, Э6(в) требуют правки `.claude/settings.json` и `.claude/skills/`. Агенту это запрещено, а поручение агента согласием не считается. Поэтому каждый такой шаг — ручной шаг владельца с файлом-предложением. Без него G9–G18, G24, G26, G41 закрыть нельзя: это не ABANDON, а ожидание владельца.

**(vii) Красная команда.** Отдельной роли «красная команда» в списке агентов нет. Исполняет proveryayushchiy (sonnet) в отдельной копии, с заданием из шаблона и канарейкой. Успех — канарейка или число из тестового периода в выводе; сверяет скрипт, а не агент.

**(viii) Текст Ж-025 в ветке отсутствует.** 21 ось и 11 метрик по именам мне неизвестны. Э9 и G38–G39 зависят от того, как pisar дословно внесёт блок в Э1. Ось без формулы в тексте → `tolkovaniya.md` с ❓.

## Порядок и исполнители — коротко

1. Слить #291 (дирижёр, после вердикта) → перебазировать `kanon-v24`.
2. Э1 (pisar + kodir) → Э2 (arhitektor + kodir) → Э3 (arhitektor + kodir + proveryayushchiy + владелец) → Э4 (kodir + владелец).
3. Параллельно, в окна между прогонами E: Э5 (arhitektor + kodir + proveryayushchiy) → Э6 (pisar + kodir + владелец) → Э7 (kodir + proveryayushchiy).
4. Э9 (arhitektor + kodir) — после Э1, в любой момент.
5. Э8 (pisar + arhitektor) → Э10 (proveryayushchiy + arhitektor).
