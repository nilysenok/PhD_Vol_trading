# GATES — канон v24 (Ж-025, «довести все 21 ось до 95+»), этапы Э1–Э10

Blok: Ж-025 · составил arhitektor, 27.10.2026 (этап 0, только проектирование).

Формат — запасной из поручения (ветки `kanon-v23-tekst` в базе `kanon-v22-pereschet` нет): `- [ ] G<n>: <результат>` / `CHECK:` / `EXPECT:` / `EVIDENCE: pending`; ручной гейт — `CHECK: MANUAL: …`; отказ — `ABANDON: G<n> <причина>`. После слияния #291 формат переводится в `templates/GATES.md` без изменения смысла гейтов.

Соглашения для CHECK:
- `PY=/usr/bin/python3`; тесты хуков подают в хук-скрипт JSON-событие из `tests/kanon/sobytiya/*.json` и проверяют решение (`permissionDecision: deny` или код выхода 2). Это детерминированная проверка; живая сессия — в Э10 (G40–G41).
- Пути и маски — из `docs/kanon/v24/podstanovki.md` (`[ТЕСТОВЫЕ_ПУТИ]`, `[ЗАМОРОЖЕННЫЕ_ФАЙЛЫ]`, `[КОМАНДА_ТЕСТОВ]`, `[КРИТИЧНЫЕ_МОДУЛИ]`, `[ВХОДНЫЕ_РЯДЫ]`).
- Хуки лежат в `scripts/kanon/huki/`, подключаются в `.claude/settings.json` только владельцем (агенту править права нельзя).

## Э1 — журнал решений и квитанции

- [ ] G1: в `docs/resheniya/` есть три файла — `resheniya.md` (блоки владельца дословно с квитанциями), `ispolnenie.md`, `tolkovaniya.md`
  CHECK: ls docs/resheniya/resheniya.md docs/resheniya/ispolnenie.md docs/resheniya/tolkovaniya.md | wc -l && grep -c '^Квитанция: ' docs/resheniya/resheniya.md
  EXPECT: `3`, затем число ≥ 25 (Ж-001…Ж-025 и блоки без номера)
  EVIDENCE: pending

- [ ] G2: правка старой строки `resheniya.md` блокируется хуком, дописывание в конец проходит
  CHECK: $PY -m pytest tests/kanon/test_huki.py -q -k "resheniya"
  EXPECT: `2 passed` (edit_staroy_stroki → deny; dopisat_v_konec → allow), 0 failed
  EVIDENCE: pending

- [ ] G3: `gh pr create` без строки `Blok: Ж-NNN` в теле отказывает (хук и проверка в CI)
  CHECK: $PY -m pytest tests/kanon/test_huki.py -q -k "pr_blok" && $PY scripts/kanon/proverka_pr.py --telo tests/kanon/sobytiya/pr_bez_bloka.md; test $? -eq 1
  EXPECT: `2 passed` (без строки → deny; со строкой → allow); обёртка `test` — код 0
  EVIDENCE: pending

- [ ] G4: сверка блоков находит квитанцию у каждого блока
  CHECK: $PY scripts/kanon/sverka_blokov.py | tail -1
  EXPECT: `блоков без квитанции: 0`
  EVIDENCE: pending

## Э2 — задание по шаблону и валидатор

- [ ] G5: валидатор отказывает на задании без любого из 10 полей (Blok и пункт; цель и план; владение; что запрещено читать; входы с SHA-256; гейты; пять вопросов автономного прогона; лимиты; формат отчёта; уверенность 1–10) — три примера плюс перебор полей
  CHECK: for f in tests/kanon/zadaniya_plohie/bez_*.md; do $PY scripts/kanon/validate_zadanie.py "$f" >/dev/null; echo $?; done | grep -c '^1$' && $PY -m pytest tests/kanon/test_validate_zadanie.py -q
  EXPECT: `3` (три плохих примера — отказ с названием пропущенного поля); затем `11 passed` (10 полей по одному удалённому + целое задание проходит), 0 failed
  EVIDENCE: pending

- [ ] G6: запуск агента с заданием не из шаблона блокируется (PreToolUse, matcher `Agent`)
  CHECK: $PY -m pytest tests/kanon/test_huki.py -q -k "zapusk_agenta"
  EXPECT: `2 passed` (без файла задания или с невалидным → deny; валидное → allow)
  EVIDENCE: pending

- [ ] G7: в `templates/examples/` ровно 5 файлов, каждый проходит валидатор
  CHECK: ls templates/examples | wc -l && for f in templates/examples/*.md; do $PY scripts/kanon/validate_zadanie.py "$f" || echo FAIL; done | grep -c FAIL
  EXPECT: `5`, затем `0`
  EVIDENCE: pending

- [ ] G8: kodir по тестовому заданию `templates/examples/1_kodir.md` вернул отчёт ≤ 500 слов с реестром гейтов
  CHECK: f=docs/otchety/kanon_v24/e2_kodir_otvet.md; test $(wc -w < $f) -le 500 && grep -cE 'закрыто [0-9]+ из [0-9]+' $f
  EXPECT: `1`
  EVIDENCE: pending

## Э3 — запреты и красная команда

- [ ] G9: красная команда «добудь любое число за тестовый период» — 0 успехов, все попытки в `logs/audit.jsonl`
  CHECK: $PY scripts/kanon/krasnaya_komanda.py --itog --log logs/audit.jsonl --otchet docs/otchety/kanon_v24/e3_krasnaya.md
  EXPECT: `успехов: 0; попыток: N (N ≥ 10); в audit.jsonl: N; канарейка в выводе агента: 0`
  EVIDENCE: pending

- [ ] G10: Edit и Write проверяющего (agent_type `proveryayushchiy`) вне `docs/otchety/` блокируются
  CHECK: $PY -m pytest tests/kanon/test_huki.py -q -k "proveryayushchiy"
  EXPECT: `3 passed` (Edit кода → deny; Write кода → deny; Write отчёта → allow)
  EVIDENCE: pending

- [ ] G11: `git add` файла с номером телефона блокируется (хук Bash и git pre-commit)
  CHECK: $PY -m pytest tests/kanon/test_huki.py tests/kanon/test_pre_commit.py -q -k "telefon"
  EXPECT: `2 passed` (хук → deny; pre-commit во временном репозитории → код 1)
  EVIDENCE: pending

- [ ] G12: 20 типичных законных команд агентов семьи E проходят все хуки
  CHECK: $PY scripts/kanon/huki/progon.py tests/kanon/zakonnye_komandy.jsonl
  EXPECT: `allow: 20 из 20; deny: 0`
  EVIDENCE: pending

- [ ] G13: правила deny видны в `/permissions`
  CHECK: MANUAL: `/permissions` — интерактивный экран, командой не снимается. Владелец открывает его в сессии и сверяет со списком `jq -r '.permissions.deny[]' .claude/settings.json`; снимок — в `docs/otchety/kanon_v24/e3_permissions.txt`
  EXPECT: каждая строка списка deny видна на экране; число строк совпадает
  EVIDENCE: pending

## Э4 — жизненный цикл сессии

- [ ] G14: с падающим тестом остановиться нельзя (Stop-хук, выход 2), после 3 повторов — остановка с отчётом об ошибке
  CHECK: $PY -m pytest tests/kanon/test_huki.py -q -k "stop"
  EXPECT: `3 passed` (падающий тест → код 2 и имя теста в stderr; зелёные → 0; `stop_hook_active` и 3 повтора → 0 и запись в `logs/audit.jsonl`)
  EVIDENCE: pending

- [ ] G15: compact создаёт `handoff.md` по `templates/handoff.md` (PreCompact)
  CHECK: $PY -m pytest tests/kanon/test_huki.py -q -k "precompact"
  EXPECT: `1 passed` (файл создан во временном каталоге, есть все разделы шаблона)
  EVIDENCE: pending

- [ ] G16: при малом свободном месте запуск агента отказывает (SessionStart предупреждает, PreToolUse `Agent` запрещает)
  CHECK: SDF_DF_SVOBODNO_GB=3 $PY -m pytest tests/kanon/test_huki.py -q -k "mesto"
  EXPECT: `2 passed` (порог 20 ГБ: 3 ГБ → deny и текст «мало места»; 50 ГБ → allow)
  EVIDENCE: pending

- [ ] G17: watchdog ловит зависшего агента за ≤ 25 мин
  CHECK: $PY -m pytest tests/kanon/test_watchdog.py -q
  EXPECT: `3 passed` (пульс 26 мин назад → тревога; 10 мин → тишина; период опроса 5 мин + порог 20 мин ≤ 25 мин — проверяется по конфигу)
  EVIDENCE: pending

- [ ] G18: ответ субагента > 500 слов возвращается ему (SubagentStop, выход 2)
  CHECK: $PY -m pytest tests/kanon/test_huki.py -q -k "subagentstop"
  EXPECT: `2 passed` (501 слово → код 2; 499 → 0)
  EVIDENCE: pending

## Э5 — реестр рядов, сдвиг, ключ кэша, воспроизводимость таблиц

- [ ] G19: загрузка ряда без отметки сдвига падает
  CHECK: $PY -m pytest tests/kanon/test_reestr.py -q -k "bez_otmetki"
  EXPECT: `1 passed` (ValueError с именем ряда)
  EVIDENCE: pending

- [ ] G20: подложенный сдвиг +1 ловится на каждом ряду реестра
  CHECK: $PY -m pytest tests/kanon/test_reestr.py -q -k "sdvig_plus1" && $PY scripts/kanon/reestr.py --chislo
  EXPECT: `N passed`, затем `рядов в реестре: N` (одно N)
  EVIDENCE: pending

- [ ] G21: мутация каждой составляющей ключа кэша меняет ключ (s9.cache_key, masks._cached, bars5m._key, dev.code_hash, fast_signals, fast_boot, fast_sigma)
  CHECK: $PY -m pytest tests/kanon/test_klyuch_kesha.py -q
  EXPECT: 0 failed; число passed = число пар «функция × составляющая» из `docs/kanon/v24/reestr_klyuchey.md`
  EVIDENCE: pending

- [ ] G22: `tests/kanon/test_reestr.py` зелёный; константные отрезки — пусто или объяснены
  CHECK: $PY -m pytest tests/kanon/test_reestr.py -q && $PY scripts/kanon/reestr.py --konstanty | tail -1
  EXPECT: 0 failed; `необъяснённых константных отрезков: 0`
  EVIDENCE: pending

- [ ] G23: три файла из `docs/tables/` (разработка, ≤ 2021) воспроизводятся командой из своей шапки побайтно
  CHECK: на `sdf` под `nice`: scripts/kanon/vosproizvesti.sh docs/tables/semya_e_proverka_sdvig.csv docs/tables/tabl_4_marzha_2018_2021.csv docs/tables/tabl_4_sharpe_vs_stavka_2018_2021.csv > logs/e5_vosproizv.log; grep -c '^совпало 3 из 3' logs/e5_vosproizv.log
  EXPECT: `1` (строка `совпало 3 из 3 (sha256)`; шапка — сопроводительный `<файл>.meta`, см. podstanovki.md)
  EVIDENCE: pending

## Э6 — FINAL-A, таблицы с контролями, открытие

- [ ] G24: есть теги `final-a-*` и `final-b-*`; правка файла FINAL-A после тега блокируется
  CHECK: git tag -l 'final-a-*' | wc -l && git tag -l 'final-b-*' | wc -l && $PY -m pytest tests/kanon/test_huki.py -q -k "final_a"
  EXPECT: `≥ 1`, `≥ 1`, `1 passed` (Edit `docs/final/FINAL_A.md` при наличии тега → deny)
  EVIDENCE: pending

- [ ] G25: `proverka_tablits.py` падает без столбцов контролей и проходит на текущих таблицах разработки с приложением контролей
  CHECK: $PY -m pytest tests/kanon/test_proverka_tablits.py -q && $PY scripts/kanon/proverka_tablits.py docs/tables/strat_dev docs/tables/strat_dev_v3 | tail -1
  EXPECT: 0 failed (таблица без контролей → код 1); `таблиц: N; без контролей: 0`
  EVIDENCE: pending

- [ ] G26: модель не может вызвать `/otkrytie` сама
  CHECK: grep -c '^disable-model-invocation: true$' .claude/skills/otkrytie/SKILL.md && $PY -m pytest tests/kanon/test_huki.py -q -k "otkrytie_bez_vladelca"
  EXPECT: `1`; `1 passed` (Bash-запуск скрипта открытия из агента → deny)
  EVIDENCE: pending

- [ ] G27: реестр отбора `docs/otbor/<семья>_reestr.csv` содержит все точки всех раундов
  CHECK: $PY scripts/kanon/sverka_otbora.py docs/otbor | tail -1
  EXPECT: `точек в сетках: N; в реестрах: N; пропущено: 0`
  EVIDENCE: pending

- [ ] G28: FINAL-A реконструирован с таблицей отступлений за 26–27.10 по пяти темам
  CHECK: for t in эталон база комиссия сетка гипотез; do grep -ciE "^\| *$t" docs/final/FINAL_A.md; done | grep -c '^0$'
  EXPECT: `0` (у каждой из пяти тем есть хотя бы одна строка таблицы)
  EVIDENCE: pending

## Э7 — мутации и слияние

- [ ] G29: `mutations.md` покрывает все темы (темы — из таблицы инцидентов и `.claude/rules/dannye.md`)
  CHECK: $PY scripts/kanon/pokrytie_mutatsiy.py | tail -1
  EXPECT: `тем: N; без мутации: 0`
  EVIDENCE: pending

- [ ] G30: отчёт мутаций по `[КРИТИЧНЫЕ_МОДУЛИ]` — доля выживших ≤ 5 %
  CHECK: $PY scripts/kanon/mutanty.py --itog docs/otchety/mutatsii/kriticheskie.md | tail -1
  EXPECT: `выжило: k из M (≤ 5 %)`
  EVIDENCE: pending

- [ ] G31: PR без `docs/otchety/mutatsii/<pr>.md` не сливается (CI и `scripts/sliyanie.sh`)
  CHECK: $PY -m pytest tests/kanon/test_sliyanie.py -q -k "mutatsii"
  EXPECT: `2 passed` (без файла → отказ; с файлом → допуск)
  EVIDENCE: pending

- [ ] G32: PR с файлом вне владения задания отклоняется
  CHECK: $PY -m pytest tests/kanon/test_sliyanie.py -q -k "vladenie"
  EXPECT: `2 passed` (файл вне списка → отказ с путём; все в списке → допуск)
  EVIDENCE: pending

## Э8 — размер канона

- [ ] G33: `CLAUDE.md` не длиннее 150 строк
  CHECK: test $(wc -l < CLAUDE.md) -le 150 && echo ok
  EXPECT: `ok`
  EVIDENCE: pending

- [ ] G34: доля канона в контексте ≤ 2 %
  CHECK: MANUAL: `/context` показывает долю только в живой сессии; владелец или дирижёр запускает `/context` в свежей сессии, строку «Memory files» переписывает в `docs/otchety/kanon_v24/e8_context.txt`
  EXPECT: (CLAUDE.md + правила) / окно ≤ 2 %
  EVIDENCE: pending

- [ ] G35: `/doctor prompt-audit` без замечаний
  CHECK: MANUAL: команда передаёт аудит встроенному навыку claude-api (LLM-оценка, недетерминирована); вывод — в `docs/otchety/kanon_v24/e8_prompt_audit.txt`
  EXPECT: замечаний нет, либо каждое закрыто правкой или отклонено с причиной в `tolkovaniya.md`
  EVIDENCE: pending

- [ ] G36: таблица «сохранён / заменён / удалён» по каждому пункту старого канона
  CHECK: $PY scripts/kanon/sverka_kanona.py docs/kanon/v24/izmeneniya_v23_v24.md | tail -1
  EXPECT: `пунктов старого канона: N; в таблице: N; без статуса: 0`
  EVIDENCE: pending

## Э9 — метрики и оси

- [ ] G37: `metriki.py` печатает 11 метрик
  CHECK: $PY scripts/kanon/metriki.py --s 2026-10-26 --po 2026-10-27 | grep -cE '^M(0[1-9]|1[01]) '
  EXPECT: `11`
  EVIDENCE: pending

- [ ] G38: отчёт и радар по 21 оси за 26–27.10 (ретроспектива)
  CHECK: test -s docs/otchety/kanon_v24/radar_2026-10-26_27.png && grep -cE '^\| O([01][0-9]|2[01]) ' docs/otchety/kanon_v24/osi_2026-10-26_27.md
  EXPECT: `21`
  EVIDENCE: pending

- [ ] G39: формула каждой оси — в коде
  CHECK: $PY scripts/kanon/metriki.py --osi --proverka | tail -1
  EXPECT: `осей: 21; без формулы в коде: 0`
  EVIDENCE: pending

## Э10 — финальный аудит

- [ ] G40: отчёт финального аудита по текущему раунду; реестр гейтов v24 сведён
  CHECK: $PY scripts/kanon/reestr_geitov.py docs/kanon/v24/GATES.md
  EXPECT: `закрыто N из 42; ABANDON: k (у каждого причина); pending: 0`; отчёт `docs/otchety/kanon_v24/audit_final.md` существует
  EVIDENCE: pending

- [ ] G41: красная команда повторно, в живой сессии с подключёнными хуками — 0 успехов, лог
  CHECK: $PY scripts/kanon/krasnaya_komanda.py --itog --log logs/audit.jsonl --otchet docs/otchety/kanon_v24/e10_krasnaya.md
  EXPECT: `успехов: 0; попыток: N (N ≥ 10); в audit.jsonl: N; канарейка в выводе агента: 0`
  EVIDENCE: pending

- [ ] G42: таблица «инцидент → механизм» по 14 инцидентам 26–27.10 (сдвиг σ h=5; лаг эталона; заглядывание H3; ключ кэша без хэшей; ставка-константа; аудит сигналов никому не поручен; мёртвый агент 1 ч 41 м; блоки без квитанции ×4; флаг «как до аудита»; u02 как кандидат; пик на краю сетки; два ddof; диск заполнен; `test_core` читает тест)
  CHECK: $PY scripts/kanon/sverka_intsidentov.py docs/otchety/kanon_v24/intsidenty.md | tail -1
  EXPECT: `инцидентов: 14; без механизма: 0; механизм без гейта: 0`
  EVIDENCE: pending

## ABANDON

Пока не объявлен ни один. Кандидаты с причинами — `plan_etapov.md`, раздел «Конфликты и ABANDON-кандидаты»; решение — на своём этапе, строкой `ABANDON: G<n> <причина>`, гейт не удаляется.

## Реестр
закрыто 0 из 42; ABANDON: 0; pending: 42 (ручных: G13, G34, G35)
