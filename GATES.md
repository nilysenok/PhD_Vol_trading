# GATES — Ж-025 Э1 (текстовая часть) и Ж-026 ось 29, pisar, 27.10.2026

Blok: Ж-025, Ж-026 · Punkt: Ж-025#1 · ветка `kanon-v24-e1-zhurnaly` от `kanon-v22-pereschet` (`a2f0127`). Формат — `templates/GATES.md` (канон v23). Гейты G1–G4 перенесены из `docs/kanon/v24/GATES.md` (Приёмка 1: а — G1, г — G4; б, в — G2, G3: хуки, вне владения этого задания). Гейты G5–G11 — свои.

## G1 — три файла журнала существуют, квитанций не меньше 25
CHECK: ls docs/resheniya/resheniya.md docs/resheniya/ispolnenie.md docs/resheniya/tolkovaniya.md | wc -l && grep -c '^Квитанция: ' docs/resheniya/resheniya.md
EXPECT: `3`, затем число ≥ 25
EVIDENCE: `3`, затем `87` (код возврата 0)

## G2 — правка старой строки `resheniya.md` блокируется хуком, дописывание проходит
CHECK: /usr/bin/python3 -m pytest tests/kanon/test_huki.py -q -k "resheniya"
EXPECT: `2 passed`, 0 failed
EVIDENCE: pending (хук и тест — kodir, PR #294; в этой ветке файла `tests/kanon/test_huki.py` нет)

## G3 — `gh pr create` без `Blok: Ж-NNN` отказывает
CHECK: /usr/bin/python3 -m pytest tests/kanon/test_huki.py -q -k "pr_blok"
EXPECT: `2 passed`
EVIDENCE: pending (хук и проверка в CI — kodir, PR #294)

## G4 — сверка блоков находит квитанцию у каждого блока
CHECK: /usr/bin/python3 scripts/kanon/sverka_blokov.py | tail -1
EXPECT: `блоков без квитанции: 0`
EVIDENCE: pending (скрипт — kodir, PR #294; в этой ветке `scripts/kanon/sverka_blokov.py` нет). Замена до слияния — G9 ниже

## G5 — все блоки владельца 26–27.10 из плана перенесены дословно
CHECK: /usr/bin/python3 -I -c "import re;r=open('docs/resheniya/resheniya.md',encoding='utf-8').read();L=[l.split('- ',1)[1].rstrip('\n') for l in open('docs/zadaniya/2026-10-25_pereschet.md',encoding='utf-8') if re.match(r'^\s*- Ъ Владелец 2[67]\.10',l)];print(len(L),'в плане; не найдено:',sum(1 for t in L if t not in r))"
EXPECT: `78 в плане; не найдено: 0`
EVIDENCE: `78 в плане; не найдено: 0` (код возврата 0)

## G6 — `ispolnenie.md`: строка на каждый Ж-001…Ж-026
CHECK: grep -cE '^\| Ж-0(0[1-9]|1[0-9]|2[0-6]) ' docs/resheniya/ispolnenie.md
EXPECT: `26`
EVIDENCE: `26`

## G7 — `tolkovaniya.md`: толкования из плана и от 27.10 ~17:00 внесены
CHECK: grep -c '^- \[' docs/resheniya/tolkovaniya.md && grep -cE '^- \[Ж-02[56]' docs/resheniya/tolkovaniya.md
EXPECT: `≥ 21`, затем `≥ 5` (watchdog, README, `kodir`, `config/dokumenty.toml`, E0)
EVIDENCE: `26`, затем `10`

## G8 — `STATUS.md` в корне с тремя разделами
CHECK: grep -c '^## ' STATUS.md
EXPECT: `3` (Что идёт · Сколько осталось · Следующая развилка)
EVIDENCE: `3`

## G9 — число блоков равно числу квитанций (замена G4 до слияния #294)
CHECK: test "$(grep -c '^### ' docs/resheniya/resheniya.md)" = "$(grep -c '^Квитанция: ' docs/resheniya/resheniya.md)" && echo ravno
EXPECT: `ravno`
EVIDENCE: `ravno` (87 = 87)

## G10 — `zhurnal.md` помечен «заменён», файл на месте; шаблон вопроса — не больше трёх
CHECK: grep -c 'заменён `resheniya.md`' docs/resheniya/zhurnal.md && grep -c 'не больше трёх' templates/vopros.md
EXPECT: `1`, затем `1`
EVIDENCE: `1`, затем `1`

## G11 — замечания проверяющего к #293 исправлены (а, б, в)
CHECK: grep -c '^- ↺ — единый' docs/kanon/v24/plan_etapov.md && grep -c '^| [0-9]* |' docs/kanon/v24/plan_etapov.md && grep -c 'vne_chasov' docs/kanon/v24/GATES.md && grep -c 'Замена mutmut' docs/kanon/v24/GATES.md
EXPECT: `1`, `21`, `≥ 1`, `1`
EVIDENCE: `1`, `21`, `2`, `1` (в `plan_etapov.md` строк с `| N |` ровно 21 — таблица осей 1–21; строки этапов начинаются с `| Э`)

## ABANDON (только с причиной; гейт не удалять и не переписывать)
ABANDON: нет. G2, G3, G4 не закрыты и не отменены: pending до слияния PR #294 (kodir); причина — файлы вне владения этого задания.

## Реестр
закрыто 8 из 11; ABANDON: 0; pending: 3 (G2, G3, G4 — ждут #294)
