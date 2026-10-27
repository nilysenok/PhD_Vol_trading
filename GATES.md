# GATES — Ж-025 часть 1 (Э1, Приёмка 1 б, в, г), kodir, 27.10

## G1 — тесты трёх скриптов проходят на исправленном коде
CHECK: /usr/bin/python3 -m unittest discover -s tests/kanon -p 'test_*.py'
EXPECT: Ran 14 tests, OK
EVIDENCE: 17 tests, OK (код возврата 0); добавлено по Ж-025 п. 2.4: пустой/отсутствующий/без шапки журнал → код 1 «журнал пуст или не найден», мутант «пустой → 0» ловится. Ветки kanon-v24-e1-zhurnaly на origin нет (ls-remote пуст), G5 pending

## G2 — мутации в памяти роняют проверку (квитанция не засчитывается, правка старой строки проходит, PR без Blok проходит, пропуск номера не ловится)
CHECK: те же тесты: test_mutant_receipt_not_counted, test_mutant_any_receipt_number, test_mutant_gap_not_caught, test_mutant_edit_passes, test_mutant_only_length_check, test_mutant_blok_not_required, test_mutant_no_check
EXPECT: каждый мутант даёт непустой список погрешностей; тест проходит
EVIDENCE: 7 мутационных тестов из 14, все OK

## G3 — proverka_append: правка/удаление старой строки — код 2 со ссылкой «Ж-025 п. 1.1»; дописывание — код 0
CHECK: test_exit_codes в tests/kanon/test_proverka_append.py
EXPECT: коды 0 и 2, в stderr «Ж-025 п. 1.1»
EVIDENCE: OK в составе G1

## G4 — proverka_pr_blok: нет Blok или Punkt — код 2
CHECK: test_exit_codes_stdin в tests/kanon/test_proverka_pr_blok.py
EXPECT: коды 0 и 2
EVIDENCE: OK в составе G1

## G5 — sverka_blokov на реальном resheniya.md
CHECK: /usr/bin/python3 scripts/kanon/sverka_blokov.py docs/resheniya/resheniya.md
EXPECT: «блоков всего: N», «без квитанции: 0», код 0
EVIDENCE: pending — файла нет ни в базе (a2f0127), ни в origin/kanon-v24-e1-zhurnaly (ветки на 27.10 нет). Пробный прогон на docs/resheniya/zhurnal.md (формат таблицей, блоков `## Ж-NNN · ЧЧ:ММ` нет): «блоков всего: 0», код 0.

## ABANDON

## Реестр
закрыто 4 из 5; ABANDON: 0; pending: 1 (G5, ждёт файла писаря)
