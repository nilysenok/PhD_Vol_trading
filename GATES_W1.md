# GATES W1 (пакет переноса wf; пп. 1 и 5; п. 4 отменён дирижёром, пп. 2–3 — листы W1b, W1c)
G1 12 отвергнутых объяснены (7 заменяются усечёнными копиями по --do, 5 не нужны коду) | CHECK: описание PR + pytest tests/test_wf_paket.py -k usech | EXPECT: passed | EVIDENCE: 4 passed (usech_do_wf_pit, mutaciya_usech_po_2021, usech_pq_i_json, usech_main_bez_otmetki)
G2 манифест wf без отметки — отказ, с отметкой — принят; мутация ловится | CHECK: pytest tests/test_wf_paket.py -k "otmetki or manifest" | EXPECT: passed | EVIDENCE: passed
G3 тесты семейства | CHECK: pytest tests/*perenos*.py tests/*wf*.py tests/test_usech_2021.py | EXPECT: 0 failed | EVIDENCE: 3 failed (устарели: test_kalibr_wf2b::test_otmetki_v_repozitorii_net, test_period_wf::test_g4_* — ждут отсутствия отметки, отметка теперь есть; тесты не менялись), 210 passed
G4 (отменён) пакет собран/развёрнут | EVIDENCE: пробно собран и развёрнут дважды (OK perenos, OK razvernut, OK [режим 1] proverka_yakor); не сдаётся
