# GATES W1 (пакет переноса wf; пп. 1, 4, 5 — пп. 2, 3 переданы листам W1b, W1c)
G1 12 отвергнутых объяснены, 8 законно заменены усечёнными копиями | CHECK: pytest tests/test_wf_paket.py -k "otvergnuto or manifest_wf_ne" | EXPECT: passed | EVIDENCE: pending
G2 манифест wf закоммичен, max ≤ 2026-02-28 | CHECK: pytest tests/test_wf_paket.py | EXPECT: 10 passed | EVIDENCE: pending
G3 пакет собран, OK perenos | CHECK: sobrat_perenos.sh HEAD (PERENOS_DO=2026-02-28) | EXPECT: строка OK perenos | EVIDENCE: pending
G4 развёрнут, proverka_yakor OK | CHECK: razvernut.sh; proverka_yakor.sh | EXPECT: OK | EVIDENCE: pending
G5 zapusk_sobol.sh wf: приём отметки / отказы | CHECK: вызовы в развёрнутом корне | EXPECT: отказы названы, отметка принята | EVIDENCE: pending
G6 семейство тестов зелёное | CHECK: pytest tests/*perenos*.py tests/*wf*.py tests/test_usech_2021.py | EXPECT: 0 failed | EVIDENCE: pending
