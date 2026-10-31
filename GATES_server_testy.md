# GATES — Т-срв: pytest на сервере по ветке, kodir, 31.10.2026

## G0 — тест падает на базе (скриптов нет)
CHECK: python3 -m pytest tests/test_sobrat_testy.py -q  (на первом коммите, до скриптов)
EXPECT: failed > 0
EVIDENCE: 4 failed in 0.08s (первый коммит)

## G1 — sobrat_testy.sh HEAD: tar < 50 МБ, есть tests/ и fastbt/, нет файлов из исключений 2022+
CHECK: python3 -m pytest tests/test_sobrat_testy.py -q -k g1
EXPECT: passed
EVIDENCE: passed; вручную: 1502 файла, 3,2 МБ, вырезано 2 файла (fastbt/rates/key_rate_cbr_vladelec.csv, scripts/analysis/tables_text_rychagi_vvod.csv)

## G2 — мутация «фильтр 2022+ снят» роняет проверку
CHECK: python3 -m pytest tests/test_sobrat_testy.py -q -k g2
EXPECT: passed (мутант отвергнут скриптом, tar мутанта содержит запрещённые файлы)
EVIDENCE: passed

## G3 — testy.sh в сухом режиме (TESTY_SUHOJ=1): итог-файл в формате
CHECK: python3 -m pytest tests/test_sobrat_testy.py -q -k g3
EXPECT: passed
EVIDENCE: passed (4 теста g1..g3 + zabrat --testy; всего 5 passed в наборе, 22 passed с соседними test_perenos_sh/test_server_skripty/test_ledger_agregaty)

## G4 — bash -n обоих скриптов (+ zabrat.sh); shellcheck, если есть
CHECK: bash -n scripts/server/sobrat_testy.sh scripts/server/perenos/testy.sh scripts/server/perenos/zabrat.sh
EXPECT: код 0
EVIDENCE: BASHN_OK; shellcheck на Маке не установлен (не проверено)

## G5 — в отчёте ровно три команды владельцу с $SRV
CHECK: РУЧНОЙ — текст отчёта
EXPECT: три команды
EVIDENCE: в отчёте

## Реестр
закрыто 6 из 6; ABANDON: 0; pending: 0
