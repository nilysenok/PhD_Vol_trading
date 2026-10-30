# GATES — Т-срв: pytest на сервере по ветке, kodir, 31.10.2026

## G0 — тест падает на базе (скриптов нет)
CHECK: python3 -m pytest tests/test_sobrat_testy.py -q  (на первом коммите, до скриптов)
EXPECT: failed > 0
EVIDENCE: pending

## G1 — sobrat_testy.sh HEAD: tar < 50 МБ, есть tests/ и fastbt/, нет файлов из исключений 2022+
CHECK: python3 -m pytest tests/test_sobrat_testy.py -q -k g1
EXPECT: passed
EVIDENCE: pending

## G2 — мутация «фильтр 2022+ снят» роняет проверку
CHECK: python3 -m pytest tests/test_sobrat_testy.py -q -k g2
EXPECT: passed (мутант содержит запрещённые файлы, оригинал нет)
EVIDENCE: pending

## G3 — testy.sh в сухом режиме (TESTY_SUHOJ=1): итог-файл в формате
CHECK: python3 -m pytest tests/test_sobrat_testy.py -q -k g3
EXPECT: passed
EVIDENCE: pending

## G4 — bash -n обоих скриптов; shellcheck, если есть
CHECK: bash -n scripts/server/sobrat_testy.sh scripts/server/perenos/testy.sh
EXPECT: код 0
EVIDENCE: pending

## G5 — в отчёте ровно три команды владельцу с $SRV
CHECK: РУЧНОЙ — текст отчёта
EXPECT: три команды
EVIDENCE: pending

## Реестр
закрыто 0 из 6; ABANDON: 0; pending: 6
