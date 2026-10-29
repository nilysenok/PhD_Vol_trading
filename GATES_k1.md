# GATES — К-1 пул калибратора на ledger 2.1 + ключ "v2", kodir, 29.10

## G1 — ключ PERIODY: "v2" есть, "V2" нет, kalibrovat(…, "v2") не падает
CHECK: python3 -m pytest tests/test_kalibr_v5_21.py -k klyuch_perioda -q
EXPECT: passed
EVIDENCE: 1 passed (в составе «6 passed, 1 deselected» для -k "klyuch_perioda or raven or mutaciya or otkaz", после слияния базы)

## G2 — top-50 и плато на суммах = пулу 2.0 (Шарп ≤ 1e-12, те же ключи, тот же порядок)
CHECK: python3 -m pytest tests/test_kalibr_v5_21.py -k raven_pulu_20 -q
EXPECT: passed
EVIDENCE: passed (2 теста: полный пул и досчитанные соседи)

## G3 — окно не содержит год t (мутации № 8, 21): сдвиг сегмента j+1 и j+4 ловится
CHECK: python3 -m pytest tests/test_kalibr_v5_21.py -k mutaciya -q
EXPECT: passed
EVIDENCE: passed (2 теста: окно с годом t; выход на конце окна не оплачен)

## G4 — отказы: ledger 2.0 / без poz_konec, окно вне 2016–2019
CHECK: python3 -m pytest tests/test_kalibr_v5_21.py -k otkaz -q
EXPECT: passed
EVIDENCE: passed

## G5 — регрессия соседей
CHECK: python3 -m pytest tests/test_kalibr_v5.py tests/test_svodka_ledger_v5.py tests/test_kalibr_v5_21.py -q
EXPECT: passed
EVIDENCE: 28 passed (до слияния базы); `cargo xtask check` — не применим (Rust-проекта нет, CLAUDE.md)
Примечание: tests/test_kalibr_v5_golos_v2.py — 6 failed и на чистой базе (проверено откатом правок), не от К-1.

## Реестр
закрыто 5 из 5; ABANDON: 0; pending: 0
