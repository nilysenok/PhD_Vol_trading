# GATES — конвейер B/C/D (к2, к3, лестница σ), T3, кодер, 27.10

Владение: `fastbt/konveyer_bcd.py`, `scripts/konveyer_bcd.sh`, `fastbt/designs/sweep/{B,D}_setka_e3.yaml`, `tests/test_konveyer_bcd*.py`, `docs/tables/konveyer_bcd/chernovik/`. Код `s9*.py` не меняется. Только train 2018–2019.

## G1 — сетки B и D объявлены до счёта, N совпадает с #296 (B 54, D 396)
CHECK: python3 -m pytest -q tests/test_konveyer_bcd.py -k setki
EXPECT: passed
EVIDENCE: 2 passed, 17 deselected

## G2 — к2 с лагом 0 даёт исходный ряд (мутация № 9)
CHECK: python3 -m pytest -q tests/test_konveyer_bcd.py -k "k2"
EXPECT: passed (проверка на верном коде + падение на мутации «лаг 0 в повторах»)
EVIDENCE: 3 passed (test_k2_k3_ok, test_k2_lag_zero_caught, сопутствующий)

## G3 — к3 обнуляет ровно S5/S6 (мутация № 10)
CHECK: python3 -m pytest -q tests/test_konveyer_bcd.py -k "k3"
EXPECT: passed (+ падение при к3 с оставленной S5/S6)
EVIDENCE: 2 passed

## G4 — σ̂, сдвинутая на +1 день, ловится (мутация № 1)
CHECK: python3 -m pytest -q tests/test_konveyer_bcd.py -k "sigma"
EXPECT: passed (+ падение на мутации +1 день)
EVIDENCE: 6 passed (B, C, D × верный код / мутация)

## G5 — смена A-yaml меняет ключ кэша (Д5)
CHECK: python3 -m pytest -q tests/test_konveyer_bcd.py -k "klyuch"
EXPECT: passed
EVIDENCE: 2 passed

## G6 — конвейер на синтетике: B/C/D × лестница, колонки Шарп/метрики/к2/к3/Δ к A/p ЛВ есть, выбора точек нет
CHECK: python3 -m pytest -q tests/test_konveyer_bcd.py
EXPECT: passed
EVIDENCE: 19 passed

## G7 — период только train: годы 2018–2019, 2020+ в строках отсутствуют, годы > 2019 — ошибка
CHECK: python3 -m pytest -q tests/test_konveyer_bcd.py -k "train"
EXPECT: passed
EVIDENCE: 1 passed

## G8 — черновые таблицы B/C/D записаны, в шапке «черновик, текущий A, train 2018–2019, без выбора точек», версия кода и хэш входов
CHECK: head -5 docs/tables/konveyer_bcd/chernovik/D_tochki.csv
EXPECT: строки «черновик…», «код:», «входы:»
EVIDENCE: 6 файлов (B/C/D × tochki/lestnitsa), 114/114/726/582/625/426 строк; шапка есть: «код: 224f207 · fastbt 894a72f2d94c», «входы: A-yaml main.yaml 690aa10d01aa; …»

## G9 — одна команда перезапуска
CHECK: scripts/konveyer_bcd.sh --help
EXPECT: код 0, строка использования с <A-yaml>
EVIDENCE: rc=0, «использование: scripts/konveyer_bcd.sh <A-yaml> [опции]»; полный прогон scripts/konveyer_bcd.sh fastbt/designs/final/main.yaml — код 0

## G10 — py_compile, длина файлов, s9*.py и sweep_drivers.py не тронуты
CHECK: python3 -m py_compile fastbt/konveyer_bcd.py; wc -l; git diff --stat origin/kanon-v22-pereschet -- fastbt/s9.py fastbt/s9_stats.py fastbt/sweep_drivers.py
EXPECT: компиляция без ошибок; ≤ 200 строк; пустой diff
EVIDENCE: compiled; konveyer_bcd.py 200, test 200; diff пуст

## ABANDON
(нет)

## Реестр
закрыто 10 из 10; ABANDON: 0; pending: 0
