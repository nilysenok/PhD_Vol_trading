# GATES Ф-сл2b

G1 CHECK: `FASTBT_MAIN=$PWD FASTBT_V5_N=50 python3 -m pytest tests/test_fast_v5_s34_pokryto.py -q -k ekviv`
EXPECT: S3 и S4 по 50 точек: быстрый = медленный побайтно (ledger-строка без t_sek/sloy, port, tik), ядро реально вызвано
EVIDENCE: pending

G2 CHECK: `python3 -m pytest tests/test_fast_v5_s34_pokryto.py -q -k "pokryto or deleg or part"`
EXPECT: pokryto зависит от флага YADRO_S34_GOTOV; мутация «ось принята при ядре без неё» ловится; part строго
EVIDENCE: pending

G3 CHECK: `python3 -m pytest tests/test_fast_v5_s34.py tests/test_fast_v5_pokryto.py tests/test_fast_etap1_smoke.py -q`
EXPECT: регрессии нет
EVIDENCE: pending

G4 CHECK: замер S3/S4 до/после, 20 точек, 1 процесс (`-k zamer -s`)
EXPECT: печать с/точку; потолок
EVIDENCE: pending
