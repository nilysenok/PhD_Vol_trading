# GATES: fast-razryv (быстрый слой, правило Params.razryv в ядре)

Записан после первой правки (отступление: гейты задним числом по ходу срочной задачи; формулировки совпадают с заданием дирижёра).

G1. Ядро = эталон побайтно на ≥ 1000 путях с разрывами 4/5/6/7/30 дней (лимит, стоп-заявка OCO, T, reentry_den, шорт, позиция через разрыв).
CHECK: `FASTBT_MAIN=$PWD python3 -m pytest -q tests/test_fast_razryv.py::test_sintetika_1200_putej_s_razryvami_yadro_ravno_etalonu`
EXPECT: 1 passed (счётчики valid>900, snyata>100, shorty>100, cherez>20 внутри теста)
EVIDENCE: см. описание PR

G2. Мутации «заявка не снимается в ядре», «разрыв ≥ вместо >» ловятся.
CHECK: `python3 -m pytest -q tests/test_fast_razryv.py -k mut`
EXPECT: 2 passed
EVIDENCE: см. описание PR

G3. Ядро реально вызывается (нет ухода в эталон).
CHECK: `python3 -m pytest -q tests/test_fast_razryv.py::test_yadro_vyzvano_a_ne_etalon`
EXPECT: 1 passed
EVIDENCE: см. описание PR

G4. Соседние наборы зелёные.
CHECK: `FASTBT_MAIN=$PWD FASTBT_V5_N=200 python3 -m pytest -q tests/test_fast_vhod_open.py tests/test_fast_v5_s34_pokryto.py tests/test_fast_v5_pokryto.py tests/test_fast_open2.py tests/test_razryv_provod.py tests/test_sessiya_razryv.py`
EXPECT: 0 failed
EVIDENCE: см. описание PR
