# GATES — П7 калибратор v5 `fastbt/kalibr_v5.py`, kodir, 29.10.2026

## G1 — тесты калибратора зелёные
CHECK: python3 -m pytest tests/test_kalibr_v5.py -q -p no:cacheprovider
EXPECT: N passed, 0 failed
EVIDENCE: `10 passed in 0.66s`, rc=0 (29.10 12:24, после merge origin/kanon-v22-pereschet @ 43c29715)

## G2 — известный ответ: в окне выбирается точка с лучшим Шарпом
CHECK: python3 -m pytest tests/test_kalibr_v5.py -q -p no:cacheprovider -k izvestnyj_otvet
EXPECT: passed, 0 failed
EVIDENCE: `1 passed, 9 deselected`, rc=0

## G3 — голосование = approach_a (ApproachA.ensemble) на тех же входах побайтно
CHECK: python3 -m pytest tests/test_kalibr_v5.py -q -p no:cacheprovider -k golosovanie
EXPECT: passed, 0 failed
EVIDENCE: `1 passed, 9 deselected`, rc=0

## G4 — мутация «окно включает 1 января года t» ловится
CHECK: python3 -m pytest tests/test_kalibr_v5.py -q -p no:cacheprovider -k pervoe_yanvarya
EXPECT: passed, 0 failed
EVIDENCE: `1 passed, 9 deselected`, rc=0 (детектор True на коде, False на мутации `granica + 1`; первый бар 2018 в синтетике — 01.01.2018)

## G5 — мутация «top-10 по году t вместо t−1» ловится
CHECK: python3 -m pytest tests/test_kalibr_v5.py -q -p no:cacheprovider -k god_t_vmesto
EXPECT: passed, 0 failed
EVIDENCE: `1 passed, 9 deselected`, rc=0 (мутация `granica = year_bounds(t)[1]`, маска [1])

## G6 — мутация № 27 «плато заменено пиком» ловится
CHECK: python3 -m pytest tests/test_kalibr_v5.py -q -p no:cacheprovider -k plato_zamenyeno_pikom
EXPECT: passed, 0 failed
EVIDENCE: `1 passed, 9 deselected`, rc=0 (мутация `sweep_plato.plateau_value = obj`)

## G7 — Шарп окна = sr_dnevnoy·√252 (Д7, одно определение)
CHECK: python3 -m pytest tests/test_kalibr_v5.py -q -p no:cacheprovider -k sharp_okna
EXPECT: passed, 0 failed
EVIDENCE: `2 passed, 8 deselected`, rc=0 (`test_sharp_okna_ravno_sr_dnevnoy` и совпавший по имени `test_izvestnyj_otvet_luchshij_sharp_okna_vybran`)

## G8 — соседние модули не сломаны (approach_a, sweep_plato, ledger_v2)
CHECK: python3 -m pytest tests/test_sweep_modul_mut.py tests/test_ledger_v2.py -q -p no:cacheprovider
EXPECT: 0 failed
EVIDENCE: `28 passed, 7 warnings`, rc=0

## G9 — компиляция и размер (≤ 200 строк, функции ≤ 40)
CHECK: python3 -m py_compile fastbt/kalibr_v5.py tests/test_kalibr_v5.py && wc -l fastbt/kalibr_v5.py tests/test_kalibr_v5.py
EXPECT: код 0; каждый файл ≤ 200 строк
EVIDENCE: rc=0; `183 fastbt/kalibr_v5.py`, `161 tests/test_kalibr_v5.py`

## ABANDON (только с причиной; гейт не удалять и не переписывать)

## Реестр
закрыто 9 из 9; ABANDON: 0; pending: 0
