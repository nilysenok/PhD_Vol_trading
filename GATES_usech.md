# GATES_usech — усечённые копии ≤ 2021 (Ж-131 п. 2)

G1 2022+ не попадает в копию (csv, parquet, граница 31.12 / 01.01)
CHECK: python3 -m pytest tests/test_usech_2021.py -q
EXPECT: 9 passed
EVIDENCE: 18 passed вместе с tests/test_manifest_perenos.py (9 + 9)

G2 фильтр на чтении (read_table/read запрещены моком; пакеты)
CHECK: pytest tests/test_usech_2021.py::test_parquet_filtr_na_chtenii_polnyj_fajl_ne_gruzitsya
EXPECT: passed
EVIDENCE: passed

G3 point-in-time: пара (date, known_from) — обе ≤ 2021-12-31
CHECK: pytest tests/test_usech_2021.py::test_pit_data_obyavleniya_i_sobytiya; мутация `ds[0] <= CUT` вместо `all(...)`
EXPECT: passed; на мутации failed
EVIDENCE: passed; на мутации failed (проверено)

G4 манифест по копиям: max дата ≤ 2021-12-31, istochnik + sha256_istochnika
CHECK: python3 scripts/server/usech_2021.py --koren <основная> --vyhod <worktree>/data/server_2021
EXPECT: последняя строка «файлов 193 … max дата 2021-12-30, копий принято 48 из 48»
EVIDENCE: файлов 193, объём 1135.8 МБ, max дата 2021-12-30, копий принято 48 из 48

G5 (а) IMOEX ≤ 2021, (б) σ кубов v5, (в) ставка/дивиденды, список 2022+ для tar
CHECK: ручной — docs/server/manifest_dannyh_2021.md; scripts/server/tar_iskluchit_2022plus.txt
EXPECT: записано
EVIDENCE: записано (eval_v5.py в базе нет — (б) частично)
