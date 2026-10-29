# GATES_manifest

G1 · скрипт отвергает файл с датой 2022+ по содержимому, не по имени
CHECK: python3 -m pytest tests/test_manifest_perenos.py -q
EXPECT: 9 passed
EVIDENCE: 9 passed in 0.26s

G2 · test_locked и test1_2022_2024 не читаются (мок open/ParquetFile/read_csv падает при попытке) — тот же прогон, test_test_locked_ne_chitaetsya_vovse
CHECK: python3 -m pytest tests/test_manifest_perenos.py -q -k test_locked
EXPECT: 1 passed
EVIDENCE: входит в 9 passed (G1)

G3 · манифест создан, max дата <= 2021-12-31
CHECK: python3 scripts/server/manifest_perenos.py --koren <основная копия> --spisok scripts/server/perenos_spisok.tsv
EXPECT: файлов 145, max дата 2021-12-30, отвергнуто 91
EVIDENCE: файлов 145, объём 1039.7 МБ, max дата 2021-12-30, отвергнуто 91, исключено 0
