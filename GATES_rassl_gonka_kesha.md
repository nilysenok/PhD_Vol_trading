# GATES: rassl-gonka-kesha (Р-7, расследование гонки кэшей)

G1. Гипотезы (а)–(д) перечислены, каждая проверена и принята/отброшена явно.
CHECK: раздел «Гипотезы» в описании PR
EXPECT: 5 строк с вердиктом
EVIDENCE: закрыт, см. описание PR (rassl_gonka_kesha_pr.md)

G2. Причина воспроизведена тестом, падающим до правки.
CHECK: `python3 -m pytest -q tests/test_rassl_gonka_kesha.py`
EXPECT: до правки — 2 failed (FileNotFoundError в os.replace); после — 2 passed
EVIDENCE: до правки 2 failed; после 2 passed

G3. Правка: уникальные tmp + os.replace в bars5m.py и cache.save.
CHECK: `python3 -m pytest -q tests/test_rassl_gonka_kesha.py tests/test_bars5m.py tests/test_klyuchi_kesha.py`
EXPECT: 0 failed
EVIDENCE: 19 passed

G4. Реальная гонка 3 процессов (пустые кэши, 15 точек S3 покрытых ядром, train4): после правки расхождений нет.
CHECK: scratchpad par.sh b 3 m3 + sver.py b1 b2 b3
EXPECT: fast==slow 15/15 во всех; хэши между процессами равны
EVIDENCE: 15/15 ×3, slow и fast равны между процессами. До правки (a1,a2,a3 на пустых кэшах): 2 из 3 процессов упали FileNotFoundError на bars5m .tmp
