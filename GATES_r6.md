# GATES Р-6 (красные тесты базы; ветка r6-krasnye)
(Составлен по ходу работы, до коммита; расследование шло бисектом, правки — после названия причины.)

G1 полный прогон базы e6e9221f — список красных. CHECK: `python3 -m pytest -q -p no:randomly tests -rf`. EXPECT: список. EVIDENCE: 45 failed, 5938 passed, 37 skipped, 4 xfailed (2439 с) — закрыт.
G2 golden RM — причина названа, эталон в архив, новый. CHECK: `python3 -m pytest -q tests/test_rm_sqrt_rv_golden.py`. EVIDENCE: 7 passed (причина #370, бисект d7bb98ea зелёный / 9bec9184 красный) — закрыт.
G3 окружение и порядок: progon_urovni, rassl_s56 (утечка s9.V2 из test_ekviv_kand), guard vpred (lru_cache). EVIDENCE: 103 passed, 1 xfailed на затронутых файлах — закрыт.
G4 test_07 split_by_hand (#370), test_reestr_v1 (append-only реестр), test_a_weights_future/test_22 (#297, #370). EVIDENCE: там же — закрыт; S3-2021 — xfail(strict), причина не названа.
G5 НЕ закрыто, передача: test_s9_scope::dev_reproduces_final3_tables (#678; эталоны заморожены), ::test_test_path_on_dev_years (класс A), test_yakor_ansambl_s9_train4 (6, устаревший кэш в data вне git, #678), test_v6_nez_eval (14; тест #672 vs #679), test_fast_v5_* (другой лист).
