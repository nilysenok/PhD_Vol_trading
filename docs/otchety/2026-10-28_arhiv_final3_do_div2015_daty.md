# Архив эталонов s9 `final3` до дивидендов 2015–2016 и дат отсечки T+2

Статус: разработка 2018–2021, эталоны до пересчёта (Ж-072 п. 3). Коммит базы: `fbb4c564d8e4a8148788ef9479dc7f14e6618737` (`kanon-v22-pereschet`, слияние #400).

Причина архивации: на базе после #394 (дивиденды 2015–2016 в s9 через прогрев, бисекция #402) и #400 (даты отсечки 2017–2021 по T+2 баров) эталоны не совпадают с `tests/test_s9_scope.py::test_dev_reproduces_final3_tables`. Владелец, Ж-072: «Пересчёт final3 один раз, после слияния #400. Старые эталоны — в архив с хэшами.»

Файлы лежат вне git: `arhiv/final3_do_div2015_daty_2026-10-28/` (копии `docs/tables/strat_dev/s9_*_final3.csv` с базы + `SHA256SUMS.txt`). Хэши сверены с blob'ами базы: рабочее дерево на `fbb4c56` чисто (`git status`), файлы совпадают с зафиксированными.

## sha256 (11 файлов)
```
796fb74773693a1cc07dc179a1061a86a18ffc24727a51a6ae500b46d4381ecd  s9_B_targetirovanie_final3.csv
34577ad117663881327f7ef147d5a74a482cb301737143e4f9155f76468a2d47  s9_C_rezhim_final3.csv
4ac0458e317d39df27dca5853328a0f07f6f1878d740aabd38c7084fb70c3d14  s9_C3_tochnost_v_rezultat_final3.csv
a960fc385465ff58715de4ec743c3f19d9af3b243d70a1eb119ccb93751f494d  s9_D_zatvor_final3.csv
6b69cb77cedc8861a8e0fa57e6cd7d804235e2db3379ad38e4f1fca2fd37d422  s9_D_zatvor_harj_final3.csv
ade6891970154839c8a1d8ce30058981d7f50dba03da2c2a0a9a01375f1fc775  s9_P1_vesa_tikerov_final3.csv
8c36dff38265d545dc5adfa233004d130564f15792ef40eda28ca567169099f2  s9_P2_multistrategiya_final3.csv
6885daf7cd1e80a3af73c3600e6a4cd8a42495fbabec304c8eec7f3523a86b13  s9_P3_fko_final3.csv
5ae75958c24e82f77beab27a1d995cb7d538596b0d83efdbd98a881823bf5941  s9_sravnenie_sigma_final3.csv
bf5e596512cd4298d30b8db4d5978808883d511bf3a30c3ae5d7b4972a2b36cb  s9_T1_komissii_final3.csv
cad5d20980235ab54a8c4a67a645cb859eb6f169ede56cdae70568b898bdbf67  s9_T2_podperiody_final3.csv
```
Строки: B 39, C 61, C3 13, D 114, D_harj 58, P1 13, P2 9, P3 16, sravnenie_sigma 97, T1 61, T2 50.
