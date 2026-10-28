# Архив эталонов s9 `final3` до правки дивидендов (Ж-069 п. 1)

- **Дата:** 28.10.2026.
- **Решение владельца:** Ж-069 п. 1 — снять замок на эталонах `final3` для пересчёта на коде #370; старые эталоны — в архив с хэшами; значения 2020–2021 не выводятся.
- **Причина пересчёта:** Ж-057 п. 1, Ж-059 п. 2, Ж-060 (PR #370):
  - дивиденды в PnL (D·0,87);
  - сигналы на скорректированном ряду;
  - стоп через отсечку сдвигается на D.
- **Код до правки:** коммит `117f4b4`.
- **Новые эталоны:** PR #375, коммит `d25b52d`. Команды:
  1. `python3 -m fastbt.s9 cache --workers 3 --sigma-h1 final3_full --suffix _final3pt`
  2. `run --boot-reps 1000`
- **Где файлы:** старые эталоны лежат в `arhiv/final3_do_dividendov_2026-10-28/` основной копии, вне git. Рядом — `SHA256SUMS.txt`.
- **Сверка хэшей:** 28.10 09:44 сверены с `git show 117f4b4:docs/tables/strat_dev/<файл>` — совпали 10 из 10.
- **Почему манифест здесь, а не в `docs/tables/`:** у дирижёра каталог `docs/tables/` закрыт правами.
- **Не пересчитывался:** `s9_sravnenie_sigma_final3.csv` — в #375 не менялся.
- **Известное следствие:** выбор C по С8 на новых эталонах не совпадает с замороженным `TEST_C`.
  - Три теста `tests/test_s9_scope.py` красные: `test_frozen_keys_equal_dev_choice`, `test_test_path_on_dev_years`, `test_s9_pseudotest_frozen_windows`. Защита `s9.py` отказывает с `SystemExit`.
  - По Ж-069 п. 2 выбран вариант (а): `TEST_C` заморожен; сверку правит отдельный PR.
  - На базе до #370 все 20 тестов файла зелёные (28.10 09:46).

## SHA256 старых эталонов

```
f4a71818ad99db821da19cbd0d414f7f103f5d584dd28ff571c857af78d224fa  s9_B_targetirovanie_final3.csv
cc131497ad8a875d47a50c604ada4b4dda003a37f50cf32718c71b3677bf3bcf  s9_C_rezhim_final3.csv
b84ac27e4c6d4584eeec022a7d6f79b1a64a3c61dc2a786365b3049d2b9a6bd5  s9_C3_tochnost_v_rezultat_final3.csv
263170f40b8f0966a984fba18a9308b711332a75c97837cbe96a01b47d826354  s9_D_zatvor_final3.csv
f42b808a0e96ad4e0d9785ed7bb6ce05b73a5de83d08f2dcaa4ca47900b74c53  s9_D_zatvor_harj_final3.csv
d2d7c8d23132f9373ac6069d6178e1c3d42a30cead9e7bf981fa2627288eef03  s9_P1_vesa_tikerov_final3.csv
ea79c145f99284476bf9d03ba93d5bd5403d8b07de3b3c585b5f0a8215285df5  s9_P2_multistrategiya_final3.csv
8680472d21138564f6d9642a8397e1108c985b641a537e57d2bbda9e5ae02d87  s9_P3_fko_final3.csv
06319c8509b7bdee99e5b42b9455062d60c5d6b294ecaad54ded3ab78cf87bcc  s9_T1_komissii_final3.csv
e6757fa615cb41a101dc0cb48954acec0aab02320a22d455a30eac7c4662a9df  s9_T2_podperiody_final3.csv
```
