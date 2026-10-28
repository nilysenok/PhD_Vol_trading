**Первый прогон теста — ДО ИСПРАВЛЕНИЯ ВЫРАВНИВАНИЯ. Числа не для текста, только журнал. Эталон HAR-J отставал на день.**

# Журнал первого прогона теста главы 3

Статус всех файлов ниже — **«до исправления выравнивания»** (решение владельца 26.10, 14:20; FINAL_DESIGN, разд. 14; дефект — `docs/otchety/2026-10-26_proverka_laga_harj.md`). Файлы не удаляются и не правятся по числам: это история. SHA-256 сняты 26.10 около 14:20 по Маку на ветке от `kanon-v22-pereschet` @ 5255b5c. У четырёх md-файлов с пометкой (раздел 5) SHA сняты уже после того, как пометка добавлена.

## 1. Запуск

- Команда: `python3 scripts/analysis/final_test.py --scope test --src data/regen/test_locked/final3`, один раз, 26.10 08:57.
- Коммит с выходом: 002a74c.
- Финалы: h=1 — D22_h1; h=5 — D12_h5; h=22 — HAR-J с поправкой.
- Эталоны: HAR-J (а) `har_raw` и HAR-J (б) `har_corr` — оба с RV до дня d−1.

## 2. Входные тестовые прогнозы (общий `data/`)

| файл | SHA-256 |
|---|---|
| `data/regen/test_locked/final3/final_h1_test.parquet` | 4e56f6252a48cb47f6027a0ff86c1f197102a5e02f61123265287b119cf4e7d0 |
| `data/regen/test_locked/final3/final_h5_test.parquet` | 4694991372bd95d3e2a7a091c2548444d77050ca4c75cc40170f5bd78b8652fc |
| `data/regen/test_locked/final3/final_h22_test.parquet` | 91f2a48e5f4ad431ac75c85bdd3dd7a58a449e4e0c9d731fed6dbd0a4937b059 |
| `data/regen/sigma_pred_h1_final3.parquet` (√rv_pred h=1, вход главы 4) | 5bed7be9eb7ebcfb0466271e3f06670c13bda380d06be1db53828c787b6b0dbb |
| `data/regen/sigma_pred_h1_final3_full.parquet` (та же копия) | 5bed7be9eb7ebcfb0466271e3f06670c13bda380d06be1db53828c787b6b0dbb |

## 3. Код

| файл | SHA-256 |
|---|---|
| `scripts/analysis/final_test.py` (= FINAL_DESIGN, поле 4) | ea67239fe302f56603df7cc98561e6472f7eb477e920799bc456cce9a1da57a2 |
| `scripts/analysis/ftp_stats.py` | ce705f96f533763314b656f3eb6babe97fa77fbdb67014f662f7638c5c05a310 |
| `scripts/analysis/ftp_tables.py` | 2feca8238c617a6664b4db34414a1a5d64e1395f3b06e40a983d270d7cdf21f8 |
| `scripts/analysis/tables_test_format.py` (формат) | c5dfc38da9f54d9ef97da3449c9bbb39637a203d68b93f8843eb76bd52043e1a |
| `scripts/analysis/test_dop.py` (срезы 09:05, 09:15, терцили v3) | 62a798eeb280a8eb5df98fa87eea9146ac7e8bda8829503b817a763dedaa6f7c |
| `scripts/analysis/test_dop2.py` (закрытие биржи, знаковый тест) | 17860d3919838972f2932a84ad57d0ced94846662850b4700b890b5043c3efe6 |
| `scripts/analysis/tables_dlya_teksta.py` (таблицы для текста) | 9c5d5b961c7bc556fc461c1f9f5ad6a1ab1e804fdd5ef666b8a2750452920201 |

## 4. Выход запуска и срезы из его прогнозов (`docs/tables/`)

### 4.1. Журнал и таблицы запуска — исходники (`_raw`, байты запуска) и формат владельца

| файл | SHA-256 |
|---|---|
| `docs/otchety/final_test_run.log` | b542ea3c08b9819d10f663b42db6a9565f38a1968d6f8c69cf678c8947b5df10 |
| `table_3_2_test_raw.csv` | e929114a4259fee742344283875b662403854205815239bccff5f2e30a6ae001 |
| `table_3_2_test.csv` | 14fecb0d2c12e67aa7fd95e2507009510d0f5cf1792e640cec40cbbfbf64d71b |
| `table_3_3_test_raw.csv` | ceba48ae424a5fb6d731634fbe1a068367c4c6089c6d14526702c849e32a4838 |
| `table_3_3_test.csv` | 706bf322403589fba7738329fb10e46e03f06abd3e7652c8a0621a0b945ea261 |
| `table_3_4_test_raw.csv` | a0ada5c017d8414170a441262e8b6dc618887e4d76db4a86a1a2037f80d72188 |
| `table_3_4_test.csv` | bbc2ee5e5f4f7485ec0241ce01dab0bb5a7c71bcd09b4e7ed076b65557443302 |
| `table_3_4_rmse_test_raw.csv` | fb9b0ab45723c0c3d2b7113808bfce337cdf689af25d342690b35ad4aa7cdb0d |
| `table_3_4_rmse_test.csv` | 6ce8d6e6a4a56136e2369fa79435084f10edf16ad41e10ea1a1764737c792ae1 |
| `table_3_4_mae_test_raw.csv` | 28b76da7886d4bd2cdd4eb1d40c8e5106a5eed608c3e31ba074475d6f0144859 |
| `table_3_4_mae_test.csv` | 301acb8f36ff2bfd644d5323c096e9d34bf91522797b1650020cf0ee2530f5e6 |
| `table_3_6_test_raw.csv` | 1926d36ee6edf78e00b63fc9b3b4359a2edf3f988df6f5ef92d4ede28a4e9d2f |
| `table_3_6_test.csv` | 4a1565a8e2354c5c7f917a74bd0551c4b85c2753c8c0d12fae4aa61138d28e52 |
| `table_3_6_rmse_test_raw.csv` | 2c9cabc1c423da19af1bcfb2c8fb532514d1832c236404c33d2f63c7e8dcac44 |
| `table_3_6_rmse_test.csv` | 9c7203d7ed80a7b93d1fdd07184fc386dcbfeb4f8eacfcd1836d9d81611d4fb1 |
| `table_3_6_mae_test_raw.csv` | 068a766ce5a48fe9316c94085d43e65c769d4d1e19b87b287af04b825bea4413 |
| `table_3_6_mae_test.csv` | 9fc1b1ceb38bfb12c47320b7e570908a0affb602961b7d8d7bd81d0ce1515d80 |
| `table_3_7_test_raw.csv` | cbd5fb033efb2fd3db686b18ed8a7af4103ca636f7d21ad6ca9675b43009f347 |
| `table_3_7_test.csv` | eba994d5557e26ccb03418948a0213dc7b5f49cad5b2bb521611d6e18a62f488 |
| `table_3_7_rmse_test_raw.csv` | ae1366b14438216aa07c8f23486382a0b5d5bdb77a8228973ed892855dd59c77 |
| `table_3_7_rmse_test.csv` | 9a42c29ed64963d851aa16b2b8c2a8fae1677cb69b5f3be6a7634e9f88e76394 |
| `table_3_7_mae_test_raw.csv` | f2b36bd2758fdfccdb13e046da8e6e2fcd39b870f8eac1307d34555ce7d7643c |
| `table_3_7_mae_test.csv` | f58f6bb98aacfe218fb890ab623e01c0082298306dfb22ac7819cd7cc55e7771 |
| `table_3_8_test_raw.csv` | f576693d4451ae5a5934662682838553b2527e28628230355d97d567ade26532 |
| `table_3_8_test.csv` | ab8bcfc6f868591e679afa9f86538f5e4fb9cb9012c1c79e5e7e82007bb1f27d |
| `table_3_9_test_raw.csv` | 4a5fafe65841523306124750af16d832e4c7301f16f1c1372354c71d36d1529d |
| `table_3_9_test.csv` | ceb96f458d078a4ff13bde266d7e3fe5c13e01ba991c43b5084146ccbaab4a3c |
| `table_3_final_po_godam_test_raw.csv` | f34b0b36a2fd31fb6ee0fffb5d51412aaa8a1a1026fc0a9004c101303693e40d |
| `table_3_final_po_godam_test.csv` | 92ec875dc381eca6d2a26c67b5b3ec4c1c8c202567d2262e6a8b0f55bf7f191d |
| `table_3_raskladka_test_raw.csv` | 483a3eb3f940a761e0872a5a76df593960b78050845616cd0f7d4c6fcb3e8fbc |
| `table_3_raskladka_test.csv` | 015caac52677829016ec2713dee7512f77f3110c03d86304ff82c94f6652db9d |
| `table_3_mz_test_raw.csv` | 28a52d575aa37b11cb7fe6aacb7de7437588789634c6cb8f545a3ea654861e5a |
| `table_3_mz_test.csv` | 8a5d86de3ab774f8eac4f6239db4fec0afc3778b233264b7dbdd7d87a393011c |
| `table_3_tickers_test_raw.csv` | dbe3f5fafc8de76336482526ef4fe6e7368666cf775faab87e885626cc31b827 |
| `table_3_tickers_test.csv` | 9b4cc033296c6d456406e93e59cf3d0d7ba44e913a6831ec4551b36365c25a84 |
| `table_3_subperiods_test_raw.csv` | 4943a3916c821035881692dac55931e3d4b9b5f88e4975c9fcd4b739f3b8ed2a |
| `table_3_subperiods_test.csv` | 9693829e098c772d5ef85af614c59423c24d0f03b919b0542f1bfb6713c0ab9e |
| `table_3_terciles_test_raw.csv` (терцили по v2) | 2812b4a3d6c8e4a8c4b6688608baff7dc6f88e60a42b5fd4c6f8f186c34895a5 |
| `table_3_terciles_test.csv` | 6c1bbf9b51d0b1e367f52d51ce3dddd70ed642d6a9915f16661db7e4edb96b1a |
| `table_3_encompassing_test_raw.csv` | 2e9de1b434f14132fa59dee6fa7301fc1175057a9e5f0dfb3363d1bdac777b6b |
| `table_3_encompassing_test.csv` | 68712656469c322d4dc42be66fbc58639d773e69c0ce86b970b3a3be461c7421 |
| `table_3_mcs_test_raw.csv` = `table_3_mcs_test.csv` | 61a365258b9fc8d843c3e2682734d33a5815c66dc845c21c547aab503ba75803 |

### 4.2. Срезы из сохранённых прогнозов (поручения 09:05, 09:15, PR #114)

| файл | SHA-256 |
|---|---|
| `table_3_dop_dm_h1h5_test.csv` | fc37a79e988fca43bc63d91328b2af0a497b6362b405d4bf376f2270add399b7 |
| `table_3_dop_subperiods_test.csv` | f4f82e22ee21a76e47cffaa1960aabbf056cb9b63cf619e78408c12d67b9eb84 |
| `table_3_dop_terciles_v3_test.csv` | fb35c80c17c3fbf3566175e909ee933127912a593ba768f7990320ec3afc1598 |
| `table_3_kontrol_svodka_test.csv` | 7eee0ab81f12480c68b6657e3de812e2e0e2fe93029dcbee799baf4816fbfaf8 |
| `table_3_kontrol_dm_test.csv` | 7c7c7a322b392d671b6b8048e244b68c40084e78b892e0164ab348265a6ece9b |
| `table_3_kontrol_mcs_test.csv` | fb95463a421cb455ca9c3056540720f46cfc1b04f2b9864edaca5cbf4eb12d26 |
| `table_3_kontrol_tickers_test.csv` | a4878166c9881ef1e045472a9036b5e98f09ed439d7947e4de18a5816fb8dbd3 |
| `table_3_kontrol_subperiods_test.csv` | bef8896779676f26dd24c261253389418c7c9e8cea0f38f9bc8707e1513f75bf |
| `table_3_neotkr_dm_test.csv` | 5300f7a4d0a9eb3ff3aa1ba4cfbb171700abbcd2be33f4e5ded36fed2dbb23d3 |
| `table_3_neotkr_mcs_test.csv` | 1307b3001d351b40a4de9e5d3a2789768f3bf21eee549657676549ccd430f8f6 |
| `table_3_neotkr_tickers_test.csv` | cec0748f0fdd6ec6caf165d7dc795226fb2ae19699260246d1e512331c4c7736 |
| `table_3_neotkr_subperiods_test.csv` | f9bd414b8d9c3e431cc68841dfd543036fe602802bf424672302c594716318b4 |

### 4.3. Правило «закрытие биржи» и знаковый тест (поручение 09:55, PR #120)

| файл | SHA-256 |
|---|---|
| `table_3_zakrytie_isklyucheno.csv` | da5b1382c7e6e5bb3bb62ce1c0272081811bbf9d841c4837331719c6eb1077c6 |
| `table_3_vse_svodka_test_bez_zakrytiya.csv` | dbb7f564a3d14a5d8703c92274991540c096fed57bc8e39d155039b4c1b42318 |
| `table_3_vse_svodka_test_s_zakrytiem.csv` | 9aec5c3d12af5fc41731e90ad9b7d5a3e1dae81e904e2f45e6b6b95c02c0c1ed |
| `table_3_vse_dm_test_bez_zakrytiya.csv` | 6bfa95b318b8501983842c78b70b83956027cecc5d4a011a8917eb232f19437f |
| `table_3_vse_dm_test_s_zakrytiem.csv` | ee68c92d75ada91f835b478054938faa88797771ffb63fe4d5f37ce653e6b059 |
| `table_3_vse_mcs_test_bez_zakrytiya.csv` | 6a6a62cbee09226aba013e4a01ff2c9f87caa86aa28e932af2fdec7456232fff |
| `table_3_vse_mcs_test_s_zakrytiem.csv` | d0d6bff6f440a0800472712fa998dfae8c9845606bc9e7ba0bac9ef841b47837 |
| `table_3_vse_tickers_test_bez_zakrytiya.csv` | ef4d69a4c4d7849de3d00eb128988a7c401b69a8a5713227bd57ff6731d1d493 |
| `table_3_vse_tickers_test_s_zakrytiem.csv` | a69cc966f3ecffcd9f6b526e843037b4c9d81fdd417fd78b22e7d191e774cf72 |
| `table_3_vse_subperiods_test_bez_zakrytiya.csv` | aa1cb4930ad5990f628d637495a9aef62e79c84526ae3bcd586fbcfaceb6e7b4 |
| `table_3_vse_subperiods_test_s_zakrytiem.csv` | 56e9149a684af7cc206545c61bcfd6a799c52c257276162e579620c716b112b0 |
| `table_3_vse_znak_test_bez_zakrytiya.csv` | 7c8f194e8cab40b188bad93ec484848988322b54cacb18f1962c2379076e8e91 |
| `table_3_vse_znak_test_s_zakrytiem.csv` | 9e007d1754d898af32931fa977bb432749f01a1c3ee0ab740b2c73b8723db4ed |
| `table_3_kontrol_svodka_test_bez_zakrytiya.csv` | 5f1d0c8054d71e943e4f2dcbc56e8da81ca083c9221029da4580201649acfece |
| `table_3_kontrol_svodka_test_s_zakrytiem.csv` | 0a32ad7a5d6092d559d9efbcfe2afaf0058b2339a9a6ab568cb51b84e3cbfa37 |
| `table_3_kontrol_dm_test_bez_zakrytiya.csv` | c4ddf74f12fea370c884c5897ebb9c63b754370b486538767b87c7907ed20734 |
| `table_3_kontrol_dm_test_s_zakrytiem.csv` | d32e395ba0b01b26d7665e951cf9a50c761890aa150f668599f280a5797b06e7 |
| `table_3_kontrol_mcs_test_bez_zakrytiya.csv` | fa176b4acfdff9f4c90c26048c43f30f54de7ca8bccf2582818d59ec08b0ec84 |
| `table_3_kontrol_mcs_test_s_zakrytiem.csv` | 9b41d4250f40f8b81296ddfdd3dc4f95dd829f0b860bbbba0fd00c8370665425 |
| `table_3_kontrol_tickers_test_bez_zakrytiya.csv` | 9a3845dbf6ca88185b58ba9656b3d15b7dcda698a5d275bf8aaa070d4d8984b0 |
| `table_3_kontrol_tickers_test_s_zakrytiem.csv` | a07ceed68e28ac3c6a382f3e41242dd85875b25de27b8aabf1f499e4cc1809a7 |
| `table_3_kontrol_subperiods_test_bez_zakrytiya.csv` | 6584a8d0da15ae259af662d820baa1e423e440d4b73da1d0b472b92cc9a831df |
| `table_3_kontrol_subperiods_test_s_zakrytiem.csv` | 1fbce0fdb32afd0d9859d36bd23e2171aac702be02134d0c1c2eda0d072dd732 |
| `table_3_kontrol_znak_test_bez_zakrytiya.csv` | dac7200bc17df4c4028188d2e307892632d7a529d4918bf972831f0347de9fb9 |
| `table_3_kontrol_znak_test_s_zakrytiem.csv` | 444828a72314564a49d4a86d45f0e5d188c8ead75e214282fd8ddff6b5534ed9 |
| `table_3_neotkr_svodka_test_bez_zakrytiya.csv` | fde57d979da31775d68a407a5dfe898123f7555da290437209036a01df45935a |
| `table_3_neotkr_svodka_test_s_zakrytiem.csv` | 5e219d0f28e51cb619e6d48a1fa4852caeadce38abe96e5769f094f483159ee3 |
| `table_3_neotkr_dm_test_bez_zakrytiya.csv` | 2bd96a3767ceb02b711410139cc82c0af1f921e0361ff13fe4cb37aeb6f455fe |
| `table_3_neotkr_dm_test_s_zakrytiem.csv` | 9e31cf6f970b5ddba892c558d568cc4886b1dbd620976f04e6b38ce913ec1016 |
| `table_3_neotkr_mcs_test_bez_zakrytiya.csv` | df52f08b1e92ddf3306dcc8cb647582dc5dd64078ff573840e0e4f31ccedf39a |
| `table_3_neotkr_mcs_test_s_zakrytiem.csv` | 2163c35e1593b478db6b8506b5279f23a18e7af60f279098a3822c428444d3aa |
| `table_3_neotkr_tickers_test_bez_zakrytiya.csv` | 77db06fafa7f324cb05276611cdbb65f454fe6f3eda6b121485dcce871179b59 |
| `table_3_neotkr_tickers_test_s_zakrytiem.csv` | 0c91b0e48223c8f92a2f75c233ee449b2304a1b10ce99c76a99567e7b7c965a4 |
| `table_3_neotkr_subperiods_test_bez_zakrytiya.csv` | 375171b0f657df39bf7c3bf4597a711435e0742a238ed24b7fedf6b0b804889c |
| `table_3_neotkr_subperiods_test_s_zakrytiem.csv` | ba7931d57b53f973eef56334f5ab9c35e09b7b9ea58349e7a1a2f7174bd2f546 |
| `table_3_neotkr_znak_test_bez_zakrytiya.csv` | 99f206eced974404687058ef14d0a1dabca2c3c96844d881703724f7c34df29a |
| `table_3_neotkr_znak_test_s_zakrytiem.csv` | 42999e545fe333d7f0235cf286de3d75e384c53f11bd0d4ecd333890abfd337e |

### 4.4. Таблицы для текста (`docs/tables/dlya_teksta/`), посчитанные из прогнозов первого прогона

| файл | SHA-256 |
|---|---|
| `table_3_4.csv` | efeae6b5ad804ace2ec49e6aa8dcf764b4f123d1ffb761a5c9ffec44b608ce0f |
| `table_3_4_dm.csv` | 179c31227e87dd2797382611d153b64a2981c160f5241331b7da8ec1106feb8a |
| `table_3_4_v_1035.csv` | bf730d8daffea723ddb4b10717b2934e0d78cc37c6b022628c94be9309d92216 |
| `table_3_4_v_prezhn.csv` | d216ad4c255530b721ed7638e96d6b2d42a3e3773f328adc638c471486e23ea3 |
| `table_3_5.csv` | 797bb4400396d7420148208a26fb037c774f69062c2d06694fb132c16cc9d742 |
| `table_3_6.csv` | f1a1cdd1c2da27c340545d347842114988e92fd951abc8bd7694e5648228bd2f |
| `table_3_7.csv` | d559c9795882c59d299ec440f5cb1e811c1da3e16164298ee5943bbb12bc148c |
| `table_3_8.csv` | 6c21899f96fcb84d0085d9d7d4d1be7d813dedc90d5794d9b809df70a377d5f3 |
| `table_3_9.csv` | 475ec63d37246fd97cea723603b4005dc30fbf7e24213f6938588231be2cb6db |
| `table_3_pril_zakrytie.csv` | f7b0aca850d594103b87b571f217cefd6a857a076dcdc0bb8ed31b818e18c0cd |
| `istochnik_3_4_test.csv` | e922cd0dec83206357167ff8c6f41f6b2b1bf449d877bf308413e642fce885e0 |
| `istochnik_lestnica_test.csv` | b9ba5daacc5a519f01f6ae14df4b22387ba661ae5e0f0a73005f66d74bd8a50b |
| `istochnik_sravnenie_test.csv` | 3e61c2cc1855cae0b35042ed409faf1e492846a0b210668fd5f7cb1cd56b216a |
| `istochnik_srez_test.csv` | 6c3c6a75c0c66523af99de017d5a6e52f97e2c307ea485f769f662f199087b5d |
| `istochnik_srez_zakrytie_test.csv` | 7f27c3599effbb7d1f25ac1c3e905aefaa44f16268cc778e8988ab4f6809b334 |
| `prilozhenie_ablyaciya_buster.csv` | 2ac96b14e63e9617bd62520da5e0e1d39c1e887143a678a9e7109f0c38ab63bb |
| `prilozhenie_buster_s_popravkoy.csv` | 114d4b34ffa8103300f02d291cc008337b4ba1ac6a88d00f77106cac591361eb |
| `prilozhenie_lestnica.csv` | 9f66cecb59bb9be5a560423b32e3714c3d104e9e9320e9da92c94ec9304f9c9f |
| `prilozhenie_nogi_gibrida.csv` | 25c9f083a59ce1d44a59b5b40909bd5638469d6e50aa0bb694ff2a5a6424a133 |
| `v1/table_3_4.csv` | 32b1e0d96139390c3b050e4318a7d74bfc6410418e334dc71cf3945fcdde91e9 |
| `v1/table_3_5.csv` | d842e591f9c4fa295c3edc82a44559158dc9db4f080d168f3154b80012874c11 |
| `v1/table_3_6.csv` | 83edddbe5971d7dfe36ae03fe3bd1dbfd6fc8089fe77f2a1a78ae9031a764ba1 |
| `v1/table_3_7.csv` | 64e2021cdb6fabb74d2fdc1685edc5239f047d95523c7311ee763d0274d0e391 |
| `v1/table_3_8.csv` | 506f5a63ba925aeda606256023ff499b170a35e5f555e8b386d2d90a27f8b599 |
| `v1/table_3_9.csv` | d726860b99ce2a34e140369202abf50b96f8e7fc8a444fcbc41cf04c4930f89c |

Не из первого прогона, а из разработочного набора: `prilozhenie_otvergnutye.csv` и `prilozhenie_spravochnye.csv`. В `spravochnye` эталон HAR-J тоже в версии до исправления.

## 5. Отчёты и черновики с числами первого прогона (с пометкой)

| файл | SHA-256 (после пометки) |
|---|---|
| `docs/otchety/2026-10-26_test_glava3.md` | 265cee0aba03f5597f83f34e813e2673cfb11d677fe5e7032a09823a3e2cc7e9 |
| `docs/otchety/2026-10-26_abzacy_3_2.md` | e01a2ebf66b0e227b122e0540faaf20d6706ec05f89b2605859889624cb72dcd |
| `docs/tables/dlya_teksta/README.md` | 0cbfda77f43bf733965e519847a1d1b4a622693c5268ee91a261f4c451b0be48 |
| `docs/tables/README_zakrytie.md` | 54253da2f30e18eb150c6eed73bedd10f6fdbe77fc0c5ae932d22c6fca25b5c5 |

Черновики `docs/tekst_chernoviki/2_1_5_okno.md`, `2_2_1_popravka.md` и `2_2_5_struktura_gibrida.md` на числа первого прогона не опираются: в них только разработочный набор. Но HAR-J в них в версии до исправления, это помечено в каждом файле.

## 6. Связанное, вне главы 3 (для агента главы 4)

- Глава 4 берёт σ финала из `sigma_pred_h1_final3*.parquet` (раздел 2) и σ HAR-J `sigma_pred_h1_harj_final2` (√har_corr, отстаёт на день). H2 по решению 14:20 — «до исправления выравнивания».
- Черновик `docs/tekst_chernoviki/4_test.md` здесь не помечался: он за агентом главы 4.
- Ранний расчёт 2019, `table_3_*_test2019_v3*.csv` (коммит 815e24f, 25.10), — отдельный прогон, но с тем же HAR-J (`rv_d` = RV_{d−1}).
