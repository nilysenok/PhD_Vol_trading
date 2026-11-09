# Архив версии 1 (результаты 1–7 и E0 до исправления согласования z)

**Причина.** дефект согласования z (Ж-237, «Моя ошибка [V7]» 18:13): в версии 1 строка даты года y брала z от энкодера года y, так что при переоценке в году Y корректор учился на парах прошлых лет в координатах других энкодеров (знак, порядок и масштаб компонент не согласованы; `docs/otchety/harj_hybrid_v7/FX_DIAG.csv`), а у пар 2017 года z не было совсем (F = H до конца 2018).

Что здесь: прогнозы 1–7 (перенесены), таблицы `1b_*.csv`, `2a_QLIKE.csv`, отчёт `LEAF1B_REPORT.md` (копии). Ключи CHECKPOINT: `1b_v1`, `2_v1` (те же sha256).

| файл | sha256 |
|---|---|
| `1b/cfg1.parquet` | 3dd6f38ef04c167c781151a19235b6c88eb0c45453e38a04c3884498bb1b3dda |
| `1b/cfg2.parquet` | 5df146b4f03a9807468820bfff2bbb3928257f2118a3328745124b4e89dafa61 |
| `1b/cfg3.parquet` | d048e44870fcdc555204b46468d182cbf037f9e2bf2180575826beab8bdd6fd4 |
| `2/cfg4.parquet` | 8ae89529a9faf6e968850aa165a4ec599046d8f306a66d4a13f821fd0e01e9d0 |
| `2/cfg5.parquet` | 4ccf44668e13edb3415ff55c6a0d142e19b72b597874fdbe9d7b3acc7804b3b9 |
| `2/cfg6.parquet` | 8569ca60e52bd5dbec778e978b61936557bbc3d1495134d22f67ddcc448a86af |
| `2/cfg7.parquet` | 2969bec593b7fd407433008a7a9d3ffa88239a3bca850826ee78c18c3bd8a94e |
| `otchety/1b_B_QLIKE.csv` | a9af986751e8c2263eab05a1d21a43e4d3ab3902649e73a956554cdcbb114e89 |
| `otchety/1b_E0.csv` | fb0207d416d94caf1d8bc612086537bc73b5329b44f7bc89749bc76baa4b9256 |
| `otchety/1b_GO.csv` | 36fe2882c80dac0d62bf1b89de1629e7d9f1988f5060ac94b79741d2a83ce2c4 |
| `otchety/1b_MCS.csv` | fb704a8a7d415006a57790c21028943552964b8a385ebfeccff591fdd3f89b0e |
| `otchety/1b_METRICS.csv` | 9d71cbb58a0db2da845f3b60438a7de57f3980878ad3bb6e999e21580d56b796 |
| `otchety/1b_NEWINFO.csv` | 65917a19bac9c4cf5b6af8c8d3bd586daa008d04dd70a0102c8a8c0ff78904a8 |
| `otchety/1b_STRESS.csv` | 399582fe94620e646041a79e4eb8ce5e5c11363458155bca462e4cbde4d1a50b |
| `otchety/1b_TESTS.csv` | 02d68cb78a49bac6cc9c20f1546e734bee053ef8ca8b40c1a36a0b347076365e |
| `otchety/1b_YEARS.csv` | ea2d359b4355818a8cf4772bfa50e6e13b0475dfe4e8f4dfc38214d80a144a54 |
| `otchety/2a_QLIKE.csv` | b06655bf475659038484d83762c0098a6666b4a666ed79470286037923c9c180 |
| `otchety/LEAF1B_REPORT.md` | 257b5b63cb4d9be2e9a05cada7e71c8964347b9dd0c21deb7cbe37463cb0a375 |
