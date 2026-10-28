# Пересчёт строк vhod = rynok новым кодом (rynok по open d+1)

Разработка train4 (2016–2019), после аудита, **потолок**: выбор и счёт на одних данных; в отбор не идёт. Ж-086 п. 6, #474.

- число строк vhod = rynok, t_vyhod ≠ close, status ok, проходы 1–2: **7998** (по Sk: {'S1': 1330, 'S2': 1337, 'S3': 1335, 'S4': 1330, 'S5': 1331, 'S6': 1335}); пересчитано ok: 7998, не ok: 0 ({})
- целевая Ж-090: mSR train4 за весь период, плата за шорт 5 %, устойчивость = mSR каждого года > 0

| | старые (rynok по close d) | новые (rynok по open d+1) |
|---|---|---|
| mSR, квантили 5/25/50/75/95 % | -1.61 / -0.82 / -0.54 / -0.23 / 0.40 | -1.75 / -0.88 / -0.60 / -0.27 / 0.33 |
| устойчивых | 8 | 7 |
| устойчивых и mSR ≥ 1,5 | 0 | 1 |
| медиана Δ mSR (новая − старая) | | -0.060 |
| mSR не определён (nan) | 5015 | 4863 |
| строк, где mSR изменился | | 3213 из 7998 |

Ledger не в git: `/private/tmp/claude-501/-Users-nilysenok-Desktop--------PhD-Vol-trading/b58320a0-0dac-4690-a8f8-b32419175dd9/scratchpad/_ledger_v2_rynok` (счёт шёл в `docs/tables/progon_e1/_ledger_v2_rynok/` копии, перенесён из-за чистоты дерева; нужен дирижёру — перенести в основную копию), 693 МБ по `du` (> 200 МБ — только sha) (старый: `/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/docs/tables/progon_e1/_ledger_v2`). sha256:

- `S1/SHA256SUMS.txt` 9b556c1d8f2784d3c4e60123b5a22962c7bab3a30ec88316ba1724c83267fda9
- `S1/SHA256SUMS_zaem.txt` bdd266ee55ec244b0a6a5b792dcfbe42533e1b4be143a7825dbffddefc22e23d
- `S1/manifest.json` bf0eaafdbf4533ae8dcc9d30123d22e32d04a22ab41ee762651a4a0284c56185
- `S2/SHA256SUMS.txt` 284d81a823581413568b9fdb99f863847dfa035e1407b92c0920eb058b400491
- `S2/SHA256SUMS_zaem.txt` fb2c3017da9839a7951517c29df161e1ee0b70f5b5337f525ff714c5372a3ec6
- `S2/manifest.json` 5fc7b2fdae7b2615e83918f8a0d4ffd0598dfc5198c89414daa115190bb5ab71
- `S3/SHA256SUMS.txt` b1c4d97f4c0e1d152da1da49e618eee02a385b0a764877808545b1ad33353af0
- `S3/SHA256SUMS_zaem.txt` 56cec298661a9a09b57327382bb4f4b5939f6320e2a761a5381f4f61cefe98c8
- `S3/manifest.json` f1205aa6fbbe1c8300661a66e79f68617a1e6d805264e4557c947138f30a453b
- `S4/SHA256SUMS.txt` 37e3205efb8753bd871172ba417061444ee975183c2d8ffb1da2f2591c951efa
- `S4/SHA256SUMS_zaem.txt` 70424b898668869d48bf47280a7e44f4fd7cf459eb68e100be77273e28fd0d3a
- `S4/manifest.json` f89941aeb7fe806fddb528a25fe710c243ffed1c02f8008a674953cd3c6f2f1d
- `S5/SHA256SUMS.txt` c5ad7fea3384cbabdc4a79fd90879ae459dce050446cc94e2492f62de33a23dd
- `S5/SHA256SUMS_zaem.txt` d63a01e5864faa65602380bffc64d68b29168662677a9170c9adbd606cc0a233
- `S5/manifest.json` ecdb1aafca4af9eec9a97f859bef7612178aa23c39195b18ee8eeca768174813
- `S6/SHA256SUMS.txt` 11e78f48720f51f6cc83646460466de1970295cf299778ca91bb24627d298e1f
- `S6/SHA256SUMS_zaem.txt` e237286668ecb0e4515ed3adfa4e1ce4d8576b923c3f4877c2c478d53e0c3eab
- `S6/manifest.json` bfc8c1771c309fff3358c793aac5da0da7e461548f61c0e71b33f8f4833f2907

sha256 от отсортированного списка sha256 всех `p*/rows.jsonl` (6 стратегий, 7998 строк): `2abcc431bd5c824faced1124eb707c563fe79cea5eb5da7c4ae89211274869f5`. Шарды port/tik/zaem — в `SHA256SUMS*.txt` выше.

## Проверка

- Код счёта: `kod = eb67fc08a81a0c1746c17e7b538d6d315c4d490c` (ветка `kanon-v22-pereschet` на c2333b42, `RYNOK_ISPOLNENIE = "open_d1"`). После счёта ветка слита с более новым `kanon-v22-pereschet` (позже #487, #489, #490 и др.) — ledger им не пересчитан; kod старой пробы `rynok_open` в manifest тот же.
- Проба: 20 случайных точек (seed 20261028) → `_ledger_v2_rynok/_probe`, 20 из 20 ok, исключений 0. Полный счёт: 4 процесса, 14 мин 50 с, ошибок оценщика 0, медленный путь (`sloy = slow`).
- Сверка (шаг 3): 20 случайных строк `t_vyhod = close`, `vhod` — лимит (не rynok); новый код = старая строка по всем полям, кроме `kod`, `kod_slit`, `klyuch_sha`, `t_sek`, `excess`, `sloy`, `istochnik`, `vhody`; порт побайтно. Расхождений 0. `vhody` пропущено намеренно: `cache.key` включает хэш кода (`fastbt/cache.py:38`), у старого и нового кода он разный. Первый запуск без этого пропуска дал 20 из 20 расхождений (причина — `vhody`), порт совпадал побайтно: проверка реагирует.
- Целевая: `fastbt.tselevaya.tselevaya_train4` (порт и `zaem_baza` из ledger, плата 5 %). Квантили — по строкам с определённым mSR; у 5015 старых и 4863 новых строк mSR не определён (в таблице «nan»), устойчивыми они не считаются.
- Это потолок: те же точки Соболя, train4, выбор и счёт на одних данных. Устойчивых 8 и 7 из 7998 — не статистика.
- Мутация (`proverka.md` п. 1): код пайплайна не менялся, мутаций не добавлено; сверка выше — проверка, что изменился только rynok.
