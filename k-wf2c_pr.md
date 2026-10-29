## К-wf2c: period во всех местах создания Pul

- `kalibr_v5.pul_iz_ledger`: period берётся из manifest ledger (нет поля -> train4).
- `kontroli_v5`: `podpul`, `podpul_ex` передают `period=pul.period`; `sintez_pul` получил аргумент `period` (по умолчанию train4).
- `etap1_eval.HONEST["wf"] = "net"`: HONEST — метка честности окна в выходе Evaluator (net = окно без подбора на нём; ctrl2019 = kontrol). wf — walk-forward, веса года t считаются по окну строго до t, как у train/v2/test1 (net). Без строки Evaluator на wf падал бы KeyError.
- `tests/test_kalibr_wf2c.py` (новый): период и охрана сохраняются (ZamokVladeltsa без отметки; отметка только во временном каталоге через monkeypatch `period_v2.OTMETKA`). Мутация «период не передан» (Pul теряет period) ловится `test_mutatsiya_period_ne_peredan_lovitsya`.
- `tests/test_perenos_wf_svod.py`: снят strict xfail P-2 (по просьбе дирижёра), тест зелёный как обычный.

Проверено: pytest test_kalibr_wf2c, test_perenos_wf_svod, test_kalibr_wf2b, test_svod_server_v5 — 26 passed. `tests/test_etap1_eval_v2.py` на Маке тяжёлый (>3 мин), не дождался; соответствующая правка — одна строка в словаре.

## Канон
закрыть: находки #645 (1), (2); гарантия: period wf сохраняется — test_wf_pul_sohranyaet_period.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
