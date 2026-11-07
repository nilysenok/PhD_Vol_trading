# state: harj_panel_hybrid (Ж-215 / Ж-217), листы P0, P1b, P2

Статус 07.11.2026: P0 (G1–G6), P1b (G8) выполнены. P2 выполнен: G9 (XGB) и G10 (LGBM) закрыты (EVIDENCE в `docs/otchety/harj_panel_hybrid/GATES_PANEL.md`), отчёт — `docs/otchety/harj_panel_hybrid/P2_REPORT.md`. P1a (K2/K3, G7) — ABANDON (Ж-218). Следующий лист — P3 (выбор по R9, статистика, LOTO, диагностика); не начат.
Код: `models_v2/s10/` — hp_engine (движок), hp_ctrl (K4–K7), `hp_hyb_core|b|c|d|a4|out|run.py` (гибриды P2), стадия `run_hp --stage hyb --module xgb|lgbm [--check]`; тесты `tests/test_v2_hp_hyb_*.py` (26), всего `test_v2_hp_*` 88 passed.
Данные `data/regen/harj_panel_hybrid/` (не в git): engine_panel 565a479b25033d1e, k4567 2ceb56c1e38df0b2, panel_base 6e2988c6a8615ac6, hyb_xgb c864e31cfe764676, hyb_lgbm c5d7cbcb3a4c4948, hyb_a4_feats d2b9cef1aa89dc9b; кэш проходов `hyb_run_*.parquet` (ключ = хэш входов и кода движка).
Бюджет: used 20 из 24 (10 XGB + 10 LGBM; резерв 4 нетронут), N_registry 605, leaf P2. Реестр — `docs/otchety/harj_panel_hybrid/ALL_TRIALS.csv` (585 строк Ж-214 без изменений + 20).
Повтор счёта: `--stage hyb --module <m>` идемпотентен (проходы из кэша, строки реестра не дублируются).
