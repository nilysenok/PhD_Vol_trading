# state: harj_panel_hybrid (Ж-215 / Ж-217), листы P0, P1b

Статус 07.11.2026: P0 выполнен (G1–G6). P1b выполнен: G8 закрыт (EVIDENCE в `docs/otchety/harj_panel_hybrid/GATES_PANEL.md`), отчёт — `docs/otchety/harj_panel_hybrid/P1b_REPORT.md`. P1a (K2/K3, G7) — другой лист.
Код `models_v2/s10/` (hp_engine — движок, hp_ctrl — K4–K7), тесты `tests/test_v2_hp_*.py`, данные `data/regen/harj_panel_hybrid/` (panel_base 6e2988c6a8615ac6; engine_panel 565a479b25033d1e; k4567 2ceb56c1e38df0b2).
Запуск: `python3 -W ignore -m models_v2.s10.run_hp --stage inv|daily|har|shift|port|panel|k4567|k23 [--check]`. k4567 считается ~10 мин (XGB/LGBM, 6 потоков).
Бюджет: used 0 из 24 (10 XGB / 10 LGBM / 4 резерв), N_registry_start 585, leaf P1.
Для P2: движок принимает `base` (уровень Ĥ), не ln; панель — `engine_panel.parquet` (K1 с 07.2015); умолчания 2016 — глубина 2, c = 1, λ = 1; `finalize(..., lams=, default=)`.
Открыто: предложения У-8…У-10 из P1b_REPORT (коммитит дирижёр).
