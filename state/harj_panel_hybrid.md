# state: harj_panel_hybrid (Ж-215 / Ж-217), лист P0

Статус 07.11.2026: P0 выполнен, гейты G1–G6 закрыты (EVIDENCE в `docs/otchety/harj_panel_hybrid/GATES_PANEL.md`), отчёт — `docs/otchety/harj_panel_hybrid/P0_REPORT.md`.
Код `models_v2/s10/`, тесты `tests/test_v2_hp_*.py`, данные `data/regen/harj_panel_hybrid/` (panel_base.parquet SHA-256[:16] 6e2988c6a8615ac6, 59 146 строк).
Запуск: `python3 -W ignore -m models_v2.s10.run_hp --stage inv|daily|har|shift|port|panel [--check]` (порядок inv, daily, har, shift, port, panel).
Бюджет: used 0 из 24 (10 XGB / 10 LGBM / 4 резерв), N_registry_start 585.
Открыто до P1: предложения У-1…У-5 в P0_REPORT (коммитит дирижёр); обёртка K2/K3 на бумаги кроме SBER.
