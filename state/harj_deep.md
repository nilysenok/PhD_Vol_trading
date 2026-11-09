# Состояние: HAR-J Hybrid SBER Deep Structural Extension (Ж-239; тема Ж-238), только SBER, h = 1

- Открыта 09.11.2026 22:36 MSK. Гейты — `docs/otchety/harj_deep/GATES_DE.md`. Ветка `harj-deep` от `kanon-v22-pereschet` (f5b0aa3c).
- Бюджет расширения (≤ 8) не утверждён: обучений нет. Израсходовано: 0. Бюджеты V7 (завершена, PR #948) и V7+ (Ж-237, план позже) отдельны.
- Лист DE0 (реестр V1–V7, разложение разрыва к XGBoost, диагностика остаточной ошибки) — задание `docs/zadaniya/2026-11-09_harj_deep_0.md`. Затем — план расширения (дирижёр).
- Лист DE0 выполнен 09.11.2026 22:50 (Мак, kodir): `models_v2/s15/`, `tests/test_v2_de0_*.py` (29 passed), `docs/otchety/harj_deep/DE0_*.csv`, `DE0_REPORT.md`, `CHECKPOINT.json` ключ `0`; `run_de --stage 0 --check` — 0 расхождений; конфигураций израсходовано 0; выходы V6/V7 побайтно прежние. Ждёт перепроверки дирижёром (G1–G5).
- Лист E1 выполнен 09.11.2026 23:32 (Мак, kodir): реестр X1–X5 (`ALL_TRIALS.csv`, 23:25:50) до прогноза; X1 посчитана (`data/regen/harj_deep/E1/x1.parquet`, `E1_REPORT.md`, `E1_*.csv`, CHECKPOINT `E1.*`); X1 против NTP2 (6.2) — «нет», «go» — «нет»; израсходовано 1 из 5 (X2–X5 не тронуты); `run_de --stage 0|E1 --check` 0 расхождений, 46 passed; G8–G11 в `GATES_DE.md` — ждёт перепроверки дирижёром.
