# Волна D (Ж-204), kodir, 05.11.2026 — ВЫПОЛНЕНО, ждёт перепроверки дирижёра

Все стадии `run_d.py` (control, comp, d1, robust, d2, report) выполнены, EVIDENCE G0–G11 в `docs/otchety/sber_h1_deep_v1/GATES_D.md`. Не закоммичено. Итог: `SBER_H1_BOOSTING_DEEP_IMPROVEMENT_FINAL.md`, блок D в `README.md`.

Код: `models_v2/s5/d/` (+ новые модули ctx_d, stages_d, robust_d, d2_d, report_d, doc_d, doc_txt, run_d); тест `tests/test_v2_deep_d_tables.py`; текст вывода `docs/otchety/sber_h1_deep_v1/D_NARRATIVE.md` (читается report). Перепроверка: `python3 models_v2/s5/d/run_d.py --stage control`; `pytest tests/test_v2_deep_d_*.py`; кэш d — `data/regen/v2_sber_h1_deep_v1/d/` (повторный `--stage report` занимает ~15 с).
