# Ж-212 hybrid_struct — состояние: лист 2 ВЫПОЛНЕН (kodir). Не коммитить.

Гейты G0–G17 закрыты (`docs/otchety/hybrid_struct_v1/GATES_HS.md`); G11 (E) — «не запускалось по правилу», вопрос владельцу.
Использовано trials: B 4/12, C 4/12, D 10/12, E 0/8, F 2/6 = 20/50; N = 541. Уровни: у всех 6 финалистов «нет»; уровень 4 — не проверялся.
Итог: `docs/otchety/hybrid_struct_v1/HAR_BOOSTING_STRUCTURAL_REDESIGN_FINAL.md` (ответы 1–7), `FINAL_STRUCT_CONFIG.json`, `REPRODUCTION.md`.
Код: `models_v2/s8/` (hs_ledger, hs_feat, hs_ctx, hs_leak, hs_store, hs_eval, hs_rules, hs_bfit/bstage, hs_cfit/cstage, hs_dfit/dstage, hs_efit/estage, hs_ffit/fstage, hs_gstage, hs_levels, hs_final).
Тесты: `pytest tests/test_v2_hs_*.py -q` → 85 passed, 1 skipped.
Открытый вопрос: HS11 для C прочитан как «побить лучшую B И HAR-J» (стоп). При чтении «хотя бы HAR-J» продолжить Cb, Cd, Ce, Cs ×2 (8 trials) и затем E — требует решения владельца (правка правила в коде `hs_cstage.ctl_rule`).
