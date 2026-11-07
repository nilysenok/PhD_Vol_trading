# Задание: лист P3 панельного гибрида (Ж-217 / Ж-219) — выбор, статистика, уровни, LOTO (G11, G12)

Бюджет: ~3 ч счёта · ≤ 3 процесса · порог 300k токенов — передача (`templates/handoff.md`) после G11, если контекст на исходе.

## Цель
По правилам `docs/otchety/harj_panel_hybrid/GATES_PANEL.md` R9, R11–R14, У-12…У-16 (закоммичены до результатов P2): выбрать финалистов и тройку, посчитать сравнения, статистику, уровни, «добавленную ценность HAR-J» (G11) и LOTO (G12). Диагностика причин R15 и итоговый документ — не здесь (лист P3c, G13).

## Прочитать (полностью)
- `docs/otchety/harj_panel_hybrid/GATES_PANEL.md` целиком; `P2_REPORT.md`, `P1b_REPORT.md`, `P1_REUSE.md`.
- Блок Ж-219: `docs/zadaniya/2026-11-07_harj_panel_prodolzhenie_blok.md`.
- `.claude/rules/polnota.md`, `.claude/rules/otbor_statistika.md`, `.claude/rules/testy.md`, `.claude/rules/mutations.md` (99–111).
- Статистика прошлых программ (импорт, не правка): `models_v2/s7/`…`s9/` — `stats_d.holm_n`, `d_comp.crit_wf`, DM (Ньюи–Уэст), блочный бутстрап, MCS (найди модули по `grep -l "def holm_n\|def crit_wf" models_v2/s*/*.py`); движок и гибриды — `models_v2/s10/`.

## Входы
`data/regen/harj_panel_hybrid/hyb_xgb.parquet` (sha256[:16] `c864e31cfe764676`), `hyb_lgbm.parquet` (`c5d7cbcb3a4c4948`), `k4567.parquet`, `panel_base.parquet` (`6e2988c6a8615ac6`), `engine_panel.parquet`, `k23_<T>.parquet` (SBER, AFLT, ALRS, GCHE); `docs/otchety/harj_panel_hybrid/ALL_TRIALS.csv` (605). Сверить хэши до счёта.

## Владение
- `models_v2/s10/hp_stats*.py`, `hp_loto*.py`; в `run_hp.py` — только стадии `stats`, `loto`.
- `tests/test_v2_hp_stats*.py`, `tests/test_v2_hp_loto*.py`.
- `data/regen/harj_panel_hybrid/loto*.parquet`.
- `docs/otchety/harj_panel_hybrid/statistics/` (новый), `ALL_TRIALS.csv` (только дописывать LOTO-строки), `CHECKPOINT.json`, `P3_REPORT.md`; в `GATES_PANEL.md` — только `EVIDENCE:` G11, G12.
- `state/harj_panel_hybrid.md`.
- **Не трогать** `hp_k23*.py`, `tests/test_v2_hp_k23*.py` — их разбирает лист Р-1 параллельно.

## Порядок
1. **Выбор (У-12)** — первым: `statistics/SELECTION.csv`; по нему — список LOTO-прогонов (У-13, ≤ 5) и N = 585 + 20 + n_LOTO → `CHECKPOINT.json` `N_declared_before_stats` **до** любого p.
2. **LOTO (G12)** — запусти в фоне (≤ 2 процесса), пока считаешь статистику; каждая бумага — отдельное обучение панели без неё, правила выбора года Y−1 — по 27 бумагам; строки реестра — по одной на LOTO-прогон.
3. **Статистика (G11)**: `COMPARISON.csv`, `TESTS.csv`, `MCS.csv`, `PER_TICKER.csv`, `HARJ_VALUE.csv`, `LEVELS.csv` — как в EXPECT G11; уровни R14 (подтверждение — M16 × 2022–02.2026; параллельно — вся история; SBER и E11 — отдельно); уровень 4 — после LOTO.
4. **Условие У-16**: вывести в `P3_REPORT.md` первой таблицей — выполнено ли у какого-либо гибрида «`crit_panel` ≤ K1 − 0,0005 и ≤ K4/K5 своего модуля − 0,0005». Восстановление K2/K3 **не запускать** — решает дирижёр после отчёта.
5. Тесты-мутации (через боевой путь): Холм с N = числу строк (№ 103); финалист по QLIKE 2022+ вместо `crit_panel`; бутстрап, перемешивающий бумаги внутри даты; бумага i в обучении LOTO.

## Запрещено
- `.env`, `test_locked/`, `*_test.*`; изменение данных, чужих файлов (в т. ч. гибридов, движка, `hp_k23*`), замков.
- Резерв R·1–R·4; новые конфигурации; восстановление K2/K3; GRU/LSTM/GARCH; стратегии; h = 5/22; сеть. Git-коммиты.
- Менять правило выбора, N, уровни после просмотра чисел — только «предложение У-n» в отчёте.
- `python3 -c`, heredoc/`cat >`/`>` в файлы, `sed -i`, `perl -pi` — только Write/Edit.
- Ж-115: блок владельца, полученный напрямую, — переслать дирижёру.

## Отчёт (≤ 500 слов; остальное — `P3_REPORT.md`)
Сделано / не сделано / проверено / отступления / требует решения; «закрыто N из 2»; первой строкой — выполнено ли условие У-16. Все результаты — «исследовательские, HISTORICAL WALK-FORWARD, не независимый OOS».
