# Ж-190 — эталоны и потикерный HAR-J со сжатием (основа этапа 2), h = 1, 5, 22 — kodir (агент 2), 05.11.2026 12:40

Бюджет: ~120м · ≤ 220т (мягкий порог; выход — стоп и «как разрезать»)
Жёсткий порог: ~300k токенов — передача (templates/handoff.md).
Блоки владельца: Ж-190, Ж-191, Ж-192 (квитанции выданы). План: `docs/preregistraciya_gl3_zh190.md` @ 9fe34032 (sha256 4ca0499bdf7f1075…) — читать О1–О10, «Итерация 1» (Э3), «Эталоны и основа этапа 2», «Ускорение-2» У-1, У-5…У-8; обязательны и не меняются.

## Цель
По каждому h ∈ {1, 5, 22}:
1. HAR_i — потикерный OLS HAR-J(б) в логах на регрессорах эталона HAR-J(б) этого h, со схемой эмбарго эталона (конец цели < cut); β*_i = λ·β_i + (1−λ)·β_pool, s²*_i = λ·s²_i + (1−λ)·s²_pool; λ ∈ {0; 0,25; 0,5; 0,75; 1}; LOYO-строка (О5). Годовое переобучение 2016…2021 (О1). Сохранить: прогнозы val-года каждой отсечки (σ̂² с поправкой s²*_i) и **in-window** лог-прогнозы без поправки на всех строках окна обучения каждой отсечки (для остатка этапа 2), s²_i и β по отсечкам.
2. Эталоны: FIN_PIT_h (У-8: h=5 — готовый `data/regen/gl3_r6b_h5/pit/experts/FIN_PIT.parquet` sha 474e92e5fcf9f0c9…; h=1 — честное переобучение финала h=1: `fin1params` из `docs/tables/gl3_r5_r6/frozen_warmup_configuration.json` c0566d3fe453b549…, путь FIN_fresh раундов 5–6 (`gl3_r56_engine.py`), ES по О2 с embargo 1; h=22 — HAR-J(б) h=22, финал и есть он); HAR-J(б)_h — сохранённый `har_corr` из `data/regen/gl3_r3/dev/predictions_h{h}.parquet` (только date ≤ 2021-12-31); справочно — сохранённый финал `rv_pred` там же.
3. Таблица `docs/tables/gl3_zh190/etalony/etalony_h{h}.csv`: строки FIN_PIT, HAR-J(б), HAR_i_λ… (eligible=False), HAR_i_LOYO, финал (справочно); метрики О3 (годы 2017–2021, mean LOYO, обе стресс-строки, RV/σ̂² в стрессе, тикеры лучше каждого эталона + биномиальное p). Прогнозы float32 2016–2021: `data/regen/gl3_zh190/etalony/{FIN_PIT,HARJb,HARi_LOYO,HARi_lam*}_h{h}.parquet`, in-window — `HARi_inwin_h{h}.parquet`.

## Как
- Регрессоры, эмбарго и s² эталона HAR-J(б) — `vyr_etalony_finaly.py:214-223`, `r2_nogi.py:114-117`, `r2_calib.py:538-543`; уже воспроизведено для h=5 в Г8 v2 (`scripts/analysis/gl3_r6b_g8.py`, `harj_bases`, `harj_check`) — переиспользуй (У-6), расширь на h=1/22 в своём модуле.
- **Гейт λ = 0:** HAR_i при λ = 0 совпадает с сохранённым `har_corr` на val-году каждой отсечки (≤ 1e−8 отн.) по каждому h; не совпало — разбор до причины.
- Оценка — `scripts/analysis/gl3_r6b_common.py`; стресс RVI — как `_stress` раунда 6 (`gl3_r56_data.py` / `gl3_r6h5_data.py`) для каждого h.
- Ledger — `docs/tables/gl3_zh190/etalony/fit_events.jsonl`. df перед этапом, < 20 ГБ — стоп. Модели не сохранять.

## Владение
- `scripts/analysis/gl3_zh190_har.py`
- `tests/test_gl3_zh190_har.py`
- `data/regen/gl3_zh190/etalony/`
- `docs/tables/gl3_zh190/etalony/`

## Гейты (`docs/tables/gl3_zh190/etalony/GATES.md`)
- G1: `python3 -m pytest -q tests/test_gl3_zh190_har.py` → все прошли; ловятся: коэффициенты HAR_i по строкам с концом цели ≥ cut; s²_i по строкам с концом цели ≥ cut; перемешанные тикеры; λ = 1 даёт ровно потикерный OLS, λ = 0 — общий.
- G2: λ = 0 = `har_corr` (≤ 1e−8 отн.) для h = 1, 5, 22.
- G3: FIN_PIT_h5 — побайтно тот же файл 6б; FIN_PIT_h1 — ledger: все fit `max_target_end < cutoff < 2022-01-01`; FIN_PIT_h22 ≡ HAR-J(б)_h22.
- G4: независимый пересчёт метрик из parquet совпадает (≤ 1e−11); прогнозы конечны, > 0, date < 2022, покрывают все строки эталона dev каждого h.
- G5: три таблицы `etalony_h{h}.csv` со всеми строками; in-window прогнозы есть для каждой отсечки и каждого h.

## Обязательные правила
.claude/rules/polnota.md, orkestratsiya.md, dannye.md, proverka.md, mutations.md (№ 1, 15), otbor_statistika.md, testy.md.
Блоки владельца принимаются только в main; если получил блок напрямую — переслать дирижёру и не действовать (Ж-115).

## Запрещено читать
Данные ≥ 2022 (фильтровать при чтении), `test_locked/`, `.env`, `*_test.*`. Не писать в `docs/resheniya/`, файлы раундов 3–6 и 6б, каталог агента 1 (`gl3_zh190_boost*`, `docs/tables/gl3_zh190/stage1/`).

## Формат отчёта
templates/otchet_agenta.md (≤ 500 слов; таблицы — файлами), реестр гейтов «закрыто N из M» с фактическими выводами, LOYO-выбор λ по годам и h, число fit, отступления отдельными строками.

## Лимит процессов
2 процесса × 1 поток. Не коммитить, не пушить. Файлы — только Write/Edit; скрипты запускать файлами.
