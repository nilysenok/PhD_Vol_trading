# GATES — Ж-031 Э1-в п. 4, выходы в единицах реализованной √RV (S1–S6), T1 kodir, 27.10

Запуск: `FASTBT_MAIN=<своя копия> python3 -m pytest tests/test_rm_sqrt_rv.py tests/test_rm_sqrt_rv_golden.py -q`
(нужны `data/fastbt/daily/ohlcv_daily_1850.parquet`, `data/regen/features_pit_v3.parquet` в своей копии; RV читается только ≤ 2021-12-31).

## G1 — старые пути ATR/pivot/vwap не изменились ни на байт
CHECK: pytest tests/test_rm_sqrt_rv_golden.py -q
EXPECT: 7 passed (S1–S6 × {база, узел RM}, SBER/LKOH/MGNT, позиции до 2019-01-01 + PnL pnl(), sha256; golden снят коммитом 899c662 ДО правки)
EVIDENCE: 7 passed (после правки sweep_post / sweep_setka)

## G2 — уровень стопа/тейка/трейла = k·σ на синтетике с известной σ
CHECK: pytest tests/test_rm_sqrt_rv.py -k "stop or teyk or trejl or bezubytok or gamma" -q
EXPECT: passed (σ = 0,02: стоп 96 при k_sl=2; тейк 108 при ratio 2; трейл 114 при k_trail=3; безубыток после хода k_be·σ)
EVIDENCE: passed (36 passed в двух файлах)

## G3 — мутация «RV с текущим днём» ловится
CHECK: pytest tests/test_rm_sqrt_rv.py -k "tekushchim or tekushchiy" -q
EXPECT: passed (детектор ok_no_lookahead(mut) == False; на уровне сделки мутация меняет выход)
EVIDENCE: passed

## G4 — мутация «уровень по σ дня d+1» ловится
CHECK: pytest tests/test_rm_sqrt_rv.py -k "d_plus_1" -q
EXPECT: passed
EVIDENCE: passed

## G5 — ключи блока `rm_sqrt_rv` и переключатели ушли из «вне v1», split принимает все значения сетки
CHECK: pytest tests/test_rm_sqrt_rv_ocenshchik.py -q
EXPECT: passed для S1–S6 (defaults(sid) == ishodnoe сетки; vne_v1 без sqrt_rv/okno_rv/k_sl/rm_edinitsa)
EVIDENCE: passed

## G6 — дымовой прогон OAT S3 с блоком rm_sqrt_rv, 2018 (setup), без таблиц в коммите
CHECK: oat_sweep по rm_sqrt_rv.* на SkEvaluator("S3"), период setup
EXPECT: 5 параметров, все Шарпы конечны, ключи меняют результат
EVIDENCE: OK 1 с (кэш сигналов тёплый); Шарп 2018 (0,04 %, цифры — потолок, не результат): k_sl 6 знач., k_trail 6, k_be 5, okno_rv 4 — меняют; gamma_hold — не меняет Шарп S3 в 2018 (сделки короче порога; на LKOH меняет число сделок)

## Отступления (↺, класс B)
- σ̄ для gamma_hold — медиана σ по барам до 2018-01-01, не пересчитывается (без заглядывания); база временного стопа тренда при gamma_hold — 30 дней (сетка: «30 дней ∝ (σ̄/σ)^γ»).
- Нет σ на баре входа (мало истории RV) — входа нет; константа 0,02 ядра не подставляется (Д4).
- RV дня d = строка `rv` даты d из features_pit_v3 (как `sigma_e.rv_dev`); в σ бара t — строки строго раньше t.
- Для S5/S6 при sqrt_rv max_hold = 8, cooldown 0 (сетка не даёт им ручек в этом режиме); для S1/S2 — max_hold и cooldown_bars из безусловного блока `rm`.
