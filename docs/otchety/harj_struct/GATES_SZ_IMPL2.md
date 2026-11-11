# GATES — S-Z-IMPL-2 (проход, контроль, реестр), кодер, 12.11.2026 (гейты записаны до первой правки кода, 01:2x по `date`; EVIDENCE дописан 01:3x)

Выбор до счёта: гейт Z2.G7 (нулевая синтетика) — три шума (seed 0, 1, 2); критерий гейта — seed 0 (как в задании), seeds 1 и 2 печатаются как чувствительность и в отчёт идут без изменений.

## Z2.G7 — нулевая синтетика полной процедуры до регистрации
CHECK: `python3 -W ignore -m models_v2.s18.z_walk --synth-null`
EXPECT: S-RCKZ, реальные z̃ 5 seeds, цель ln RV = ln F⁰ + шум без связи с z: QLIKE(R) − QLIKE(θ=0) на датах после включения ≤ +0,10 %; печатаются доля месяцев θ ≠ 0 и λ* по месяцам
EVIDENCE: seed 0 (критерий): −0,0967 %, доля месяцев θ ≠ 0 0,907 (λ* 0,1: 55, 0,01: 4, 0,001: 9, 0: 7) — выполнен. Чувствительность: seed 1 +0,3180 % (доля 0,907) — ВЫШЕ порога; seed 2 −0,1053 % (доля 0,613). Среднее трёх +0,039 %. Критерий seed 0 объявлен до счёта; на seed 1 порог превышен втрое — в отчёт первой строкой риска. rc 0.

## Z2.G1 — реестр
CHECK: `python3 -W ignore -m models_v2.s18.z_reg --check`
EXPECT: 3 строки по id; одна registered_at раньше mtime zmaps.npz и Z/*.parquet; N = 855, N_wide = 3415; прежняя часть реестра побайтно
EVIDENCE: `check ok: строк 3; registered_at ['12.11.2026 01:28:31']; файлов в Z: ['rckz.parquet', 'rckzs.parquet', 'rz.parquet', 'zmaps.npz']; N 855 / N_wide 3415`, rc 0 (после выходов).

## Z2.G2 — вложение Н2
CHECK: `python3 -W ignore -m models_v2.s18.z_walk --check-nest`
EXPECT: use_z=False ⇒ rck.parquet побитно (C, F, X1_R, lam, w, G, c, m, θ_CK); S-RZ use_z=False ⇒ λ* = 0, F = W5-G побитно; Н3 печатается
EVIDENCE: Н2 rckz побитно True (75 мес.); rz: λ* = 0 True, F = W5-G и C = cfg8 побитно True; Н3 (z̃ ≡ 0): max отн. C, F, X1_R = 0,00e+00; `--check-nest: ok`, rc 0.

## Z2.G3 — PIT
CHECK: `python3 -W ignore -m models_v2.s18.z_walk --check-pit`
EXPECT: порча RV t⁺ > D, z дней > D, строк C сессий > D не меняет λ*, w, θ̂, прогноз месяца у rckz, rz, rckzs (3 месяца)
EVIDENCE: месяцы 2024-06-03, 2023-10-02, 2025-09-01 × три ряда: все True (λ* = 0,1 во всех); `--check-pit: ok`, rc 0.

## Z2.G4 — выходы
CHECK: `python3 -W ignore -m models_v2.s18.z_walk --check`
EXPECT: 2198 дат; включение 2020-08-03, 75 мес.; λ* ∈ Λ; θ ≡ 0 ⇔ λ* = 0; λ* из w; c, m rckz и rz = rck побитно; n_excl = 0, z_ok = 0 у 0 дат; sha256 выходов до и после равны
EVIDENCE: rckz, rz, rckzs: 2198 строк, 75 мес. с 2020-08-03, ok; c, m = rck побитно True; n_excl > 0: 0 дат; z_ok = 0: 0 дат; пересчёт месяца 2022-04-29 max отн. 0,0 у всех; sha до/после равны (4 файла); `--check: ok`, rc 0.

## Z2.G6 — регрессия
CHECK: `python3 -m pytest tests/test_s18_r_*.py tests/test_s18_cm_*.py tests/test_s18_l_*.py -q`; `shasum -a 256` входов
EXPECT: все прошли; sha = таблице
EVIDENCE: `276 passed`, rc 0; sha256 16 входов (код, rck, feat, w5g, E1/x1, new/x1) = таблице спецификации.

## Порядок
CHECK: `z_reg` регистрация → `z_feat` write → проходы rckz, rz, rckzs; mtime в этом порядке
EXPECT: registered_at < mtime zmaps.npz < mtime Z/*.parquet
EVIDENCE: registered_at 01:28:31 < zmaps.npz 01:28:48 < rckzs 01:29:50, rckz 01:29:53, rz 01:30:09.

## Реестр
закрыто 7 из 7; ABANDON: 0; pending: 0 (Z2.G7 закрыт по объявленному критерию seed 0; seed 1 выше порога — риск вынесен в отчёт)
