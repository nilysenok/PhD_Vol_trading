# GATES — спецификация S-RCD до счёта (Ж-259), arhitektor, 12.11.2026

Файл записан до первого проверочного прогона (02:25 по `date`); EVIDENCE — из прогонов после записи (02:26–02:34).
`<sp>` = `/private/tmp/claude-501/-Users-nilysenok-Desktop--------PhD-Vol-trading/5cfa4f4e-a264-412c-a614-0a8af6830acb/scratchpad/s_rcd`. Скрипты только читают сохранённые ряды и код; запуск из корня проекта: `python3 -W ignore <sp>/<скрипт>.py`.

## G1 — S_RCD_SPEC.md существует, разделы 1–12
CHECK: grep -c "^## " docs/otchety/harj_struct/S_RCD_SPEC.md
EXPECT: 12
EVIDENCE: 12 (02:33)

## G2 — обязательные элементы поручения в спецификации: три ряда, пометка «после просмотра», N 858 / 3421, вложение, предсказание
CHECK: grep -c -E "S-RCD|S-CDF|S-CKF|после просмотра результатов S-R, S-D, S-L, S-Z|855 \+ 3 = 858|3415 \+ 2·3 = 3421|\(Н1\)|## 9\. Предсказание" docs/otchety/harj_struct/S_RCD_SPEC.md
EXPECT: ≥ 8
EVIDENCE: 104; по отдельности (grep -c -F): «после просмотра результатов S-R, S-D, S-L, S-Z» 2, «855 + 3 = 858» 1, «3415 + 2·3 = 3421» 1, «(Н1)» 1, «## 9. Предсказание» 1, «S-CKF» 28 (02:33)

## G3 — статус в реестре: S-RCD, S-CDF, S-CKF нет; S-CD есть (контроль); N программы до регистрации — 855 / 3415, максимум по S.*.registered
CHECK: python3 -W ignore <sp>/reg.py
EXPECT: S-RCD 0, S-CDF 0, S-CKF 0, S-CD 1 (статус «контроль»); max n = 855 (S.Z.registered), n_wide 3415
EVIDENCE: «S-RCD: строк 0»; «S-CDF: строк 0»; «S-CKF: строк 0»; «S-CD: строк 1; статус «контроль»; registered_at 11.11.2026 17:59:40»; «max n = 855 (S.Z.registered), n_wide 3415»; ключ S.status есть (пишет параллельный кодер); rc 0 (02:26)

## G4 — формула S-CD восстановлена по коду: сохранённый C ряда `CM/cd.parquet` воспроизводится `cm_model.forward(θ месяца, ·, "CD")`; уровень потери обучения — C
CHECK: python3 -W ignore <sp>/cd_formula.py
EXPECT: max отн. |C − cd.C| ≤ 1e-15 на 2198 датах; до 2018-07-02 C = cfg8 побитно; λ ∈ {1e-4, 1e-3, 1e-2}; проекционный градиент потери уровня C в сохранённом θ ≪ градиента потери уровня F (3 месяца)
EVIDENCE: «max отн. 0.00e+00 на 2198 датах; побитно True»; «до включения 2018-07-02: дат 124; C == cfg8 побитно: True»; «активных месяцев 100; λ: 1e-4 — 18, 1e-2 — 82»; «λ ⊂ {1e-4, 1e-3, 1e-2}: True»; ∇ уровня C / F: 2019-05-02 5,71e-07 / 9,36e-03; 2022-10-03 1,94e-06 / 1,69e-02; 2026-11-02 1,01e-06 / 1,68e-02; rc 0 (02:27)

## G5 — вложение: путь C-D при θ = 0 = путь C-K побитно; F(0) = W5-G побитно; сборка R при θ ≡ 0 даёт F = W5-G и X1_R = X1 побитно
CHECK: python3 -W ignore <sp>/nest.py
EXPECT: ln C (CD, θ = 0) == ln C (CK, θ = 0) побитно; exp(ln F0) против w5g.F — max отн. 0,0; assemble: F == w5g.F, X1_R == x1 побитно
EVIDENCE: «ln C (CD, θ = 0) == ln C (CK, θ = 0) побитно: True на 2198 датах»; «exp(ln F0) пути C-D против w5g.F: max отн. 3.33e-16; Ctx.lnf0 == ln F0 (C-D) побитно: True»; «assemble при θ ≡ 0: F == w5g.F побитно: True; X1_R == x1.F побитно: True; C == cfg8 побитно: True»; ∂Σ ln C/∂θ0 = 2198; rc 0 (02:28). **С EXPECT не совпало в одном пункте:** exp(ln F0) — 3,33e-16, а не 0,0. EXPECT ошибочен: 0,0 в `S_R_SPEC.md` получен по форме степеней, а не через exp(ln F). Гейт не закрыт; закрывает G11.

## G6 — R-CD флагом не реализуется: C-K зашит в `r_loss.lnc_of`, `r_loss._starts`, `r_walk.names`; границы CD и CK различаются
CHECK: grep -n -E 'CM.forward\(v, V, "CK"\)|CM._x0\("CK", s\)|CM.NAMES\["CK"\]|^UNIT' models_v2/s18/r_loss.py models_v2/s18/r_walk.py models_v2/s18/cm_model.py
EXPECT: r_loss.py:67, r_loss.py:123, r_walk.py:43, cm_model.py:26 (UNIT CD = (1, 2), CK = (0, 1))
EVIDENCE: «models_v2/s18/r_walk.py:43: … list(CM.NAMES["CK"])»; «models_v2/s18/cm_model.py:26:UNIT = {"CD": (1, 2), "CK": (0, 1), …}»; «models_v2/s18/r_loss.py:67: return torch.log(V.c8 * CM.forward(v, V, "CK")["M"])»; «models_v2/s18/r_loss.py:123: return [CM._x0("CK", s) for s in seeds], CM.bounds("CK")»; rc 0 (02:28)

## G7 — числа сохранённых рядов в спецификации воспроизводятся из parquet
CHECK: python3 -W ignore <sp>/nums.py
EXPECT: QLIKE W5-G 0,215476645644; S-CD 0,218098439243; S-CK 0,217893959139; S-RCK 0,215026781817; S-RCKf 0,215350280363; S-RCKb 0,215204937389 (отклонение от таблиц ≤ 1e-12); n = 2198
EVIDENCE: W5-G 0.215476645644 (|Δ| 4,4e-13); S-CD 0.218098439243 (2,8e-13); S-CK 0.217893959139 (3,2e-13); S-RCK 0.215026781817 (5,3e-14); S-RCKf 0.215350280363 (1,8e-13); S-RCKb 0.215204937389 (4,0e-13); S-RCKZ 0.214876816097 (3,0e-13); n 2198, t⁺ 2018-01-04…2026-11-10; S-CK/S-CD −0,0938 %, S-RCK/S-CK −1,3159 %, S-RCK/S-RCKf −0,1502 %; включение S-CD 2018-07-02, S-RCK 2020-08-03; S-RCK λ*: 0 — 16, 1e-3 — 3, 1e-2 — 4, 1e-1 — 52 из 75; rc 0 (02:27). Дополнительно: `win.py` — общее активное окно n 1549, h1 1010, h2 1188, 2018 — 253; `pool.py` — пул S-C = пул R в 106 из 106 месяцев, |P| ≥ 630 — 75, ≥ 252 — 94, активных S-C — 100 (02:29–02:30)

## G8 — ни одного прогноза S-RCD, S-CDF, S-CKF не посчитано
CHECK: ls data/regen/harj_struct
EXPECT: A C CM D L R Z (каталога RD нет)
EVIDENCE: A C CM D L R Z (02:33)

## G9 — записаны только файлы владения
CHECK: find docs/otchety/harj_struct docs/otchety/harj_x1 models_v2 data/regen/harj_struct tests -type f -newer <sp>/reg.py
EXPECT: только docs/otchety/harj_struct/S_RCD_SPEC.md и docs/otchety/harj_struct/GATES_SRCD.md (и, возможно, файлы параллельного кодера `S.status` — CHECKPOINT.json, STATUS-файлы; их архитектор не пишет)
EVIDENCE: «docs/otchety/harj_struct/S_RCD_SPEC.md», «docs/otchety/harj_struct/GATES_SRCD.md»; других файлов нет; rc 0 (02:34)

## G10 — инструменты записи архитектора применялись только к файлам владения (ручной гейт: авторство записи командой не проверяется)
CHECK: РУЧНОЙ — по журналу вызовов Write/Edit этой сессии
EXPECT: Write/Edit — только S_RCD_SPEC.md, GATES_SRCD.md и `<sp>/*.py`
EVIDENCE: Write — `GATES_SRCD.md` (2 раза), `<sp>/reg.py`, `cd_formula.py`, `nest.py`, `nums.py`, `win.py`, `pool.py`, `S_RCD_SPEC.md`; Edit — `S_RCD_SPEC.md` (6 раз), `GATES_SRCD.md` (3 раза, EVIDENCE G9, G10, G11 и реестр). Других записей нет; реестр и CHECKPOINT не открывались на запись (02:35)

## G11 — вложение с исправленным EXPECT (вместо пункта G5 про exp(ln F0)): путь, которым пользуется R, совпадает с W5-G побитно; путь exp(ln F) — в пределах 1 ulp
CHECK: python3 -W ignore <sp>/nest.py
EXPECT: «Ctx.lnf0 == ln F0 (C-D) побитно: True»; «assemble при θ ≡ 0: F == w5g.F побитно: True»; exp(ln F0) против w5g.F ≤ 4,5e-16 (2 ulp)
EVIDENCE: «Ctx.lnf0 == ln F0 (C-D) побитно: True»; «assemble при θ ≡ 0: F == w5g.F побитно: True; X1_R == x1.F побитно: True; C == cfg8 побитно: True»; «exp(ln F0) … max отн. 3.33e-16»; rc 0 (02:34)

## Реестр
закрыто 10 из 11 (G1–G4, G6–G11); ABANDON: 0; pending: 0; не совпало с EXPECT: 1 — G5 (exp(ln F0) 3,33e-16 вместо 0,0: ошибка в EXPECT, закрыто G11). G10 — ручной.
