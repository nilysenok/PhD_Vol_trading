# GATES — тесты режимов Э1-в (PR #318), тестировщик T1, 27.10

Файлы: `tests/test_sweep_iter.py`, `tests/test_sweep_iter_n.py`, `tests/test_sweep_iter_util.py` (помощники). Синтетика `SynthEvaluator`, tmp_path, сеть и данные не читаются.
Схема: `ok_*` — проверка, `test_*` — проходит на коде, `test_mut_*` — та же проверка при подложенной ошибке (monkeypatch) обязана не пройти.
Общий CHECK: `python3 -m pytest tests/test_sweep_iter.py tests/test_sweep_iter_n.py -q`; EXPECT: `37 passed`; EVIDENCE: `37 passed in 3.79s`.

| Гейт | Результат | Тест / мутация |
|---|---|---|
| G1 | Уровни факторного плана: центр ± шаг; край сетки — три ближайших узла; база вне сетки — `ValueError` | `test_urovni_centr_plyus_minus_shag`, `test_baza_na_kraju_...`, `test_baza_vne_setki_otkaz`; мутации: `levels` = `v[:3]` (2), `levels` без зажима на краю (1) |
| G2 | Комиссия 0,035 не увеличивает N | `test_komissiya_0035_ne_uvelichivaet_N`; мутация: пересчёт пишет ключ в ledger |
| G3 | 2019 (control) не в цели: `objective` = Шарп 2018, `ctrl_*` отдельно | `test_2019_ne_v_celi`; мутация: цель по `train` |
| G4 | N накапливается OAT → factorial → optuna; повтор той же точки не задваивает | `test_N_nakaplivaetsya_...`, `test_povtor_toi_zhe_tochki_...`; мутации: ledger на каждый запуск, новый ключ на каждый повтор |
| G5 | `n_jobs=7`, `jobs=7`, `n_jobs=0` — отказ; 6 допустимо и доходит до Optuna | `test_limit_processov_6`, `test_n_jobs_6_...`; мутация: `MAX_JOBS`=8 |
| G6 | Optuna с одним seed — одинаковый результат | `test_optuna_odin_seed_...`; мутация: случайный seed |
| G7 | `top3` дополняется (по режимам), повтор режима не дублирует | `test_top3_dopolnyaetsya_...`; мутация: файл удаляется перед записью |
| G8 | Неактивный условный параметр не раздувает N (optuna, morris) | `test_neaktivnyi_...`; мутация: `make_fix` не сводит к базе |
| G9 | Validation без `open_validation` — отказ во всех трёх режимах (флаг сетки → `ValueError`; чтение 2020–2021 → `PermissionError`) | `test_validation_*`; мутации: `periods` игнорирует флаг, `_val_open=True` |

Мутаций поймано 13 из 13 (17 прогонов `test_mut_*`: validation — по 3 режима на каждую из двух мутаций).
Не покрыто: Optuna при `n_jobs>1` (по докстрингу недетерминирована), реальный `SkEvaluator`.

## Реестр
закрыто 9 из 9; ABANDON: 0; pending: 0
