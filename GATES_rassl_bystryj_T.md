# GATES — расследование: быстрый слой v6 ≠ эталонный путь (поле T), kodir, 31.10.2026

## G0 — причина названа и воспроизведена на точке v6bystr__S1__15
CHECK: python3 scratchpad/kodir_T/tocka.py S1 15
EXPECT: fast: status oshibka, TypeError: run() got an unexpected keyword argument 'cena_T'
EVIDENCE: получено (до правки): `fast None None oshibka TypeError: run() got an unexpected keyword argument 'cena_T'`; T эталона 1010

## G1 — тест tests/test_rassl_bystryj_T.py падает на базе 0abd93eb
CHECK: python3 -m pytest tests/test_rassl_bystryj_T.py -q
EXPECT: failed (TypeError / сигнатура)
EVIDENCE: pending

## G2 — после правки тест зелёный; мутация «правка снята» роняет
CHECK: pytest tests/test_rassl_bystryj_T.py -q; затем с возвращённой строкой
EVIDENCE: pending

## G3 — точка S1/15 сходится + ещё 5; узкий набор зелёный
CHECK: tocka.py; pytest tests/test_fast*.py tests/test_exec_e*.py tests/test_eval_v6*.py tests/test_eval_v7*.py tests/test_etalony*.py
EVIDENCE: pending

## Реестр
закрыто 1 из 4; pending: 3
