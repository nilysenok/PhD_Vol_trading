# GATES_P6 (листы П3b и П6)

G1 П3b выход = parabolic_exit на синтетике (25 путей x 2 стороны x 4 уровня x 2 сигнала)
CHECK: python3 -m pytest tests/test_exec_e_parabolic.py -q
EXPECT: 18 passed
EVIDENCE: 18 passed in 0.81s

G2 П3b ось выключена - прежнее побайтно (быстрое ядро == автомат); соседние тесты зелёные
CHECK: python3 -m pytest tests/test_exec_e*.py tests/test_vyhod_parabolic.py tests/test_fast_etap1*.py -q
EXPECT: все passed
EVIDENCE: 656 passed in 91.81s

G3 П3b мутации № 3, № 4, заглядывание в бар, бар входа - ловятся (тесты test_mutatsiya_* в G1)
CHECK: см. G1
EXPECT: passed
EVIDENCE: входят в 18 passed

G4 П6 голосование v2 == execlayer.vote на v4-эквиваленте побайтно; мутация d+1 ловится
CHECK: python3 -m pytest tests/test_kalibr_v5_golos_v2.py tests/test_kalibr_v5.py -q
EXPECT: passed
EVIDENCE: 29 passed in 2.64s
