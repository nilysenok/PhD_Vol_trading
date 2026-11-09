"""V7-DX (G41): полный `run_v7 --stage DX --check` (пересчёт повтора 1, 4, 6, энкодеров, регрессий; ≈ 2–3 мин): rc 0, хэш выходов до и после равен."""
from models_v2.s14 import run_v7, v7_0_io as IO


def test_dx_check_reads_only_and_passes():
    before = IO.outputs_digest()
    assert run_v7.main(["--stage", "DX", "--check"]) == 0
    assert IO.outputs_digest() == before
