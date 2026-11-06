"""Ж-214: реестр trials (541 + новые), лимиты 8/12/16/8/16/10/10 = 80, «отличие_от», N поправок. Мутации: N Холма = числу тестов,
лимит превышен, qlike_harj без отличия, реестр не побайтно."""
import numpy as np
import pytest

from models_v2.s5.d import stats_d
from models_v2.s9 import hv2_ledger as L


@pytest.fixture()
def paths(tmp_path):
    return tmp_path / "ALL_TRIALS.csv", tmp_path / "CHECKPOINT.json"


def ref_key(dst):
    return L.load(dst)["key"].iloc[0]


def test_first_541_rows_byte_equal_and_n_final(paths):
    dst, chk = paths
    L.init(dst=dst)
    assert dst.read_bytes() == L.SRC.read_bytes() and len(dst.read_bytes().splitlines()) == 542
    st = L.write_check(L.load(dst), path=chk)
    assert st["used_total"] == 0 and st["N_final"] == 541


def test_mutation_prior_registry_not_bytewise_is_caught(paths, tmp_path):
    dst, _ = paths
    bad_src = tmp_path / "src.csv"
    lines = L.SRC.read_bytes().splitlines(keepends=True)
    bad_src.write_bytes(b"".join(lines[:-1]))                  # потеряна строка: 540
    with pytest.raises(ValueError):
        L.init(src=bad_src, dst=dst)


def test_qlike_harj_needs_existing_difference_key(paths):
    dst, chk = paths
    with pytest.raises(ValueError):
        L.register("qlike_harj", {"m": "x"}, dst, chk)
    with pytest.raises(ValueError):
        L.register("qlike_harj", {"m": "x", L.DIFF: "Ж-999:нет"}, dst, chk)
    k = L.register("qlike_harj", {"m": "x", L.DIFF: ref_key(dst)}, dst, chk)
    assert k.startswith("Ж-214:qlike_harj:")
    with pytest.raises(ValueError):                              # structural_ml — тоже требует отличие от существующего ключа
        L.register("structural_ml", {"m": "n1"}, dst, chk)
    assert L.register("structural_ml", {"m": "n1", L.DIFF: ref_key(dst)}, dst, chk).startswith("Ж-214:structural_ml:")


def test_repeat_config_is_not_a_new_trial_and_limit_stops(paths):
    dst, chk = paths
    ref = ref_key(dst)
    k1 = L.register("qlike_harj", {"i": 0, L.DIFF: ref}, dst, chk)
    assert L.register("qlike_harj", {"i": 0, L.DIFF: ref}, dst, chk) == k1
    for i in range(1, 8):
        L.register("qlike_harj", {"i": i, L.DIFF: ref}, dst, chk)
    assert L.used(L.load(dst))["qlike_harj"] == 8
    with pytest.raises(ValueError):
        L.register("qlike_harj", {"i": 8, L.DIFF: ref}, dst, chk)
    with pytest.raises(ValueError):
        L.register("neizvestnoe", {}, dst, chk)


def test_limits_sum_to_80_and_n_for_holm_is_541_plus_used(paths):
    dst, chk = paths
    assert sum(L.LIMITS.values()) == L.TOTAL == 80
    for i in range(3):
        L.register("market", {"i": i}, dst, chk)
    df = L.load(dst)
    assert L.n_final(df) == 544
    p = np.array([0.001, 0.01, 0.2])
    right = stats_d.holm_n(p, L.n_final(df))
    wrong = stats_d.holm_n(p, len(p))                          # мутация «N = числу тестов»
    assert (right > wrong).all() and right[0] == pytest.approx(0.001 * 544)
