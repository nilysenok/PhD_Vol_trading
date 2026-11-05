"""B1-1 (Ж-204): 28 trials объявлены, id уникальны, лимиты первого прохода (≤ половины бюджета направления)."""
from models_v2.s5.b1 import configs


def test_counts_and_limits():
    c = configs.counts()
    assert c == {configs.OBJ: 8, configs.HAR: 11, configs.REG: 9}
    assert all(c[k] <= configs.LIMITS[k] for k in c)
    assert configs.LIMITS == {configs.OBJ: 17, configs.HAR: 25, configs.REG: 25}   # половина 35 / 50 / 50 (округл. вверх)


def test_ids_unique_and_stable():
    ids = [configs.trial_id(r) for r in configs.trials()]
    assert len(ids) == len(set(ids)) == 28
    assert ids == [configs.trial_id(r) for r in configs.trials()]
    assert len({r["row_id"] for r in configs.ROWS}) == len(configs.ROWS)


def test_id_changes_with_configuration():
    r = dict(configs.BY_ID["obj_l4"])
    r2 = dict(r, c=0.5)
    assert configs.trial_id(r) != configs.trial_id(r2)


def test_every_stream_row_has_block_stream_or_is_rule():
    for r in configs.ROWS:
        if r["kind"] == "stream":
            assert r["stream"] in configs.STREAMS_BLOCK
    assert set(configs.H2_CAND) <= set(configs.BY_ID)
