"""V7-1a-E, G13/G14: предобучение (Stage A) на синтетике. Энкодер года Y — только сессии < 01.01.Y (реальный путь отбора `train_rows`),
детерминизм (тот же seed — побитно), метка ln(r²_{j+1}+ε), восстановление состояния torch. Мутация: бары ≥ Y в обучении — ловится."""
import numpy as np
import pandas as pd
import pytest
import torch

from models_v2.s14 import v7_1e_net as NET
from models_v2.s14 import v7_1e_pre as P
from models_v2.s14 import v7_1e_syn as SYN

Y = 2020
KW = dict(max_epochs=3, min_rows=30)


@pytest.fixture(scope="module")
def pack():
    return SYN.synth_pack(n=130, W=5, seed=0, start="2019-08-01")


def weights(m):
    return torch.cat([p.flatten() for p in m.parameters()])


def spy_train(pack, monkeypatch, kind="tcn", seed=0):
    """Обучение с перехватом реально использованных строк: -> (модель, максимальная дата дня в обучении/валидации)."""
    used = []
    orig = P._batch

    def spy(pk, rows, y, ok, idx):
        used.append(rows[idx])
        return orig(pk, rows, y, ok, idx)
    monkeypatch.setattr(P, "_batch", spy)
    m = P.train_encoder(kind, 5, Y, seed, pack, **KW)
    r = np.unique(np.concatenate(used))
    return m, pd.Timestamp(np.asarray(pack["days"])[r].max())


def test_train_rows_known_answer(pack):
    rows = P.train_rows(pack, Y)
    d = pd.DatetimeIndex(pack["days"])[rows]
    assert d.max() < pd.Timestamp(Y, 1, 1) and (d >= pd.Timestamp(Y, 1, 1)).sum() == 0
    assert len(rows) == ((pd.DatetimeIndex(pack["days"]) < pd.Timestamp(Y, 1, 1)) & pack["norm_ok"]).sum() > 60


@pytest.mark.parametrize("kind", ["tcn", "cnn"])
def test_pretrain_uses_only_sessions_before_year(pack, monkeypatch, kind):
    m, dmax = spy_train(pack, monkeypatch, kind)
    assert dmax < pd.Timestamp(Y, 1, 1)
    assert pd.DatetimeIndex(m.meta["train_days"]).max() < pd.Timestamp(Y, 1, 1)


def test_mutation_bars_from_year_in_training_caught(pack, monkeypatch):
    def leaky(pk, year):                                       # граница сдвинута: сессии до 01.02.Y
        return np.flatnonzero((pd.DatetimeIndex(pk["days"]) < pd.Timestamp(year, 2, 1)) & pk["norm_ok"])
    monkeypatch.setattr(P, "train_rows", leaky)
    _, dmax = spy_train(pack, monkeypatch)
    assert dmax >= pd.Timestamp(Y, 1, 1)


def test_future_bars_do_not_change_trained_weights(pack):
    """Бары ≥ 01.01.Y (X, маска, r²) заменены мусором — веса энкодера года Y побитно те же."""
    pk2 = {k: v.copy() for k, v in pack.items()}
    fut = pd.DatetimeIndex(pk2["days"]) >= pd.Timestamp(Y, 1, 1)
    rng = np.random.default_rng(5)
    pk2["X"][fut] = rng.normal(size=pk2["X"][fut].shape).astype(np.float32)
    pk2["r2"][fut] = rng.random(pk2["r2"][fut].shape) * 50
    a, b = P.train_encoder("tcn", 5, Y, 1, pack, **KW), P.train_encoder("tcn", 5, Y, 1, pk2, **KW)
    assert torch.equal(weights(a), weights(b))


def test_mutation_future_garbage_changes_weights_if_selection_leaks(pack, monkeypatch):
    pk2 = {k: v.copy() for k, v in pack.items()}
    fut = pd.DatetimeIndex(pk2["days"]) >= pd.Timestamp(Y, 1, 1)
    pk2["r2"][fut] = np.random.default_rng(5).random(pk2["r2"][fut].shape) * 50
    monkeypatch.setattr(P, "train_rows", lambda pk, year: np.flatnonzero(pk["norm_ok"]))
    a, b = P.train_encoder("tcn", 5, Y, 1, pack, **KW), P.train_encoder("tcn", 5, Y, 1, pk2, **KW)
    assert not torch.equal(weights(a), weights(b))


def test_same_seed_bitwise_different_seed_differs(pack):
    a, b, c = (P.train_encoder("cnn", 5, Y, s, pack, **KW) for s in (0, 0, 1))
    assert torch.equal(weights(a), weights(b)) and not torch.equal(weights(a), weights(c))
    X, M = pack["X"][50:60], pack["mask"][50:60]
    from models_v2.s14.v7_1e_rep import encode
    assert np.array_equal(encode(a, X, M), encode(b, X, M)) and not np.array_equal(encode(a, X, M), encode(c, X, M))


def test_target_known_answer_and_eps():
    r2 = np.full((2, NET.SLOTS), np.nan)
    r2[0, :4] = [1.0, 4.0, 9.0, 16.0]
    r2[1, 0], r2[1, 2] = 2.0, 3.0                              # бар 1 пропущен -> ни s=0, ни s=1 без метки
    pk = dict(r2=r2)
    y, ok = P.targets(pk, np.array([0, 1]), 0.5)
    assert ok[0].tolist()[:5] == [True, True, True, False, False] and not ok[1].any() and not ok[:, -1].any()
    assert np.allclose(y[0, :3], np.log(np.array([4.0, 9.0, 16.0]) + 0.5), atol=1e-6)
    assert P.eps_of(dict(r2=np.full((3, NET.SLOTS), 4.0)), np.arange(3)) == pytest.approx(4e-6)
    assert P.eps_of(dict(r2=np.zeros((3, NET.SLOTS))), np.arange(3)) == P.EPS_REL


def test_pretrain_learns_something(pack):
    m = P.train_encoder("tcn", 5, Y, 0, pack, max_epochs=25, min_rows=30)
    rows = P.train_rows(pack, Y)
    y, ok = P.targets(pack, rows, m.meta["eps"])
    v = np.arange(len(rows) - m.meta["n_val"], len(rows))
    base = float(((y[v] - y[v][ok[v]].mean()) ** 2)[ok[v]].mean())
    assert m.meta["best_val"] < base and m.meta["n_params"] == NET.n_params(m) <= 1000


def test_errors_and_torch_state_restored(pack):
    with pytest.raises(ValueError):
        P.train_encoder("tcn", 1, Y, 0, pack, **KW)            # окно pack = 5
    with pytest.raises(ValueError):
        P.train_encoder("tcn", 5, 2019, 0, pack, **KW)         # прошлого меньше min_rows (по умолчанию 60)
    thr, det = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    P.train_encoder("cnn", 5, Y, 0, pack, **KW)
    assert (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()) == (thr, det)
    with pytest.raises(RuntimeError), P.deterministic():
        assert torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled()
        raise RuntimeError
    assert (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()) == (thr, det)
