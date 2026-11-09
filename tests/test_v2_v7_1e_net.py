"""V7-1a-E, G13: энкодеры TCN и CNN. Причинность (выход на баре j не зависит от баров > j), число параметров ≤ 1000 и равно объявленному,
маска слотов без бара, порядок дней в окне. Мутация: непричинное (симметричное) дополнение — ловится."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from models_v2.s14 import v7_1e_net as NET

L = 5 * NET.SLOTS


def causal_ok(kind: str, js=(0, 1, 52, 53, 130, 200)) -> bool:
    """Правда, если для каждого j изменение входов на позициях > j не меняет скрытое состояние и прогноз на позициях ≤ j."""
    torch.manual_seed(3)
    m = NET.build(kind).eval()
    g = torch.Generator().manual_seed(7)
    x = torch.randn(2, NET.K, L, generator=g)
    with torch.no_grad():
        for j in js:
            x2 = x.clone()
            x2[:, :, j + 1:] = torch.randn_like(x2[:, :, j + 1:])
            same_h = torch.allclose(m.hidden(x)[:, :, :j + 1], m.hidden(x2)[:, :, :j + 1], atol=1e-6, rtol=0)
            same_p = torch.allclose(NET.forward_pred(m, x)[:, :j + 1], NET.forward_pred(m, x2)[:, :j + 1], atol=1e-6, rtol=0)
            x3 = x.clone()
            x3[:, :, j + 1] += 5.0                           # k = 1: ровно следующий бар
            same_1 = torch.allclose(m.hidden(x)[:, :, :j + 1], m.hidden(x3)[:, :, :j + 1], atol=1e-6, rtol=0)
            if not (same_h and same_p and same_1):
                return False
    return True


def pad_symmetric(x, p):
    return F.pad(x, (p // 2, p - p // 2))


@pytest.mark.parametrize("kind", ["tcn", "cnn"])
def test_causal(kind):
    assert causal_ok(kind)


@pytest.mark.parametrize("kind", ["tcn", "cnn"])
def test_mutation_noncausal_padding_caught(kind, monkeypatch):
    monkeypatch.setattr(NET, "pad_left", pad_symmetric)
    assert not causal_ok(kind)


@pytest.mark.parametrize("kind", ["tcn", "cnn"])
def test_future_changes_output_at_later_positions(kind):
    """Проверка не вырожденная: изменение входа меняет выход на позициях ≥ j (сеть не константа)."""
    torch.manual_seed(1)
    m = NET.build(kind).eval()
    x = torch.randn(1, NET.K, L)
    x2 = x.clone()
    x2[:, :, 100] += 3.0
    with torch.no_grad():
        d = (m.hidden(x) - m.hidden(x2)).abs().amax(dim=(0, 1))
    assert d[100] > 1e-3 and d[:100].max() == 0


def test_param_counts_declared_and_within_plan():
    got = {k: NET.n_params(NET.build(k)) for k in ("tcn", "cnn")}
    assert got == NET.PARAMS == {"tcn": 905, "cnn": 785}
    assert all(v <= NET.MAX_PARAMS == 1000 for v in got.values())


def test_param_count_known_answer():
    # TCN: 1×1 11→8 (96) + 4×(8·8·3+8) + голова 9; CNN: 11·8·5+8 + 8·8·5+8 + 9
    assert 96 + 4 * 200 + 9 == NET.n_params(NET.build("tcn"))
    assert 448 + 328 + 9 == NET.n_params(NET.build("cnn"))


def test_dilations_and_kernels_as_declared():
    t, c = NET.build("tcn"), NET.build("cnn")
    assert [b.dilation[0] for b in t.blocks] == [1, 2, 4, 8] and all(b.kernel_size == (3,) for b in t.blocks)
    assert c.c1.kernel_size == c.c2.kernel_size == (5,) and (c.c1.in_channels, c.c1.out_channels, c.c2.out_channels) == (11, 8, 8)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        NET.build("lstm")


def test_to_seq_layout_and_mask_zeroing():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2, 5, NET.SLOTS, NET.K))
    M = rng.random((2, 5, NET.SLOTS)) > 0.2
    s = NET.to_seq(X, M)
    assert tuple(s.shape) == (2, NET.K, L)
    # день w занимает позиции w·53 … w·53+52 (предыдущие дни — раньше дня d)
    assert np.allclose(s[1, :, 3 * NET.SLOTS + 7].numpy(), (X[1, 3, 7] * M[1, 3, 7]).astype(np.float32), atol=1e-6)
    assert (s[0, :, :].transpose(0, 1)[~torch.as_tensor(M[0].reshape(-1))] == 0).all()


@pytest.mark.parametrize("kind", ["tcn", "cnn"])
def test_garbage_in_masked_slots_does_not_matter(kind):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1, 5, NET.SLOTS, NET.K))
    M = rng.random((1, 5, NET.SLOTS)) > 0.3
    X2 = np.where(M[..., None], X, 1e6)
    torch.manual_seed(0)
    m = NET.build(kind).eval()
    with torch.no_grad():
        assert torch.equal(m.hidden(NET.to_seq(X, M)), m.hidden(NET.to_seq(X2, M)))


def test_same_seed_same_init_different_seed_differs():
    def w(seed):
        torch.manual_seed(seed)
        return torch.cat([p.flatten() for p in NET.build("tcn").parameters()])
    assert torch.equal(w(0), w(0)) and not torch.equal(w(0), w(1))
