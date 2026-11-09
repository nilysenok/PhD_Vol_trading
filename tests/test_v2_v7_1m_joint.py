"""V7 лист 1a-M, Stage B: адаптер к энкодерам 1a-E (TCN/CNN, PCA). Дифференцируемый z = z листа 1a-E; F = H при нулевой голове; градиент доходит до энкодера; штраф λ_e стягивает к весам предобучения. Синтетика, данных SBER нет."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from models_v2.s14 import v7_1e_net as NET, v7_1e_rep as REP, v7_1m_fit as FT, v7_1m_joint as JT, v7_1m_loss as LS


def synth(n=40, w=1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, w, NET.SLOTS, NET.K)).astype(np.float32)
    M = np.ones((n, w, NET.SLOTS), bool)
    M[::3, :, -5:] = False                                                    # неполные дни
    M[1::7, :, :] = False
    M[1::7, :, :3] = True
    return X, M


def adapter(kind="tcn", seed=0):
    torch.manual_seed(seed)
    base = NET.build(kind)
    X, M = synth()
    raw = REP.encode(base, X, M)
    pca = REP.fit_pca_past(raw, np.array(["2015-01-01"] * len(raw), dtype="datetime64[ns]"), 2016)
    return base, pca, raw, X, M


@pytest.mark.parametrize("kind", ["tcn", "cnn"])
def test_adapter_reproduces_1aE_z(kind):
    base, pca, raw, X, M = adapter(kind)
    ref = REP.transform_pca(pca, raw)
    enc = JT.PCAEncoder(base.double(), pca)
    with torch.no_grad():
        z = enc(torch.as_tensor(JT.pack_seq(X, M)))
    assert z.shape == (len(X), 4) and np.allclose(z.numpy(), ref, atol=1e-4)


def test_mutation_adapter_ignoring_mask_differs():
    base, pca, raw, X, M = adapter()
    xm = torch.as_tensor(JT.pack_seq(X, M))
    enc = JT.PCAEncoder(base.double(), pca)
    xm2 = xm.clone()
    xm2[:, -1] = 1.0                                                          # мутант: маска слотов потеряна
    with torch.no_grad():
        assert not torch.allclose(enc(xm), enc(xm2))


def test_joint_zero_head_gives_F_equals_H_and_gradient_reaches_encoder():
    base, pca, raw, X, M = adapter()
    xm = torch.as_tensor(JT.pack_seq(X, M))
    spec = dict(kind="moe", router="B", encoder=lambda: JT.PCAEncoder(base, pca))
    m = FT.make(spec, 0)
    n = len(X)
    g = torch.Generator().manual_seed(1)
    b = dict(Zg=torch.randn(n, 9, generator=g, dtype=torch.float64), z=xm, ell=torch.zeros(n, dtype=torch.float64), H=torch.ones(n, dtype=torch.float64) * 2.0, rv=torch.exp(torch.randn(n, generator=g, dtype=torch.float64)))
    assert torch.equal(b["H"] * torch.exp(m(b["Zg"], b["z"], b["ell"])[0]), b["H"])
    with torch.no_grad():
        m.head.a.fill_(0.3)                                                   # голова ненулевая -> градиент идёт и в энкодер
    LS.loss(m, b, 0.1, 0.0).backward()
    assert any(float(p.grad.abs().sum()) > 0 for p in m.encoder.base.parameters())
    assert all(not b.requires_grad for b in m.encoder.buffers())               # PCA заморожена


def test_joint_penalty_pulls_encoder_to_pretrained_weights():
    def run(lam_e):
        base, pca, raw, X, M = adapter()
        xm = torch.as_tensor(JT.pack_seq(X, M))
        n = len(X)
        g = torch.Generator().manual_seed(2)
        b = dict(Zg=torch.randn(n, 9, generator=g, dtype=torch.float64), z=xm, ell=torch.zeros(n, dtype=torch.float64), H=torch.ones(n, dtype=torch.float64), rv=torch.exp(0.5 * torch.randn(n, generator=g, dtype=torch.float64)))
        spec = dict(kind="moe", router="B", encoder=lambda: JT.PCAEncoder(base, pca), lam_e=lam_e)
        m, _, _ = FT.train_one(spec, 0, 0.1, 0.0, b, None, 30)
        return float(sum(((p - p0) ** 2).sum() for p, p0 in zip(m.encoder.parameters(), m.phi0)))
    assert run(0.0) > 1e-8 and run(1e4) < 0.05 * run(0.0)
