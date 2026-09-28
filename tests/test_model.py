from __future__ import annotations

import random

import numpy as np
import torch

from m3._engine.model import M3_model
from m3_perturbseq.model import CenteredLineHead, PCAContextLinear


def test_head_zero_init_is_exact_identity():
    torch.manual_seed(1)
    shared = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    head = CenteredLineHead(shared)
    x = torch.randn(18, 2)
    lines = torch.arange(18) % 6
    assert torch.equal(shared(x), head(x, lines))
    assert head.A.shape == (6, 2)
    assert head.b.shape == (6,)


def test_context_shapes_zero_A_and_private_rng():
    random.seed(71)
    np.random.seed(72)
    torch.manual_seed(73)
    base = M3_model(
        nfeatures=[2000],
        hidden_features=[30],
        z_dim=30,
        cty_classify_dim=50,
        batch_classify_dim=2,
        condition_dim=[2],
    )
    before = torch.get_rng_state().clone()

    mean = np.zeros(2000, dtype=np.float64)
    basis = np.zeros((2000, 64), dtype=np.float64)
    basis[:64, :] = np.eye(64)
    layer = PCAContextLinear(base.encoder.encoders_mean[0][0], mean, basis)

    assert torch.equal(before, torch.get_rng_state())
    assert layer.A.shape == (6, 2, 4)
    assert layer.B.shape == (4, 64)
    assert torch.count_nonzero(layer.A).item() == 0
    assert layer.A.numel() + layer.B.numel() == 304
