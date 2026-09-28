"""Final M3 Perturb-seq architecture.

This is a direct, readable implementation of the frozen development winner:
- rank-64 unwhitened PCA restriction on RNA mean coordinates 26:28,
- rank-4 cell-line context residual on those two coordinates,
- RNA mean-branch dropout p=0.5,
- centered cell-line-specific affine residual on condition logits.

It deliberately contains no monkey-patching or experiment-runtime adapters.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from m3._engine.model import M3_model
from m3._engine.util import poe

LINE_ORDER = ("A549", "BXPC3", "HAP1", "HT29", "K562", "MCF7")
CONDITION_SLICE = slice(26, 28)


class PCAContextLinear(nn.Module):
    """Replacement for the first RNA posterior-mean Linear(2000, 30).

    The 28 non-condition rows are retained exactly. Rows 26:28 are replaced by
    a trainable affine map of fixed rank-64 PCA scores, plus a rank-4
    cell-line-specific residual:

        s = (x - m) @ V
        z = C s + d + A[line] @ (B s)

    Parameter/state names intentionally mirror the audited winner so archived
    winner checkpoints can be strict-loaded into this clean implementation.
    """

    def __init__(
        self,
        original: nn.Linear,
        pca_mean: np.ndarray | torch.Tensor,
        pca_basis: np.ndarray | torch.Tensor,
        *,
        line_order: Sequence[str] = LINE_ORDER,
        private_seed: int = 20260927,
    ) -> None:
        super().__init__()

        if not isinstance(original, nn.Linear):
            raise TypeError("expected the original RNA mean first layer to be nn.Linear")
        if original.bias is not None or tuple(original.weight.shape) != (30, 2000):
            raise ValueError("expected Linear(2000, 30, bias=False)")

        W0 = original.weight.detach().clone()
        device, dtype = W0.device, W0.dtype

        m = torch.as_tensor(pca_mean, dtype=torch.float64, device=device)
        V = torch.as_tensor(pca_basis, dtype=torch.float64, device=device)
        if tuple(m.shape) != (2000,) or tuple(V.shape) != (2000, 64):
            raise ValueError("expected PCA mean (2000,) and basis (2000, 64)")
        if tuple(line_order) != LINE_ORDER:
            raise ValueError(f"line order must be exactly {LINE_ORDER}")

        keep = [i for i in range(30) if i not in (26, 27)]
        self.retained = nn.Parameter(W0[keep].clone())

        W_target = W0[26:28].detach().to(torch.float64)
        C0 = W_target @ V
        d0 = W_target @ m
        self.C = nn.Parameter(C0.to(dtype=dtype))
        self.d = nn.Parameter(d0.to(dtype=dtype))

        # Private CPU generator exactly matches the audited initialization while
        # leaving the global Torch RNG untouched.
        private = torch.Generator(device="cpu").manual_seed(private_seed)
        B = torch.empty((4, 64), dtype=dtype, device="cpu")
        nn.init.kaiming_uniform_(B, a=math.sqrt(5), generator=private)
        self.B = nn.Parameter(B.to(device=device, dtype=dtype))
        self.A = nn.Parameter(torch.zeros((6, 2, 4), dtype=dtype, device=device))

        # Persistent buffers intentionally use the same names as the archived
        # winner's replacement module for strict state_dict compatibility.
        self.register_buffer("m_float64", m.clone())
        self.register_buffer("V_float64", V.clone())
        self.register_buffer("W0_target_float64", W_target.clone())
        self.register_buffer("C0_float64", C0.clone())
        self.register_buffer("d0_float64", d0.clone())
        self.register_buffer("m_cast", m.to(dtype=dtype))
        self.register_buffer("V_cast", V.to(dtype=dtype))
        self.register_buffer("C0_cast", C0.to(dtype=dtype))
        self.register_buffer("d0_cast", d0.to(dtype=dtype))
        self.register_buffer(
            "_keep_index",
            torch.tensor(keep, device=device, dtype=torch.long),
            persistent=False,
        )
        self.original_dtype = str(dtype)
        self.last_residual: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, line_ids: torch.Tensor) -> torch.Tensor:
        if line_ids.ndim != 1 or len(line_ids) != len(x):
            raise ValueError("line_ids must be a length-N vector aligned to x")
        line = line_ids.to(device=x.device, dtype=torch.long)
        if bool(torch.any(line < 0)) or bool(torch.any(line >= len(LINE_ORDER))):
            raise ValueError("line_ids contain an index outside the six-line vocabulary")

        retained = F.linear(x, self.retained)
        scores = (x - self.m_cast) @ self.V_cast
        z = F.linear(scores, self.C, self.d)

        shared = F.linear(scores, self.B)
        delta = torch.bmm(self.A[line], shared.unsqueeze(-1)).squeeze(-1)
        self.last_residual = delta
        z = z + delta

        out = x.new_empty((x.shape[0], 30))
        out[:, self._keep_index] = retained
        out[:, 26:28] = z
        if not torch.isfinite(out).all():
            raise FloatingPointError("non-finite PCA condition-mean preactivation")
        return out


class CenteredLineHead(nn.Module):
    """Centered affine cell-line residual on top of the shared condition head."""

    def __init__(self, shared: nn.Module) -> None:
        super().__init__()
        self.shared = shared
        self.A = nn.Parameter(torch.zeros(6, 2))
        self.b = nn.Parameter(torch.zeros(6))

    def forward(self, z: torch.Tensor, line_ids: torch.Tensor) -> torch.Tensor:
        logits = self.shared(z)

        if line_ids.ndim != 1 or len(line_ids) != len(z):
            raise ValueError("line_ids must be a length-N vector aligned to z")
        line = line_ids.to(device=z.device, dtype=torch.long)
        if bool(torch.any(line < 0)) or bool(torch.any(line >= len(LINE_ORDER))):
            raise ValueError("line_ids contain an index outside the six-line vocabulary")

        A_centered = self.A - self.A.mean(dim=0, keepdim=True)
        b_centered = self.b - self.b.mean()
        d = (A_centered[line] * z).sum(dim=1) + b_centered[line]
        return logits + torch.stack((-0.5 * d, 0.5 * d), dim=1)


class WinnerM3Model(M3_model):
    """M3 generator with the frozen Perturb-seq winner architecture."""

    def __init__(
        self,
        pca_mean: np.ndarray | torch.Tensor,
        pca_basis: np.ndarray | torch.Tensor,
        *,
        batch_classify_dim: int = 2,
        cty_classify_dim: int = 50,
    ) -> None:
        # Construct the ordinary M3 model first. This preserves the exact base
        # initialization/order used by the development run.
        super().__init__(
            nfeatures=[2000],
            hidden_features=[30],
            z_dim=30,
            cty_classify_dim=cty_classify_dim,
            batch_classify_dim=batch_classify_dim,
            condition_dim=[2],
        )

        original = self.encoder.encoders_mean[0][0]
        self.encoder.encoders_mean[0][0] = PCAContextLinear(
            original,
            pca_mean,
            pca_basis,
        )

        # Winner changes only RNA posterior-mean dropout from 0.2 -> 0.5.
        dropout = self.encoder.encoders_mean[0][3]
        if not isinstance(dropout, nn.Dropout):
            raise TypeError("unexpected M3 RNA mean encoder layout")
        dropout.p = 0.5

        self.condition_classifiers[0] = CenteredLineHead(self.condition_classifiers[0])

    @property
    def context_layer(self) -> PCAContextLinear:
        layer = self.encoder.encoders_mean[0][0]
        if not isinstance(layer, PCAContextLinear):
            raise RuntimeError("winner context layer is missing")
        return layer

    @property
    def condition_head(self) -> CenteredLineHead:
        head = self.condition_classifiers[0]
        if not isinstance(head, CenteredLineHead):
            raise RuntimeError("winner condition head is missing")
        return head

    def _encode_experts(
        self,
        x: torch.Tensor,
        line_ids: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self.encoder.n_modalities != 1 or self.encoder.feature_splits != [2000]:
            raise RuntimeError("the final Perturb-seq model is RNA-only with 2,000 HVGs")

        block = self.encoder.encoders_mean[0]
        h = block[0](x, line_ids)
        for module in list(block)[1:]:
            h = module(h)

        logvar = self.encoder.encoders_var[0](x)
        return [h], [logvar]

    def posterior(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor,
        line_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu_list, logvar_list = self._encode_experts(x, line_ids)

        b = F.one_hot(
            batch.long(),
            num_classes=self.batch_classify_dim,
        ).float()
        batch_mu, batch_logvar = self.batch_encoder(b)
        return poe(mu_list + [batch_mu], logvar_list + [batch_logvar], mask)

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor,
        line_ids: torch.Tensor,
    ):
        # This is the original M3 forward written explicitly so cell-line
        # context is a normal model input rather than hidden mutable state.
        mu, logvar = self.posterior(x, batch, mask, line_ids)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        z_embedding = z[:, :26]
        z_conditions = [z[:, 26:28]]
        z_batch = z[:, -2:]

        x_recon = self.decoder(z)
        x_batch_recon = self.batch_decoder(z_batch)

        cla_cty = self.cty1_classify(z_embedding)
        cla_batch = self.batch1_classify(torch.cat([z_embedding] + z_conditions, dim=1))
        cla_conditions = [self.condition_head(z_conditions[0], line_ids)]

        return (
            x_recon,
            x_batch_recon,
            z_embedding,
            z_batch,
            z_conditions,
            cla_cty,
            cla_batch,
            cla_conditions,
            mu,
            logvar,
        )

    @torch.no_grad()
    def condition_logits_from_posterior_mean(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor,
        line_ids: torch.Tensor,
    ) -> torch.Tensor:
        mu, _ = self.posterior(x, batch, mask, line_ids)
        return self.condition_head(mu[:, CONDITION_SLICE], line_ids)


def build_winner_model(
    pca_mean: np.ndarray,
    pca_basis: np.ndarray,
    *,
    batch_classify_dim: int = 2,
    device: torch.device | str | None = None,
) -> WinnerM3Model:
    model = WinnerM3Model(
        pca_mean,
        pca_basis,
        batch_classify_dim=batch_classify_dim,
    )
    if device is not None:
        model = model.to(device)
    return model
