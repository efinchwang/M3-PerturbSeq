"""Perturb-seq architectural modifications for M3."""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

from m3._engine.util import poe


LATENT_DIM = 30
CONDITION_SLICE = slice(26, 28)
PCA_RANK = 64
CONTEXT_RANK = 4


class PCAContextLinear(nn.Module):
    """M3's first RNA mean layer with PCA-restricted condition coordinates."""

    def __init__(
        self,
        original: nn.Linear,
        pca_mean: np.ndarray | torch.Tensor,
        pca_basis: np.ndarray | torch.Tensor,
        n_cell_lines: int,
    ) -> None:
        super().__init__()

        if n_cell_lines < 1:
            raise ValueError("n_cell_lines must be positive")

        weight = original.weight.detach().clone()
        if original.bias is not None or weight.shape[0] != LATENT_DIM:
            raise ValueError(
                f"expected M3 Linear(n_features, {LATENT_DIM}, bias=False)"
            )

        dtype = weight.dtype
        device = weight.device
        n_features = weight.shape[1]

        mean64 = torch.as_tensor(pca_mean, dtype=torch.float64, device=device)
        basis64 = torch.as_tensor(pca_basis, dtype=torch.float64, device=device)
        if tuple(mean64.shape) != (n_features,):
            raise ValueError(f"expected PCA mean shape ({n_features},)")
        if tuple(basis64.shape) != (n_features, PCA_RANK):
            raise ValueError(
                f"expected PCA basis shape ({n_features}, {PCA_RANK})"
            )

        self.n_cell_lines = n_cell_lines

        # The other 28 latent coordinates keep the original M3 weights.
        self.retained = nn.Parameter(
            torch.cat(
                [
                    weight[: CONDITION_SLICE.start],
                    weight[CONDITION_SLICE.stop :],
                ],
                dim=0,
            )
        )

        # Initialize the two condition coordinates as the projection of the
        # original M3 weights into the fixed PCA subspace.
        target64 = weight[CONDITION_SLICE].to(torch.float64)
        self.C = nn.Parameter((target64 @ basis64).to(dtype=dtype))
        self.d = nn.Parameter((target64 @ mean64).to(dtype=dtype))

        # Rank-4 cell-line-specific residual. A starts at zero, so the residual
        # is initially inactive. A private generator avoids changing M3's RNG.
        generator = torch.Generator(device="cpu").manual_seed(20260927)
        B = torch.empty((CONTEXT_RANK, PCA_RANK), dtype=dtype, device="cpu")
        nn.init.kaiming_uniform_(B, a=math.sqrt(5), generator=generator)
        self.B = nn.Parameter(B.to(device=device))
        self.A = nn.Parameter(
            torch.zeros(
                (
                    n_cell_lines,
                    CONDITION_SLICE.stop - CONDITION_SLICE.start,
                    CONTEXT_RANK,
                ),
                dtype=dtype,
                device=device,
            )
        )

        self.register_buffer("pca_mean", mean64.to(dtype=dtype))
        self.register_buffer("pca_basis", basis64.to(dtype=dtype))

    def forward(self, x: torch.Tensor, cell_line: torch.Tensor) -> torch.Tensor:
        cell_line = cell_line.to(device=x.device, dtype=torch.long)
        if cell_line.ndim != 1 or len(cell_line) != len(x):
            raise ValueError("cell_line must be a length-N vector")
        if torch.any((cell_line < 0) | (cell_line >= self.n_cell_lines)):
            raise ValueError("invalid cell-line index")

        scores = (x - self.pca_mean) @ self.pca_basis

        condition = F.linear(scores, self.C, self.d)
        context = F.linear(scores, self.B)
        condition = condition + torch.bmm(
            self.A[cell_line], context.unsqueeze(-1)
        ).squeeze(-1)

        retained = F.linear(x, self.retained)
        return torch.cat(
            [
                retained[:, : CONDITION_SLICE.start],
                condition,
                retained[:, CONDITION_SLICE.start :],
            ],
            dim=1,
        )



class M3PerturbSeq(nn.Module):
    """Thin wrapper around an already-created M3_model.

    All original M3 modules are retained. Only the RNA mean layer and its
    dropout are modified; the original shared condition classifier is retained.
    """

    requires_cell_line = True

    def __init__(
        self,
        base_model: nn.Module,
        pca_mean: np.ndarray,
        pca_basis: np.ndarray,
        n_cell_lines: int,
    ) -> None:
        super().__init__()

        if list(base_model.condition_dim) != [2]:
            raise ValueError(
                "M3PerturbSeq currently requires one binary condition"
            )
        if len(base_model.encoder.encoders_mean) != 1:
            raise ValueError(
                "M3PerturbSeq currently expects RNA as the only modality"
            )

        # Reuse the modules that M3 itself just initialized. This avoids
        # reimplementing or reinitializing the rest of the architecture.
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.batch_encoder = base_model.batch_encoder
        self.batch_decoder = base_model.batch_decoder
        self.cty1_classify = base_model.cty1_classify
        self.batch1_classify = base_model.batch1_classify
        self.condition_classifiers = base_model.condition_classifiers

        self.norm = base_model.norm
        self.drop = base_model.drop
        self.batch_classify_dim = base_model.batch_classify_dim
        self.condition_dim = base_model.condition_dim

        self.encoder.encoders_mean[0][0] = PCAContextLinear(
            self.encoder.encoders_mean[0][0],
            pca_mean,
            pca_basis,
            n_cell_lines=n_cell_lines,
        )

        # Final selected RNA mean dropout.
        self.encoder.encoders_mean[0][3].p = 0.5

    def _encode_rna(
        self,
        x: torch.Tensor,
        cell_line: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        mean_encoder = self.encoder.encoders_mean[0]

        mean = mean_encoder[0](x, cell_line)
        for layer in list(mean_encoder)[1:]:
            mean = layer(mean)

        logvar = self.encoder.encoders_var[0](x)
        return [mean], [logvar]

    def posterior(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor,
        cell_line: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        means, logvars = self._encode_rna(x, cell_line)

        batch_onehot = F.one_hot(
            batch.long(),
            num_classes=self.batch_classify_dim,
        ).float()
        batch_mean, batch_logvar = self.batch_encoder(batch_onehot)

        return poe(
            means + [batch_mean],
            logvars + [batch_logvar],
            mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor,
        cell_line: torch.Tensor,
    ):
        mean, logvar = self.posterior(x, batch, mask, cell_line)

        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

        z_cell = z[:, : CONDITION_SLICE.start]
        z_condition = z[:, CONDITION_SLICE]
        z_batch = z[:, CONDITION_SLICE.stop :]

        return (
            self.decoder(z),
            self.batch_decoder(z_batch),
            z_cell,
            z_batch,
            [z_condition],
            self.cty1_classify(z_cell),
            self.batch1_classify(torch.cat([z_cell, z_condition], dim=1)),
            [self.condition_classifiers[0](z_condition)],
            mean,
            logvar,
        )

    @torch.no_grad()
    def condition_logits(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor,
        cell_line: torch.Tensor,
    ) -> torch.Tensor:
        mean, _ = self.posterior(x, batch, mask, cell_line)
        return self.condition_classifiers[0](
            mean[:, CONDITION_SLICE]
        )


def _fit_rank64_pca(
    expression: torch.Tensor,
    train_indices,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the fixed unwhitened rank-64 PCA on M3's training cells."""
    indices = np.asarray(train_indices, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError("at least two training cells are required for PCA")

    x = (
        expression[indices]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )

    mean = x.mean(axis=0, dtype=np.float64)
    centered = x - mean
    covariance = (centered.T @ centered) / (len(x) - 1)

    n_features = covariance.shape[0]
    if PCA_RANK > n_features:
        raise ValueError(
            f"PCA rank {PCA_RANK} exceeds {n_features} input features"
        )

    eigenvalues, eigenvectors = scipy.linalg.eigh(
        covariance,
        subset_by_index=[n_features - PCA_RANK, n_features - 1],
    )

    order = np.argsort(eigenvalues)[::-1]
    return mean, eigenvectors[:, order], eigenvalues[order]


def build_perturbseq_model(
    base_model: nn.Module,
    transformed_dataset,
    train_indices,
    validation_indices,
    n_cell_lines: int,
) -> M3PerturbSeq:
    """Architecture hook called by the minimally modified M3 training engine."""
    mean, basis, eigenvalues = _fit_rank64_pca(
        transformed_dataset.data,
        train_indices,
    )

    model = M3PerturbSeq(
        base_model,
        pca_mean=mean,
        pca_basis=basis,
        n_cell_lines=n_cell_lines,
    )

    # Non-parameter metadata for saving the exact internal split/PCA later.
    model.train_indices = tuple(int(i) for i in train_indices)
    model.validation_indices = tuple(int(i) for i in validation_indices)
    model.pca_eigenvalues = eigenvalues

    return model
