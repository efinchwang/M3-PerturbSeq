"""Training loop for the frozen winner without runtime monkey-patching."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .model import WinnerM3Model


def set_requires_grad(model: WinnerM3Model, trainable_names: set[str]) -> None:
    names = (
        "cty1_classify",
        "batch1_classify",
        "condition_classifiers",
        "encoder",
        "decoder",
        "batch_encoder",
        "batch_decoder",
    )
    for name in names:
        module = getattr(model, name)
        enabled = name in trainable_names
        for p in module.parameters():
            p.requires_grad = enabled


def process_batch(
    batch_data: dict,
    device: torch.device,
    model: WinnerM3Model,
    criterion_ce: nn.Module,
    criterion_mse_mean: nn.Module,
    criterion_kl: nn.Module,
    *,
    batch_classify_dim: int,
    weight_modality: float = 1.0,
):
    data = batch_data["data"].to(device).reshape(len(batch_data["data"]), -1)
    mask_recon = batch_data["mask_recon"].to(device).reshape(len(data), -1)
    mask_poe = batch_data["mask_poe"].to(device).reshape(len(data), -1)
    label = batch_data["label"].to(device)
    batch = batch_data["batch"].to(device)
    condition = batch_data["conditions"][0].to(device)
    is_reference = batch_data["all_train_query_info"].to(device).float()
    line_ids = batch_data["line_id"].to(device)

    batch_onehot = F.one_hot(batch.long(), num_classes=batch_classify_dim).float()

    (
        x_recon,
        x_batch,
        _,
        _,
        _,
        cla_cty,
        cla_batch,
        cla_conditions,
        mu,
        logvar,
    ) = model(data, batch, mask_poe, line_ids)

    cty_loss = criterion_ce(cla_cty, label)
    batch_loss = criterion_ce(cla_batch, batch)
    kl_loss = criterion_kl(mu, logvar)

    mse_none = F.mse_loss(x_recon, data, reduction="none")
    ae_loss = (weight_modality * mse_none * mask_recon).sum() / (
        mask_recon.sum() + 1e-8
    )
    batch_ae_loss = criterion_mse_mean(batch_onehot, x_batch)

    ce = F.cross_entropy(cla_conditions[0], condition, reduction="none")
    condition_loss = (ce * is_reference).sum() / (is_reference.sum() + 1e-8)

    return cty_loss, batch_loss, ae_loss, batch_ae_loss, [condition_loss], kl_loss


@torch.no_grad()
def validation_loss(
    val_loader,
    device: torch.device,
    model: WinnerM3Model,
    criterion_ce: nn.Module,
    criterion_mse_mean: nn.Module,
    criterion_kl: nn.Module,
    *,
    batch_classify_dim: int,
) -> float:
    model.eval()
    total = 0.0
    for batch_data in val_loader:
        _, _, ae, batch_ae, _, _ = process_batch(
            batch_data,
            device,
            model,
            criterion_ce,
            criterion_mse_mean,
            criterion_kl,
            batch_classify_dim=batch_classify_dim,
        )
        total += float((ae + batch_ae).item())
    return total / len(val_loader)


def train_winner(
    train_loader,
    val_loader,
    model: WinnerM3Model,
    *,
    device: torch.device,
    num_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    weight_batch_ae: float = 1.0,
    early_stop_patience: int = 100,
    min_delta: float = 0.0,
) -> tuple[WinnerM3Model, list[dict]]:
    """Train with the exact M3 loss schedule used by the frozen winner.

    Important historical detail preserved here: ``set_requires_grad`` is called
    after each forward, matching upstream M3. Consequently the first main
    minibatch of epochs 2..100 is built while the encoder is still frozen from
    the preceding adversary phase. This reproduces the audited 99 graph-absent
    main minibatches instead of silently "fixing" the training schedule.
    """
    from m3._engine.util import KL_loss

    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse_mean = nn.MSELoss().to(device)
    criterion_kl = KL_loss()

    optimizer = torch.optim.AdamW(
        [{"params": model.parameters()}],
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in tqdm(range(1, num_epochs + 1)):
        batch_weight = max(1.0, 40.0 * (1.0 - epoch / 50.0))
        model.train()

        for batch_data in train_loader:
            losses = process_batch(
                batch_data,
                device,
                model,
                criterion_ce,
                criterion_mse_mean,
                criterion_kl,
                batch_classify_dim=model.batch_classify_dim,
            )
            _, batch_loss, ae_loss, batch_ae_loss, condition_losses, kl_loss = losses

            # Preserve upstream ordering: forward first, requires-grad toggle second.
            set_requires_grad(
                model,
                {
                    "cty1_classify",
                    "condition_classifiers",
                    "encoder",
                    "decoder",
                    "batch_encoder",
                    "batch_decoder",
                },
            )

            loss = (
                ae_loss
                + weight_batch_ae * batch_ae_loss
                + 0.0001 * kl_loss
                + sum(condition_losses)
                - batch_weight * batch_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.train()
        for _ in range(3):
            for batch_data in train_loader:
                losses = process_batch(
                    batch_data,
                    device,
                    model,
                    criterion_ce,
                    criterion_mse_mean,
                    criterion_kl,
                    batch_classify_dim=model.batch_classify_dim,
                )
                batch_loss = losses[1]
                set_requires_grad(model, {"batch1_classify"})
                loss = batch_weight * batch_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        val = validation_loss(
            val_loader,
            device,
            model,
            criterion_ce,
            criterion_mse_mean,
            criterion_kl,
            batch_classify_dim=model.batch_classify_dim,
        )
        history.append({"epoch": epoch, "validation_loss": val})
        print(f"Epoch {epoch}, Validation Loss: {val:.4f}")

        if best_val - val > min_delta:
            best_val = val
            epochs_no_improve = 0
            best_state = deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                if best_state is not None:
                    model.load_state_dict(best_state)
                return model, history

    return model, history
