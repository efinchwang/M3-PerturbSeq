"""Evaluation helpers for the final M3 Perturb-seq model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .model import LINE_ORDER, WinnerM3Model


@torch.no_grad()
def posterior_mean_logits(
    model: WinnerM3Model,
    data: torch.Tensor,
    batch: torch.Tensor,
    mask_poe: torch.Tensor,
    metadata: pd.DataFrame,
    *,
    chunk_size: int = 512,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    line_index = {name: i for i, name in enumerate(LINE_ORDER)}
    cell_lines = metadata["cell_type"].astype(str).tolist()
    if any(x not in line_index for x in cell_lines):
        raise ValueError("metadata contains a cell line outside the fixed six-line order")
    line_ids = torch.tensor([line_index[x] for x in cell_lines], dtype=torch.long)

    chunks = []
    for start in range(0, len(metadata), chunk_size):
        stop = min(start + chunk_size, len(metadata))
        logits = model.condition_logits_from_posterior_mean(
            data[start:stop].to(device),
            batch[start:stop].to(device),
            mask_poe[start:stop].to(device),
            line_ids[start:stop].to(device),
        )
        chunks.append(logits.cpu().numpy())

    out = np.concatenate(chunks, axis=0)
    if out.shape != (len(metadata), 2) or not np.isfinite(out).all():
        raise ValueError("expected finite N x 2 condition logits")
    return out


def compute_qtrain_metrics(
    metadata: pd.DataFrame,
    logits: np.ndarray,
    truth: pd.DataFrame,
) -> list[dict]:
    meta = metadata.copy()
    truth = truth[["cell_id", "true_gene"]].copy()
    meta["cell_id"] = meta["cell_id"].astype(str)
    truth["cell_id"] = truth["cell_id"].astype(str)

    if meta["cell_id"].duplicated().any() or truth["cell_id"].duplicated().any():
        raise ValueError("cell_id must be unique")

    joined = meta.merge(
        truth,
        how="left",
        on="cell_id",
        validate="one_to_one",
        indicator=True,
    )
    q = joined[
        joined["query_split"].astype(str).eq("query")
        & joined["internal_split"].astype(str).eq("Q_train")
    ].copy()

    if len(q) != 7037 or q["_merge"].ne("both").any():
        raise ValueError("Q_train truth join does not match the frozen 7,037-cell membership")
    if logits.shape != (len(meta), 2):
        raise ValueError("logits must be aligned to metadata and have shape N x 2")

    row_by_id = {cid: i for i, cid in enumerate(meta["cell_id"])}
    indices = np.asarray([row_by_id[cid] for cid in q["cell_id"]], dtype=int)
    margins = logits[:, 1] - logits[:, 0]
    y = q["true_gene"].eq("TGFBR1").to_numpy(dtype=np.int8)

    if set(q["true_gene"].dropna().astype(str)) != {"NT", "TGFBR1"}:
        raise ValueError("Q_train truth must contain only NT and TGFBR1")

    rows = []
    groups = [("OVERALL", np.ones(len(q), dtype=bool))]
    groups += [
        (line, q["cell_type"].astype(str).to_numpy() == line)
        for line in LINE_ORDER
    ]

    for name, mask in groups:
        yy = y[mask]
        ss = margins[indices[mask]]
        if len(np.unique(yy)) != 2:
            auc = ba = None
        else:
            auc = float(roc_auc_score(yy, ss))
            ba = float(balanced_accuracy_score(yy, ss >= 0))
        rows.append(
            {
                "group": name,
                "n": int(mask.sum()),
                "AUROC": auc,
                "balanced_accuracy": ba,
            }
        )
    return rows
