#!/usr/bin/env python3
"""Train the frozen M3 Perturb-seq development winner from a sanitized H5AD."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = repo_root()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/M3/src")]

from m3_perturbseq.data import prepare_h5ad
from m3_perturbseq.evaluation import posterior_mean_logits
from m3_perturbseq.model import build_winner_model
from m3_perturbseq.training import train_winner


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-h5ad", required=True, type=Path)
    p.add_argument("--basis-dir", type=Path, default=ROOT / "artifacts/pca")
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()

    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)

    basis_dir = args.basis_dir.resolve()
    mean_path = basis_dir / "pca_mean_float64.npy"
    basis_path = basis_dir / "pca_basis_float64.npy"
    if not mean_path.is_file() or not basis_path.is_file():
        raise FileNotFoundError(
            "basis directory must contain pca_mean_float64.npy and pca_basis_float64.npy"
        )
    mean = np.load(mean_path, allow_pickle=False)
    basis = np.load(basis_path, allow_pickle=False)

    prepared = prepare_h5ad(args.input_h5ad, batch_size=256, seed=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_winner_model(
        mean,
        basis,
        batch_classify_dim=prepared.n_unique_batch,
        device=device,
    )
    model, history = train_winner(
        prepared.train_loader,
        prepared.val_loader,
        model,
        device=device,
        num_epochs=100,
        lr=1e-4,
        weight_decay=0.05,
        early_stop_patience=100,
        min_delta=0.0,
    )

    torch.save(model.state_dict(), out / "terminal_generator.state_dict.pt")
    torch.save(
        {
            "data": prepared.full_data.detach().cpu(),
            "batch": prepared.full_batch.detach().cpu(),
            "mask_poe": prepared.full_mask_poe.detach().cpu(),
            "cell_ids": prepared.metadata["cell_id"].astype(str).tolist(),
        },
        out / "replay_inputs.pt",
    )
    prepared.metadata.to_csv(out / "cell_metadata.csv", index=False)
    (out / "locked_hvg_order.json").write_text(
        json.dumps(prepared.hvg_names, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )

    shutil.copy2(mean_path, out / mean_path.name)
    shutil.copy2(basis_path, out / basis_path.name)

    logits = posterior_mean_logits(
        model,
        prepared.full_data,
        prepared.full_batch,
        prepared.full_mask_poe,
        prepared.metadata,
        device=device,
    )
    np.save(out / "condition_logits_full.npy", logits)

    config = {
        "experiment_id": "pca_context_head_residual_rank64_ctx4_drop50_wd5e2_epoch100_seed0",
        "training": {
            "seed": 0,
            "epochs": 100,
            "lr": 1e-4,
            "batch_size": 256,
            "weight_decay": 0.05,
            "balance_batches": False,
        },
        "pca": {"rank": 64, "whiten": False},
        "context": {
            "rank": 4,
            "B_shape": [4, 64],
            "A_shape": [6, 2, 4],
            "private_seed": 20260927,
        },
        "condition_head": {
            "A_shape": [6, 2],
            "b_shape": [6],
            "stored_parameters": 18,
            "identifiable_degrees_of_freedom": 15,
        },
        "line_order": ["A549", "BXPC3", "HAP1", "HT29", "K562", "MCF7"],
        "split_counts": prepared.split_counts,
        "Q_truth_used_for_training": False,
    }
    (out / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved final run to: {out}")


if __name__ == "__main__":
    main()
