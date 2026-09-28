#!/usr/bin/env python3
"""Evaluate a clean final-model run on Q_train."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/M3/src")]

from m3_perturbseq.evaluation import compute_qtrain_metrics, posterior_mean_logits
from m3_perturbseq.model import build_winner_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--truth-csv", required=True, type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    run = args.run_dir.resolve()
    meta = pd.read_csv(
        run / "cell_metadata.csv",
        dtype={"cell_id": str, "cell_type": str, "query_split": str, "internal_split": str},
    )
    truth = pd.read_csv(args.truth_csv, dtype={"cell_id": str, "true_gene": str})
    saved = torch.load(run / "replay_inputs.pt", map_location="cpu", weights_only=True)

    mean = np.load(run / "pca_mean_float64.npy", allow_pickle=False)
    basis = np.load(run / "pca_basis_float64.npy", allow_pickle=False)
    n_batch = int(saved["batch"].max().item()) + 1
    model = build_winner_model(mean, basis, batch_classify_dim=n_batch, device="cpu")
    model.load_state_dict(
        torch.load(run / "terminal_generator.state_dict.pt", map_location="cpu", weights_only=True),
        strict=True,
    )

    logits = posterior_mean_logits(
        model,
        saved["data"],
        saved["batch"],
        saved["mask_poe"],
        meta,
        device="cpu",
    )
    rows = compute_qtrain_metrics(meta, logits, truth)

    print(f"{'GROUP':<10} {'N':>6} {'AUROC':>10} {'BALANCED ACCURACY':>20}")
    for row in rows:
        auc = "NA" if row["AUROC"] is None else f"{row['AUROC']:.6f}"
        ba = "NA" if row["balanced_accuracy"] is None else f"{row['balanced_accuracy']:.6f}"
        print(f"{row['group']:<10} {row['n']:>6d} {auc:>10} {ba:>20}")

    output = args.output or (run / "final_metrics.csv")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    else:
        with output.open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=("group", "n", "AUROC", "balanced_accuracy"),
            )
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
