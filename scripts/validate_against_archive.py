#!/usr/bin/env python3
"""Check that the clean architecture strict-loads and reproduces archived logits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/M3/src")]

from m3_perturbseq.evaluation import posterior_mean_logits
from m3_perturbseq.model import build_winner_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive-run", required=True, type=Path)
    args = p.parse_args()
    run = args.archive_run.resolve()

    meta = pd.read_csv(
        run / "cell_metadata.csv",
        dtype={"cell_id": str, "cell_type": str, "query_split": str, "internal_split": str},
    )
    replay = torch.load(run / "replay_inputs.pt", map_location="cpu", weights_only=True)
    mean = np.load(run / "pca_mean_float64.npy", allow_pickle=False)
    basis = np.load(run / "pca_basis_float64.npy", allow_pickle=False)

    model = build_winner_model(
        mean,
        basis,
        batch_classify_dim=int(replay["batch"].max().item()) + 1,
        device="cpu",
    )
    state = torch.load(
        run / "terminal_generator.state_dict.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)

    logits = posterior_mean_logits(
        model,
        replay["data"],
        replay["batch"],
        replay["mask_poe"],
        meta,
        device="cpu",
    )
    archived = np.load(run / "condition_logits_full.npy", allow_pickle=False)

    max_abs = float(np.max(np.abs(logits - archived)))
    if not np.allclose(logits, archived, rtol=1e-5, atol=1e-6):
        raise RuntimeError(
            f"clean implementation disagrees with archived logits; max_abs={max_abs}"
        )
    print(f"PASS: strict checkpoint load and archived-logit reproduction; max_abs={max_abs:.3e}")


if __name__ == "__main__":
    main()
