"""Data preparation for the fixed Jiang24 TGF-beta half-holdout experiment."""

from __future__ import annotations

import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset, random_split

from .model import LINE_ORDER


class QueryDataset(TorchDataset):
    """Training rows with explicit cell-line routing and R/Q supervision mask."""

    def __init__(
        self,
        data: torch.Tensor,
        mask_recon: torch.Tensor,
        mask_poe: torch.Tensor,
        celltype: torch.Tensor,
        batch: torch.Tensor,
        conditions: list[torch.Tensor],
        is_reference: torch.Tensor,
        line_ids: torch.Tensor,
    ) -> None:
        self.data = data
        self.mask_recon = mask_recon
        self.mask_poe = mask_poe
        self.celltype = celltype
        self.batch = batch
        self.conditions = conditions
        self.is_reference = is_reference
        self.line_ids = line_ids

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        return {
            "data": self.data[index],
            "mask_recon": self.mask_recon[index],
            "mask_poe": self.mask_poe[index],
            "label": self.celltype[index],
            "batch": self.batch[index],
            "conditions": [c[index] for c in self.conditions],
            "all_train_query_info": self.is_reference[index],
            "line_id": self.line_ids[index],
            "row_id": torch.tensor(index, dtype=torch.long),
        }


@dataclass
class PreparedData:
    train_loader: DataLoader
    val_loader: DataLoader
    full_data: torch.Tensor
    full_batch: torch.Tensor
    full_mask_poe: torch.Tensor
    metadata: pd.DataFrame
    train_indices: list[int]
    val_indices: list[int]
    n_unique_batch: int
    hvg_names: list[str]
    split_counts: dict[str, int]


def _process_rna_counts(count: torch.Tensor, count_list, hvg_num: int = 2000):
    """Match M3 process_count_matrix while also returning the selected HVG mask."""
    from m3._engine.util import process_highly_variable_genes

    count_valid = [x for x in count_list if x is not None]
    count_concat = torch.cat(count_valid, dim=0)
    nonzero_mask = count_concat.sum(dim=1) != 0
    count_nonzero = count_concat[nonzero_mask]
    _, hvg_mask = process_highly_variable_genes(count_nonzero, hvg_num)
    hvg_mask = np.asarray(hvg_mask, dtype=bool)
    return count[:, hvg_mask], hvg_mask


def prepare_h5ad(
    input_h5ad: Path | str,
    *,
    batch_size: int = 256,
    seed: int = 0,
    val_percentage: float = 0.1,
    condition_key: str = "gene",
    celltype_key: str = "cell_type",
    batch_key: str = "batch",
    query_key: str = "query_split",
    query_value: str = "query",
) -> PreparedData:
    """Build the exact RNA-only R/Q substrate used by the winner.

    The input must be the sanitized development H5AD: query condition labels
    are absent. A reference condition placeholder is inserted only after that
    absence is verified; Q condition CE remains masked during training.
    """
    import scanpy as sc
    from m3._dataset import Dataset
    from m3 import _bridge
    from m3._engine.util import (
        convert_to_longtensor,
        fill_and_concat_available_lists,
        get_ref_query_data,
        load_and_merge_metadata,
        load_if_available,
        process_ref_count,
        setup_seed,
    )

    input_h5ad = Path(input_h5ad).resolve()
    adata = sc.read_h5ad(input_h5ad)
    required = {"cell_id", batch_key, celltype_key, query_key, condition_key}
    missing = required - set(adata.obs.columns)
    if missing:
        raise ValueError(f"input H5AD is missing required obs columns: {sorted(missing)}")
    if adata.obs["cell_id"].astype(str).duplicated().any():
        raise ValueError("cell_id must be unique")

    qmask_original = adata.obs[query_key].astype(str).eq(query_value).to_numpy()
    if qmask_original.sum() != 7859:
        raise ValueError(f"expected 7,859 external query cells, got {int(qmask_original.sum())}")
    if adata.obs.loc[qmask_original, condition_key].notna().any():
        raise ValueError("query condition truth must be absent from the sanitized training H5AD")

    reference_vocab = sorted(
        adata.obs.loc[~qmask_original, condition_key].dropna().astype(str).unique()
    )
    if reference_vocab != ["NT", "TGFBR1"]:
        raise ValueError(f"expected reference condition vocabulary ['NT','TGFBR1'], got {reference_vocab}")

    # Match M3.train(): fill an unlabelled Q condition with a reference placeholder
    # solely so the categorical tensor can be constructed. The Q CE mask is zero.
    obs = adata.obs.copy()
    obs[condition_key] = obs[condition_key].astype(object)
    obs.loc[qmask_original, condition_key] = reference_vocab[0]

    ds = Dataset(
        modalities={"rna": sp.csr_matrix(adata.X)},
        obs=obs,
        var={"rna": adata.var_names.copy()},
        present={"rna": np.ones(adata.n_obs, dtype=bool)},
    )

    setup_seed(seed)
    marshalled = _bridge.marshal(ds)
    try:
        metadata_paths = marshalled["metadata_paths"]
        modality_paths = marshalled["modality_paths"]

        label, batch = load_and_merge_metadata(metadata_paths)
        celltype = convert_to_longtensor(label[celltype_key])
        conditions = [convert_to_longtensor(label[condition_key])]

        rna_list = load_if_available(modality_paths["rna"])
        rna, _, _ = fill_and_concat_available_lists(rna_list, None, None)
        rna, hvg_mask = _process_rna_counts(rna, rna_list, 2000)

        if int(hvg_mask.sum()) != 2000:
            raise ValueError("M3 preprocessing did not select exactly 2,000 HVGs")
        hvg_names = [str(x) for x in adata.var_names[hvg_mask]]

        query_keep = torch.tensor(
            label[query_key].astype(str).eq(query_value).to_numpy(),
            dtype=torch.bool,
        )
        ref_keep = ~query_keep

        ref = get_ref_query_data(
            batch,
            conditions,
            celltype,
            rna,
            None,
            None,
            label,
            keep_mask=ref_keep,
            preserve_batch_codes=True,
        )
        query = get_ref_query_data(
            batch,
            conditions,
            celltype,
            rna,
            None,
            None,
            label,
            keep_mask=query_keep,
            preserve_batch_codes=True,
        )

        (
            ref_b,
            ref_c,
            ref_cty,
            ref_rna,
            _,
            _,
            ref_meta,
        ) = ref
        (
            query_b,
            query_c,
            query_cty,
            query_rna,
            _,
            _,
            query_meta,
        ) = query

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ref_mpoe, ref_mrec, ref_data_list = [], [], []
        ref_mpoe, ref_mrec, ref_data_list, ref_batch_mask = process_ref_count(
            ref_rna, device, ref_mpoe, ref_mrec, ref_data_list
        )
        query_mpoe, query_mrec, query_data_list = [], [], []
        query_mpoe, query_mrec, query_data_list, query_batch_mask = process_ref_count(
            query_rna, device, query_mpoe, query_mrec, query_data_list
        )

        ref_data = torch.nan_to_num(ref_data_list[0], nan=0.0)
        query_data = torch.nan_to_num(query_data_list[0], nan=0.0)
        ref_mask_poe = torch.cat(
            [ref_mpoe[0], ref_batch_mask[:, 1].unsqueeze(1).to(device)],
            dim=1,
        )
        query_mask_poe = torch.cat(
            [query_mpoe[0], query_batch_mask[:, 1].unsqueeze(1).to(device)],
            dim=1,
        )
        ref_mask_recon = ref_mrec[0]
        query_mask_recon = query_mrec[0]

        full_data = torch.cat([ref_data, query_data], dim=0)
        full_mask_recon = torch.cat([ref_mask_recon, query_mask_recon], dim=0)
        full_mask_poe = torch.cat([ref_mask_poe, query_mask_poe], dim=0)
        full_celltype = torch.cat([ref_cty, query_cty], dim=0)
        full_batch = torch.cat([ref_b, query_b], dim=0)
        full_conditions = [
            torch.cat([r.reshape(-1), q.to(dtype=r.dtype).reshape(-1)], dim=0)
            for r, q in zip(ref_c, query_c)
        ]
        is_reference = torch.cat(
            [
                torch.ones(len(ref_data), dtype=torch.long),
                torch.zeros(len(query_data), dtype=torch.long),
            ]
        )

        metadata = pd.concat(
            [ref_meta.reset_index(drop=True), query_meta.reset_index(drop=True)],
            ignore_index=True,
        )
        metadata["cell_id"] = metadata["cell_id"].astype(str)
        if len(metadata) != 15715 or metadata["cell_id"].duplicated().any():
            raise ValueError("expected the unique 15,715-cell substrate")

        line_lookup = {name: i for i, name in enumerate(LINE_ORDER)}
        unknown_lines = set(metadata[celltype_key].astype(str)) - set(LINE_ORDER)
        if unknown_lines:
            raise ValueError(f"unknown cell lines: {sorted(unknown_lines)}")
        line_ids = torch.tensor(
            [line_lookup[x] for x in metadata[celltype_key].astype(str)],
            dtype=torch.long,
        )

        dataset = QueryDataset(
            full_data,
            full_mask_recon,
            full_mask_poe,
            full_celltype,
            full_batch,
            full_conditions,
            is_reference,
            line_ids,
        )

        total_len = len(dataset)
        val_len = int(val_percentage * total_len)
        train_len = total_len - val_len
        train_subset, val_subset = random_split(dataset, [train_len, val_len])
        train_indices = list(train_subset.indices)
        val_indices = list(val_subset.indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        metadata = metadata[
            ["cell_id", batch_key, celltype_key, query_key]
        ].copy()
        metadata["internal_split"] = None
        for idx in train_indices:
            metadata.loc[idx, "internal_split"] = (
                "R_train" if int(is_reference[idx]) == 1 else "Q_train"
            )
        for idx in val_indices:
            metadata.loc[idx, "internal_split"] = (
                "R_val" if int(is_reference[idx]) == 1 else "Q_val"
            )

        split_counts = metadata["internal_split"].value_counts().to_dict()
        expected = {"R_train": 7107, "Q_train": 7037, "R_val": 749, "Q_val": 822}
        if split_counts != expected:
            raise ValueError(
                f"internal split differs from the frozen seed-0 split: {split_counts} != {expected}"
            )

        n_unique_batch = int(full_batch.max().item()) + 1
        if n_unique_batch != 2:
            raise ValueError(f"expected two original batches, got {n_unique_batch}")

        return PreparedData(
            train_loader=train_loader,
            val_loader=val_loader,
            full_data=full_data,
            full_batch=full_batch,
            full_mask_poe=full_mask_poe,
            metadata=metadata,
            train_indices=train_indices,
            val_indices=val_indices,
            n_unique_batch=n_unique_batch,
            hvg_names=hvg_names,
            split_counts=expected,
        )
    finally:
        shutil.rmtree(marshalled["tmpdir"], ignore_errors=True)

