# M3 Perturb-seq

Clean implementation of the frozen M3 Perturb-seq development winner on the
Jiang et al. TGF-beta subset.

## Architecture

The model keeps the ordinary M3 generator and changes only:

1. the RNA posterior-mean first layer:
   - coordinates other than 26/27 retain their original trainable weights;
   - condition coordinates 26/27 are parameterised through a fixed,
     unwhitened rank-64 PCA basis;
   - a rank-4 cell-line residual is added:
     `z = C s + d + A[line] @ (B s)`;
2. RNA posterior-mean dropout is `0.5` rather than `0.2`;
3. the shared NT/TGFBR1 classifier receives a centered six-line affine
   log-odds residual.

Cell-line order is fixed to:
`A549, BXPC3, HAP1, HT29, K562, MCF7`.

The implementation has no Codex runtime adapter, no function monkey-patching,
and no query-gradient ablation.

## M3 dependency

`vendor/M3` should point to the `query-cell-holdout` branch of the M3 fork. That
branch differs from official M3 only in preserving original batch IDs for
sample-level query holdouts.

## Required PCA artifacts

Copy these two authenticated development artifacts into `artifacts/pca/`:

- `pca_mean_float64.npy`
- `pca_basis_float64.npy`

They must come from the frozen rank-64 development basis used by the winner.
The basis is not refit by this implementation.

## Validate the clean architecture against the archived winner first

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH="$PWD\src;$PWD\vendor\M3\src"

python .\scripts\validate_against_archive.py `
  --archive-run "C:\Users\ethan\Perturb-seq\Jiang24\experiments\pca_context_head_residual\runs\pca_context_head_residual_rank64_ctx4_drop50_wd5e2_epoch100_seed0"
```

This must:
- strict-load the archived checkpoint into the clean model; and
- reproduce the archived full condition logits within `rtol=1e-5, atol=1e-6`.

Do this before retraining.

## Tests

```powershell
$env:PYTHONPATH="$PWD\src;$PWD\vendor\M3\src"
python -m pytest .\tests -q
```

## Train

Training requires the same sanitized H5AD used by the development winner:
query `gene` values must be absent.

```powershell
python .\scripts\train.py `
  --input-h5ad "PATH\TO\SANITIZED_INPUT.h5ad" `
  --basis-dir ".\artifacts\pca" `
  --output-dir ".\runs\final_seed0"
```

The fixed recipe is seed 0, 100 epochs, batch 256, LR `1e-4`, AdamW weight
decay `0.05`, 2,000 HVGs, rank-64 PCA, context rank 4, and no batch balancing.

## Evaluate Q_train

```powershell
python .\scripts\evaluate.py `
  --run-dir ".\runs\final_seed0" `
  --truth-csv "PATH\TO\query_truth.csv"
```

The evaluator treats TGFBR1/class 1 as positive and reports overall and
per-cell-line AUROC and balanced accuracy.

## Provenance

The exploratory/autonomous experiment machinery is intentionally not part of
this repository. Keep the original `Perturb-seq` workspace frozen as the
historical provenance archive.
