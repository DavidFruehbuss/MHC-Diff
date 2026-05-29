# Sidechain evaluation (PDBFixer vs DLPacker)

Benchmarks sidechain reconstruction on 50 X-ray structures from the 8K
checkpoint set (lowest best-of-10 Cα RMSD per structure).

| Method | Script | Description |
|---|---|---|
| **CA-only input** | — | Baseline: predicted Cα only (no sidechains) |
| **PDBFixer + OpenMM** | `../../adding_sidechains.py` | Physics-based: add missing atoms, minimize with restrained backbone |
| **DLPacker** | `run_dlpacker.py` | Deep-learning rotamer packing on peptide backbone |

## Pipeline

```bash
export MHC_DIFF_ROOT=/path/to/MHC-Diff

# 1. Select 50 lowest-CA-RMSD X-ray structures from 8K CV folds
python analysis/sidechain_eval/select_subset.py \
    --checkpoints-dir checkpoints/pmhc_8K_xray_aligned_egnn_1 \
    --output selection.csv --n 50

# 2. Build CA-only inputs + ground-truth crystal PDBs
python analysis/sidechain_eval/build_inputs.py \
    --selection selection.csv --data-dir data/8k --out-dir work/inputs

# 3a. PDBFixer path (uses full backbone when N,C,O present)
python adding_sidechains.py work/inputs_bb/

# 3b. DLPacker path (requires DLPacker installed + weights)
python analysis/sidechain_eval/run_dlpacker.py \
    --input-dir work/inputs_ca --gt-dir work/ground_truth --out-dir work/outputs_dlpacker

# 4. Score all methods vs crystal (Cα, backbone, sidechain, all-heavy RMSD)
python analysis/sidechain_eval/evaluate_vs_gt.py \
    --selection selection.csv --gt-dir work/ground_truth \
    --methods ca,pdbfixer,dlpacker --work-dir work/
```

Large runtime outputs live on scratch, e.g.
`/gpfs/scratch1/shared/dfruhbus/MHC-Diff_sidechain_eval/`.

## Slurm

```bash
export MHC_DIFF_ROOT=/path/to/MHC-Diff
sbatch analysis/sidechain_eval/run_pdbfixer.job
sbatch analysis/sidechain_eval/run_dlpacker.job
```

Set `EVAL` in the job scripts to your scratch eval directory.

## Dependencies

- **PDBFixer + OpenMM**: conda `openmm`, `pdbfixer`
- **DLPacker**: separate install; set weights path in `run_dlpacker.py`
- **BioPython**, **h5py**, **torch** (for `build_inputs.py` / `select_subset.py`)

## Metrics

`rmsd_utils.py` reports peptide RMSD after MHC superposition:
Cα, backbone (N/CA/C/O), sidechain (non-backbone heavy), all-heavy.

Committed reference numbers: `reference_results/`.

## Appendix

- `APPENDIX_sidechain_benchmark.tex` — LaTeX table for paper appendix
- `rebuttal_text.md` — draft rebuttal paragraph
