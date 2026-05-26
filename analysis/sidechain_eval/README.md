# Sidechain evaluation (PDBFixer vs DLPacker)

Benchmarks sidechain reconstruction on 50 X-ray structures from the 8K checkpoint set (lowest best-of-10 Cα RMSD per structure).

## Pipeline

1. `select_subset.py` — pick 50 structures from folds 2–10 checkpoints
2. `build_inputs.py` — ground truth, CA-only inputs, fixed-backbone inputs (`inputs_bb/`, `inputs_packer_peptide/`)
3. `adding_sidechains.py` on `inputs_bb/` — PDBFixer + OpenMM (repo root)
4. `run_dlpacker.py` — DLPacker packing + merge with crystal MHC
5. `evaluate_vs_gt.py` — RMSD vs crystal (`reference_results/` for committed summary)

Large runtime outputs live on scratch, e.g. `/gpfs/scratch1/shared/dfruhbus/MHC-Diff_sidechain_eval/`.

## Slurm

```bash
export MHC_DIFF_ROOT=/path/to/MHC-Diff
sbatch analysis/sidechain_eval/run_pdbfixer.job
sbatch analysis/sidechain_eval/run_dlpacker.job
```

Set `EVAL` in the job scripts to your scratch eval directory.

## Appendix

- `APPENDIX_sidechain_benchmark.tex` — LaTeX table for paper appendix
- `rebuttal_text.md` — draft rebuttal paragraph
