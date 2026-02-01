# MHC-Diff Result Analysis

This folder contains scripts for analyzing the output of `test.py`, which generates structure predictions using the trained MHC-Diff model.

## Scripts

### `analyze_8k_results.py`

Analyzes results from the 8K dataset (10-fold cross-validation on HLA-A*02:01 9-mer peptides).

**Features:**
- Computes RMSD statistics for X-ray and PANDORA structures
- Calculates both best-of-10 and average-of-10 (ensemble) metrics
- Aggregates results across all 10 folds
- Outputs summary statistics and per-fold breakdowns

**Usage:**
```bash
python analysis/analyze_8k_results.py \
    --results-dir ./checkpoints \
    --output-dir ./analysis_output
```

**Expected input:** `samples.pkl.gz` files in fold directories (e.g., `checkpoints/fold_1/samples.pkl.gz`)

**Output:**
- `8k_results_summary.csv`: Per-fold statistics table
- `8k_results_detailed.pkl`: Full results with RMSD distributions

---

### `analyze_100k_results.py`

Analyzes results from the 100K dataset (multi-allele, G-domain clustered).

**Features:**
- Overall RMSD statistics for X-ray vs PANDORA structures
- Per-cluster breakdown (G-domain clusters 1-10)
- Per-peptide-length analysis
- Both best-of-10 and average-of-10 metrics

**Usage:**
```bash
python analysis/analyze_100k_results.py \
    --results-dir ./checkpoints/100k \
    --output-dir ./analysis_output \
    --cluster-mapping data_processing/pdb_cluster_mapping.tsv \
    --allele-info data_processing/mhci_alleles.tsv
```

**Expected input:** `samples.pkl.gz` files (e.g., `samples_test.pkl.gz`)

**Output:**
- `100k_results_summary.csv`: Overall statistics
- `100k_results_detailed.pkl`: Full results with per-cluster and per-length breakdowns

---

## Workflow

After running inference with `test.py`:

1. **Run analysis:**
   ```bash
   # For 8K results (10-fold CV)
   python analysis/analyze_8k_results.py --results-dir ./checkpoints --output-dir ./results/analysis
   
   # For 100K results
   python analysis/analyze_100k_results.py --results-dir ./checkpoints/100k --output-dir ./results/analysis
   ```

2. **View summary:**
   The scripts print a summary to stdout and save CSV files for further analysis.

---

## Metrics

### Best-of-10
For each test structure, 10 samples are generated. The best-of-10 metric reports the minimum RMSD among the 10 samples.

### Average-of-10 (Ensemble Average)
The average-of-10 metric computes the mean RMSD across all 10 samples, excluding outliers above a cutoff (default: 10 Å for 8K, 20 Å for 100K).

### X-ray vs PANDORA
- **X-ray structures**: Experimentally determined structures from the PDB
- **PANDORA structures**: Computationally modeled structures (identified by 'BA' prefix)

---

## Dependencies

- Python 3.8+
- PyTorch
- pandas
- numpy

---

## Output Format

The `samples.pkl.gz` files from `test.py` contain:

```python
{
    'graph_name': list,      # Structure identifiers
    'x_target': dict,        # Ground truth coordinates
    'x_predicted': dict,     # Predicted coordinates (10 per sample)
    'h': dict,               # Node features
    'rmse': list,            # All 10 RMSD values per sample
    'rmse_mean': tensor,     # Mean RMSD per sample
    'rmse_best': tensor,     # Best (min) RMSD per sample
}
```

