#!/usr/bin/env python3
"""
Analyze 8K pMHC-I structure prediction results.

This script analyzes the output from test.py for the 8K dataset (HLA-A*02:01, 9-mer).
It computes RMSD statistics for X-ray and PANDORA structures, both for best-of-10 
and average-of-10 sampling strategies.

Usage:
    python analyze_8k_results.py --results-dir /path/to/results [--output-dir /path/to/output]
    
Example:
    python analyze_8k_results.py --results-dir ./checkpoints --output-dir ./analysis_output
"""

import argparse
import os
import pickle
import gzip
import re
import io
from pathlib import Path

import torch
import pandas as pd
import numpy as np


class CPU_Unpickler(pickle.Unpickler):
    """Unpickler that maps CUDA tensors to CPU."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_samples(file_path: str) -> dict:
    """Load a samples.pkl.gz file, handling CUDA tensors on CPU."""
    import io
    
    with gzip.open(file_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
    
    # Use custom unpickler to handle CUDA tensors
    return CPU_Unpickler(buffer).load()


def decode_graph_name(name) -> str:
    """Decode graph name from bytes if necessary."""
    if isinstance(name, bytes):
        return name.decode('utf-8')
    return str(name)


def is_xray_structure(name: str) -> bool:
    """Check if a structure is X-ray (not PANDORA)."""
    decoded = decode_graph_name(name)
    return not decoded.startswith('BA')


def compute_docked_average(rmse_list: list, cutoff: float = 10.0) -> tuple:
    """
    Compute docked average: mean RMSD excluding outliers above cutoff.
    This is the average-of-10 metric used in the paper.
    
    Returns:
        avg: tensor of per-sample averages
        divergent_counts: tensor of divergent sample counts per structure
    """
    avg = torch.zeros(len(rmse_list))
    divergent_counts = torch.zeros(len(rmse_list))
    
    for i, rmse in enumerate(rmse_list):
        if isinstance(rmse, torch.Tensor):
            valid = rmse[rmse < cutoff]
            avg[i] = valid.mean() if len(valid) > 0 else rmse.mean()
            # Count how many of the 10 samples are divergent (> cutoff)
            divergent_counts[i] = (rmse > cutoff).sum().item()
        else:
            avg[i] = float(rmse)
    
    return avg, divergent_counts


def analyze_fold(samples: dict, fold_name: str) -> dict:
    """Analyze results for a single fold."""
    results = {
        'fold': fold_name,
        'n_samples': len(samples.get('rmse_best', [])),
    }
    
    # Separate X-ray and PANDORA structures
    xray_best, pandora_best = [], []
    xray_avg, pandora_avg = [], []
    
    graph_names = samples.get('graph_name', [])
    rmse_best = samples.get('rmse_best', [])
    rmse_all = samples.get('rmse', [])
    
    n = min(len(graph_names), len(rmse_best))
    
    for i in range(n):
        name = graph_names[i]
        
        if is_xray_structure(name):
            xray_best.append(rmse_best[i])
            if i < len(rmse_all):
                xray_avg.append(rmse_all[i])
        else:
            pandora_best.append(rmse_best[i])
            if i < len(rmse_all):
                pandora_avg.append(rmse_all[i])
    
    # Best-of-10 statistics
    if xray_best:
        xray_tensor = torch.tensor(xray_best)
        results['xray_best_mean'] = xray_tensor.mean().item()
        results['xray_best_median'] = xray_tensor.median().item()
        results['xray_best_min'] = xray_tensor.min().item()
        results['xray_best_max'] = xray_tensor.max().item()
        results['xray_best_values'] = xray_tensor
        results['n_xray'] = len(xray_best)
    
    if pandora_best:
        pandora_tensor = torch.tensor(pandora_best)
        results['pandora_best_mean'] = pandora_tensor.mean().item()
        results['pandora_best_median'] = pandora_tensor.median().item()
        results['pandora_best_min'] = pandora_tensor.min().item()
        results['pandora_best_max'] = pandora_tensor.max().item()
        results['pandora_best_values'] = pandora_tensor
        results['n_pandora'] = len(pandora_best)
    
    # Average-of-10 statistics (docked average)
    if xray_avg:
        xray_docked, xray_div_counts = compute_docked_average(xray_avg)
        results['xray_avg_mean'] = xray_docked.mean().item()
        results['xray_avg_median'] = xray_docked.median().item()
        results['xray_avg_min'] = xray_docked.min().item()
        results['xray_avg_max'] = xray_docked.max().item()
        results['xray_avg_values'] = xray_docked
        # Average number of divergent samples per structure (out of 10)
        results['xray_divergent_avg'] = xray_div_counts.mean().item()
    
    if pandora_avg:
        pandora_docked, pandora_div_counts = compute_docked_average(pandora_avg)
        results['pandora_avg_mean'] = pandora_docked.mean().item()
        results['pandora_avg_median'] = pandora_docked.median().item()
        results['pandora_avg_min'] = pandora_docked.min().item()
        results['pandora_avg_max'] = pandora_docked.max().item()
        results['pandora_avg_values'] = pandora_docked
        # Average number of divergent samples per structure (out of 10)
        results['pandora_divergent_avg'] = pandora_div_counts.mean().item()
    
    return results


def find_sample_files(results_dir: str) -> list:
    """Find all samples.pkl.gz files in the results directory."""
    results_dir = Path(results_dir)
    sample_files = []
    
    # Check for fold directories
    for fold_dir in sorted(results_dir.glob('fold_*')):
        for sample_file in fold_dir.glob('samples*.pkl.gz'):
            sample_files.append(sample_file)
    
    # Check for direct sample files
    for sample_file in results_dir.glob('samples*.pkl.gz'):
        sample_files.append(sample_file)
    
    return sorted(set(sample_files))


def main():
    parser = argparse.ArgumentParser(description='Analyze 8K pMHC structure prediction results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing samples.pkl.gz files')
    parser.add_argument('--output-dir', type=str, default='./analysis_output',
                        help='Directory to save analysis results')
    parser.add_argument('--cutoff', type=float, default=10.0,
                        help='RMSD cutoff for docked average calculation')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find sample files
    sample_files = find_sample_files(args.results_dir)
    
    if not sample_files:
        print(f"No sample files found in {args.results_dir}")
        print("Looking for files matching: samples*.pkl.gz")
        return
    
    print(f"Found {len(sample_files)} sample file(s)")
    
    # Analyze each fold
    all_results = []
    all_xray_best = []
    all_xray_avg = []
    
    for sample_file in sample_files:
        print(f"\nAnalyzing: {sample_file}")
        
        try:
            samples = load_samples(sample_file)
        except Exception as e:
            print(f"  Error loading file: {e}")
            continue
        
        # Extract fold name from path or filename
        fold_name = sample_file.parent.name if 'fold' in sample_file.parent.name else sample_file.stem
        fold_match = re.search(r'(\d+)', fold_name)
        fold_num = int(fold_match.group(1)) if fold_match else 0
        
        results = analyze_fold(samples, fold_name)
        all_results.append(results)
        
        # Aggregate X-ray values
        if 'xray_best_values' in results:
            all_xray_best.append(results['xray_best_values'])
        if 'xray_avg_values' in results:
            all_xray_avg.append(results['xray_avg_values'])
        
        # Print fold summary
        print(f"  Fold: {fold_name}")
        print(f"  Samples: {results.get('n_samples', 0)}")
        print(f"  X-ray: {results.get('n_xray', 0)}, PANDORA: {results.get('n_pandora', 0)}")
        if 'xray_best_mean' in results:
            print(f"  X-ray RMSD (best-of-10): mean={results['xray_best_mean']:.3f}, "
                  f"median={results['xray_best_median']:.3f}")
        if 'xray_avg_mean' in results:
            print(f"  X-ray RMSD (avg-of-10):  mean={results['xray_avg_mean']:.3f}, "
                  f"median={results['xray_avg_median']:.3f}")
            if 'xray_divergent_avg' in results:
                print(f"  Divergent samples (>10Å): avg {results['xray_divergent_avg']:.2f}/10 per structure")
    
    # Create summary DataFrame
    df_data = []
    for r in all_results:
        row = {
            'fold': r.get('fold', ''),
            'n_samples': r.get('n_samples', 0),
            'n_xray': r.get('n_xray', 0),
            'n_pandora': r.get('n_pandora', 0),
            'xray_best_mean': r.get('xray_best_mean', float('nan')),
            'xray_best_median': r.get('xray_best_median', float('nan')),
            'xray_avg_mean': r.get('xray_avg_mean', float('nan')),
            'xray_avg_median': r.get('xray_avg_median', float('nan')),
            'pandora_best_mean': r.get('pandora_best_mean', float('nan')),
            'pandora_best_median': r.get('pandora_best_median', float('nan')),
            'pandora_avg_mean': r.get('pandora_avg_mean', float('nan')),
            'pandora_avg_median': r.get('pandora_avg_median', float('nan')),
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Print overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY (8K Dataset - HLA-A*02:01 9-mer)")
    print("="*60)
    
    if len(all_xray_best) > 0:
        combined_best = torch.cat(all_xray_best)
        print(f"\nX-ray RMSD (best-of-10) across all folds:")
        print(f"  N:      {len(combined_best)}")
        print(f"  Mean:   {combined_best.mean().item():.3f} Å")
        print(f"  Median: {combined_best.median().item():.3f} Å")
        print(f"  Min:    {combined_best.min().item():.3f} Å")
        print(f"  Max:    {combined_best.max().item():.3f} Å")
    
    if len(all_xray_avg) > 0:
        combined_avg = torch.cat(all_xray_avg)
        print(f"\nX-ray RMSD (avg-of-10) across all folds:")
        print(f"  N:      {len(combined_avg)}")
        print(f"  Mean:   {combined_avg.mean().item():.3f} Å")
        print(f"  Median: {combined_avg.median().item():.3f} Å")
        print(f"  Min:    {combined_avg.min().item():.3f} Å")
        print(f"  Max:    {combined_avg.max().item():.3f} Å")
    
    # Cross-fold averages
    print(f"\nPer-fold averages:")
    print(f"  X-ray best-of-10 mean:   {df['xray_best_mean'].mean():.3f} Å")
    print(f"  X-ray best-of-10 median: {df['xray_best_median'].mean():.3f} Å")
    print(f"  X-ray avg-of-10 mean:    {df['xray_avg_mean'].mean():.3f} Å")
    print(f"  PANDORA best-of-10 mean: {df['pandora_best_mean'].mean():.3f} Å")
    
    # Save results
    csv_path = os.path.join(args.output_dir, '8k_results_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save detailed results
    detailed_path = os.path.join(args.output_dir, '8k_results_detailed.pkl')
    with open(detailed_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Detailed results saved to: {detailed_path}")


if __name__ == '__main__':
    main()

