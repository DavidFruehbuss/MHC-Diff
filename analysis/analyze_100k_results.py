#!/usr/bin/env python3
"""
Analyze 100K pMHC-I structure prediction results.

This script analyzes the output from test.py for the 100K dataset (multi-allele).
It computes RMSD statistics for X-ray and PANDORA structures, with breakdowns by:
- Overall statistics
- Per-cluster statistics (G-domain clusters 1-10)
- Per-peptide-length statistics

Usage:
    python analyze_100k_results.py --results-dir /path/to/results [--output-dir /path/to/output]
    
Example:
    python analyze_100k_results.py --results-dir ./checkpoints/100k --output-dir ./analysis_output
"""

import argparse
import os
import pickle
import gzip
import re
import io
from pathlib import Path
from collections import defaultdict

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
    with gzip.open(file_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
    
    # Use custom unpickler to handle CUDA tensors
    return CPU_Unpickler(buffer).load()


def decode_graph_name(name) -> str:
    """Decode graph name from bytes if necessary."""
    if isinstance(name, bytes):
        return name.decode('utf-8')
    return str(name)


def normalize_pdb_id(name: str) -> str:
    """Normalize PDB ID for matching."""
    name = decode_graph_name(name)
    # Extract base name (before any underscore)
    base = name.split('_')[0]
    # Handle PANDORA format: BA-12345 -> BA-12345
    base = re.sub(r'^(BA)[-_]?(\d+)$', r'\1-\2', base, flags=re.IGNORECASE)
    return base.upper()


def is_xray_structure(name: str) -> bool:
    """Check if a structure is X-ray (not PANDORA)."""
    decoded = decode_graph_name(name)
    return not decoded.startswith('BA')


def compute_docked_average(rmse_list: list, cutoff: float = 20.0) -> tuple:
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


def load_cluster_mapping(mapping_file: str) -> dict:
    """Load PDB to cluster mapping from TSV file."""
    pdb_to_cluster = {}
    
    if not os.path.exists(mapping_file):
        print(f"Warning: Cluster mapping file not found: {mapping_file}")
        return pdb_to_cluster
    
    df = pd.read_csv(mapping_file, sep='\t')
    
    # Check for different possible column names
    if 'PDB_ID' in df.columns and 'Cluster_ID' in df.columns:
        for _, row in df.iterrows():
            pdb_id = normalize_pdb_id(str(row['PDB_ID']))
            cluster_id = int(row['Cluster_ID'])
            pdb_to_cluster[pdb_id] = cluster_id
    elif 'PDB_IDs' in df.columns and 'Cluster_ID' in df.columns:
        for _, row in df.iterrows():
            cluster_id = int(row['Cluster_ID'])
            for pdb_id in str(row['PDB_IDs']).split(';'):
                pdb_id = pdb_id.strip()
                if pdb_id:
                    pdb_to_cluster[normalize_pdb_id(pdb_id)] = cluster_id
    
    return pdb_to_cluster


def load_allele_info(allele_file: str) -> dict:
    """Load allele and peptide length information."""
    pdb_to_info = {}
    
    if not os.path.exists(allele_file):
        print(f"Warning: Allele info file not found: {allele_file}")
        return pdb_to_info
    
    df = pd.read_csv(allele_file, sep='\t')
    
    # Common column names
    file_col = None
    allele_col = None
    length_col = None
    
    for col in df.columns:
        if 'file' in col.lower() or col == 'PDB_ID':
            file_col = col
        if 'allele' in col.lower():
            allele_col = col
        if 'peptide' in col.lower() and 'length' in col.lower():
            length_col = col
    
    if file_col and allele_col:
        for _, row in df.iterrows():
            pdb_id = normalize_pdb_id(str(row[file_col]).split('.')[0])
            info = {'allele': row[allele_col] if allele_col else None}
            if length_col:
                info['peptide_length'] = int(row[length_col])
            pdb_to_info[pdb_id] = info
    
    return pdb_to_info


def analyze_samples(samples: dict, pdb_to_cluster: dict = None, pdb_to_info: dict = None) -> dict:
    """Analyze results from a samples file."""
    results = {
        'n_total': 0,
        'n_xray': 0,
        'n_pandora': 0,
    }
    
    # Separate by type
    xray_best, pandora_best = [], []
    xray_avg, pandora_avg = [], []
    
    # Per-cluster results
    cluster_xray_best = defaultdict(list)
    cluster_pandora_best = defaultdict(list)
    cluster_xray_avg = defaultdict(list)
    cluster_pandora_avg = defaultdict(list)
    
    # Per-peptide-length results
    length_xray_best = defaultdict(list)
    length_xray_avg = defaultdict(list)
    
    graph_names = samples.get('graph_name', [])
    rmse_best = samples.get('rmse_best', [])
    rmse_all = samples.get('rmse', [])
    
    n = min(len(graph_names), len(rmse_best))
    results['n_total'] = n
    
    for i in range(n):
        name = graph_names[i]
        pdb_id = normalize_pdb_id(name)
        
        # Get cluster ID
        cluster_id = pdb_to_cluster.get(pdb_id, -1) if pdb_to_cluster else -1
        
        # Get peptide length
        info = pdb_to_info.get(pdb_id, {}) if pdb_to_info else {}
        peptide_length = info.get('peptide_length', -1)
        
        if is_xray_structure(name):
            xray_best.append(rmse_best[i])
            if i < len(rmse_all):
                xray_avg.append(rmse_all[i])
            
            if cluster_id > 0:
                cluster_xray_best[cluster_id].append(rmse_best[i])
                if i < len(rmse_all):
                    cluster_xray_avg[cluster_id].append(rmse_all[i])
            
            if peptide_length > 0:
                length_xray_best[peptide_length].append(rmse_best[i])
                if i < len(rmse_all):
                    length_xray_avg[peptide_length].append(rmse_all[i])
        else:
            pandora_best.append(rmse_best[i])
            if i < len(rmse_all):
                pandora_avg.append(rmse_all[i])
            
            if cluster_id > 0:
                cluster_pandora_best[cluster_id].append(rmse_best[i])
                if i < len(rmse_all):
                    cluster_pandora_avg[cluster_id].append(rmse_all[i])
    
    results['n_xray'] = len(xray_best)
    results['n_pandora'] = len(pandora_best)
    
    # Overall statistics
    if xray_best:
        xray_tensor = torch.tensor(xray_best)
        results['xray_best_mean'] = xray_tensor.mean().item()
        results['xray_best_median'] = xray_tensor.median().item()
        results['xray_best_min'] = xray_tensor.min().item()
        results['xray_best_max'] = xray_tensor.max().item()
        results['xray_best_values'] = xray_tensor
    
    if pandora_best:
        pandora_tensor = torch.tensor(pandora_best)
        results['pandora_best_mean'] = pandora_tensor.mean().item()
        results['pandora_best_median'] = pandora_tensor.median().item()
        results['pandora_best_values'] = pandora_tensor
    
    if xray_avg:
        xray_docked, xray_div_counts = compute_docked_average(xray_avg)
        results['xray_avg_mean'] = xray_docked.mean().item()
        results['xray_avg_median'] = xray_docked.median().item()
        results['xray_avg_values'] = xray_docked
        results['xray_divergent_avg'] = xray_div_counts.mean().item()
    
    if pandora_avg:
        pandora_docked, pandora_div_counts = compute_docked_average(pandora_avg)
        results['pandora_avg_mean'] = pandora_docked.mean().item()
        results['pandora_avg_median'] = pandora_docked.median().item()
        results['pandora_avg_values'] = pandora_docked
        results['pandora_divergent_avg'] = pandora_div_counts.mean().item()
    
    # Per-cluster statistics
    results['per_cluster'] = {}
    for cluster_id in sorted(set(cluster_xray_best.keys()) | set(cluster_pandora_best.keys())):
        cluster_results = {'cluster_id': cluster_id}
        
        if cluster_xray_best[cluster_id]:
            t = torch.tensor(cluster_xray_best[cluster_id])
            cluster_results['xray_best_mean'] = t.mean().item()
            cluster_results['xray_best_median'] = t.median().item()
            cluster_results['n_xray'] = len(cluster_xray_best[cluster_id])
        
        if cluster_pandora_best[cluster_id]:
            t = torch.tensor(cluster_pandora_best[cluster_id])
            cluster_results['pandora_best_mean'] = t.mean().item()
            cluster_results['pandora_best_median'] = t.median().item()
            cluster_results['n_pandora'] = len(cluster_pandora_best[cluster_id])
        
        if cluster_xray_avg[cluster_id]:
            t, div_counts = compute_docked_average(cluster_xray_avg[cluster_id])
            cluster_results['xray_avg_mean'] = t.mean().item()
            cluster_results['xray_avg_median'] = t.median().item()
            cluster_results['xray_divergent_avg'] = div_counts.mean().item()
        
        if cluster_pandora_avg[cluster_id]:
            t, div_counts = compute_docked_average(cluster_pandora_avg[cluster_id])
            cluster_results['pandora_avg_mean'] = t.mean().item()
            cluster_results['pandora_avg_median'] = t.median().item()
            cluster_results['pandora_divergent_avg'] = div_counts.mean().item()
        
        results['per_cluster'][cluster_id] = cluster_results
    
    # Per-peptide-length statistics
    results['per_length'] = {}
    for length in sorted(length_xray_best.keys()):
        length_results = {'peptide_length': length}
        
        if length_xray_best[length]:
            t = torch.tensor(length_xray_best[length])
            length_results['xray_best_mean'] = t.mean().item()
            length_results['xray_best_median'] = t.median().item()
            length_results['n_xray'] = len(length_xray_best[length])
        
        if length_xray_avg[length]:
            t, div_counts = compute_docked_average(length_xray_avg[length])
            length_results['xray_avg_mean'] = t.mean().item()
            length_results['xray_avg_median'] = t.median().item()
            length_results['xray_divergent_avg'] = div_counts.mean().item()
        
        results['per_length'][length] = length_results
    
    return results


def find_sample_files(results_dir: str) -> list:
    """Find all samples.pkl.gz files in the results directory."""
    results_dir = Path(results_dir)
    sample_files = []
    
    # Check for direct sample files
    for sample_file in results_dir.glob('samples*.pkl.gz'):
        sample_files.append(sample_file)
    
    # Check subdirectories
    for sample_file in results_dir.rglob('samples*.pkl.gz'):
        sample_files.append(sample_file)
    
    return sorted(set(sample_files))


def main():
    parser = argparse.ArgumentParser(description='Analyze 100K pMHC structure prediction results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing samples.pkl.gz files')
    parser.add_argument('--output-dir', type=str, default='./analysis_output',
                        help='Directory to save analysis results')
    parser.add_argument('--cluster-mapping', type=str, 
                        default='data_processing/pdb_cluster_mapping.tsv',
                        help='Path to cluster mapping TSV file')
    parser.add_argument('--allele-info', type=str,
                        default='data_processing/mhci_alleles.tsv',
                        help='Path to allele info TSV file')
    parser.add_argument('--cutoff', type=float, default=20.0,
                        help='RMSD cutoff for docked average calculation')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load mappings
    pdb_to_cluster = load_cluster_mapping(args.cluster_mapping)
    pdb_to_info = load_allele_info(args.allele_info)
    
    print(f"Loaded {len(pdb_to_cluster)} PDB-to-cluster mappings")
    print(f"Loaded {len(pdb_to_info)} PDB-to-info mappings")
    
    # Find sample files
    sample_files = find_sample_files(args.results_dir)
    
    if not sample_files:
        print(f"No sample files found in {args.results_dir}")
        return
    
    print(f"Found {len(sample_files)} sample file(s)")
    
    # Analyze each file
    all_results = []
    
    for sample_file in sample_files:
        print(f"\nAnalyzing: {sample_file}")
        
        try:
            samples = load_samples(sample_file)
        except Exception as e:
            print(f"  Error loading file: {e}")
            continue
        
        results = analyze_samples(samples, pdb_to_cluster, pdb_to_info)
        results['file'] = str(sample_file)
        all_results.append(results)
        
        # Print summary
        print(f"  Total samples: {results['n_total']}")
        print(f"  X-ray: {results['n_xray']}, PANDORA: {results['n_pandora']}")
        
        if 'xray_best_mean' in results:
            print(f"  X-ray RMSD (best-of-10): mean={results['xray_best_mean']:.3f} Å, "
                  f"median={results['xray_best_median']:.3f} Å")
        if 'xray_avg_mean' in results:
            print(f"  X-ray RMSD (avg-of-10):  mean={results['xray_avg_mean']:.3f} Å, "
                  f"median={results['xray_avg_median']:.3f} Å")
            if 'xray_divergent_avg' in results:
                print(f"  Divergent samples (>20Å): X-ray avg {results['xray_divergent_avg']:.2f}/10, "
                      f"PANDORA avg {results.get('pandora_divergent_avg', 0):.2f}/10 per structure")
        
        # Per-cluster summary (test clusters: 3, 4, 6)
        if results['per_cluster']:
            print(f"\n  Per-cluster (test clusters 3, 4, 6):")
            for cid in [3, 4, 6]:
                if cid in results['per_cluster']:
                    cr = results['per_cluster'][cid]
                    n_xray = cr.get('n_xray', 0)
                    mean_best = cr.get('xray_best_mean', float('nan'))
                    mean_avg = cr.get('xray_avg_mean', float('nan'))
                    print(f"    Cluster {cid}: n={n_xray}, best={mean_best:.3f} Å, avg={mean_avg:.3f} Å")
        
        # Per-length summary
        if results['per_length']:
            print(f"\n  Per-peptide-length:")
            for length in sorted(results['per_length'].keys()):
                lr = results['per_length'][length]
                n_xray = lr.get('n_xray', 0)
                mean_best = lr.get('xray_best_mean', float('nan'))
                mean_avg = lr.get('xray_avg_mean', float('nan'))
                print(f"    Length {length}: n={n_xray}, best={mean_best:.3f} Å, avg={mean_avg:.3f} Å")
    
    # Overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY (100K Dataset - Multi-allele)")
    print("="*60)
    
    # Combine results across files
    combined_xray_best = []
    combined_xray_avg = []
    
    for r in all_results:
        if 'xray_best_values' in r:
            combined_xray_best.append(r['xray_best_values'])
        if 'xray_avg_values' in r:
            combined_xray_avg.append(r['xray_avg_values'])
    
    if combined_xray_best:
        combined = torch.cat(combined_xray_best)
        print(f"\nCombined X-ray RMSD (best-of-10):")
        print(f"  N:      {len(combined)}")
        print(f"  Mean:   {combined.mean().item():.3f} Å")
        print(f"  Median: {combined.median().item():.3f} Å")
    
    if combined_xray_avg:
        combined = torch.cat(combined_xray_avg)
        print(f"\nCombined X-ray RMSD (avg-of-10):")
        print(f"  N:      {len(combined)}")
        print(f"  Mean:   {combined.mean().item():.3f} Å")
        print(f"  Median: {combined.median().item():.3f} Å")
    
    # Save results
    detailed_path = os.path.join(args.output_dir, '100k_results_detailed.pkl')
    with open(detailed_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nDetailed results saved to: {detailed_path}")
    
    # Save summary CSV
    summary_data = []
    for r in all_results:
        row = {
            'file': r.get('file', ''),
            'n_total': r.get('n_total', 0),
            'n_xray': r.get('n_xray', 0),
            'n_pandora': r.get('n_pandora', 0),
            'xray_best_mean': r.get('xray_best_mean', float('nan')),
            'xray_best_median': r.get('xray_best_median', float('nan')),
            'xray_avg_mean': r.get('xray_avg_mean', float('nan')),
            'xray_avg_median': r.get('xray_avg_median', float('nan')),
            'pandora_best_mean': r.get('pandora_best_mean', float('nan')),
            'pandora_avg_mean': r.get('pandora_avg_mean', float('nan')),
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(args.output_dir, '100k_results_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Summary saved to: {csv_path}")


if __name__ == '__main__':
    main()

