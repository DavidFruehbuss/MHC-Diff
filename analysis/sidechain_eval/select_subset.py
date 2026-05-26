#!/usr/bin/env python3
"""Select X-ray subset with lowest best-of-10 CA RMSD for sidechain evaluation."""

import argparse
import csv
import gzip
import io
import os
import sys

REPO_ROOT = os.environ.get("MHC_DIFF_ROOT", "/home/dfruhbus/MHC-Diff")
sys.path.insert(0, REPO_ROOT)

from analysis.analyze_8k_results import CPU_Unpickler, decode_graph_name, is_xray_structure


def load_fold_xray(checkpoints_dir, fold):
    pkl_path = os.path.join(checkpoints_dir, f"fold_{fold}", "samples.pkl.gz")
    if not os.path.exists(pkl_path):
        return []
    with gzip.open(pkl_path, "rb") as f:
        data = CPU_Unpickler(io.BytesIO(f.read())).load()
    entries = []
    rmse_all = data.get("rmse", [])
    for idx, name in enumerate(data["graph_name"]):
        pdb_id = decode_graph_name(name)
        if not is_xray_structure(pdb_id):
            continue
        rmse_tensor = rmse_all[idx] if idx < len(rmse_all) else None
        if rmse_tensor is not None and hasattr(rmse_tensor, "argmin"):
            best_sample_idx = int(rmse_tensor.argmin().item())
            ca_rmsd_best = float(rmse_tensor.min().item())
        else:
            best_sample_idx = 0
            ca_rmsd_best = float("inf")
        entries.append(
            {
                "pdb_id": pdb_id,
                "fold": fold,
                "graph_index": idx,
                "best_sample_idx": best_sample_idx,
                "ca_rmsd_best": ca_rmsd_best,
            }
        )
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Select lowest-CA-RMSD X-ray structures (best-of-10 per structure)"
    )
    parser.add_argument("--checkpoints-dir", default=os.path.join(REPO_ROOT, "checkpoints"))
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "selected_50.csv"))
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--fold-min", type=int, default=2)
    parser.add_argument("--fold-max", type=int, default=10)
    args = parser.parse_args()

    seen = {}
    for fold in range(args.fold_min, args.fold_max + 1):
        for entry in load_fold_xray(args.checkpoints_dir, fold):
            if entry["pdb_id"] not in seen or entry["ca_rmsd_best"] < seen[entry["pdb_id"]]["ca_rmsd_best"]:
                seen[entry["pdb_id"]] = entry

    pool = sorted(seen.values(), key=lambda e: e["ca_rmsd_best"])
    print(f"Unique X-ray structures in folds {args.fold_min}-{args.fold_max}: {len(pool)}")
    if len(pool) < args.n:
        raise SystemExit(f"Only {len(pool)} structures available; reduce --n")

    selected = pool[: args.n]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["pdb_id", "fold", "graph_index", "best_sample_idx", "ca_rmsd_best"],
        )
        w.writeheader()
        w.writerows(selected)

    rmsds = [e["ca_rmsd_best"] for e in selected]
    print(f"Wrote {len(selected)} structures to {args.output}")
    print(f"CA RMSD range: {min(rmsds):.3f} - {max(rmsds):.3f} A (mean {sum(rmsds)/len(rmsds):.3f})")


if __name__ == "__main__":
    main()
