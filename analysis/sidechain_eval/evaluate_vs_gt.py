#!/usr/bin/env python3
"""Evaluate sidechain reconstruction vs crystal ground truth."""

import argparse
import csv
import os
import sys

REPO_ROOT = os.environ.get("MHC_DIFF_ROOT", "/home/dfruhbus/MHC-Diff")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from rmsd_utils import compute_peptide_rmsds_vs_gt, summarize


def fmt(v):
    return "" if v is None else f"{v:.3f}"


def evaluate_method(selection_csv, gt_dir, pred_dir, pred_suffix, method_name):
    rows = []
    with open(selection_csv, newline="") as f:
        selection = list(csv.DictReader(f))

    for row in selection:
        pdb_id = row["pdb_id"]
        gt_path = os.path.join(gt_dir, f"{pdb_id}.pdb")
        pred_path = os.path.join(pred_dir, f"{pdb_id}{pred_suffix}")
        if not os.path.isfile(pred_path):
            print(f"[SKIP] missing {pred_path}")
            continue
        m = compute_peptide_rmsds_vs_gt(gt_path, pred_path)
        out = {
            "pdb_id": pdb_id,
            "method": method_name,
            "fold": row["fold"],
            "ca_rmsd_model": row.get("ca_rmsd_best", ""),
            "pred_chain": m.get("pred_chain", ""),
            "ca_rmsd": fmt(m["ca_rmsd"]),
            "bb_rmsd": fmt(m["bb_rmsd"]),
            "sc_rmsd": fmt(m["sc_rmsd"]),
            "all_heavy_rmsd": fmt(m["all_heavy_rmsd"]),
            "n_bb": m["n_bb"],
            "n_sc": m["n_sc"],
            "n_all": m["n_all"],
        }
        rows.append(out)
        print(
            f"[{method_name}] {pdb_id} chain={out['pred_chain']}: "
            f"CA={out['ca_rmsd']} BB={out['bb_rmsd']} SC={out['sc_rmsd']} ALL={out['all_heavy_rmsd']}"
        )

    summary = {
        "method": method_name,
        "pdb_id": "SUMMARY",
        "fold": "",
        "ca_rmsd_model": "",
        "pred_chain": "",
        "n_bb": "",
        "n_sc": "",
        "n_all": "",
    }
    for key in ("ca_rmsd", "bb_rmsd", "sc_rmsd", "all_heavy_rmsd"):
        vals = [float(r[key]) for r in rows if r.get(key)]
        mean, median = summarize(vals)
        summary[key] = fmt(mean)
        summary[f"{key}_mean"] = fmt(mean)
        summary[f"{key}_median"] = fmt(median)

    return rows, summary


def write_results(path, rows, summary):
    if not rows:
        print(f"No rows to write for {path}")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        w.writerow({k: summary.get(k, "") for k in fieldnames})
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default=os.path.dirname(__file__))
    parser.add_argument("--selection", default=os.path.join(os.path.dirname(__file__), "selected_50.csv"))
    parser.add_argument(
        "--methods", nargs="+", default=["ca_input", "pdbfixer"],
        choices=["ca_input", "pdbfixer", "dlpacker"],
    )
    args = parser.parse_args()

    gt_dir = os.path.join(args.eval_dir, "ground_truth")
    method_paths = {
        "ca_input": (os.path.join(args.eval_dir, "inputs_ca"), ".pdb"),
        "pdbfixer": (os.path.join(args.eval_dir, "inputs_bb_new"), "_full_complex_min.pdb"),
        "dlpacker": (os.path.join(args.eval_dir, "outputs_dlpacker"), "_merged.pdb"),
    }

    all_summaries = []
    for method in args.methods:
        pred_dir, suffix = method_paths[method]
        rows, summary = evaluate_method(
            args.selection, gt_dir, pred_dir, suffix, method
        )
        write_results(os.path.join(args.eval_dir, f"results_{method}.csv"), rows, summary)
        all_summaries.append(summary)

    if len(all_summaries) > 1:
        comp_path = os.path.join(args.eval_dir, "results_comparison.csv")
        keys = ["method", "ca_rmsd", "bb_rmsd", "sc_rmsd", "all_heavy_rmsd"]
        with open(comp_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for s in all_summaries:
                w.writerow({k: s.get(k, "") for k in keys})
        print(f"Wrote {comp_path}")

    summary_txt = os.path.join(args.eval_dir, "results_summary.txt")
    with open(summary_txt, "w") as f:
        for s in all_summaries:
            f.write(
                f"{s['method']}: CA={s.get('ca_rmsd', 'NA')} A, "
                f"BB={s.get('bb_rmsd', 'NA')} A, "
                f"SC={s.get('sc_rmsd', 'NA')} A, "
                f"ALL={s.get('all_heavy_rmsd', 'NA')} A (mean vs crystal)\n"
            )
    print(f"Wrote {summary_txt}")


if __name__ == "__main__":
    main()
