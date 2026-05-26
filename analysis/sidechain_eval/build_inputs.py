#!/usr/bin/env python3
"""Build CA-only inputs (preserve crystal residue numbering) and ground truth."""

import argparse
import csv
import gzip
import io as iolib
import os
import sys

import h5py
import numpy as np
import torch
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model

REPO_ROOT = os.environ.get("MHC_DIFF_ROOT", "/home/dfruhbus/MHC-Diff")
sys.path.insert(0, REPO_ROOT)

from analysis.analyze_8k_results import CPU_Unpickler


def load_pdb_string(data_dir, graph_index):
    with h5py.File(os.path.join(data_dir, "test.hdf5"), "r") as f5:
        return f5["pdb_strings"][graph_index].decode("utf-8")


def load_best_ca_coords(checkpoints_dir, fold, graph_index, best_sample_idx):
    pkl_path = os.path.join(checkpoints_dir, f"fold_{fold}", "samples.pkl.gz")
    with gzip.open(pkl_path, "rb") as f:
        data = CPU_Unpickler(iolib.BytesIO(f.read())).load()
    coords = data["x_predicted"][graph_index][best_sample_idx]
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    return np.asarray(coords, dtype=float)


def write_ca_preserving_ids(pdb_string, pred_ca_coords, out_path, chain_id="P"):
    """Replace peptide CA coords only; keep original residue numbers (matches benchmark RMSE)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("", iolib.StringIO(pdb_string))
    model = structure[0]
    peptide_chain = model[chain_id]
    new_chain = Chain(chain_id)
    coords = list(pred_ca_coords)
    for i, residue in enumerate(peptide_chain):
        if i >= len(coords) or "CA" not in residue:
            continue
        ca = residue["CA"].copy()
        ca.set_coord(coords[i])
        new_res = Residue(residue.get_id(), residue.get_resname(), "")
        new_res.add(ca)
        new_chain.add(new_res)
    model.detach_child(chain_id)
    model.add(new_chain)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)


def build_peptide_backbone_chain(gt_pdb, pred_ca_pdb, chain_id="P"):
    """Peptide chain with crystal N,C,O and predicted CA."""
    parser = PDBParser(QUIET=True)
    gt = parser.get_structure("gt", gt_pdb)
    pred = parser.get_structure("pr", pred_ca_pdb)
    gt_res = [r for r in gt[0][chain_id] if "CA" in r]
    pred_res = [r for r in pred[0][chain_id] if "CA" in r]
    new_chain = Chain(chain_id)
    for gr, pr in zip(gt_res, pred_res):
        new_res = Residue(gr.get_id(), gr.get_resname(), "")
        for name in ("N", "CA", "C", "O"):
            if name == "CA":
                if "CA" not in pr:
                    continue
                atom = pr["CA"].copy()
            elif name in gr:
                atom = gr[name].copy()
            else:
                continue
            new_res.add(atom)
        new_chain.add(new_res)
    return new_chain


def write_peptide_backbone_for_packing(gt_pdb, pred_ca_pdb, out_pdb, chain_id="P"):
    """Peptide-only PDB (N,C,O from crystal, CA from model) for DLPacker."""
    new_chain = build_peptide_backbone_chain(gt_pdb, pred_ca_pdb, chain_id)
    structure = Structure("pep")
    structure.add(Model(0))
    structure[0].add(new_chain)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb)


def write_bb_complex(gt_pdb, pred_ca_pdb, out_pdb, chain_id="P"):
    """Full complex: crystal MHC + peptide backbone (for PDBFixer sidechain addition)."""
    parser = PDBParser(QUIET=True)
    new_chain = build_peptide_backbone_chain(gt_pdb, pred_ca_pdb, chain_id)
    structure = parser.get_structure("bb", gt_pdb)
    model = structure[0]
    if chain_id in [c.id for c in model]:
        model.detach_child(chain_id)
    model.add(new_chain)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default=os.path.join(os.path.dirname(__file__), "selected_50.csv"))
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data", "pmhc_xray_8K_aligned", "folds"))
    parser.add_argument("--checkpoints-dir", default=os.path.join(REPO_ROOT, "checkpoints"))
    parser.add_argument("--out-dir", default=os.path.dirname(__file__))
    args = parser.parse_args()

    inputs_ca = os.path.join(args.out_dir, "inputs_ca")
    ground_truth = os.path.join(args.out_dir, "ground_truth")
    packer_in = os.path.join(args.out_dir, "inputs_packer_peptide")
    inputs_bb = os.path.join(args.out_dir, "inputs_bb")
    os.makedirs(inputs_ca, exist_ok=True)
    os.makedirs(ground_truth, exist_ok=True)
    os.makedirs(packer_in, exist_ok=True)
    os.makedirs(inputs_bb, exist_ok=True)

    with open(args.selection, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        pdb_id = row["pdb_id"]
        fold = int(row["fold"])
        graph_index = int(row["graph_index"])
        best_sample_idx = int(row["best_sample_idx"])

        data_dir = os.path.join(args.data_root, f"fold_{fold}")
        pdb_string = load_pdb_string(data_dir, graph_index)
        coords = load_best_ca_coords(args.checkpoints_dir, fold, graph_index, best_sample_idx)

        gt_path = os.path.join(ground_truth, f"{pdb_id}.pdb")
        ca_path = os.path.join(inputs_ca, f"{pdb_id}.pdb")
        with open(gt_path, "w") as out:
            out.write(pdb_string)
        write_ca_preserving_ids(pdb_string, coords, ca_path)
        packer_path = os.path.join(packer_in, f"{pdb_id}_peptide.pdb")
        bb_path = os.path.join(inputs_bb, f"{pdb_id}.pdb")
        write_peptide_backbone_for_packing(gt_path, ca_path, packer_path)
        write_bb_complex(gt_path, ca_path, bb_path)
        print(f"Built {pdb_id} fold={fold} ca_rmsd_best={row['ca_rmsd_best']}")

    print(f"Wrote {len(rows)} structures")


if __name__ == "__main__":
    main()
