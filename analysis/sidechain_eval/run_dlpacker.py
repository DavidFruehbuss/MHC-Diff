#!/usr/bin/env python3
"""Run DLPacker on peptide backbone inputs; merge with crystal MHC."""

import argparse
import copy
import csv
import os
import sys

from Bio.PDB import PDBIO, PDBParser

REPO_ROOT = os.environ.get("MHC_DIFF_ROOT", "/home/dfruhbus/MHC-Diff")
sys.path.insert(0, REPO_ROOT)

PEPTIDE_CHAIN = "P"


def merge_peptide_with_mhc(gt_pdb, peptide_pdb, out_pdb, pep_chain=PEPTIDE_CHAIN):
    parser = PDBParser(QUIET=True)
    gt = parser.get_structure("gt", gt_pdb)
    pep = parser.get_structure("pep", peptide_pdb)
    model = gt[0]
    if pep_chain in [c.id for c in model]:
        model.detach_child(pep_chain)
    pchain = None
    if pep_chain in [c.id for c in model]:
        pchain = pep[0][pep_chain]
    else:
        for chain in pep[0]:
            n = sum(1 for r in chain if r.get_resname() in {
                "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
            })
            if 8 <= n <= 11:
                pchain = chain
                break
    if pchain is None:
        pchain = list(pep[0].get_chains())[0]
    rcopy = copy.deepcopy(pchain)
    rcopy.id = pep_chain
    model.add(rcopy)
    io = PDBIO()
    io.set_structure(gt)
    io.save(out_pdb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default=os.path.dirname(__file__))
    parser.add_argument("--selection", default=os.path.join(os.path.dirname(__file__), "selected_50.csv"))
    args = parser.parse_args()

    from dlpacker import DLPacker

    packer_in = os.path.join(args.eval_dir, "inputs_packer_peptide")
    gt_dir = os.path.join(args.eval_dir, "ground_truth")
    work = os.path.join(args.eval_dir, "dlpacker_work")
    out_dir = os.path.join(args.eval_dir, "outputs_dlpacker")
    os.makedirs(work, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    with open(args.selection, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        pdb_id = row["pdb_id"]
        pep_in = os.path.join(packer_in, f"{pdb_id}_peptide.pdb")
        pep_out = os.path.join(work, f"{pdb_id}_peptide_packed.pdb")
        merged = os.path.join(out_dir, f"{pdb_id}_merged.pdb")
        gt_path = os.path.join(gt_dir, f"{pdb_id}.pdb")

        dlp = DLPacker(pep_in)
        dlp.reconstruct_protein(order="sequence", output_filename=pep_out)
        merge_peptide_with_mhc(gt_path, pep_out, merged)
        print(f"[OK] {pdb_id} -> {merged}")

    print(f"Done: {len(rows)} structures in {out_dir}")


if __name__ == "__main__":
    main()
