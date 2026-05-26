"""Peptide RMSD vs crystal ground truth."""

import numpy as np
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.Polypeptide import is_aa

import os
import sys

REPO_ROOT = os.environ.get("MHC_DIFF_ROOT", "/home/dfruhbus/MHC-Diff")
sys.path.insert(0, REPO_ROOT)

from adding_sidechains import (
    PEPTIDE_CHAIN_ID,
    collect_atom_dict,
    coords_from_atoms,
    match_atom_lists,
    rmsd_after_rotran,
    count_std_residues,
    residue_key,
)


def load_structure(pdb_path):
    return PDBParser(QUIET=True).get_structure("s", pdb_path)


def guess_peptide_chain_id(structure, expected_len=None, preferred=(PEPTIDE_CHAIN_ID,)):
    model = structure[0]
    chain_ids = [c.id for c in model]
    for cid in preferred:
        if cid in chain_ids:
            return cid
    best_cid = None
    best_score = None
    for chain in model:
        n = count_std_residues(structure, chain.id)
        if n == 0:
            continue
        score = (abs(n - expected_len), n) if expected_len is not None else (n,)
        if best_score is None or score < best_score:
            best_score = score
            best_cid = chain.id
    return best_cid or PEPTIDE_CHAIN_ID


def match_by_sequence_order(gt, pred, gt_chain, pred_chain, mode, atom_name=None):
    """Match residues in chain order (same length peptides in aligned datasets)."""
    gt_atoms, pred_atoms = [], []
    gt_res = [r for r in gt[0][gt_chain] if is_aa(r, standard=True)]
    pred_res = [r for r in pred[0][pred_chain] if is_aa(r, standard=True)]
    n = min(len(gt_res), len(pred_res))
    for i in range(n):
        gr, pr = gt_res[i], pred_res[i]
        if mode == "ca":
            if "CA" in gr and "CA" in pr:
                gt_atoms.append(gr["CA"])
                pred_atoms.append(pr["CA"])
        elif mode == "backbone":
            for name in ("N", "CA", "C", "O"):
                if name in gr and name in pr:
                    gt_atoms.append(gr[name])
                    pred_atoms.append(pr[name])
        elif mode == "sidechain":
            for atom in gr:
                if atom.name.startswith("H") or atom.name in {"N", "CA", "C", "O"}:
                    continue
                if atom.name in pr:
                    gt_atoms.append(atom)
                    pred_atoms.append(pr[atom.name])
        elif mode == "allheavy":
            for atom in gr:
                if atom.name.startswith("H"):
                    continue
                if atom.name in pr:
                    gt_atoms.append(atom)
                    pred_atoms.append(pr[atom.name])
    return gt_atoms, pred_atoms, len(gt_atoms)


def benchmark_ca_rmsd(gt_pdb, pred_pdb, gt_chain=PEPTIDE_CHAIN_ID, pred_chain=None):
    """
    Same as test.py RMSE: sqrt(mean((CA_pred - CA_true)^2)) in the shared frame,
    no superposition. Residues matched in sequence order.
    """
    gt = load_structure(gt_pdb)
    pred = load_structure(pred_pdb)
    gt_len = count_std_residues(gt, gt_chain)
    if pred_chain is None:
        pred_chain = guess_peptide_chain_id(pred, expected_len=gt_len)

    gt_ca, pred_ca, n = match_by_sequence_order(gt, pred, gt_chain, pred_chain, "ca")
    if n == 0:
        return None, pred_chain
    diff = np.array([a.get_coord() for a in gt_ca]) - np.array([a.get_coord() for a in pred_ca])
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))), pred_chain


def compute_peptide_rmsds_vs_gt(gt_pdb, pred_pdb, gt_chain=PEPTIDE_CHAIN_ID, pred_chain=None):
    """
    CA: benchmark-style (no alignment).
    BB/SC/ALL: superimpose on CA atoms, then measure (fair for sidechain packing).
    """
    gt = load_structure(gt_pdb)
    pred = load_structure(pred_pdb)
    gt_len = count_std_residues(gt, gt_chain)
    if pred_chain is None:
        pred_chain = guess_peptide_chain_id(pred, expected_len=gt_len)

    ca_rmsd, _ = benchmark_ca_rmsd(gt_pdb, pred_pdb, gt_chain, pred_chain)

    gt_ca, pred_ca, n_ca = match_by_sequence_order(gt, pred, gt_chain, pred_chain, "ca")
    if n_ca < 3:
        return {
            "bb_rmsd": None,
            "sc_rmsd": None,
            "all_heavy_rmsd": None,
            "ca_rmsd": ca_rmsd,
            "n_bb": 0,
            "n_sc": 0,
            "n_all": 0,
            "n_ca": n_ca,
            "pred_chain": pred_chain,
        }

    sup = Superimposer()
    sup.set_atoms(gt_ca, pred_ca)
    R, t = sup.rotran

    gt_bb, pred_bb, n_bb = match_by_sequence_order(gt, pred, gt_chain, pred_chain, "backbone")
    bb_rmsd = rmsd_after_rotran(coords_from_atoms(gt_bb), coords_from_atoms(pred_bb), R, t) if n_bb >= 3 else None

    gt_sc, pred_sc, n_sc = match_by_sequence_order(gt, pred, gt_chain, pred_chain, "sidechain")
    sc_rmsd = (
        rmsd_after_rotran(coords_from_atoms(gt_sc), coords_from_atoms(pred_sc), R, t)
        if n_sc >= 1
        else None
    )

    gt_all, pred_all, n_all = match_by_sequence_order(gt, pred, gt_chain, pred_chain, "allheavy")
    all_rmsd = (
        rmsd_after_rotran(coords_from_atoms(gt_all), coords_from_atoms(pred_all), R, t)
        if n_all >= 3
        else None
    )

    return {
        "bb_rmsd": bb_rmsd,
        "sc_rmsd": sc_rmsd,
        "all_heavy_rmsd": all_rmsd,
        "ca_rmsd": ca_rmsd,
        "n_bb": len(gt_bb),
        "n_sc": len(gt_sc),
        "n_all": len(gt_all),
        "n_ca": n_ca,
        "pred_chain": pred_chain,
    }


def summarize(values):
    arr = np.array([v for v in values if v is not None], dtype=float)
    if len(arr) == 0:
        return None, None
    return float(np.mean(arr)), float(np.median(arr))
