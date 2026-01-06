#!/usr/bin/env python

from typing import Dict, List
import sys
import os
from uuid import uuid4
from tempfile import gettempdir

from Bio.PDB.Residue import Residue
from Bio.PDB.PDBParser import PDBParser
from pymol import cmd as pymol_cmd


# pdb parser from biopython
pdb_parser = PDBParser()

# directory for storing temporary files
tmp_dir = gettempdir()

# reference pth path is assumed to be in ../data/ref.pdb
reference_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/ref.pdb")


def align(mobile_path: str) -> str:
    """
    Calls pymol to align the input structure to the reference structure.

    Args:
        mobile_path: to unknown PDB file
    Returns:
        alignment path to file in clustal format
    """

    # where to store the alignment
    alignment_name = uuid4()
    alignment_path = os.path.join(tmp_dir, f"{alignment_name}.aln")

    # init pymol
    pymol_cmd.reinitialize()

    # load mobile structure
    pymol_cmd.load(mobile_path, 'mobile')
    pymol_cmd.load(reference_path, 'reference')

    # align
    r = pymol_cmd.align("mobile", "reference", object="alignment")
    if r[1] == 0:
        raise ValueError("No residues aligned")

    # save
    pymol_cmd.save(alignment_path, selection="alignment", format="aln")

    # clean up
    pymol_cmd.remove("all")

    return alignment_path


def parse_clustal(path: str) -> Dict[str, str]:
    """
    Parses clustal alignment format file

    Args:
        path: to alignment file
    Returns:
        alignment dictionary: {id 1: aligned sequence 1, id 2: aligned sequence 2, ...}
    """

    alignment = {}
    with open(path, 'rt') as f:
        for line in f:
            # line format: key and sequence separated by whitespaces
            # key starts on first line

            i = line.find(' ')
            k = line[:i].strip()
            v = line[i:].strip()

            if len(k) > 0 and len(v) > 0:
                alignment[k] = alignment.get(k, '') + v

    return alignment


def list_aligned_residues(alignment_path: str, pdb_path: str) -> List[Residue]:
    """
    List all residues that were aligned to the reference structure

    Args:
        alignment_path: to alignment file in clustal format
        pdb_path: the pdb file that was used as a mobile structure
    Returns:
        List of biopython pdb residues in the mobile structure, that were aligned
    """

    # get residues from pdb structure
    structure = pdb_parser.get_structure('mobile', pdb_path)
    model = list(structure.get_models())[0]
    residues = []
    for chain in sorted(model.get_chains(), key=lambda c: c.get_id()):
        residues += list(chain.get_residues())

    # get reference and mobile aligned sequences
    alignment = parse_clustal(alignment_path)
    aligned_reference = alignment['reference']
    aligned_mobile = alignment['mobile']

    # find all aligned residues
    aligned_index = []
    for i in range(len(aligned_reference)):
        if aligned_reference[i] != '-':
            aligned_index.append(len(aligned_mobile[:i].replace('-', '')))

    # isolate aligned residues
    aligned_residues = [residues[i] for i in aligned_index]

    return aligned_residues


def find_gdomain(pdb_path):

        alignment_path = align(pdb_path)
        residues = list_aligned_residues(alignment_path, pdb_path)

        return residues

        # print only first and last residue (chain id + residue number)
        # print(pdb_path, f"{residues[0].get_parent().get_id()}{residues[0].get_id()[1]}-{residues[-1].get_parent().get_id()}{residues[-1].get_id()[1]}")
