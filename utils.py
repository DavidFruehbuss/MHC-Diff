from pathlib import Path

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

import re
import glob
import os
import pickle
import h5py
import numpy as np
from io import StringIO
from typing import Sequence


def create_new_pdb_hdf5(
        peptide, peptide_idx, graph_name, run_id, data_dir, time_step, sample_id
):
    hdf5_file = h5py.File(f'{data_dir}/test.hdf5', 'r')
        
    pdb_names = hdf5_file['pdb_names'][:]
    pdb_strings = hdf5_file['pdb_strings'][:]
    pdb_string = pdb_strings[pdb_names.tolist().index(graph_name)].decode('utf-8')

    # Create a temporary file or use StringIO to make the string readable by parser
    pdb_fh = StringIO(pdb_string)
    
    pdb_output_path = f'./results/structures/{run_id}/{graph_name}_{time_step}_{sample_id}.pdb'

    directory = os.path.dirname(pdb_output_path)
    if not os.path.exists(directory):
         os.makedirs(directory)

    write_updated_peptide_coords_pdb(peptide, peptide_idx, pdb_fh, pdb_output_path)

def create_new_pdb_hdf5_100k(
    peptide: np.ndarray,
    peptide_idx: Sequence[int],
    graph_name: str,
    run_id: str,
    data_dir: str,
    time_step: int,
    sample_id: int
):
    """
    Saves a new PDB for non-BA entries only.
    Loads the original PDB string via group[()] decoding,
    then overwrites the P-chain CA coords with `peptide`.
    """
    # 1) Only handle non-BA
    if graph_name.startswith("BA"):
        return

    # 2) Read the PDB string from the group
    hdf5_path = Path(data_dir) / '100k_test.hdf5'
    with h5py.File(hdf5_path, 'r') as f5:
        if graph_name not in f5:
            raise KeyError(f"{graph_name} not found in {hdf5_path}")
        group = f5[graph_name]
        pdb_string = group[()].decode('utf-8')

    # 3) Prepare in-memory file for parser
    pdb_fh = StringIO(pdb_string)

    # 4) Build output path
    out_dir = Path('results') / 'structures' / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_output_path = out_dir / f"{graph_name}_{time_step}_{sample_id}.pdb"

    # 5) Write updated PDB: replaces only P-chain CA coords
    write_updated_peptide_coords_pdb(
        peptide=peptide,
        peptide_idx=peptide_idx,
        pdb_reference_path_or_stream=pdb_fh,
        pdb_output_path=str(pdb_output_path),
        atom_level=False
    )
    
def write_updated_peptide_coords_pdb(
    peptide, peptide_idx, pdb_reference_path_or_stream, pdb_output_path, atom_level=False
):
    """
    Function from https://github.com/steusink/DiffSBDD.git

    Takes an existing pdb file with peptide and mhc and creates a new one
    with the same mhc pocket and the peptide with updated atom/residue
    coordinates given by the model.
    :param peptide: peptide with updated coordinates
    :param decoder: decoder, from index to atom/residue
    :param pdb_reference_path: path to the reference pdb file
    :param pdb_output_path: path to the output pdb file
    :param atom_level: whether to use atoms or residues

    :return: None
    """
    # Read the reference pdb file
    parser = PDBParser(QUIET=True)
    pdb_models = parser.get_structure("", pdb_reference_path_or_stream)

    # Get the peptide chain
    peptide_chain = pdb_models[0]["P"]

    if not atom_level:
        peptide_chain_new = Chain("P")

    # Get the peptide atoms/residues
    if atom_level:
        peptide_elements = peptide_chain.get_atoms()
    else:
        peptide_elements = peptide_chain.get_residues()

    # Update the peptide coordinates
    for i, element in enumerate(peptide_elements):
        if atom_level:
            element.set_coord(peptide[i])
        else:
            ca_atom = element["CA"] # might need to switch this to "CB"
            ca_atom.set_coord(peptide[i])
            id = element.get_id()
            id = (' ', int(peptide_idx[i]), ' ')
            new_residue = Residue(id, element.get_resname(), "")
            new_residue.add(ca_atom)
            peptide_chain_new.add(new_residue)

    # Write the new pdb file
    if not atom_level:
        pdb_models[0].detach_child("P")
        pdb_models[0].add(peptide_chain_new)

        # Write the new pdb file
    io = PDBIO()
    io.set_structure(pdb_models)
    io.save(str(pdb_output_path))