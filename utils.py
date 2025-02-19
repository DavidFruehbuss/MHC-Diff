from pathlib import Path
import logging

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

import re
import glob
import os
import pickle
import h5py
from io import StringIO


_log = logging.getLogger(__name__)


def assure_string(x):
    if isinstance(x, bytes):
        return x.decode('utf-8')

    return x


def create_new_pdb_hdf5(
        peptide, peptide_idx, graph_name, run_id, data_dir, time_step, sample_id
):
    # TODO: modify this to be adaptable
    hdf5_file = h5py.File(f'{data_dir}/test.hdf5', 'r')
        
    pdb_names = hdf5_file['pdb_names'][:]
    pdb_strings = hdf5_file['pdb_strings'][:]
    pdb_string = pdb_strings[pdb_names.tolist().index(graph_name)].decode('utf-8')

    # Create a temporary file or use StringIO to make the string readable by parser
    pdb_fh = StringIO(pdb_string)

    graph_name = assure_string(graph_name)

    pdb_output_path = f'./results/structures/{run_id}/{graph_name}_{time_step}_{sample_id}.pdb'

    directory = os.path.dirname(pdb_output_path)
    if not os.path.exists(directory):
         os.makedirs(directory)

    _log.debug(f"write pdb to {pdb_output_path}")

    write_updated_peptide_coords_pdb(peptide, peptide_idx, pdb_fh, pdb_output_path)
    
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
