from pathlib import Path

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

import re
import glob
import os
import pickle
import h5py
from io import StringIO

def create_new_pdb_hdf5(
        peptide, peptide_idx, graph_name, run_id, time_step
):
    # TODO: modify this to be adaptable
    hdf5_file = h5py.File('./data/pmhc_xray_8K_aligned/folds/fold_1/test.hdf5', 'r')
        
    pdb_names = hdf5_file['pdb_names'][:]
    pdb_strings = hdf5_file['pdb_strings'][:]
    pdb_string = pdb_strings[pdb_names.tolist().index(graph_name)].decode('utf-8')

    # Create a temporary file or use StringIO to make the string readable by parser
    pdb_fh = StringIO(pdb_string)
    
    pdb_output_path = f'./results/sampled_pmhcs/{run_id}/{graph_name}_{time_step}.pdb'

    directory = os.path.dirname(pdb_output_path)
    if not os.path.exists(directory):
         os.makedirs(directory)

    write_updated_peptide_coords_pdb(peptide, peptide_idx, pdb_fh, pdb_output_path)

def create_new_pdb(
        peptide, peptide_idx, graph_name, run_id, time_step
):
    # print(graph_name)
    pdb_number = extract_pdb_number(graph_name)
    # print(pdb_number)
    pdb_reference_path_or_stream = find_pdb_filepath(pdb_number)

    # pdb_output_path
    pdb_output_path = f'./results/sampled_pmhcs/{run_id}/BA-{pdb_number}_{time_step}.pdb'

    directory = os.path.dirname(pdb_output_path)
    if not os.path.exists(directory):
         os.makedirs(directory)

    write_updated_peptide_coords_pdb(peptide, peptide_idx, pdb_reference_path_or_stream, pdb_output_path)
    
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

def extract_pdb_number(input_string):
    # Use regular expression to find sequences of digits in the input string
    match = re.search(r'(?<=[-\s])\d+', input_string)
    if match:
        return match.group(0)
    else:
        return None

def find_pdb_filepath(pdb_number):
    # # Generate the pattern to match the file paths
    # pattern = f'/projects/0/einf2380/data/pMHCI/db2_selected_models/BA/*/*/pdb/*{pdb_number}*.pdb'
    # # Use glob to find files that match the pattern
    # file_paths = glob.glob(pattern)

    if not os.path.exists('./Data/Peptide_data/pdb_index.pkl'):
        root_dir = '/projects/0/einf2380/data/pMHCI/db2_selected_models/'
        pdb_dict = build_pdb_index(root_dir)
        save_pdb_index(pdb_dict)

    pdb_dict = load_pdb_index()

    return pdb_dict.get(pdb_number)
    
def build_pdb_index(root_dir):
    pdb_dict = {}
    pattern = re.compile(r'BA-(\d+)\.pdb$')  # Ensure the regex matches the full filename and ends with .pdb

    # Define the specific path pattern to search only necessary directories
    search_pattern = os.path.join(root_dir, 'BA', '*', '*', 'pdb', '*.pdb')
    
    # Use glob to find all pdb files in the specified pattern
    for file_path in glob.glob(search_pattern, recursive=True):
        filename = os.path.basename(file_path)
        match = pattern.search(filename)
        if match:
            pdb_number = match.group(1)
            pdb_dict[pdb_number] = file_path
    return pdb_dict

def save_pdb_index(pdb_dict, file_path='./results/pdb_index.pkl'):
    with open(file_path, 'wb') as f:
        pickle.dump(pdb_dict, f)

def load_pdb_index(file_path='./results/pdb_index.pkl'):
    with open(file_path, 'rb') as f:
        pdb_dict = pickle.load(f)
    return pdb_dict