import os
import h5py
import numpy as np
from io import StringIO
from pathlib import Path
from Bio import PDB
import pickle
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bar

AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
INDEX_TO_AA = {idx: aa for idx, aa in enumerate(AA_LIST)}

def decode_sequence(aatype_array):
    """Convert numerical amino acid encoding to a string sequence."""
    return "".join(INDEX_TO_AA.get(idx, "X") for idx in aatype_array)

def extract_sequence(structure, chain_id):
    """Extract amino acid sequence from a specified chain in a PDB structure."""
    sequence = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if PDB.is_aa(residue):
                        try:
                            sequence.append(PDB.Polypeptide.three_to_one(residue.resname))
                        except KeyError:
                            sequence.append("X")
                return "".join(sequence)
    return ""

def extract_g_domains_from_hdf5(file_paths):
    """Extract unique G-domain (chain M) sequences and track their PDBs & source files with a progress bar."""
    g_domain_mapping = {}  # Dictionary to store {sequence: {"pdb_ids": set(), "source_files": set()}}
    parser = PDB.PDBParser(QUIET=True)
    total_structures = 0

    # Count total keys for progress bar
    total_pdbs = sum(len(h5py.File(fp, 'r')) for fp in file_paths if os.path.exists(fp))
    progress_bar = tqdm(total=total_pdbs, desc="Processing PDB structures")

    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue

        with h5py.File(file_path, 'r') as f5:
            for pdb_key in f5.keys():
                graph = f5[pdb_key]

                if isinstance(graph, h5py.Group) and 'protein' in graph:
                    # HDF5 Graph Format
                    protein_seq = decode_sequence(graph['protein']['aatype'][:])
                else:
                    # Raw PDB format
                    pdb_data = f5[pdb_key][()].decode("utf-8")
                    structure = parser.get_structure(pdb_key, StringIO(pdb_data))
                    protein_seq = extract_sequence(structure, 'M')

                if protein_seq:
                    if protein_seq not in g_domain_mapping:
                        g_domain_mapping[protein_seq] = {"pdb_ids": set(), "source_files": set()}
                    g_domain_mapping[protein_seq]["pdb_ids"].add(pdb_key)
                    g_domain_mapping[protein_seq]["source_files"].add(os.path.basename(file_path))
                    total_structures += 1
                
                progress_bar.update(1)  # Update progress bar

    progress_bar.close()  # Close the progress bar when done

    print(f"? Total structures processed: {total_structures}")
    print(f"? Unique G-domain sequences found: {len(g_domain_mapping)}")
    return g_domain_mapping

# Example usage
# NOTE: Update these paths to match your local filesystem layout
file_paths = [
    # X-ray structures from PANDORA database (NOT in MHC-Diff repo)
    "/gpfs/home4/dfruhbus/PANDORA_database/data/PDBs/xray_dataset.hdf5",
    # 100k Pandora structure HDF5s (repurposed containers; names are legacy)
    # Update these paths to point to where your 100k_train.hdf5, 100k_valid.hdf5, 100k_test.hdf5 files are located
    "./100k_test.hdf5",
    "./100k_train.hdf5",
    "./100k_valid.hdf5",
]

g_domain_mapping = extract_g_domains_from_hdf5(file_paths)

# Save mapping as a pickle file
with open("g_domain_mapping.pkl", "wb") as f:
    pickle.dump(g_domain_mapping, f)

# Save as a TSV file
df = pd.DataFrame([
    {"Sequence": seq, 
     "PDB_IDs": ";".join(sorted(data["pdb_ids"])), 
     "Source_Files": ";".join(sorted(data["source_files"]))}
    for seq, data in g_domain_mapping.items()
])

df.to_csv("g_domain_mapping.tsv", sep="\t", index=False)

print("? Saved g_domain_mapping.pkl and g_domain_mapping.tsv successfully!")
