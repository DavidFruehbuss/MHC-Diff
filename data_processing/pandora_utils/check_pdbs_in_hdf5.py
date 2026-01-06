import h5py
from Bio import PDB
import os

# Path to the HDF5 file containing the PDBs
hdf5_file_path = "./xray_dataset.hdf5"

# Initialize PDB parser
parser = PDB.PDBParser(QUIET=True)

# Open the HDF5 file
with h5py.File(hdf5_file_path, "r") as hdf5_file:
    for pdb_key in hdf5_file.keys():
        pdb_data = hdf5_file[pdb_key][()].decode("utf-8")  # Decode bytes to string

        # Save PDB data to a temp file for Biopython parsing
        temp_pdb_path = f"temp_{pdb_key}.pdb"
        with open(temp_pdb_path, "w", encoding="utf-8") as temp_file:
            temp_file.write(pdb_data)

        # Parse PDB and count C-alpha atoms in chain M
        structure = parser.get_structure(pdb_key, temp_pdb_path)
        m_chain = None
        ca_count = 0

        for model in structure:
            for chain in model:
                if chain.get_id() == "M":
                    m_chain = chain
                    break

        if m_chain:
            # Count C-alpha atoms in M chain
            ca_count = sum(1 for atom in m_chain.get_atoms() if atom.get_name() == "CA")

        # Print the result for the current PDB
        print(f"PDB {pdb_key}: {ca_count} C-alpha atoms in chain M")

        # Cleanup temp file
        os.remove(temp_pdb_path)

print("Check completed for all PDBs in the HDF5 file.")
