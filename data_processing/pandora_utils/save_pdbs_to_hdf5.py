import os
import h5py

# Define input folder and output file
input_folder = "./aligned_pdbs"  # Folder containing aligned PDB files
output_hdf5_file = "./xray_dataset.hdf5"  # Output HDF5 file

# Get all PDB files in the folder
pdb_files = [f for f in os.listdir(input_folder) if f.endswith('.pdb')]

# Step 1: Save all PDBs into HDF5
with h5py.File(output_hdf5_file, "w") as hdf5_file:
    for pdb_file in pdb_files:
        pdb_path = os.path.join(input_folder, pdb_file)

        with open(pdb_path, "r", encoding="utf-8") as f:
            pdb_content = f.read()  # Read PDB as string

        pdb_key = os.path.splitext(pdb_file)[0]  # Remove .pdb extension
        hdf5_file.create_dataset(pdb_key, data=pdb_content, dtype=h5py.string_dtype(encoding="utf-8"))  # Store as string

        print(f"Saved {pdb_file} as {pdb_key} in HDF5.")

print("All PDBs have been saved to HDF5.")
