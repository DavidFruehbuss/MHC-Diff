import os
import pymol
from pymol import cmd

# Initialize PyMOL in headless mode
pymol.finish_launching(['pymol', '-cq'])

# Define input and output folders
input_folder = "./processed_pdbs"  # Folder containing PDB files to align
output_folder = "./aligned_pdbs"  # Folder to save aligned PDBs
reference_pdb_path = "./reference_structure.pdb"  # Path to reference structure

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Get list of PDB files
pdb_files = [f for f in os.listdir(input_folder) if f.endswith('.pdb')]

for pdb_file in pdb_files:
    input_pdb_path = os.path.join(input_folder, pdb_file)
    output_pdb_path = os.path.join(output_folder, pdb_file)

    try:
        # Load reference and mobile structures
        cmd.load(reference_pdb_path, "reference")
        cmd.load(input_pdb_path, "mobile")

        # Align both chains P and M to the reference
        cmd.align("mobile and (chain P or chain M)", "reference and (chain P or chain M)")

        # Save aligned structure
        cmd.save(output_pdb_path, "mobile")

        # Cleanup PyMOL session
        cmd.delete("all")

        print(f"Aligned and saved: {pdb_file}")

    except Exception as e:
        print(f"Failed to align {pdb_file}: {e}")
        continue

print("All structures aligned successfully!")
