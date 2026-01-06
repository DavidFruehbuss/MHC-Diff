import os
import sys
import subprocess
from pathlib import Path
from Bio import PDB

# Import find_gdomain from the local gdomain module
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from gdomain.scripts.find_gdomain import find_gdomain

def extract_chain(pdb_path, chain_id):
    """Extracts a specific chain from a PDB file and returns a Biopython Chain object."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_path)
    
    for model in structure:
        for chain in model:
            if chain.get_id() == chain_id:
                return chain  # Return the extracted chain object
    return None

# def save_chain(chain, output_file):
#     """Saves a Biopython Chain object to a PDB file."""
#     io = PDB.PDBIO()
#     structure = PDB.Structure.Structure("filtered")
#     model = PDB.Model.Model(0)
#     model.add(chain)
#     structure.add(model)
#     
#     io.set_structure(structure)
#     io.save(output_file)

def save_chain(chain, output_file):
    """Saves a Biopython Chain object to a PDB file without an END statement."""
    class NoEndPDBIO(PDB.PDBIO):
        def save(self, file, *args, **kwargs):
            """Overrides default save method to remove END statement."""
            super().save(file, *args, **kwargs)
            with open(file, 'r') as f:
                lines = f.readlines()
            with open(file, 'w') as f:
                for line in lines:
                    if not line.strip().startswith("END"):
                        f.write(line)

    io = NoEndPDBIO()
    structure = PDB.Structure.Structure("filtered")
    model = PDB.Model.Model(0)
    model.add(chain)
    structure.add(model)

    io.set_structure(structure)
    io.save(output_file)


def process_pdb_files(pdb_folder, output_folder):
    """Processes all PDB files, extracts chains, applies G-domain filtering, and saves results.

    Args:
        pdb_folder (str): Path to the folder containing PDB files.
        output_folder (str): Path to save processed PDB files.
    """
    os.makedirs(output_folder, exist_ok=True)

    pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith('.pdb')]

    for pdb_file in pdb_files:
        pdb_path = os.path.join(pdb_folder, pdb_file)
        key = os.path.splitext(pdb_file)[0]  # Remove extension for key

        try:
            # Extract full M and P chains
            m_chain = extract_chain(pdb_path, 'M')
            p_chain = extract_chain(pdb_path, 'P')

            if not m_chain or not p_chain:
                print(f"Could not extract chains for {pdb_file}. Skipping...")
                continue

            # Run G-domain extraction on M chain
            aligned_residues = find_gdomain(pdb_path)

            if not aligned_residues:
                print(f"No G-domain found for {pdb_file}. Skipping G-domain filtering.")
                continue

            # Create new filtered M chain
            filtered_m_chain = PDB.Chain.Chain('M')
            for residue in aligned_residues:
                filtered_m_chain.add(residue)

            filtered_pdb_m = os.path.join(output_folder, f"filtered_chain_M_{key}.pdb")
            filtered_pdb_p = os.path.join(output_folder, f"filtered_chain_P_{key}.pdb")
            combined_pdb = os.path.join(output_folder, f"{key}_combined.pdb")

            # Save chains
            save_chain(filtered_m_chain, filtered_pdb_m)
            save_chain(p_chain, filtered_pdb_p)

            # Combine M and P chains into one file
            combine_cmd = f'cat {filtered_pdb_m} {filtered_pdb_p} > {combined_pdb}'
            subprocess.run(combine_cmd, shell=True, check=True)

            # Remove temporary M and P files
            os.remove(filtered_pdb_m)
            os.remove(filtered_pdb_p)

        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")

# Example usage
pdb_folder = './pMHCI'  # Path to your PDB files folder
output_folder = './processed_pdbs'  # Path to save processed files
process_pdb_files(pdb_folder, output_folder)
