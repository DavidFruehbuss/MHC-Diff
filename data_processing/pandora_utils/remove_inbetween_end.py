import os

def clean_pdb_file(file_path):
    """Removes in-between END statements from a PDB file while keeping the final one."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for i, line in enumerate(lines):
        if line.strip() == "END":
            # Keep only the last END statement in the file
            if i != len(lines) - 1:
                continue  # Skip this END statement
        
        cleaned_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(cleaned_lines)

def process_pdb_folder(pdb_folder):
    """Processes all PDB files in the folder, cleaning unnecessary END statements."""
    pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith('.pdb')]

    for pdb_file in pdb_files:
        pdb_path = os.path.join(pdb_folder, pdb_file)
        clean_pdb_file(pdb_path)
        print(f"Processed: {pdb_file}")

# Example usage
pdb_folder = './processed_pdbs'  # Change this to your folder path
process_pdb_folder(pdb_folder)
