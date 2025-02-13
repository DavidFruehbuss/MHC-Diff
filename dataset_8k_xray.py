import os
import h5py
import torch
from torch.utils.data import Dataset
from Bio import PDB  # Biopython's PDB parser
from typing import Dict

# Define amino acid one-hot encoding for 20 standard residues
AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_LIST)}

def one_hot_encode_sequence(sequence):
    """One-hot encode the amino acid sequence based on standard residues."""
    encoding = torch.zeros((len(sequence), len(AA_LIST)), dtype=torch.float32)
    for i, aa in enumerate(sequence):
        if aa in AA_TO_INDEX:
            encoding[i, AA_TO_INDEX[aa]] = 1.0
        else:
            print(f"Warning: Non-standard amino acid '{aa}' found and ignored.")
    return encoding


AA_ORDER_3DSSL = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'U', 'Z', 'X']
NUM_AA_3DSSL = len(AA_ORDER_3DSSL)
AA_TO_3DSSL_CONVERSION = [AA_TO_INDEX_3DSSL[a] for i, a in enumerate(AA_LIST)]

def convert_amino_acid_to_3dssl(aa: torch.Tensor) -> torch.Tensor:
    """
    3D SSL uses a different amino acid one-hot encoding.
    This function converts the MHC-Diff one-hot encoded input to 3D SSL one-hot encoded output.
    """

    conversion = aa.new_tensor(AA_TO_3DSSL_CONVERSION)

    return torch.nn.functional.one_hot(conversion[aa.argmax(dim=-1)], num_classes=NUM_AA_3DSSL)


class PDB_Dataset(Dataset):

    def __init__(self, datadir, split='train'):
        """
        Args:
            datadir (str): Path to the directory where HDF5 files are located.
            split (str): Dataset split, one of 'train', 'valid', 'test'.
        """
        # Define the HDF5 file paths for the dataset split
        self.hdf5_path = os.path.join(datadir, f'{split}.hdf5')

        print(f"Loading dataset from {self.hdf5_path}...")

        # Open the HDF5 file and load the pdb_strings dataset directly
        with h5py.File(self.hdf5_path, 'r') as f5:
            self.pdb_strings = f5['pdb_strings'][:]  # Load the pdb_strings array directly
            self.pdb_names = f5['pdb_names'][:]  # Load the pdb_names array
            print(f"Loaded {len(self.pdb_strings)} pdb strings and names from {split} split.")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns a data entry from the HDF5 file for a given index.
        """
        # print(f"Loading entry at index: {index}")
        return self.get_entry(index)

    def __len__(self) -> int:
        """
        Returns the total number of entries in the dataset.
        """
        return len(self.pdb_strings)

    def get_entry(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves and processes a single entry from the HDF5 file.
        
        Args:
            entry_name (str): The name of the entry in the HDF5 file.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the processed data.
        """
        data = {}
        with h5py.File(self.hdf5_path, 'r') as f5:
            
            # Access the pdb_string directly using the index
            pdb_string = self.pdb_strings[index].decode('utf-8')
            # pdb_string = entry[0].decode('utf-8')  # Access by index to ensure it's correct

            # Parse the PDB data using Biopython
            structure = self.parse_pdb_structure(pdb_string)

            # Extract peptide (P chain) and protein (M chain) data
            peptide_chain = structure[0]['P']
            protein_chain = structure[0]['M']

            # Extract C-alpha atom coordinates and sequence for peptide and protein
            peptide_coords, peptide_seq = self.extract_ca_coords_and_sequence(peptide_chain)
            # print(f"Peptide sequence: {peptide_seq} (length: {len(peptide_seq)})")
            
            # Truncate the M chain (protein chain) to the first 178 residues
            protein_coords, protein_seq = self.extract_ca_coords_and_sequence(protein_chain, max_residues=178)
            # print(f"Protein sequence: {protein_seq} (length: {len(protein_seq)})")

            # One-hot encode sequences
            peptide_onehot = one_hot_encode_sequence(peptide_seq)
            protein_onehot = one_hot_encode_sequence(protein_seq)

            # Generate masks (assuming all residues are valid for now)
            peptide_len = peptide_coords.shape[0]
            protein_len = protein_coords.shape[0]
            peptide_mask = torch.ones(peptide_len, dtype=torch.bool)
            protein_mask = torch.ones(protein_len, dtype=torch.bool)

            # Prepare the output dictionary
            data['graph_name'] = self.pdb_names[index]
            data['peptide_idx'] = peptide_mask
            data['peptide_positions'] = peptide_coords  # 3D C-alpha coordinates for peptide
            data['peptide_features'] = peptide_onehot  # One-hot encoded peptide sequence
            data['num_peptide_residues'] = peptide_len
            data['protein_pocket_idx'] = protein_mask
            data['protein_pocket_positions'] = protein_coords  # 3D C-alpha coordinates for protein
            data['protein_pocket_features'] = protein_onehot  # One-hot encoded protein sequence
            data['num_protein_pocket_residues'] = protein_len
            data['pos_in_seq'] = torch.arange(peptide_len) + 1  # Position in the sequence

        return data

    def parse_pdb_structure(self, pdb_string: str):
        """Parses the PDB string using Biopython and returns a structure."""
        parser = PDB.PDBParser(QUIET=True)
        from io import StringIO
        pdb_io = StringIO(pdb_string)
        structure = parser.get_structure("structure", pdb_io)
        return structure

    def extract_ca_coords_and_sequence(self, chain, max_residues=None):
        """
        Extract C-alpha coordinates and amino acid sequence from a PDB chain.

        Args:
            chain (Bio.PDB.Chain): The chain object from which to extract data.
            max_residues (int, optional): Maximum number of residues to include (for truncation).

        Returns:
            coords (torch.Tensor): C-alpha coordinates (Nx3 tensor).
            sequence (str): Corresponding amino acid sequence.
        """
        ca_coords = []
        sequence = []
        
        for i, residue in enumerate(chain):
            if max_residues is not None and i >= max_residues:
                break  # Truncate if the number of residues exceeds the limit

            if 'CA' in residue:
                ca_coords.append(residue['CA'].coord)
                sequence.append(PDB.Polypeptide.three_to_one(residue.resname))
            else:
                print(f"Warning: Missing CA atom for residue {residue.resname} in chain {chain.id}")

        # Convert to torch tensors
        coords_tensor = torch.tensor(ca_coords, dtype=torch.float32)  # Shape: (N, 3)
        sequence_str = ''.join(sequence)
        
        return coords_tensor, sequence_str


    @staticmethod
    def collate_fn(batch):
        """
        Collation function to combine batch data into a single batch.
        
        Args:
            batch (list of Dict): A list of individual data entries.

        Returns:
            Dict: A dictionary containing batched data.
        """
        data_batch = {}
        for key in batch[0].keys():

            if key == 'graph_name':
                data_batch[key] = [x[key] for x in batch]
            elif key == 'num_peptide_residues' or key == 'num_protein_pocket_residues':
                data_batch[key] = torch.tensor([x[key] for x in batch])
            elif 'idx' in key:
                # Ensure that indices in the batch start at zero (needed for torch_scatter)
                data_batch[key] = torch.cat([i * torch.ones(len(x[key]), dtype=torch.long) for i, x in enumerate(batch)], dim=0)
            else:
                data_batch[key] = torch.cat([x[key] for x in batch], dim=0)

        return data_batch
