import os
import h5py
import torch
from torch.utils.data import Dataset
from Bio import PDB
from typing import Dict
from io import StringIO

# Amino acid definitions
AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_LIST)}
INDEX_TO_AA = {idx: aa for idx, aa in enumerate(AA_LIST)}

def one_hot_encode_sequence(sequence):
    encoding = torch.zeros((len(sequence), len(AA_LIST)), dtype=torch.float32)
    for i, aa in enumerate(sequence):
        if aa in AA_TO_INDEX:
            encoding[i, AA_TO_INDEX[aa]] = 1.0
        else:
            print(f"Warning: Non-standard amino acid '{aa}' found and ignored.")
    return encoding

def decode_numeric_sequence(aatype_array):
    return "".join(INDEX_TO_AA.get(idx, "X") for idx in aatype_array)

class PDB_Dataset_Mixed(Dataset):
    def __init__(self, datadir, split='train'):
        self.hdf5_path = os.path.join(datadir, f'{split}.hdf5')
        print(f"Loading dataset from {self.hdf5_path}...")
        
        with h5py.File(self.hdf5_path, 'r') as f5:
            self.pdb_names = list(f5.keys())
            print(f"Loaded {len(self.pdb_names)} pdb entries from {split} split.")

    def __len__(self):
        return len(self.pdb_names)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.get_entry(index)

    def get_entry(self, index: int) -> Dict[str, torch.Tensor]:
        data = {}
        graph_name = self.pdb_names[index]

        with h5py.File(self.hdf5_path, 'r') as f5:
            group = f5[graph_name]

            if graph_name.startswith('BA'):
                # BA entries: peptide/protein aatype and coordinates
                peptide_seq = decode_numeric_sequence(group['peptide']['aatype'][:])
                protein_seq = decode_numeric_sequence(group['protein']['aatype'][:])

                peptide_coords = torch.tensor(group['peptide']['coords'][:], dtype=torch.float32)
                protein_coords = torch.tensor(group['protein']['coords'][:], dtype=torch.float32)

            else:
                # Standard PDB string entries
                pdb_string = group[()].decode('utf-8')
                structure = self.parse_pdb_structure(pdb_string)

                peptide_chain = structure[0]['P']
                protein_chain = structure[0]['M']

                peptide_coords, peptide_seq = self.extract_ca_coords_and_sequence(peptide_chain)
                protein_coords, protein_seq = self.extract_ca_coords_and_sequence(protein_chain, max_residues=178)

            # One-hot encoding
            peptide_onehot = one_hot_encode_sequence(peptide_seq)
            protein_onehot = one_hot_encode_sequence(protein_seq)

            # Create masks
            peptide_len = peptide_coords.shape[0]
            protein_len = protein_coords.shape[0]
            peptide_mask = torch.ones(peptide_len, dtype=torch.bool)
            protein_mask = torch.ones(protein_len, dtype=torch.bool)

            # Package everything
            data['graph_name'] = graph_name
            data['peptide_idx'] = peptide_mask
            data['peptide_positions'] = peptide_coords
            data['peptide_features'] = peptide_onehot
            data['num_peptide_residues'] = peptide_len
            data['protein_pocket_idx'] = protein_mask
            data['protein_pocket_positions'] = protein_coords
            data['protein_pocket_features'] = protein_onehot
            data['num_protein_pocket_residues'] = protein_len
            data['pos_in_seq'] = torch.arange(peptide_len) + 1

        return data

    def parse_pdb_structure(self, pdb_string: str):
        parser = PDB.PDBParser(QUIET=True)
        pdb_io = StringIO(pdb_string)
        return parser.get_structure("structure", pdb_io)

    def extract_ca_coords_and_sequence(self, chain, max_residues=None):
        ca_coords = []
        sequence = []
        for i, residue in enumerate(chain):
            if max_residues is not None and i >= max_residues:
                break
            if 'CA' in residue:
                ca_coords.append(residue['CA'].coord)
                sequence.append(PDB.Polypeptide.three_to_one(residue.resname))
            else:
                print(f"Warning: Missing CA atom for residue {residue.resname} in chain {chain.id}")
        coords_tensor = torch.tensor(ca_coords, dtype=torch.float32)
        return coords_tensor, ''.join(sequence)

    @staticmethod
    def collate_fn(batch):
        data_batch = {}
        for key in batch[0].keys():
            if key == 'graph_name':
                data_batch[key] = [x[key] for x in batch]
            elif key in ['num_peptide_residues', 'num_protein_pocket_residues']:
                data_batch[key] = torch.tensor([x[key] for x in batch])
            elif 'idx' in key:
                data_batch[key] = torch.cat([i * torch.ones(len(x[key]), dtype=torch.long) for i, x in enumerate(batch)], dim=0)
            else:
                data_batch[key] = torch.cat([x[key] for x in batch], dim=0)
        return data_batch
