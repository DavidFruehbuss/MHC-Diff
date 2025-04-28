import torch
from torch.utils.data import DataLoader
from dataset_100k_xray import PDB_Dataset_Mixed # Update this if your dataset file has a different name

# Path to your HDF5 file
hdf5_path = '/gpfs/home4/dfruhbus/MHC-Diff/data/pmhc_xray_100K/'  # <<< Update this path

# Instantiate the dataset
dataset = PDB_Dataset_Mixed(hdf5_path, split="100k_train")

# Create DataLoader (batch_size=100 for testing / printing)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True, collate_fn=PDB_Dataset_Mixed.collate_fn)

# Iterate and print the first batch
for batch_idx, batch in enumerate(dataloader):
    print(f"\n?? Batch {batch_idx}")
    
    # Print Graph Name
    print("Graph name:", batch['graph_name'][:])  # Should handle both 'BA_*' and normal

    # Print Peptide Features (One-hot encoding) and its shape
    print("Peptide Features (shape):", batch['peptide_features'].shape)
    print("Peptide Features (values):", batch['peptide_features'])

    # Print Peptide Positions (Coordinates) and its shape
    print("Peptide Positions (shape):", batch['peptide_positions'].shape)
    print("Peptide Positions (values):", batch['peptide_positions'])

    # Print Protein Features (One-hot encoding) and its shape
    print("Protein Features (shape):", batch['protein_pocket_features'].shape)
    print("Protein Features (values):", batch['protein_pocket_features'])

    # Print Protein Positions (Coordinates) and its shape
    print("Protein Positions (shape):", batch['protein_pocket_positions'].shape)
    print("Protein Positions (values):", batch['protein_pocket_positions'])

    break
