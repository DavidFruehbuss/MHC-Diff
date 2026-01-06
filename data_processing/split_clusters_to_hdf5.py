import h5py
import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm  # Progress bar

# Settings
BASE_DIR = Path(__file__).resolve().parent
VALID_CLUSTERS = {7}
TEST_CLUSTERS = {3, 4, 6}

# NOTE: Update these paths to match your local filesystem layout
source_hdf5_files = [
    # X-ray structures from PANDORA database (NOT in MHC-Diff repo)
    "/gpfs/home4/dfruhbus/PANDORA_database/data/PDBs/xray_dataset.hdf5",
    # 100k Pandora structure HDF5s (repurposed containers; names are legacy)
    # Update these paths to point to where your 100k_train.hdf5, 100k_valid.hdf5, 100k_test.hdf5 files are located
    "./100k_train.hdf5",
    "./100k_valid.hdf5",
    "./100k_test.hdf5"
]

# Load cluster mapping
cluster_df = pd.read_csv(BASE_DIR / "pdb_cluster_mapping.tsv", sep="\t")

# Get cluster to split assignments
split_pdbs = {"train": [], "valid": [], "test": []}
seen_pdbs = set()

for _, row in cluster_df.iterrows():
    cluster_id = row["Cluster_ID"]
    pdb_ids = row["PDB_IDs"].split(";")
    selected_ids = [pid for pid in pdb_ids if pid not in seen_pdbs]
    seen_pdbs.update(selected_ids)

    # Assign to the correct split
    if cluster_id in VALID_CLUSTERS:
        split_pdbs["valid"].extend(selected_ids)
    elif cluster_id in TEST_CLUSTERS:
        split_pdbs["test"].extend(selected_ids)
    else:
        split_pdbs["train"].extend(selected_ids)

# Summary of the selected PDBs
print(f"\n? Selected PDBs per cluster (no limit):")
for split in split_pdbs:
    print(f"{split}: {len(split_pdbs[split])} total PDBs")

# Create output directory if it doesn't exist
output_dir = BASE_DIR / "splits"
output_dir.mkdir(exist_ok=True)

# Open output files
output_files = {
    "train": h5py.File(output_dir / "100k_train.hdf5", "w"),
    "valid": h5py.File(output_dir / "100k_valid.hdf5", "w"),
    "test": h5py.File(output_dir / "100k_test.hdf5", "w"),
}

written_counts = defaultdict(int)
written_from_source = defaultdict(lambda: defaultdict(int))  # [split][source] = count

# Track which PDBs we've already written
written_pdbs = {"train": set(), "valid": set(), "test": set()}

# Start copying from each file to distribute sources
for src_path in source_hdf5_files:
    if not os.path.exists(src_path):
        print(f"? Skipping missing source: {src_path}")
        continue

    with h5py.File(src_path, "r") as h5in:
        for split in split_pdbs:
            # Using tqdm to show progress for each split
            for pdb_id in tqdm(split_pdbs[split], desc=f"Copying {split} PDBs", position=1, leave=False):
                if pdb_id in written_pdbs[split]:
                    continue
                if pdb_id in h5in:
                    try:
                        obj = h5in[pdb_id]
                        if isinstance(obj, h5py.Dataset):
                            output_files[split].create_dataset(pdb_id, data=obj[()])
                        else:
                            h5in.copy(pdb_id, output_files[split])

                        written_counts[split] += 1
                        written_from_source[split][src_path] += 1
                        written_pdbs[split].add(pdb_id)
                    except Exception as e:
                        print(f"? Failed to copy {pdb_id} from {src_path}: {e}")

# Close files
for f in output_files.values():
    f.close()

# Report
print("\n?? Final write counts:")
for split in ["train", "valid", "test"]:
    print(f"{split.upper()}: {written_counts[split]} entries")
    for src, count in written_from_source[split].items():
        print(f"  from {os.path.basename(src)}: {count}")
