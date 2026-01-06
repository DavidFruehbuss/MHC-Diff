import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict
import glob
import os

BASE_DIR = Path(__file__).resolve().parent

# Find the latest g_domain_clusters_*.tsv file (from hierarchical clustering)
cluster_files = sorted(glob.glob(str(BASE_DIR / "g_domain_clusters_*.tsv")), key=os.path.getmtime)
if not cluster_files:
    raise FileNotFoundError(f"No g_domain_clusters_*.tsv file found in {BASE_DIR}")
cluster_file = cluster_files[-1]
print(f"Using clustering results from: {os.path.basename(cluster_file)}")

# Load the sequence-to-cluster mapping from TSV
g_domain_clusters_df = pd.read_csv(cluster_file, sep="\t")

# Convert to dictionary {sequence: cluster_id}
sequence_to_cluster = dict(zip(g_domain_clusters_df["Sequence"], g_domain_clusters_df["Cluster"]))

# Load the G-domain mapping {sequence: {"pdb_ids": set(), "source_files": set()}}
g_domain_mapping_path = BASE_DIR / "g_domain_mapping.pkl"
if not g_domain_mapping_path.exists():
    raise FileNotFoundError(f"g_domain_mapping.pkl not found at {g_domain_mapping_path}")

with open(g_domain_mapping_path, "rb") as f:
    g_domain_mapping = pickle.load(f)

# Dictionary to store {cluster_id: set(PDB IDs)}
cluster_to_pdbs = defaultdict(set)

# Assign PDBs to clusters based on the sequence-cluster mapping
for sequence, cluster_id in sequence_to_cluster.items():
    if sequence in g_domain_mapping:
        cluster_to_pdbs[cluster_id].update(g_domain_mapping[sequence]["pdb_ids"])

# Print cluster sizes
print("\n?? **Cluster PDB Distribution:**")
for cluster_id, pdbs in sorted(cluster_to_pdbs.items()):
    print(f"Cluster {cluster_id}: {len(pdbs)} PDBs")

# Save to TSV
output_path = BASE_DIR / "pdb_cluster_mapping.tsv"
df = pd.DataFrame([
    {"Cluster_ID": cluster_id, "PDB_IDs": ";".join(sorted(pdbs)), "Num_PDBs": len(pdbs)}
    for cluster_id, pdbs in sorted(cluster_to_pdbs.items())
])

df.to_csv(output_path, sep="\t", index=False)

print(f"\n? Saved `pdb_cluster_mapping.tsv` successfully to {output_path}!")
