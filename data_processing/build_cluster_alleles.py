# make_cluster_alleles_all_info.py
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ---- inputs (relative to this file) ----
BASE_DIR = Path(__file__).resolve().parent
PDB_CLUSTER_MAP = BASE_DIR / "pdb_cluster_mapping.tsv"
MHCI_ALLELES = BASE_DIR / "mhci_alleles.tsv"
EXTRACTED_PDB = BASE_DIR / "extracted_pdb_data.tsv"
GDOMAIN_MAP = BASE_DIR / "g_domain_mapping.tsv"

out_dir = BASE_DIR
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

OUT_CLUSTER = out_dir / f"cluster_alleles_by_cluster_{ts}.tsv"        # (from prior step)
OUT_A2G     = out_dir / f"allele_to_gdomain_{ts}.tsv"                 # (from prior step)
OUT_FULL    = out_dir / f"file_allele_gdomain_cluster_{ts}.tsv"       # <-- NEW (this request)

# ---- load allele maps ----
mhci = pd.read_csv(MHCI_ALLELES, sep="\t", dtype=str)
mhci = mhci[["File Name", "Allele Name"]].rename(columns={"File Name":"FileName", "Allele Name":"Allele"}).dropna()
mhci["FileName"] = mhci["FileName"].str.strip()
mhci["Allele"]   = mhci["Allele"].str.strip()
fn_to_alleles = mhci.groupby("FileName")["Allele"].apply(lambda s: set(a for a in s if a)).to_dict()

ext = pd.read_csv(EXTRACTED_PDB, sep="\t", dtype=str)
ext = ext[["Graph Name", "Allele Name"]].rename(columns={"Graph Name":"GraphName", "Allele Name":"Allele"}).dropna()
ext["GraphName"] = ext["GraphName"].str.strip()
ext["Allele"]    = ext["Allele"].str.strip()
gn_to_alleles = ext.groupby("GraphName")["Allele"].apply(lambda s: set(a for a in s if a)).to_dict()

def id_to_alleles(pdb_id: str):
    """BA-xxxxx → ext['Graph Name'];  XXXX_combined → mhci['File Name'] as XXXX.pdb"""
    pdb_id = str(pdb_id).strip()
    if not pdb_id:
        return set()
    if pdb_id.startswith("BA-"):
        return gn_to_alleles.get(pdb_id, set())
    core = pdb_id[:-9] if pdb_id.endswith("_combined") else pdb_id
    filename = f"{core.upper()}.pdb"
    return fn_to_alleles.get(filename, set())

# ---- 1) clusters → allele list (same as before) ----
cm = pd.read_csv(PDB_CLUSTER_MAP, sep="\t", dtype=str)[["Cluster_ID", "PDB_IDs"]].dropna()
cluster_to_alleles = defaultdict(set)
cm_rows = []  # keep exploded rows for step 3

for _, r in cm.iterrows():
    cid = r["Cluster_ID"].strip()
    ids = [x.strip() for x in str(r["PDB_IDs"]).split(";") if x.strip()]
    for pid in ids:
        cm_rows.append((cid, pid))
        als = id_to_alleles(pid)
        if als:
            cluster_to_alleles[cid].update(als)

out_rows_cluster = []
for cid in sorted(cluster_to_alleles, key=lambda x: int(x) if str(x).isdigit() else x):
    allele_list = sorted(cluster_to_alleles[cid])
    out_rows_cluster.append({"Cluster_ID": cid, "Alleles": ";".join(allele_list)})
pd.DataFrame(out_rows_cluster).to_csv(OUT_CLUSTER, sep="\t", index=False)

# ---- 2) Allele → g-domain via g_domain_mapping.tsv (same mapping rules as before) ----
gdm = pd.read_csv(GDOMAIN_MAP, sep="\t", dtype=str)[["Sequence", "PDB_IDs"]].dropna()
pid_to_sequences = defaultdict(set)
for _, r in gdm.iterrows():
    seq = r["Sequence"].strip()
    ids = [x.strip() for x in str(r["PDB_IDs"]).split(";") if x.strip()]
    for pid in ids:
        pid_to_sequences[pid].add(seq)

allele_to_gdomains = defaultdict(set)
for pid, seqs in pid_to_sequences.items():
    als = id_to_alleles(pid)
    for a in als:
        allele_to_gdomains[a].update(seqs)

a2g_rows = [{"Allele": a, "Sequence": s} for a, seqs in allele_to_gdomains.items() for s in seqs]
pd.DataFrame(a2g_rows).sort_values(["Allele","Sequence"]).to_csv(OUT_A2G, sep="\t", index=False)

# ---- 3) NEW: filename (PDB_ID) ↔ allele ↔ gdomain sequence ↔ cluster ----
full_rows = []
for cid, pid in cm_rows:
    alleles = sorted(id_to_alleles(pid))            # may be empty
    sequences = sorted(pid_to_sequences.get(pid, []))  # may be empty

    if alleles and sequences:
        # Cartesian product
        for a in alleles:
            for seq in sequences:
                full_rows.append({"PDB_ID": pid, "Cluster_ID": cid, "Allele": a, "Sequence": seq})
    elif alleles and not sequences:
        for a in alleles:
            full_rows.append({"PDB_ID": pid, "Cluster_ID": cid, "Allele": a, "Sequence": ""})
    elif sequences and not alleles:
        for seq in sequences:
            full_rows.append({"PDB_ID": pid, "Cluster_ID": cid, "Allele": "", "Sequence": seq})
    else:
        # neither found — keep a placeholder row so nothing disappears
        full_rows.append({"PDB_ID": pid, "Cluster_ID": cid, "Allele": "", "Sequence": ""})

pd.DataFrame(full_rows).to_csv(OUT_FULL, sep="\t", index=False)

print(f"Wrote:\n  {OUT_CLUSTER}\n  {OUT_A2G}\n  {OUT_FULL}")
