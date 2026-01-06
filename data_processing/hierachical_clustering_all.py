#!/usr/bin/env python3
# g_domain_hclust_and_distances_blosum_only.py
# Goal: average BLOSUM *distance* between clusters on a 0..1 scale
#       (0 = same cluster / closest; 1 = most different; negative raw BLOSUM -> near 1)
# NOTE: Cache is *just* a .npy matrix (no meta checks).

import numpy as np
import pickle
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import fcluster, to_tree
from datetime import datetime
from pathlib import Path
import math, glob, os
from tqdm import tqdm

# ========== CONFIG ==========
BASE_DIR = Path(__file__).resolve().parent
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
base = f"g_domain_clusters_{ts}"

# High-res figures everywhere by default
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["figure.dpi"] = 150  # screen preview

# primary outputs
clusters_tsv = f"{base}.tsv"
circ_png = f"{base}_circular_dendrogram.png"
rect_png = f"{base}_dendrogram_rect.png"
cluster_sizes_tsv = f"{base}_cluster_sizes.tsv"

# inter-cluster outputs (BLOSUM only)
intercluster_blosum_raw    = f"{base}_intercluster_blosum_raw.csv"     # RAW per-position BLOSUM SIMILARITY (higher=closer)
intercluster_blosum_norm01 = f"{base}_intercluster_blosum_norm01.csv"  # 0..1 DISTANCE (0=closest, 1=farthest)

# cross-set histogram outputs (REAL pairwise BLOSUM) — directional NN(A→B)
# ---------- filenames (replace the three old ones) ----------
hist_test_train_png = f"{base}_blosum_hist_test_vs_train.png"  # for each Test seq, best match in Train
hist_val_train_png  = f"{base}_blosum_hist_val_vs_train.png"   # for each Val  seq, best match in Train


# NEW: cross-set mean similarities (exact averages over all A×B BLOSUM pairs)
crossset_avg_tsv = f"{base}_avg_blosum_similarity_sets.tsv"

# optional: export Newick + leaf→cluster table for cleaner external circular trees (iTOL/ETE/ggtree)
EXPORT_NEWICK_FOR_EXTERNAL_TOOL = False

# BLOSUM pairwise cache (FIXED name; no metadata file)
BLOSUM_CACHE_NPY = str(BASE_DIR / "g_domain_blosum_raw_dist_sq.npy")   # stores RAW per-position similarity (symmetric)

# ========== LOAD INPUTS ==========
similarity_matrix = np.load(BASE_DIR / "g_domain_similarity_matrix.npy")
with open(BASE_DIR / "unique_g_domains.pkl", "rb") as f:
    g_domains = list(pickle.load(f))

# ========== PREP MATRICES FOR CLUSTERING ==========
np.fill_diagonal(similarity_matrix, 1.0)
distance_matrix = 1 - similarity_matrix               # 0..1 distance (0=closest, 1=farthest)
condensed_dist_matrix = ssd.squareform(distance_matrix)

# ========== HIERARCHICAL CLUSTERING (average linkage) ==========
linkage_matrix = sch.linkage(condensed_dist_matrix, method="average")

# Step 1: split into two main clusters by threshold (same as before)
initial_clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')
main_cluster_labels = sorted(set(initial_clusters))

# Step 2: split each main cluster into five subclusters (→ 10 clusters total if both sides have ≥5)
final_clusters = np.zeros_like(initial_clusters, dtype=int)
current_cluster_id = 1
for main_cluster in main_cluster_labels:
    indices = np.where(initial_clusters == main_cluster)[0]
    if len(indices) > 1:
        sub_distance_matrix = distance_matrix[np.ix_(indices, indices)]
        sub_condensed_dist_matrix = ssd.squareform(sub_distance_matrix)
        sub_linkage = sch.linkage(sub_condensed_dist_matrix, method="average")

        threshold = None
        for t in np.linspace(0, 1, 1000):
            sub_clusters = fcluster(sub_linkage, t, criterion='distance')
            if len(set(sub_clusters)) == 5:
                threshold = t
                break
        if threshold is None:
            raise ValueError(f"Could not find a threshold for exactly 5 clusters in main cluster {main_cluster}")

        sub_clusters = fcluster(sub_linkage, threshold, criterion='distance')
        for sc in np.unique(sub_clusters):
            sub_idx = indices[np.where(sub_clusters == sc)[0]]
            final_clusters[sub_idx] = current_cluster_id
            current_cluster_id += 1
    else:
        final_clusters[indices] = current_cluster_id
        current_cluster_id += 1

# ========== SAVE CLUSTER ASSIGNMENTS ==========
df_assign = pd.DataFrame({"Sequence": g_domains, "Cluster": final_clusters})
df_assign.to_csv(clusters_tsv, sep="\t", index=False)

# ========== CIRCULAR DENDROGRAM WITH ALLELE LABELS (CENTERED 2-LINE TITLE + DISTINCT COLORS + LEGEND) ==========
import colorsys
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches

MAX_ALLELES_PER_LEAF = 3  # show at most 3 allele names per g-domain

ALL_INFO_TSV_PATTERN = str(BASE_DIR / "file_allele_gdomain_cluster_*.tsv")
candidates = sorted(glob.glob(ALL_INFO_TSV_PATTERN), key=os.path.getmtime)
if not candidates:
    raise FileNotFoundError(f"No matching all-info TSV for pattern: {ALL_INFO_TSV_PATTERN}")
all_info_path = candidates[-1]

# Map sequence → set of allele names
a2g_full = pd.read_csv(all_info_path, sep="\t", dtype=str).fillna("")
seq_to_alleles = {}
for _, r in a2g_full.iterrows():
    seq = r.get("Sequence", "").strip()
    allele = r.get("Allele", "").strip()
    if not seq:
        continue
    seq_to_alleles.setdefault(seq, set())
    if allele:
        seq_to_alleles[seq].add(allele)

import re
def clean_allele(a):
    a = a.strip().upper()
    a = re.sub(r'^HLA[-_]*', '', a)      # remove HLA- or HLA_
    a = re.sub(r'[^A-Z0-9*:]', '', a)    # keep only valid symbols
    return a

alleles_clean = [clean_allele(a) for a in a2g_full["Allele"].astype(str) if a.strip()]
print("Unique alleles:", len(set(alleles_clean)))

# Build dendrogram structure (no plot) to get linkage coordinates & leaf order
ddata = sch.dendrogram(linkage_matrix, labels=None, no_plot=True)
leaves_order = ddata["leaves"]
icoord = ddata["icoord"]; dcoord = ddata["dcoord"]

# coordinate→polar helpers
all_x = [x for xs in icoord for x in xs]
min_x, max_x = min(all_x), max(all_x)
all_y = [y for ys in dcoord for y in ys]
y_max = max(all_y) if max(all_y) > 0 else 1.0

def x_to_theta(x):
    # map [min_x, max_x] → [0, 2π)
    return 2 * math.pi * (x - min_x) / (max_x - min_x + 1e-12)

def y_to_radius(y):
    # outer rim slightly > 1 for label padding; inner radius > 0 to avoid center clutter
    rim, inner = 1.05, 0.15
    if y_max == 0:
        return rim
    frac = y / y_max
    return rim - (rim - inner) * frac

def draw_polar_arc(ax, r, theta1, theta2, **kwargs):
    # draw an arc at fixed radius r between theta1..theta2
    if theta2 < theta1:
        theta1, theta2 = theta2, theta1
    thetas = np.linspace(theta1, theta2, 72)
    rs = np.full_like(thetas, r, dtype=float)
    ax.plot(thetas, rs, **kwargs)

# maximally separated cluster colors (golden-angle in HSV)
def distinct_colors(n, s=0.70, v=0.98, h0=0.18):
    phi = 0.61803398875  # golden ratio conjugate
    cols = []
    for k in range(n):
        h = (h0 + k*phi) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append((r, g, b))
    return cols

cluster_ids_sorted = sorted(np.unique(final_clusters))
palette = distinct_colors(len(cluster_ids_sorted))
cluster_to_color = {cid: palette[i] for i, cid in enumerate(cluster_ids_sorted)}

# Figure
fig = plt.figure(figsize=(12, 12))
ax = plt.subplot(111, polar=True)
ax.set_axis_off()

# Draw branches (neutral color), using proper radial + circular geometry
for xs, ys in zip(icoord, dcoord):
    x1, x2, x3, x4 = xs
    y1, y2, y3, y4 = ys

    # left radial segment (vertical in Cartesian)
    theta_left = x_to_theta(x2)
    r_lo_left  = y_to_radius(min(y1, y2))
    r_hi_left  = y_to_radius(max(y1, y2))
    ax.plot([theta_left, theta_left], [r_lo_left, r_hi_left], lw=0.7, color="0.55", alpha=0.95, zorder=1)

    # right radial segment
    theta_right = x_to_theta(x3)
    r_lo_right  = y_to_radius(min(y3, y4))
    r_hi_right  = y_to_radius(max(y3, y4))
    ax.plot([theta_right, theta_right], [r_lo_right, r_hi_right], lw=0.7, color="0.55", alpha=0.95, zorder=1)

    # top arc (horizontal in Cartesian)
    r_top = y_to_radius(y2)  # y2 == y3 for the top bar
    draw_polar_arc(ax, r_top, x_to_theta(x2), x_to_theta(x3), lw=0.7, color="0.55", alpha=0.95, zorder=1)

# Leaf positions (equally spaced around the rim in the order of 'leaves')
n = len(leaves_order)
leaf_x = np.linspace(min_x, max_x, n)
leaf_theta = [x_to_theta(x) for x in leaf_x]

# Colored labels + small colored ticks at the rim
for pos, leaf_idx in enumerate(leaves_order):
    seq = g_domains[leaf_idx]
    cid = int(final_clusters[leaf_idx])
    theta = leaf_theta[pos]
    color = cluster_to_color[cid]

    # cap to MAX_ALLELES_PER_LEAF
    alleles = sorted(seq_to_alleles.get(seq, []))
    if not alleles:
        label = "NA"
    else:
        shown = alleles[:MAX_ALLELES_PER_LEAF]
        extra = len(alleles) - len(shown)
        label = "/".join(shown) + (f"/+{extra}" if extra > 0 else "")

    deg = np.degrees(theta)
    flip = 90 < deg < 270
    rotation = deg + (180 if flip else 0)
    align = "right" if flip else "left"

    # tiny colored tick for cluster cue
    ax.plot([theta, theta], [1.08, 1.12], lw=2.4, color=color, solid_capstyle="round", zorder=3)

    # colored allele label
    ax.text(theta, 1.135, label, rotation=rotation, rotation_mode="anchor",
            ha=align, va="center", color=color, fontsize=6.0, zorder=4)

# Central 2-line title (foreground with white stroke halo for readability)
title_txt = "Circular Dendrogram\nAllele-labeled (per-cluster colors)"
ax.text(0.0, 0.0, title_txt,
        ha="center", va="center", fontsize=15, weight="bold", zorder=10,
        linespacing=1.2,
        path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])

# Legend for clusters (outside on the right)
handles = [mpatches.Patch(color=cluster_to_color[cid], label=f"C{cid}") for cid in cluster_ids_sorted]
# pick ncol adaptively for readability
nC = len(cluster_ids_sorted)
ncol = 1 if nC <= 10 else (2 if nC <= 20 else 3)
legend = ax.legend(handles=handles, title="Clusters", frameon=False, ncol=len(handles), bbox_to_anchor=(0.5, -0.08), loc="upper center", borderaxespad=0.0, handlelength=1.2, columnspacing=0.8)


# # limits & save (high-res)
# ax.set_ylim(0, 1.18)
# plt.savefig(circ_png, bbox_inches="tight", pad_inches=0.25, dpi=600)
# plt.close()
# print(f"Saved circular dendrogram to: {circ_png}")

# limits & save (high-res)
ax.set_ylim(0, 1.18)
ax.set_facecolor('none')  # make axes background transparent (incl. polar)
plt.savefig(circ_png, bbox_inches="tight", pad_inches=0.25, dpi=600, transparent=True)
plt.close()
print(f"Saved circular dendrogram to: {circ_png}")


# ========== OPTIONAL: EXPORT NEWICK + LEAF→CLUSTER MAP FOR EXTERNAL CIRCULAR PLOTTING ==========
if EXPORT_NEWICK_FOR_EXTERNAL_TOOL:
    newick_path = f"{base}.nwk"
    leafmap_tsv = f"{base}_leaf_cluster_map.tsv"

    # Prefer allele labels; fallback to raw sequence
    export_labels = []
    for i in range(len(g_domains)):
        alleles = sorted(seq_to_alleles.get(g_domains[i], []))
        export_labels.append("/".join(alleles) if alleles else g_domains[i])

    def _build_newick(node, newick, parent_dist, leaf_names):
        if node.is_leaf():
            return f"{leaf_names[node.id]}:{max(parent_dist - node.dist, 0.0):.6f}{newick}"
        else:
            left = _build_newick(node.get_left(), "", node.dist, leaf_names)
            right = _build_newick(node.get_right(), "", node.dist, leaf_names)
            return f"({left},{right}):{max(parent_dist - node.dist, 0.0):.6f}{newick}"

    root = to_tree(linkage_matrix, rd=False)
    newick = _build_newick(root, ";", root.dist, export_labels)
    with open(newick_path, "w") as f:
        f.write(newick)
    print(f"[OK] Exported Newick tree: {newick_path}")

    # Leaf → cluster id map (for coloring in iTOL/ETE/ggtree)
    leaf_rows = []
    for leaf_idx, label in zip(d2["leaves"], export_labels):
        leaf_rows.append({"leaf_label": label, "cluster_id": int(final_clusters[leaf_idx])})
    pd.DataFrame(leaf_rows).to_csv(leafmap_tsv, sep="\t", index=False)
    print(f"[OK] Exported leaf→cluster map: {leafmap_tsv}")

# ========== CLUSTER SIZES (using PDB_ID, Cluster_ID, Allele, Sequence) ==========

# 1) Base sizes from current run (unique G-domain leaves)
cluster_sizes_base = (
    pd.Series(final_clusters, name="Cluster")
      .value_counts().sort_index()
      .rename("Size")
      .reset_index()
      .rename(columns={"index": "Cluster"})
)

# 2) Use original cluster labels from TSV to count real dataset structures
af = a2g_full.copy()

required_cols = {"PDB_ID", "Cluster_ID", "Sequence"}
missing = required_cols - set(af.columns)
if missing:
    raise KeyError(f"Missing required columns in all-info TSV: {missing}")

# Clean up column types
af["PDB_ID"] = af["PDB_ID"].astype(str).str.strip()
af["Cluster_ID"] = pd.to_numeric(af["Cluster_ID"], errors="coerce").astype("Int64")
af = af[af["Cluster_ID"].notna()].copy()
af["Cluster_ID"] = af["Cluster_ID"].astype(int)

# Classify by prefix rule: BA* -> Pandora, else Xray
af["SourceClass"] = np.where(af["PDB_ID"].str.upper().str.startswith("BA"), "Pandora", "Xray")

# Count UNIQUE PDB_IDs per (Cluster_ID, SourceClass)
counts_ids = (
    af.groupby(["Cluster_ID", "SourceClass"])["PDB_ID"]
      .nunique()
      .unstack(fill_value=0)
      .reset_index()
      .rename(columns={"Cluster_ID": "Cluster"})
)

# Ensure both columns exist
for col in ["Pandora", "Xray"]:
    if col not in counts_ids.columns:
        counts_ids[col] = 0

# 3) Merge with current-run cluster sizes (for Size column)
cluster_sizes = (
    cluster_sizes_base
    .merge(counts_ids[["Cluster", "Pandora", "Xray"]], on="Cluster", how="left")
    .fillna({"Pandora": 0, "Xray": 0})
    .astype({"Pandora": int, "Xray": int})
)

# 4) Save and print summary
cluster_sizes.to_csv(cluster_sizes_tsv, sep="\t", index=False)
print(f"Saved cluster sizes (unique PDB_IDs; BA*=Pandora) to: {cluster_sizes_tsv}")

u_total   = af["PDB_ID"].nunique()
u_pandora = af.loc[af["SourceClass"] == "Pandora", "PDB_ID"].nunique()
u_xray    = af.loc[af["SourceClass"] == "Xray", "PDB_ID"].nunique()
print(f"[STATS] Unique PDB_IDs in TSV: total={u_total} | Pandora={u_pandora} | Xray={u_xray}")


# ========== BLOSUM62-BASED INTER-CLUSTER (aligned; RAW = similarity, norm01 = distance) ==========
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as mat
BLOSUM62 = mat.blosum62
GAP_OPEN = -10
GAP_EXT  = -1

def load_blosum_cache_or_none(path, expected_N):
    """Load NxN cache matrix if it exists and shape matches, else None."""
    if not os.path.exists(path):
        return None
    try:
        M = np.load(path)
        if isinstance(M, np.ndarray) and M.shape == (expected_N, expected_N):
            return M
    except Exception:
        pass
    return None

def save_blosum_cache(path, M):
    np.save(path, M)

N = len(g_domains)
blosum_raw_sim_sq = load_blosum_cache_or_none(BLOSUM_CACHE_NPY, N)

if blosum_raw_sim_sq is None:
    print("[INFO] BLOSUM cache not found/invalid. Computing pairwise BLOSUM similarities...")
    blosum_raw_sim_sq = np.zeros((N, N), dtype=float)
    total = N * (N - 1) // 2
    pbar = tqdm(total=total, desc="BLOSUM pairwise alignments", dynamic_ncols=True)

    for i in range(N):
        si = g_domains[i]
        blosum_raw_sim_sq[i, i] = 0.0  # not used for inter-cluster stats
        for j in range(i + 1, N):
            sj = g_domains[j]
            aln = pairwise2.align.globalds(si, sj, BLOSUM62, GAP_OPEN, GAP_EXT, one_alignment_only=True)
            score = aln[0][2]
            ali1, ali2 = aln[0][0], aln[0][1]
            L_align = len(ali1)  # == len(ali2)
            sim = (score / L_align) if L_align > 0 else 0.0  # RAW similarity per aligned position
            blosum_raw_sim_sq[i, j] = blosum_raw_sim_sq[j, i] = sim
            pbar.update(1)
    pbar.close()
    save_blosum_cache(BLOSUM_CACHE_NPY, blosum_raw_sim_sq)
    print(f"[INFO] Saved BLOSUM cache: {BLOSUM_CACHE_NPY}")
else:
    print(f"[INFO] Loaded BLOSUM cache: {BLOSUM_CACHE_NPY}")

# Inter-cluster RAW BLOSUM similarity (mean over all A×B pairs)
cluster_ids = sorted(np.unique(final_clusters))
nC = len(cluster_ids)
cluster_to_indices = {cid: np.where(final_clusters == cid)[0] for cid in cluster_ids}

inter_blosum_raw_sim = np.zeros((nC, nC), dtype=float)
for ii, ci in enumerate(cluster_ids):
    A = cluster_to_indices[ci]
    for jj, cj in enumerate(cluster_ids):
        if ii == jj:
            inter_blosum_raw_sim[ii, jj] = 0.0
        else:
            B = cluster_to_indices[cj]
            inter_blosum_raw_sim[ii, jj] = float(blosum_raw_sim_sq[np.ix_(A, B)].mean())

# Save RAW similarity (higher = closer)
raw_df = pd.DataFrame(inter_blosum_raw_sim, index=[f"C{c}" for c in cluster_ids],
                      columns=[f"C{c}" for c in cluster_ids])
raw_df.to_csv(intercluster_blosum_raw)
print(f"Saved BLOSUM raw (SIMILARITY) to: {intercluster_blosum_raw}")

# Normalize to 0..1 DISTANCE
off_diag_mask = ~np.eye(nC, dtype=bool)
off_vals = inter_blosum_raw_sim[off_diag_mask]
if off_vals.size > 0:
    s_min = off_vals.min()
    s_max = off_vals.max()
else:
    s_min = s_max = 0.0

if s_max > s_min:
    inter_blosum_norm = (s_max - inter_blosum_raw_sim) / (s_max - s_min)
else:
    inter_blosum_norm = np.ones_like(inter_blosum_raw_sim)
np.fill_diagonal(inter_blosum_norm, 0.0)

norm_df = pd.DataFrame(inter_blosum_norm, index=[f"C{c}" for c in cluster_ids],
                       columns=[f"C{c}" for c in cluster_ids])
norm_df.to_csv(intercluster_blosum_norm01)
print(f"Saved BLOSUM norm01 (DISTANCE) to: {intercluster_blosum_norm01}")

# ========== Directional NN(A→B) histograms from REAL pairwise BLOSUM ==========
TRAIN = {1, 2, 5, 8, 9, 10}
VAL   = {7}
TEST  = {3, 4, 6}

def gather_indices(cluster_set):
    idxs = [cluster_to_indices[c] for c in cluster_set if c in cluster_to_indices]
    if not idxs:
        return np.array([], dtype=int)
    return np.concatenate(idxs).astype(int)

idx_train = gather_indices(TRAIN)
idx_val   = gather_indices(VAL)
idx_test  = gather_indices(TEST)

print(len(idx_train), len(idx_val), len(idx_test))
print({c: len(cluster_to_indices[c]) for c in sorted(cluster_to_indices)})

def per_seq_best_to_other(M, idxA, idxB):
    if idxA.size == 0 or idxB.size == 0:
        return np.array([], dtype=float)
    sub = M[np.ix_(idxA, idxB)]
    return np.max(sub, axis=1).astype(float)

def plot_hist(vals, out_png, title, bins=60, hrange=None):
    if vals.size == 0:
        print(f"[WARN] No sequences for {title}; skipping {out_png}")
        return
    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=bins, range=hrange)
    m, s = float(np.mean(vals)), float(np.std(vals))
    mn, mx = float(np.min(vals)), float(np.max(vals))
    plt.axvline(m, linestyle="-")
    plt.title(f"{title}\nmean={m:.3f}±{s:.3f} | min={mn:.3f} max={mx:.3f} | n={vals.size}")
    plt.xlabel("Per-position BLOSUM similarity (nearest neighbor)")
    plt.ylabel("Count")

    # NEW: start x-axis at 3 and make tick labels slightly larger
    plt.xlim(left=3.0)
    plt.tick_params(axis="both", labelsize=12)

    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()
    print(f"[OK] Saved histogram: {out_png}")

# Global range from all off-diagonal entries for consistent binning
Ntot = blosum_raw_sim_sq.shape[0]
global_off = blosum_raw_sim_sq[~np.eye(Ntot, dtype=bool)]
hrange = (max(3.0, float(np.min(global_off))), float(np.max(global_off)))

vals_Te_to_T = per_seq_best_to_other(blosum_raw_sim_sq, idx_test, idx_train)  # Test → Train
vals_V_to_T  = per_seq_best_to_other(blosum_raw_sim_sq, idx_val,  idx_train)  # Val  → Train

# keep the consistent global range, but clamp visible left edge at 3.0 (bins still global)
# plot_hist(vals_Te_to_T, hist_test_train_png,
#           "Test → Train — nearest BLOSUM similarity per Test sequence",
#           bins=60, hrange=hrange)

# plot_hist(vals_V_to_T,  hist_val_train_png,
#           "Val → Train — nearest BLOSUM similarity per Val sequence",
#           bins=60, hrange=hrange)

# ========== NEW: CROSS-SET AVERAGE BLOSUM SIMILARITIES (exact; from pairwise matrix) ==========
def mean_pairwise(M, idxA, idxB):
    if idxA.size == 0 or idxB.size == 0:
        return float("nan"), 0
    sub = M[np.ix_(idxA, idxB)]
    return float(sub.mean()), int(sub.size)

rows = []
m_tv, n_tv = mean_pairwise(blosum_raw_sim_sq, idx_train, idx_val)
rows.append({"pair": "Train–Val", "mean_similarity": m_tv,
             "nA": int(idx_train.size), "nB": int(idx_val.size), "n_pairs": n_tv})

m_tt, n_tt = mean_pairwise(blosum_raw_sim_sq, idx_train, idx_test)
rows.append({"pair": "Train–Test", "mean_similarity": m_tt,
             "nA": int(idx_train.size), "nB": int(idx_test.size), "n_pairs": n_tt})

pd.DataFrame(rows).to_csv(crossset_avg_tsv, sep="\t", index=False)
print(f"Saved cross-set average BLOSUM similarities to: {crossset_avg_tsv}")

# ========== CONSOLE SUMMARY ==========
print(f"Saved cluster assignments to: {clusters_tsv}")
print(f"Saved circular dendrogram to: {circ_png}")
print(f"Saved rectangular dendrogram to: {rect_png}")
print(f"Saved cluster sizes to:       {cluster_sizes_tsv}")
print(f"Saved BLOSUM raw to:          {intercluster_blosum_raw}")
print(f"Saved BLOSUM norm01 to:       {intercluster_blosum_norm01}")
print(f"Saved cross-set averages to:  {crossset_avg_tsv}")
