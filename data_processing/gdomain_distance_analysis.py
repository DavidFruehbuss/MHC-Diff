from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as mat
import numpy as np
import pickle
from tqdm import tqdm

BLOSUM62 = mat.blosum62

# Load your unique G-domains
with open("unique_g_domains.pkl", "rb") as f:
    g_domains = list(pickle.load(f))

num_seqs = len(g_domains)
similarity_matrix = np.zeros((num_seqs, num_seqs))

# Store all raw similarity scores to determine min and max for normalization
raw_scores = []

# Compute pairwise similarities
total_comparisons = num_seqs * (num_seqs - 1) // 2
progress = tqdm(total=total_comparisons, desc="Computing pairwise similarities")

for i in range(num_seqs):
    for j in range(i + 1, num_seqs):
        seq1, seq2 = g_domains[i], g_domains[j]
        alignment = pairwise2.align.globalds(seq1, seq2, BLOSUM62, -10, -1, one_alignment_only=True)
        score = alignment[0][2]
        raw_scores.append(score)
        similarity_matrix[i, j] = score
        similarity_matrix[j, i] = score  # Symmetric matrix
        progress.update(1)

progress.close()

# Convert to numpy array
raw_scores = np.array(raw_scores)

# Normalize similarity scores between 0 and 1
if len(raw_scores) > 0:
    min_score, max_score = raw_scores.min(), raw_scores.max()
    if max_score != min_score:  # Avoid division by zero
        similarity_matrix = (similarity_matrix - min_score) / (max_score - min_score)
    else:
        similarity_matrix.fill(1)  # If all scores are the same, set everything to 1

# Save normalized similarity matrix
np.save("g_domain_similarity_matrix.npy", similarity_matrix)
