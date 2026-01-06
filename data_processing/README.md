# Data processing for G-domain clustering (pmhc_100K)

This folder contains the **scripts and small metadata tables** used to build the G-domain clustering and allele mappings for the **100k dataset**. The **large structure datasets themselves (HDF5, NPY, PKL, PNG, etc.) are *not* included here** and must be available separately on the filesystem.

## Raw structure datasets (Pandora / X-ray)

- The files `100k_train.hdf5`, `100k_valid.hdf5`, and `100k_test.hdf5` contain **Pandora structures**.
  - These HDF5 files are **repurposed from Pandora**; the train/val/test names are **legacy** and have no meaning for our processing.
  - We **do not use their splitting logic** — they could equally be named `hdf5_1`, `hdf5_2`, `hdf5_3`.
  - We only need the **structures (PDB entries) contained in them** to extract unique G-domain sequences.
- **X-ray structures** are stored in a separate HDF5 file:
  - `/gpfs/home4/dfruhbus/PANDORA_database/data/PDBs/xray_dataset.hdf5`
  - This file is **not in the MHC-Diff repository**; it lives in the external PANDORA database directory.

In the processing here, we extract unique G-domain sequences from **all structures** in these HDF5 containers, regardless of which file they came from.

## Key Python scripts (detailed explanations)

### `extract_unique_gdomains.py`

**Purpose**: Extract unique G-domain (chain M) sequences from all structure HDF5 files and build a mapping of sequences to their source PDB IDs and files.

**What it does**:
1. Iterates through multiple HDF5 files:
   - X-ray structures from `/gpfs/home4/dfruhbus/PANDORA_database/data/PDBs/xray_dataset.hdf5`
   - Pandora structures from `100k_train.hdf5`, `100k_valid.hdf5`, `100k_test.hdf5` (paths configurable)
2. For each PDB structure:
   - If the HDF5 entry is a graph format (has `'protein'` group), decodes the amino acid type array to a sequence string
   - If it's raw PDB format, parses the PDB structure and extracts chain M sequence using BioPython
3. Builds a dictionary mapping each **unique G-domain sequence** to:
   - The set of PDB IDs that contain this sequence
   - The set of source HDF5 filenames where these PDBs were found
4. Saves outputs:
   - `g_domain_mapping.pkl` (Python pickle of the dictionary)
   - `g_domain_mapping.tsv` (tabular format: Sequence, PDB_IDs (semicolon-separated), Source_Files (semicolon-separated))

**Key detail**: This script treats all HDF5 files equally — it doesn't care about train/val/test labels, it just extracts all unique sequences from all structures.

---

### `gdomain_distance_analysis.py`

**Purpose**: Compute pairwise BLOSUM62 similarity scores between all unique G-domain sequences to create a similarity matrix for clustering.

**What it does**:
1. Loads `unique_g_domains.pkl` (list of unique G-domain sequences)
2. For each pair of sequences (i, j):
   - Performs a **global pairwise alignment** using BLOSUM62 substitution matrix
   - Uses gap open penalty = -10, gap extension penalty = -1
   - Extracts the alignment score from the first (best) alignment
   - Stores the score in a symmetric matrix `similarity_matrix[i, j] = similarity_matrix[j, i]`
3. Normalizes the entire matrix to [0, 1] range:
   - Finds min and max scores across all off-diagonal entries
   - Applies linear normalization: `(score - min) / (max - min)`
   - This ensures similarity values are on a consistent scale for clustering
4. Saves `g_domain_similarity_matrix.npy` (N×N numpy array, where N = number of unique G-domains)

**Note**: This is computationally expensive (O(N²) pairwise alignments). The resulting matrix is the core input for all downstream clustering.

---

### `split_clusters_to_hdf5.py`

**Purpose**: Create train/validation/test split HDF5 files by assigning PDB structures to splits based on their cluster membership.

**What it does**:
1. **Defines the split assignment logic**:
   - `VALID_CLUSTERS = {7}` → all PDBs in cluster 7 go to validation set
   - `TEST_CLUSTERS = {3, 4, 6}` → all PDBs in clusters 3, 4, or 6 go to test set
   - **All other clusters** → go to training set
2. Loads `pdb_cluster_mapping.tsv` which maps `Cluster_ID` → semicolon-separated list of `PDB_IDs`
3. For each cluster:
   - Extracts all PDB IDs belonging to that cluster
   - Assigns them to the appropriate split (train/valid/test) based on the cluster ID
   - Tracks which PDBs have already been assigned (to avoid duplicates)
4. Copies PDB structures from source HDF5 files to output split files:
   - Reads from: `xray_dataset.hdf5`, `100k_train.hdf5`, `100k_valid.hdf5`, `100k_test.hdf5`
   - Writes to: `./splits/100k_train.hdf5`, `./splits/100k_valid.hdf5`, `./splits/100k_test.hdf5`
   - For each PDB ID assigned to a split, searches all source files until it finds the structure, then copies it to the appropriate output file
5. Reports statistics: how many PDBs were written to each split, and which source files they came from

**Key insight**: This is where the **actual train/val/test split is defined** — it's based on cluster membership, not on any pre-existing split in the Pandora HDF5 files. The Pandora files (`100k_train.hdf5`, etc.) are just containers; their names are legacy and irrelevant to this splitting logic.

---

### `cluster_to_pdb_assingment.py`

**Purpose**: Convert G-domain sequence clustering results into a PDB-to-cluster mapping.

**What it does**:
1. Loads the latest `g_domain_clusters_<timestamp>.tsv` file (from `hierachical_clustering_all.py`) which maps G-domain sequences to cluster IDs
2. Loads `g_domain_mapping.pkl` (from `extract_unique_gdomains.py`) which maps sequences to PDB IDs
3. For each sequence-cluster pair, looks up all PDB IDs that contain that sequence and assigns them to that cluster
4. Aggregates all PDB IDs per cluster
5. Saves `pdb_cluster_mapping.tsv` with columns:
   - `Cluster_ID`: The cluster number
   - `PDB_IDs`: Semicolon-separated list of all PDB IDs belonging to this cluster
   - `Num_PDBs`: Count of PDBs in this cluster

**Key detail**: This script bridges the gap between **sequence-level clustering** (G-domains) and **structure-level assignments** (PDB IDs). It's the critical step that allows downstream scripts to work with clusters at the PDB level.

---

### `build_cluster_alleles.py`

**Purpose**: Build comprehensive mappings linking clusters, PDB IDs, alleles, and G-domain sequences together.

**What it does**:
1. **Loads four metadata tables**:
   - `pdb_cluster_mapping.tsv`: Cluster_ID → PDB_IDs (created by `cluster_to_pdb_assingment.py` from clustering results)
   - `mhci_alleles.tsv`: File Name → Allele Name (for X-ray structures)
   - `extracted_pdb_data.tsv`: Graph Name → Allele Name (for Pandora structures)
   - `g_domain_mapping.tsv`: Sequence → PDB_IDs (from `extract_unique_gdomains.py`)

2. **Builds PDB ID → allele mapping**:
   - For Pandora structures (IDs starting with `BA-`): looks up allele in `extracted_pdb_data.tsv` by Graph Name
   - For X-ray structures (IDs ending with `_combined` or plain PDB codes): looks up allele in `mhci_alleles.tsv` by File Name (converts PDB code to `.pdb` filename)

3. **Produces three output TSV files** (timestamped):
   - `cluster_alleles_by_cluster_<timestamp>.tsv`: For each cluster, lists all alleles found in that cluster's PDBs
   - `allele_to_gdomain_<timestamp>.tsv`: For each allele, lists all G-domain sequences associated with that allele (via PDBs)
   - `file_allele_gdomain_cluster_<timestamp>.tsv`: **Comprehensive mapping** with columns:
     - `PDB_ID`, `Cluster_ID`, `Allele`, `Sequence`
     - Contains all combinations (Cartesian product) where a PDB has both an allele annotation and a G-domain sequence

**Key detail**: The third output (`file_allele_gdomain_cluster_*.tsv`) is the **most important** — it's used by `hierachical_clustering_all.py` to label dendrogram leaves with allele names and to compute cluster statistics.

---

### `hierachical_clustering_all.py`

**Purpose**: Perform hierarchical clustering on G-domain sequences and generate comprehensive cluster statistics, visualizations, and inter-cluster distance metrics.

**What it does** (detailed breakdown):

1. **Loads inputs**:
   - `g_domain_similarity_matrix.npy`: N×N normalized similarity matrix (values in [0,1]) from `gdomain_distance_analysis.py`
   - `unique_g_domains.pkl`: List of unique G-domain sequences, index-aligned with the similarity matrix (sequence `i` corresponds to row/column `i` in the matrix)
   - `file_allele_gdomain_cluster_*.tsv`: Latest timestamped comprehensive mapping file (automatically selected by modification time). Provides:
     - Allele annotations for labeling dendrogram leaves
     - PDB_ID → Cluster_ID mappings for computing cluster size statistics
     - Sequence → Allele mappings for each PDB

2. **Hierarchical clustering** (two-stage strategy):
   - **Preprocessing**: 
     - Sets diagonal of similarity matrix to 1.0 (self-similarity)
     - Converts to distance matrix: `distance = 1 - similarity` (so 0 = identical, 1 = maximally different)
     - Converts square distance matrix to condensed form for scipy
   
   - **Stage 1 - Main split**:
     - Performs **average linkage hierarchical clustering** on all sequences
     - Cuts the dendrogram at distance threshold **0.5** to create two main clusters
     - This initial split separates the G-domain sequences into two broad groups
   
   - **Stage 2 - Sub-clustering**:
     - For each of the two main clusters:
       - Extracts the sub-distance matrix for sequences in that cluster
       - Performs hierarchical clustering again (average linkage) on just those sequences
       - Searches through distance thresholds (0 to 1, in steps of 0.001) to find the threshold that yields **exactly 5 subclusters**
       - If no threshold yields exactly 5 clusters, raises an error
     - Assigns final cluster IDs sequentially (1, 2, 3, ...) across all subclusters
     - Result: **~10 final clusters** total (5 from each main cluster, if both sides have ≥5 subclusters)
   
   - **Output**: `g_domain_clusters_<timestamp>.tsv` with columns:
     - `Sequence`: The G-domain amino acid sequence
     - `Cluster`: The assigned cluster ID (integer)

3. **Circular dendrogram visualization**:
   - Builds a polar (circular) dendrogram from the linkage matrix
   - **Allele labeling**:
     - For each leaf (G-domain sequence), looks up associated alleles from `file_allele_gdomain_cluster_*.tsv`
     - Shows up to **3 allele names** per leaf (if more exist, shows "allele1/allele2/allele3/+N")
     - Cleans allele names (removes "HLA-" prefix, keeps only valid characters)
   - **Coloring**:
     - Assigns distinct colors to each cluster using a golden-angle HSV color scheme (maximally separated in color space)
     - Colors each leaf label and adds a small colored tick mark at the rim
   - **Layout**:
     - Leaves are equally spaced around the circle in dendrogram order
     - Text labels are rotated to be readable (flipped for bottom half)
     - Includes a centered 2-line title with white stroke halo for readability
     - Legend shows all cluster colors outside the circle
   - **Output**: `g_domain_clusters_<timestamp>_circular_dendrogram.png` (600 DPI, transparent background)

4. **Cluster size statistics**:
   - **Base counts**: Counts how many unique G-domain sequences (leaves) are in each cluster
   - **PDB-level counts**: Uses `file_allele_gdomain_cluster_*.tsv` to count **unique PDB IDs** per cluster
   - **Source classification**:
     - **Pandora structures**: PDB IDs starting with `BA-` (Pandora-generated structures)
     - **X-ray structures**: All other PDB IDs (experimental structures)
   - **Output**: `g_domain_clusters_<timestamp>_cluster_sizes.tsv` with columns:
     - `Cluster`: Cluster ID
     - `Size`: Number of unique G-domain sequences in this cluster
     - `Pandora`: Number of unique Pandora PDB IDs in this cluster
     - `Xray`: Number of unique X-ray PDB IDs in this cluster
   - Prints summary statistics: total unique PDBs, Pandora count, X-ray count

5. **Inter-cluster BLOSUM distance matrices**:
   - **BLOSUM cache**: 
     - Checks for `g_domain_blosum_raw_dist_sq.npy` (pairwise BLOSUM similarity cache)
     - If missing or wrong shape, computes all pairwise BLOSUM62 global alignments:
       - Uses BioPython `pairwise2.align.globalds()` with BLOSUM62 matrix
       - Gap open penalty = -10, gap extension = -1
       - Computes **per-position similarity**: `score / alignment_length`
       - Stores in symmetric N×N matrix (cached for future runs)
   
   - **Inter-cluster similarity**:
     - For each cluster pair (C_i, C_j):
       - Extracts all sequence pairs where one sequence is in C_i and the other in C_j
       - Computes the **mean BLOSUM similarity** over all such pairs
       - Stores in an N_clusters × N_clusters matrix
   
   - **Outputs**:
     - `g_domain_clusters_<timestamp>_intercluster_blosum_raw.csv`: 
       - Raw similarity scores (higher = more similar sequences)
       - Diagonal is 0 (self-similarity not computed)
       - Symmetric matrix with cluster labels (C1, C2, ...)
     - `g_domain_clusters_<timestamp>_intercluster_blosum_norm01.csv`:
       - Normalized to [0,1] **distance** scale
       - Formula: `distance = (max_sim - sim) / (max_sim - min_sim)`
       - 0 = closest clusters, 1 = most distant clusters
       - Diagonal is 0

6. **Train/val/test split similarity analysis**:
   - **Hard-coded cluster assignments** (matches `split_clusters_to_hdf5.py`):
     - `TRAIN = {1, 2, 5, 8, 9, 10}` → training set clusters
     - `VAL = {7}` → validation set cluster
     - `TEST = {3, 4, 6}` → test set clusters
   
   - **Mean pairwise similarity computation**:
     - Extracts all sequence indices belonging to each split
     - For each pair of splits (Train–Val, Train–Test):
       - Computes the **mean BLOSUM similarity** over all sequence pairs where one sequence is from split A and the other from split B
       - Also records: number of sequences in each split, total number of pairs evaluated
   
   - **Output**: `g_domain_clusters_<timestamp>_avg_blosum_similarity_sets.tsv` with columns:
     - `pair`: Which split pair (e.g., "Train–Val", "Train–Test")
     - `mean_similarity`: Average BLOSUM similarity over all pairs
     - `nA`, `nB`: Number of sequences in each split
     - `n_pairs`: Total number of sequence pairs evaluated
   
   - **Purpose**: Quantifies how similar/dissimilar the train/val/test splits are at the sequence level, which is important for understanding potential data leakage or generalization challenges.

**Note**: The cluster-to-split assignment (`TRAIN = {1, 2, 5, 8, 9, 10}`, `VAL = {7}`, `TEST = {3, 4, 6}`) matches the logic in `split_clusters_to_hdf5.py` (`VALID_CLUSTERS = {7}`, `TEST_CLUSTERS = {3, 4, 6}`). This ensures consistency between the clustering analysis and the actual data splits used for training.

---

## Metadata tables

- `pdb_cluster_mapping.tsv`: Maps Cluster_ID to semicolon-separated list of PDB_IDs. This defines which structures belong to which cluster. **This file is created by `cluster_to_pdb_assingment.py` from the clustering results** (not pre-existing). It's used by `build_cluster_alleles.py` and `split_clusters_to_hdf5.py`.
- `mhci_alleles.tsv`: Maps File Name (X-ray PDB filenames) to Allele Name. Used to annotate X-ray structures with their MHC alleles.
- `extracted_pdb_data.tsv`: Maps Graph Name (Pandora structure IDs) to Allele Name. Used to annotate Pandora structures with their MHC alleles.
- `g_domain_mapping.tsv`: Maps G-domain Sequence to semicolon-separated list of PDB_IDs. Output from `extract_unique_gdomains.py`, records which PDBs contain each unique sequence.
- `unique_g_domains.txt`: Plain text list of all unique G-domain sequences (one per line). Human-readable version of the sequences used for clustering.
- `gdomain_perturbation_experiments.tsv`: Additional metadata for BLOSUM-based perturbation experiments (optional, not required for core clustering workflow).

**Example outputs** (kept in this folder for reference):
- `cluster_alleles_by_cluster_20250926_160505.tsv`
- `allele_to_gdomain_20250926_160505.tsv`
- `file_allele_gdomain_cluster_20250926_160505.tsv`

## PANDORA Database utilities (`pandora_utils/`)

This folder contains utilities for working with X-ray structures from the **PANDORA Database** (see the [PANDORA Database paper](https://github.com/X-lab-3D/PANDORA_database)). These scripts are used to extract G-domain sequences from X-ray PDB structures.

### Dependencies

To use these utilities, you need:

1. **PANDORA Database**: 
   - Clone from: `https://github.com/X-lab-3D/PANDORA_database.git`
   - The database contains X-ray pMHCI structures and metadata
   - The X-ray HDF5 file (`xray_dataset.hdf5`) should be available at the path specified in `extract_unique_gdomains.py`

2. **G-domain extraction module** (from [cbaakman](https://github.com/cbaakman)):
   - The `gdomain` module is included in `pandora_utils/gdomain/`
   - It uses **PyMOL** to align structures to a reference and extract the G-domain region
   - **Requirements**: `pymol==3.0.0`, `biopython==1.84` (see `pandora_utils/gdomain/requirements.txt`)

### Files in `pandora_utils/`

- **`extract_gdomain.py`**:
  - Main script for processing PDB files to extract G-domains
  - Extracts chain M (MHC heavy chain) and chain P (peptide) from PDB files
  - Uses `gdomain.scripts.find_gdomain` to identify G-domain residues via structural alignment
  - Filters chain M to contain only G-domain residues
  - Combines filtered chain M + chain P into `*_combined.pdb` files
  - Used during preprocessing of X-ray structures before they're added to the HDF5 database

- **`gdomain/scripts/find_gdomain.py`**:
  - Core G-domain identification algorithm
  - Uses PyMOL to align input PDB structure to a reference structure (`gdomain/data/ref.pdb`)
  - Parses the alignment to identify which residues align to the reference G-domain
  - Returns a list of BioPython Residue objects corresponding to the G-domain region
  - **Reference structure**: `gdomain/data/ref.pdb` (must be present for alignment to work)

- **`gdomain/data/ref.pdb`**:
  - Reference PDB structure used for structural alignment
  - The G-domain region is defined by alignment to this reference
  - Must be present in `gdomain/data/` for `find_gdomain()` to work

### Usage

These utilities are typically used **before** running the main pipeline, during the preprocessing of X-ray structures:

1. Extract raw PDB files from the PANDORA Database
2. Run `extract_gdomain.py` to process PDBs and extract G-domains:
   ```python
   from pandora_utils.extract_gdomain import process_pdb_files
   process_pdb_files(pdb_folder="./raw_pdbs", output_folder="./processed_pdbs")
   ```
3. The processed `*_combined.pdb` files can then be loaded into HDF5 format for use in the main pipeline

**Note**: For the 100k dataset processing in this folder, the X-ray structures have already been processed and are stored in `xray_dataset.hdf5`. These utilities are included here for reference and for processing additional X-ray structures if needed.

## Canonical clustering run

Multiple historical runs of `hierachical_clustering_all.py` exist in the original `pmhc_datasets` folder, with filenames of the form `g_domain_clusters_<timestamp>.*`. The **last complete run** is:

- `g_domain_clusters_20251029_163125.*`

Those files remain in the original processing directory and are **not duplicated here**; this folder focuses on the *scripts* and minimal metadata needed to reproduce or inspect the clustering, rather than on storing all outputs.

## Running the pipeline from this folder

**Prerequisites**: 

1. **Large structure datasets (HDF5)** must exist on the filesystem:
   - The Pandora 100k HDF5 files (`100k_train.hdf5`, `100k_valid.hdf5`, `100k_test.hdf5`) should be accessible (update paths in scripts as needed).
   - The X-ray dataset HDF5 file (`/gpfs/home4/dfruhbus/PANDORA_database/data/PDBs/xray_dataset.hdf5`) must exist at that path (or update the path in `extract_unique_gdomains.py` and `split_clusters_to_hdf5.py`).

2. **PANDORA Database** (for X-ray structures):
   - The X-ray structures come from the [PANDORA Database](https://github.com/X-lab-3D/PANDORA_database)
   - The `xray_dataset.hdf5` file contains preprocessed X-ray structures
   - If you need to process additional X-ray structures, see the `pandora_utils/` section above

3. **Python dependencies**:
   - Standard scientific Python stack: numpy, scipy, pandas, matplotlib, BioPython, tqdm
   - For G-domain extraction utilities: PyMOL 3.0.0, BioPython 1.84 (see `pandora_utils/gdomain/requirements.txt`)

**Pipeline steps**:

1. **Extract unique G-domain sequences**:
   ```bash
   python extract_unique_gdomains.py
   ```
   Produces: `g_domain_mapping.pkl` and `g_domain_mapping.tsv`

2. **Generate `unique_g_domains.pkl`**:
   - This should be a Python pickle file containing a list of unique G-domain sequences (matching `unique_g_domains.txt`).
   - If you don't have it, you can create it from `unique_g_domains.txt` or copy it from your processing environment.

3. **Compute pairwise similarity matrix**:
   ```bash
   python gdomain_distance_analysis.py
   ```
   Produces: `g_domain_similarity_matrix.npy` (this step is computationally expensive, O(N²) alignments)

4. **Perform hierarchical clustering**:
   ```bash
   python hierachical_clustering_all.py
   ```
   Produces timestamped clustering outputs:
   - `g_domain_clusters_<timestamp>.tsv` (sequence → cluster assignments)
   - `g_domain_clusters_<timestamp>_cluster_sizes.tsv` (cluster statistics)
   - `g_domain_clusters_<timestamp>_intercluster_blosum_*.csv` (inter-cluster distances)
   - `g_domain_clusters_<timestamp>_avg_blosum_similarity_sets.tsv` (train/val/test similarities)
   - `g_domain_clusters_<timestamp>_circular_dendrogram.png` (visualization)
   
   **Note**: This step requires `file_allele_gdomain_cluster_*.tsv` to exist for allele labeling and cluster statistics. For the **first run**, you can:
   - Use an existing `file_allele_gdomain_cluster_*.tsv` from a previous analysis (one is included in this folder: `file_allele_gdomain_cluster_20250926_160505.tsv`)
   - Or create a preliminary one by running steps 5-6 first with an existing `pdb_cluster_mapping.tsv` (if you have one from a previous clustering run)

5. **Create PDB-to-cluster mapping**:
   ```bash
   python cluster_to_pdb_assingment.py
   ```
   Produces: `pdb_cluster_mapping.tsv` (Cluster_ID → PDB_IDs)
   
   This converts the sequence-level clustering results into structure-level (PDB) cluster assignments. The script automatically finds the latest `g_domain_clusters_<timestamp>.tsv` file.

6. **Build cluster-allele-G-domain mappings**:
   ```bash
   python build_cluster_alleles.py
   ```
   Produces timestamped outputs:
   - `cluster_alleles_by_cluster_<timestamp>.tsv`
   - `allele_to_gdomain_<timestamp>.tsv`
   - `file_allele_gdomain_cluster_<timestamp>.tsv` (most important for future clustering runs)
   
   **Note**: This step uses the `pdb_cluster_mapping.tsv` created in step 5. If you're running `hierachical_clustering_all.py` for the first time (step 4), you'll need an existing `file_allele_gdomain_cluster_*.tsv` file. After step 6, you'll have a fresh one that can be used for subsequent clustering runs.

7. **Create train/val/test split HDF5 files** (optional, if you need the split datasets):
   ```bash
   python split_clusters_to_hdf5.py
   ```
   Produces: `./splits/100k_train.hdf5`, `./splits/100k_valid.hdf5`, `./splits/100k_test.hdf5`
   
   **Note**: This step uses `pdb_cluster_mapping.tsv` (from step 5) and defines the split based on cluster membership:
   - Cluster 7 → validation
   - Clusters 3, 4, 6 → test
   - All other clusters → training

All scripts are designed to run from within this `data_processing` directory. You may need to adjust hard-coded paths (especially the X-ray dataset path) to match your local filesystem layout.
