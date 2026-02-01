
# MHC-Diff
Structure-based Equivarinat Diffusion Models for cancer immunotherapy

![Project Image](./Diffusion%20Chain.png)

## Overview

Equivariant Diffusion Model for generating peptide-MHC structures

## Installation

To set up the environment, please follow the instructions carefully. **Ensure that you install the packages in the correct versions and in the specified order. Installing them out of order can result in package conflict issues.**

1. **Create a new conda environment**:  
   ```sh
   conda create --yes --name mol python=3.10 numpy matplotlib
   conda activate mol
   ```

2. **Install PyTorch with CUDA support**:  
   ```sh
   conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
   ```

3. **Install PyG (PyTorch Geometric)**:  
   ```sh
   conda install pyg==2.3.1 -c pyg -y
   ```

4. **Install additional PyG dependencies**:  
   ```sh
   pip3 install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
   ```  
   (Note: This step may take some time.)

5. **Install other Python packages**:  
   ```sh
   pip3 install wandb
   pip install h5py
   pip3 install pytorch_lightning==1.8.6
   conda install -c conda-forge biopython=1.79
   pip install prody
   pip install pandas
   ```

## 📁 Configuration

All training and inference runs are controlled via a single YAML config file, e.g.:

```
configs/new_config.yml
```

The key parameters are grouped as follows:

### 🔧 General Training Parameters

| Parameter     | Description                               |
| ------------- | ----------------------------------------- |
| `logdir`      | Output directory for logs and checkpoints |
| `project`     | Project name (e.g. for Weights & Biases)  |
| `run_name`    | Run identifier                            |
| `entity`      | WandB user or team                        |
| `num_epochs`  | Max number of training epochs             |
| `device`      | Device to train on (e.g. `cuda`)          |
| `gpus`        | Number of GPUs                            |
| `batch_size`  | Training batch size                       |
| `lr`          | Learning rate                             |
| `num_workers` | Data loading workers                      |

---

### 🧪 Task Configuration

| Parameter                      | Description                                        |
| ------------------------------ | -------------------------------------------------- |
| `task_params.features_fixed`   | Whether node features are fixed (True = fixed)     |
| `task_params.confidence_score` | Whether to compute a confidence score (False = no) |

---

### 📚 Dataset Configuration

| Parameter                     | Description                       |
| ----------------------------- | --------------------------------- |
| `data_dir`                    | Path to training/validation data  |
| `dataset`                     | Dataset identifier                |
| `dataset_params.num_atoms`    | Max number of atoms per structure |
| `dataset_params.num_residues` | Max number of residues            |
| `dataset_params.norm_values`  | Feature normalization factors     |

---

### 🌀 Generative Model Parameters

| Parameter                                       | Description                                               |
| ----------------------------------------------- | --------------------------------------------------------- |
| `generative_model`                              | Type of model (e.g. `conditional_diffusion`)              |
| `generative_model_params.timesteps`             | Number of diffusion steps                                 |
| `generative_model_params.position_encoding`     | Use positional encoding                                   |
| `generative_model_params.position_encoding_dim` | Dimensionality of PE                                      |
| `generative_model_params.com_handling`          | How to handle center-of-mass (`peptide`, `protein`, etc.) |
| `generative_model_params.sampling_stepsize`     | Step size for reverse sampling                            |
| `generative_model_params.noise_scaling`         | Optional custom noise schedule                            |
| `generative_model_params.high_noise_training`   | Enable training with more noise                           |

---

### 🧠 Architecture Parameters

| Parameter                                                 | Description                                       |
| --------------------------------------------------------- | ------------------------------------------------- |
| `architecture`                                            | Model type (e.g. `egnn`)                          |
| `network_params.conditioned_on_time`                      | Whether to use time embedding                     |
| `network_params.joint_dim`                                | Joint latent dim for ligand + protein             |
| `network_params.hidden_dim`                               | Hidden layer dim                                  |
| `network_params.num_layers`                               | Number of layers                                  |
| `network_params.edge_embedding_dim`                       | Dim of edge features                              |
| `network_params.edge_cutoff_ligand`                       | Cutoff for ligand-ligand edges                    |
| `network_params.edge_cutoff_pocket`                       | Cutoff for pocket-pocket edges                    |
| `network_params.edge_cutoff_interaction`                  | Cutoff for ligand-pocket edges                    |
| `network_params.attention`, `tanh`, `sin_embedding`, etc. | EGNN-specific flags                               |
| `network_params.aggregation_method`                       | Method for node aggregation (`sum`, `mean`, etc.) |
| `network_params.reflection_equivariant`                   | Toggle reflection equivariance                    |

---

### 📈 Evaluation and Sampling

| Parameter                | Description                               |
| ------------------------ | ----------------------------------------- |
| `checkpoint`             | Path to trained model checkpoint          |
| `num_samples`            | Number of samples per input               |
| `sample_batch_size`      | Batch size during sampling                |
| `sampling_without_noise` | If true, performs deterministic denoising |
| `sample_savepath`        | Output directory for sampled structures   |

---

## 🏋️ Training

To train locally:

```bash
python train.py --config configs/new_config.yml
```

To train on a cluster using SLURM (e.g. in a job script):

```bash
srun python -u train.py --config /absolute/path/to/new_config.yml
```

---

## 🧪 Inference / Structure Generation

To generate samples from a trained model:

```bash
python test.py --config configs/new_config.yml
```

Or on a cluster:

```bash
srun python -u test.py --config /absolute/path/to/new_config.yml
```

---

## 📊 Result Analysis

After running inference with `test.py`, you can analyze the results using the scripts in the `analysis/` folder.

### Analyzing 8K Dataset Results (10-fold CV)

```bash
python analysis/analyze_8k_results.py \
    --results-dir ./checkpoints \
    --output-dir ./analysis_output
```

This script:
- Loads `samples.pkl.gz` files from fold directories
- Computes RMSD statistics for X-ray and PANDORA structures
- Reports both **best-of-10** and **average-of-10** (ensemble) metrics
- Tracks divergent samples (>10Å cutoff)

### Analyzing 100K Dataset Results (Multi-allele)

```bash
python analysis/analyze_100k_results.py \
    --results-dir ./checkpoints/100k \
    --output-dir ./analysis_output \
    --cluster-mapping data_processing/pdb_cluster_mapping.tsv \
    --allele-info data_processing/mhci_alleles.tsv
```

This script provides:
- Overall RMSD statistics
- Per-cluster breakdown (G-domain clusters 1-10)
- Per-peptide-length analysis
- Divergent sample tracking (>20Å cutoff)

### Output Files

| File | Description |
| ---- | ----------- |
| `*_results_summary.csv` | Per-fold/file statistics table |
| `*_results_detailed.pkl` | Full results with RMSD distributions |

### Example Output

```
============================================================
OVERALL SUMMARY (8K Dataset - HLA-A*02:01 9-mer)
============================================================

X-ray RMSD (best-of-10) across all folds:
  N:      160
  Mean:   0.457 Å
  Median: 0.448 Å

Per-fold averages:
  X-ray best-of-10 mean:   0.428 Å
  X-ray avg-of-10 mean:    1.055 Å
```
