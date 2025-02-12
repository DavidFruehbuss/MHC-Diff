
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