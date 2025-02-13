import torch
import numpy as np
import torch.nn as nn
from torch_scatter import scatter_mean

from torch_geometric.data import Data, Batch

from model.egnn import EGNN, GNN
from model.positional_encoding import sin_pE

"""
This file sets up the neural network for the generative framework.
"""

class NN_Model(nn.Module):

    def __init__(
            self,
            architecture: str,
            features_fixed: bool,
            position_encoding: bool,
            position_encoding_dim: int,
            network_params,
            num_atoms: int,
            num_residues: int,
            device: str,
    ):
        
        """
        Parameters:

        architecture: name of the neural network to use
        features_fixed: option to train to generate new peptides, instead of just structures (not implemented)
        position_encoding: switch for the positional encoding
        position_encoding_dim: size of the positional encoding
        network_params: parameters for the specific neural network architecture
        num_atoms: number of amino-acid types in peptide
        num_residues: number of amino-acid types in protein pocket
        device: hardware to run on (should be a gpu)
        
        """

        super().__init__()

        self.architecture = architecture
        self.features_fixed = features_fixed
        self.x_dim = 3
        self.act_fn = nn.SiLU()

        self.joint_dim = network_params.joint_dim
        self.hidden_dim = network_params.hidden_dim
        self.num_layers = network_params.num_layers
        self.conditioned_on_time = network_params.conditioned_on_time

        # positional encoding
        self.position_encoding = position_encoding
        self.pE_dim = position_encoding_dim

        # edge parameters
        self.edge_embedding_dim = network_params.edge_embedding_dim
        self.edge_cutoff_l = network_params.edge_cutoff_ligand
        self.edge_cutoff_p = network_params.edge_cutoff_pocket
        self.edge_cutoff_i = network_params.edge_cutoff_interaction

        if architecture == 'egnn' or 'gnn':

            # edge embedding
            self.edge_embedding_dim = network_params.edge_embedding_dim
            if self.edge_embedding_dim is not None: 
                self.edge_embedding = nn.Embedding(self.x_dim, self.edge_embedding_dim)
            else: 
                self.edge_embedding = None
            self.edge_embedding_dim = 0 if self.edge_embedding_dim is None else self.edge_embedding_dim

            # node embedding
            self.atom_encoder = nn.Sequential(
                nn.Linear(num_atoms, 2 * num_atoms),
                self.act_fn,
                nn.Linear(2 * num_atoms, self.joint_dim)
            )

            self.atom_decoder = nn.Sequential(
                nn.Linear(self.joint_dim, 2 * num_atoms),
                self.act_fn,
                nn.Linear(2 * num_atoms, num_atoms)
            )

            self.residue_encoder = nn.Sequential(
                nn.Linear(num_residues, 2 * num_residues),
                self.act_fn,
                nn.Linear(2 * num_residues, self.joint_dim)
            )

            self.residue_decoder = nn.Sequential(
                nn.Linear(self.joint_dim, 2 * num_residues),
                self.act_fn,
                nn.Linear(2 * num_residues, num_residues)
            )

            if self.position_encoding:
                self.joint_dim += self.pE_dim

            if self.conditioned_on_time:
                self.joint_dim += 1

            if architecture == 'egnn':

                self.egnn = EGNN(in_node_nf=self.joint_dim, in_edge_nf=self.edge_embedding_dim,
                                 hidden_nf=self.hidden_dim, device=device, act_fn=self.act_fn,
                                 n_layers=self.num_layers, attention=network_params.attention, tanh=network_params.tanh,
                                 norm_constant=network_params.norm_constant,
                                 inv_sublayers=network_params.inv_sublayers, sin_embedding=network_params.sin_embedding,
                                 normalization_factor=network_params.normalization_factor,
                                 aggregation_method=network_params.aggregation_method,
                                 reflection_equiv=network_params.reflection_equivariant) # edge_sin_attr=self.edge_sin_attrs

            else:
                
                self.gnn = GNN(in_node_nf=self.joint_dim + self.x_dim, in_edge_nf=self.edge_embedding_dim,
                               hidden_nf=self.hidden_dim, out_node_nf=self.x_dim + self.joint_dim,
                               device=device, act_fn=self.act_fn, n_layers=self.num_layers,
                               attention=network_params.attention, normalization_factor=network_params.normalization_factor,
                               aggregation_method=network_params.aggregation_method)

        else:
            raise Exception(f"Wrong architecture {architecture}")





    def forward(self, z_t_mol, z_t_pro, t, molecule_idx, protein_pocket_idx, molecule_pos=None):

        '''
        Takes in noised sample and outputs predicted added noise

        Inputs:
        z_t_mol: size = [batch_node_dim_mol, x_dim + num_atoms + position_encoding_dim]
        z_t_pro: size = [batch_node_dim_mol, x_dim + num_residues + position_encoding_dim]
        t: size = [batch_node_dim]
        molecule['idx']: size = [batch_node_dim]
        protein_pocket['idx']: size = [batch_node_dim]

        return epsilon_hat_mol: size = [batch_node_dim_mol, x + num_atoms], 
                epsilon_hat_pro: size = [batch_node_dim_pro, x + num_residues]
        '''

        idx_joint = torch.cat((molecule_idx, protein_pocket_idx), dim=0)
        x_mol = z_t_mol[:,:self.x_dim].clone()
        x_pro = z_t_pro[:,:self.x_dim].clone()

        # add edges to the graph (edges are determined by distance cutoffs)
        edges = self.get_edges(molecule_idx, protein_pocket_idx, x_mol, x_pro)
        assert torch.all(idx_joint[edges[0]] == idx_joint[edges[1]])

        if self.architecture == 'egnn' or self.architecture == 'gnn':

            # encode z_t_mol, z_t_pro (possible need to .clone() the inputs)
            h_mol = self.atom_encoder(z_t_mol[:,self.x_dim:]).clone()
            h_pro = self.residue_encoder(z_t_pro[:,self.x_dim:]).clone()

            # position_encoding
            if self.position_encoding:
                # TODO: molecule_pos[molecule_idx] not correct !!!
                pE = sin_pE(molecule_pos, self.pE_dim)
                h_mol = torch.cat([h_mol, pE], dim=1)
                h_pro = torch.cat([h_pro, torch.zeros((h_pro.shape[0], self.pE_dim), device=h_pro.device)], dim=1)

            # combine molecule and protein in joint space
            x_joint = torch.cat((z_t_mol[:,:self.x_dim], z_t_pro[:,:self.x_dim]), dim=0) # [batch_node_dim_mol + batch_node_dim_pro, 3]
            h_joint = torch.cat((h_mol, h_pro), dim=0) # [batch_node_dim_mol + batch_node_dim_pro, joint_dim]

            # add time conditioning
            if self.conditioned_on_time:
                h_time = t[idx_joint]
                h_joint = torch.cat([h_joint, h_time], dim=1)

            # add edge embedding and types
            if self.edge_embedding_dim > 0:
                # 0: ligand-pocket, 1: ligand-ligand, 2: pocket-pocket
                edge_types = torch.zeros(edges.size(1), dtype=int, device=edges.device)
                edge_types[(edges[0] < len(molecule_idx)) & (edges[1] < len(molecule_idx))] = 1
                edge_types[(edges[0] >= len(molecule_idx)) & (edges[1] >= len(molecule_idx))] = 2

                # Learnable embedding
                edge_types = self.edge_embedding(edge_types)
            else:
                edge_types = None

            if self.architecture == 'egnn':

                # choose whether to get protein_pocket corrdinates fixed
                protein_pocket_fixed = torch.cat((torch.ones_like(molecule_idx), torch.zeros_like(protein_pocket_idx))).unsqueeze(1)

                # neural net forward pass
                h_new, x_new = self.egnn(h_joint, x_joint, edges,
                                            update_coords_mask=protein_pocket_fixed,
                                            batch_mask=idx_joint, edge_attr=edge_types)
                
                # calculate displacement vectors
                displacement_vec = (x_new - x_joint)

            elif self.architecture == 'gnn':

                # GNN
                x_h_joint = torch.cat([x_joint, h_joint], dim=1)
                out = self.gnn(x_h_joint, edges, node_mask=None, edge_attr=edge_types)
                displacement_vec = out[:, :self.x_dim]
                h_new = out[:, self.x_dim:]

            else:
                raise Exception(f"Wrong architecture {self.architecture}")

        else:
            raise Exception(f"Wrong architecture {self.architecture}")
        
        # remove time dim
        if self.conditioned_on_time:
            # Slice off last dimension which represented time.
            h_new = h_new[:, :-1]

        # remove position information
        if self.position_encoding:
            # Slice off last dimension which represented postional encoding.
            h_new = h_new[:, :-self.pE_dim]
                
        # decode h_new
        h_new_mol = self.atom_decoder(h_new[:len(molecule_idx)])
        h_new_pro = self.residue_decoder(h_new[len(molecule_idx):])

        # might not be necessary but let's see
        if torch.any(torch.isnan(displacement_vec)):
            raise ValueError("NaN detected in EGNN output")

        # output
        epsilon_hat_mol = torch.cat((displacement_vec[:len(molecule_idx)], h_new_mol), dim=1)
        epsilon_hat_pro = torch.cat((displacement_vec[len(molecule_idx):], h_new_pro), dim=1)

        return epsilon_hat_mol, epsilon_hat_pro
    
    def get_edges(self, batch_mask_ligand, batch_mask_pocket, x_ligand, x_pocket): 
        '''
        get edges based on edge cutoff
        ''' 
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l)

        if self.edge_cutoff_p is not None:
            adj_pocket = adj_pocket & (torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p)

        if self.edge_cutoff_i is not None:
            adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i)

        adj = torch.cat((torch.cat((adj_ligand, adj_cross), dim=1),
                         torch.cat((adj_cross.T, adj_pocket), dim=1)), dim=0)
        edges = torch.stack(torch.where(adj), dim=0)

        return edges