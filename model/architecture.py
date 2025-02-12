import torch
import numpy as np
import torch.nn as nn
from torch_scatter import scatter_mean

from torch_geometric.data import Data, Batch

from model.egnn import EGNN, GNN
from model.positional_encoding import sin_pE

"""
This file sets up the neural network for the generative framework.
Before we can pass the data to our neural network we need to encode the molecule and the protein_pocket in a joint space.
"""

class NN_Model(nn.Module):

    def __init__(
            self,
            architecture: str,
            protein_pocket_fixed: bool,
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

        
        """

        super().__init__()

        self.architecture = architecture
        self.protein_pocket_fixed = protein_pocket_fixed
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

        # possible to use edge_embedding if I want to distinguish between molecule-moelcule, pocket-molecule edges, usw.

        if architecture == 'ponita':

            # here we can add positional
            self.atom_encoder = nn.Linear(num_atoms, self.joint_dim)

            self.atom_decoder = nn.Linear(self.joint_dim, num_atoms)

            self.residue_encoder = nn.Linear(num_residues, self.joint_dim)

            self.residue_decoder = nn.Linear(self.joint_dim, num_residues)

            if self.position_encoding:
                self.joint_dim += self.pE_dim

            if self.conditioned_on_time:
                self.joint_dim += 1

            # dimensions for ponita model
            in_channels_scalar = self.joint_dim
            in_channels_vec = 1 # shouldn't this be 1 or does it take the pos by default?
            # TODO: check how to properly use scalar vs vector outputs
            out_channels_scalar = self.joint_dim # updated features
            out_channels_vec = 1 # displacment vector

            self.ponita = Ponita(in_channels_scalar + in_channels_vec,
                            self.hidden_dim,
                            out_channels_scalar,
                            self.num_layers,
                            output_dim_vec=out_channels_vec,
                            radius=network_params.radius,
                            num_ori=network_params.num_ori,
                            basis_dim=network_params.basis_dim,
                            degree=network_params.degree,
                            widening_factor=network_params.widening_factor,
                            layer_scale=network_params.layer_scale,
                            task_level='node',
                            multiple_readouts=network_params.multiple_readouts,
                            lift_graph=True)
            
        elif architecture == 'egnn' or 'gnn':

            # edge embedding
            self.edge_embedding_dim = network_params.edge_embedding_dim
            if self.edge_embedding_dim is not None: 
                self.edge_embedding = nn.Embedding(self.x_dim, self.edge_embedding_dim)
            else: 
                self.edge_embedding = None
            self.edge_embedding_dim = 0 if self.edge_embedding_dim is None else self.edge_embedding_dim

            # same encoder, decoders as in [Schneuing et al. 2023]

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
        Inputs:
        z_t_mol: [batch_node_dim_mol, x + num_atoms]
        z_t_pro: [batch_node_dim_pro, x + num_residues]
        t: int
        molecule['idx']: [batch_node_dim]
        protein_pocket['idx']: [batch_node_dim]

        return epsilon_hat_mol [batch_node_dim_mol, x + num_atoms], 
                epsilon_hat_pro [batch_node_dim_pro, x + num_residues]
        '''

        idx_joint = torch.cat((molecule_idx, protein_pocket_idx), dim=0)
        x_mol = z_t_mol[:,:self.x_dim].clone()
        x_pro = z_t_pro[:,:self.x_dim].clone()

        # add edges to the graph
        edges = self.get_edges(molecule_idx, protein_pocket_idx, x_mol, x_pro)
        assert torch.all(idx_joint[edges[0]] == idx_joint[edges[1]])

        if self.architecture == 'ponita':

            # (1) need z_t_mol and z_t_pro to be of the same size but no nonlinear embedding
            h_mol = self.atom_encoder(z_t_mol[:,self.x_dim:])
            h_pro = self.residue_encoder(z_t_pro[:,self.x_dim:])
            # combine molecule and protein in joint space for displacment_vector calculation
            x_joint = torch.cat((z_t_mol[:,:self.x_dim], z_t_pro[:,:self.x_dim]), dim=0) # [batch_node_dim_mol + batch_node_dim_pro, 3]

            # position_encoding
            if self.position_encoding:
                pE = sin_pE(molecule_pos, self.pE_dim)
                h_mol = torch.cat([h_mol, pE], dim=1)
                h_pro = torch.cat([h_pro, torch.zeros((h_pro.shape[0], self.pE_dim), device=h_pro.device)], dim=1)

            # (2) add time conditioning
            if self.conditioned_on_time:
                h_time_mol = t[molecule_idx]
                h_time_pro = t[protein_pocket_idx]
                h_mol = torch.cat([h_mol, h_time_mol], dim=1)
                h_pro = torch.cat([h_pro, h_time_pro], dim=1)

            # (3) need to save [x, h, edges] as [graph.pos, graph.x, graph.edge_index]
            # (might want to make a helper function for this, can use molecule['size'] object)
            # for orientation add .vec object 
            _, counts_mol = torch.unique(molecule_idx, return_counts=True)
            _, counts_pro = torch.unique(protein_pocket_idx, return_counts=True)
            h_mol_split = torch.split(h_mol, counts_mol.tolist()) # list([graph_num_nodes, num_atoms], len(batch_size))
            h_pro_split = torch.split(h_pro, counts_pro.tolist()) # list([graph_num_nodes, num_residues], len(batch_size))
            x_mol_split = torch.split(x_mol, counts_mol.tolist()) # list([graph_num_nodes, 3], len(batch_size))
            x_pro_split = torch.split(x_pro, counts_pro.tolist()) # list([graph_num_nodes, 3], len(batch_size))
            h_split = [torch.cat((h_mol_split[i], h_pro_split[i]), dim=0) for i in range(len(h_mol_split))]
            x_split = [torch.cat((x_mol_split[i], x_pro_split[i]), dim=0) for i in range(len(x_mol_split))]

            graphs = [Data(x=h_split[i], pos=x_split[i]) for i in range(len(h_mol_split))]
            batched_graph = Batch.from_data_list(graphs)
            # add edge_index
            # TODO: make sure that adding edge_index works like that (might be wrong)
            batched_graph.edge_index = edges

            # (4) TODO: choose whether to get protein_pocket corrdinates fixed (might need to modify ponita)
            if self.protein_pocket_fixed:
                # raise NotImplementedError
                protein_pocket_fixed = torch.cat((torch.ones_like(molecule_idx), torch.zeros_like(protein_pocket_idx))).unsqueeze(1)
            else:
                protein_pocket_fixed = None

            # (5) ponita forward pass (x_new could also be the displacment vector directly)
            h_new, x_new = self.ponita(batched_graph)

            # (6) calculate displacement vectors (possibly not necessary see step 5.)
            # for orientation predict orientation noise .vec (add and need to normalize)
            # rotation_matrix or quaternions (rotations rel. to corrdinate systems) (Gramsmitth orthogonalisation of 3 vec)
            # 
            # TODO: retruns predicted noise x_new = [batch, 1, 3]
            # x_new = h_new[:,:self.x_dim]
            x_new = x_new.squeeze(1) # ([batch, 1, 3] - [batch, 3])
            displacement_vec = (x_new - x_joint)

        elif self.architecture == 'egnn' or self.architecture == 'gnn':

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

            #######################################################################
            # positional edge features (Siem)
            # if self.edge_sin_attr:
            #     edge_attr = torch.zeros(edges.shape[1], h_mol.shape[1]).to(self.device)
            #     edges_atoms = self.get_edges_molecules(molecule_idx)
            #     edge_diff = edges_atoms[0] - edges_atoms[1]
            #     atom_edge_attr = self.sin_pE(edge_diff, self.pE_dim)

            #     _, sizes = torch.unique(edges_atoms[0], return_counts=True)
            #     atom_nodes = edges_atoms[0].unique()
            #     atom_starts = torch.searchsorted(edges[0], atom_nodes)
            #     atom_index = torch.repeat_interleave(atom_starts, sizes)

            #     atom_index = atom_index + torch.concatenate(
            #         [torch.arange(s).to(self.device) for s in sizes]
            #     )
            #     edge_attr[atom_index] = atom_edge_attr
            #######################################################################

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

        # remove position information (TODO: careful with ponita (pE not added there yet))
        if self.position_encoding:
            # Slice off last dimension which represented postional encoding.
            h_new = h_new[:, :-self.pE_dim]
                
        # decode h_new
        h_new_mol = self.atom_decoder(h_new[:len(molecule_idx)])
        h_new_pro = self.residue_decoder(h_new[len(molecule_idx):])

        # might not be necessary but let's see
        if torch.any(torch.isnan(displacement_vec)):
            raise ValueError("NaN detected in EGNN output")
        
        # remove mean batch of the position only for joint
        # TODO: this might have been wrong
        if self.protein_pocket_fixed == False:
            displacement_vec = displacement_vec - scatter_mean(displacement_vec, idx_joint, dim=0)[idx_joint]

        # output
        epsilon_hat_mol = torch.cat((displacement_vec[:len(molecule_idx)], h_new_mol), dim=1)
        epsilon_hat_pro = torch.cat((displacement_vec[len(molecule_idx):], h_new_pro), dim=1)

        return epsilon_hat_mol, epsilon_hat_pro
    
    def get_edges_molecules(self, batch_mask_ligand): 
        '''
        returns peptide edges only
        first line is a very smart trick to get the maximum edge_index (not my trick)
        ''' 
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]

        edges = torch.stack(torch.where(adj_ligand), dim=0)

        return edges
    
    def get_edges(self, batch_mask_ligand, batch_mask_pocket, x_ligand, x_pocket): 
        '''
        function copied from [Schneuing et al. 2023]
        -> need to write my own function
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