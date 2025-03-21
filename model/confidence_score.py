import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """A simple Graph Convolutional Layer (without PyG)."""
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        """
        x: (num_nodes, in_dim)  - Node features
        adj: (num_nodes, num_nodes)  - Adjacency matrix
        """
        h = torch.mm(adj, x)  # Message passing (adjacency-weighted sum)
        h = self.linear(h)  # Apply learnable transformation
        return F.relu(h)  # Non-linearity

class Confidence_Score(nn.Module):
    def __init__(self, in_dim):
        super(Confidence_Score, self).__init__()
        hidden_dim = 2* in_dim
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.confidence_head = nn.Linear(hidden_dim, 1)  # Predicts confidence

    def forward(self, x, batch):
        """
        x: (num_nodes, in_dim)  - Node features
        batch: (num_nodes,)  - Graph indices per node
        """
        adj = create_full_adj(batch)  # Fully connected graph adjacency
        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)  # Apply a second GCN layer

        # Confidence head (node-wise)
        #confidence = torch.sigmoid(self.confidence_head(h))  # Normalize confidence to [0,1]
        confidence = self.confidence_head(h)
        confidence = F.softplus(confidence) / (1 + F.softplus(confidence))

        return confidence

def create_full_adj(batch):
    """
    Constructs an adjacency matrix for a batch of fully connected graphs.
    Each graph is fully connected within itself, but isolated from others.
    
    batch: (num_nodes,)  - Graph index for each node.
    Returns: (num_nodes, num_nodes) adjacency matrix.
    """
    num_nodes = batch.shape[0]
    adj = torch.zeros((num_nodes, num_nodes), device=batch.device)  # Initialize adjacency matrix
    
    for graph_id in batch.unique():  # Iterate through unique graph indices
        indices = (batch == graph_id).nonzero(as_tuple=True)[0]  # Nodes in the same graph
        adj[indices[:, None], indices] = 1  # Fully connect nodes within the graph

    return adj
