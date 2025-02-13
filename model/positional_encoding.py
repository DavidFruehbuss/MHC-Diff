import torch

def sin_pE(node_indices, num_features=10):
    """
    Compute sinusoidal node features for positional encoding of graph nodes in a batch,
    given node indices as relative positions within each graph and a batch index tensor 
    that indicates graph membership for each node.

    # Need to update this discription

    Args:
        node_indices (torch.Tensor): 1D Tensor containing node indices representing 
                                     the relative position of each node within its graph.
        batch_indices (torch.Tensor): 1D Tensor of the same length as node_indices, 
                                      indicating the graph number for each node.
        num_features (int): Number of features in the encoding.

    Returns:
        torch.Tensor: Sinusoidal node features for the given node indices.
    """
    device = node_indices.device
    num_nodes = node_indices.size(0)
    
    # Compute the frequencies for the sinusoidal functions
    frequencies = torch.pow(10000, -torch.arange(0, num_features, dtype=torch.float32, device=device) * 2 / num_features)
    
    # Expand node_indices for each feature
    node_indices_expanded = node_indices.unsqueeze(1).expand(num_nodes, num_features)

    # Compute sinusoidal features
    sinusoidal_features = torch.zeros(num_nodes, num_features, device=device)
    sinusoidal_features[:, 0::2] = torch.sin(node_indices_expanded[:, 0::2] * frequencies[0::2])  # Even indices: sine
    sinusoidal_features[:, 1::2] = torch.cos(node_indices_expanded[:, 1::2] * frequencies[0::2])  # Odd indices: cosine

    return sinusoidal_features
