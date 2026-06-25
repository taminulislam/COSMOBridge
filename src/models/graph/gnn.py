"""
Graph Neural Network for predicting ionic liquid properties from molecular graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_TORCH_GEOMETRIC_AVAILABLE = True
try:
    from torch_geometric.nn import (
        GCNConv,
        GATConv,
        GINConv,
        global_mean_pool,
        global_add_pool,
        BatchNorm,
    )
    from torch_geometric.nn import aggr
except ImportError:
    _TORCH_GEOMETRIC_AVAILABLE = False
    print(
        "WARNING: torch_geometric is not installed. "
        "MolecularGNN can be imported but cannot be instantiated. "
        "Install with: pip install torch-geometric"
    )


class MolecularGNN(nn.Module):
    """Graph Neural Network for molecular property prediction.

    Supports GCN, GAT, and GIN convolution types with configurable
    pooling strategies (mean, sum, attention).

    Parameters
    ----------
    atom_feature_dim : int
        Dimensionality of input atom features.
    bond_feature_dim : int
        Dimensionality of input bond/edge features.
    hidden_dim : int
        Hidden layer dimensionality throughout the network.
    num_layers : int
        Number of GNN convolution layers.
    conv_type : str
        Type of graph convolution: "GCN", "GAT", or "GIN".
    heads : int
        Number of attention heads (only used for GAT).
    dropout : float
        Dropout probability.
    pooling : str
        Global pooling strategy: "mean", "sum", or "attention".
    num_targets : int
        Number of output targets to predict.
    """

    def __init__(
        self,
        atom_feature_dim: int = 22,
        bond_feature_dim: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 4,
        conv_type: str = "GAT",
        heads: int = 4,
        dropout: float = 0.3,
        pooling: str = "mean",
        num_targets: int = 7,
        aux_feature_dim: int = 0,
    ):
        if not _TORCH_GEOMETRIC_AVAILABLE:
            raise RuntimeError(
                "torch_geometric is required to instantiate MolecularGNN. "
                "Install with: pip install torch-geometric"
            )

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_type = conv_type.upper()
        self.heads = heads
        self.dropout = dropout
        self.pooling = pooling

        # Initial linear projection of atom features to hidden_dim
        self.atom_projection = nn.Linear(atom_feature_dim, hidden_dim)

        # Build GNN convolution layers, batch norms, and dropout
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_dim
            # For GAT, the output of each layer is heads * out_per_head.
            # We keep out_per_head = hidden_dim // heads so output = hidden_dim.
            if self.conv_type == "GAT":
                if i > 0:
                    # Previous GAT layer outputs heads * (hidden_dim // heads)
                    in_dim = hidden_dim
                out_per_head = hidden_dim // heads
                conv = GATConv(
                    in_channels=in_dim,
                    out_channels=out_per_head,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
            elif self.conv_type == "GCN":
                conv = GCNConv(in_channels=in_dim, out_channels=hidden_dim)
            elif self.conv_type == "GIN":
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                conv = GINConv(nn=mlp)
            else:
                raise ValueError(
                    f"Unsupported conv_type '{conv_type}'. Choose from 'GCN', 'GAT', 'GIN'."
                )

            self.convs.append(conv)
            # For GAT, the effective output dim is heads * out_per_head = hidden_dim
            self.batch_norms.append(BatchNorm(hidden_dim))

        # Global pooling
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "sum":
            self.pool = global_add_pool
        elif pooling == "attention":
            self.pool = aggr.AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                ),
            )
        else:
            raise ValueError(
                f"Unsupported pooling '{pooling}'. Choose from 'mean', 'sum', 'attention'."
            )

        # Optional auxiliary feature projection (e.g., temperature features)
        self.aux_feature_dim = aux_feature_dim
        pred_input_dim = hidden_dim + aux_feature_dim

        # MLP prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(pred_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets),
        )

    def _pool_graph(self, x, batch):
        """Apply global pooling to obtain graph-level representations."""
        if self.pooling == "attention":
            return self.pool(x, index=batch)
        return self.pool(x, batch)

    def get_features(self, atom_features, edge_index, bond_features, batch):
        """Return pooled graph-level representation before the prediction head.

        Parameters
        ----------
        atom_features : Tensor of shape (N, atom_feature_dim)
        edge_index : LongTensor of shape (2, E)
        bond_features : Tensor of shape (E, bond_feature_dim)
        batch : LongTensor of shape (N,)

        Returns
        -------
        Tensor of shape (B, hidden_dim) where B is the number of graphs in the batch.
        """
        x = self.atom_projection(atom_features)

        for conv, bn in zip(self.convs, self.batch_norms):
            if self.conv_type == "GAT":
                x = conv(x, edge_index)
            elif self.conv_type == "GCN":
                x = conv(x, edge_index)
            elif self.conv_type == "GIN":
                x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_repr = self._pool_graph(x, batch)
        return graph_repr

    def forward(self, atom_features, edge_index, bond_features, batch, features=None, **kwargs):
        """Forward pass: graph encoding followed by property prediction.

        Parameters
        ----------
        atom_features : Tensor of shape (N, atom_feature_dim)
        edge_index : LongTensor of shape (2, E)
        bond_features : Tensor of shape (E, bond_feature_dim)
        batch : LongTensor of shape (N,)
        features : Tensor of shape (B, aux_feature_dim), optional
            Auxiliary features (e.g., temperature, 1/T, T², T³) concatenated
            with graph features before the prediction head.

        Returns
        -------
        Tensor of shape (B, num_targets) with predicted properties.
        """
        graph_repr = self.get_features(atom_features, edge_index, bond_features, batch)
        if features is not None and self.aux_feature_dim > 0:
            graph_repr = torch.cat([graph_repr, features], dim=-1)
        return self.prediction_head(graph_repr)
