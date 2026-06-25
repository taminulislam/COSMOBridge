"""Directed Message Passing Neural Network (D-MPNN) for molecular graphs.

Incorporates key concepts from Chemprop (Yang et al., 2019):
1. Directed message passing — messages flow along directed bonds, avoiding tottering
2. Bond-level hidden states — messages live on edges, not nodes
3. Bond features as first-class inputs — concatenated into messages directly
4. Sum pooling — preserves molecular size information

This module replaces GAT-GNN as the graph encoder in our multimodal framework,
keeping the same interface (get_features returns a graph-level vector).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import global_add_pool, global_mean_pool
except ImportError:
    raise RuntimeError("torch_geometric required")


class DirectedMPNN(nn.Module):
    """Directed Message Passing Neural Network.

    Messages are passed along DIRECTED edges. Each directed edge (u→v) maintains
    a hidden state that is updated by aggregating messages from all edges pointing
    INTO u (except the reverse edge v→u), avoiding information tottering.

    Parameters
    ----------
    atom_feature_dim : int
        Input atom feature dimension.
    bond_feature_dim : int
        Input bond feature dimension.
    hidden_dim : int
        Hidden state dimension for edge messages.
    num_layers : int
        Number of message passing iterations (depth).
    dropout : float
        Dropout probability.
    num_targets : int
        Number of output targets (0 = feature extractor only).
    """

    def __init__(
        self,
        atom_feature_dim: int = 22,
        bond_feature_dim: int = 7,
        hidden_dim: int = 300,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_targets: int = 0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Initial edge message: W_i * [atom_features(u), bond_features(u→v)]
        self.W_i = nn.Linear(atom_feature_dim + bond_feature_dim, hidden_dim, bias=False)

        # Message passing weight (shared across layers)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Node aggregation: W_o * [atom_features(v), sum(incoming messages)]
        self.W_o = nn.Linear(atom_feature_dim + hidden_dim, hidden_dim)

        # Layer norm for message passing
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Prediction head (if num_targets > 0)
        self.num_targets = num_targets
        if num_targets > 0:
            self.prediction_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_targets),
            )

    def forward(self, atom_features, edge_index, bond_features, batch,
                features=None, **kwargs):
        """Full forward: encoding + prediction."""
        h = self.get_features(atom_features, edge_index, bond_features, batch)
        if self.num_targets > 0:
            return self.prediction_head(h)
        return h

    def get_features(self, atom_features, edge_index, bond_features, batch):
        """Encode molecular graph into a fixed-size vector.

        Returns
        -------
        Tensor of shape (B, hidden_dim) — graph-level representation.
        """
        num_atoms = atom_features.size(0)
        num_edges = edge_index.size(1)

        if num_edges == 0:
            # No edges — just pool atom features
            h_atom = torch.zeros(num_atoms, self.hidden_dim,
                                  device=atom_features.device)
            return global_add_pool(h_atom, batch)

        src, dst = edge_index[0], edge_index[1]

        # ── Step 1: Initialize edge messages ──
        # h(u→v)_0 = ReLU(W_i * [f_atom(u), f_bond(u→v)])
        edge_input = torch.cat([atom_features[src], bond_features], dim=-1)
        h_edge = F.relu(self.W_i(edge_input))  # (E, hidden)

        # ── Build reverse edge index for tottering prevention ──
        # For each edge (u→v), find the reverse edge (v→u)
        # Chemprop avoids sending message from v→u back to u→v
        reverse_idx = self._build_reverse_index(edge_index)

        # ── Step 2: Message passing iterations ──
        for t in range(self.num_layers):
            # For each edge (u→v), aggregate messages from all edges (w→u)
            # EXCEPT the reverse edge (v→u) — this is the key D-MPNN innovation
            h_edge_new = self._directed_message_pass(
                h_edge, edge_index, reverse_idx, num_atoms)
            h_edge_new = self.W_h(h_edge_new)
            h_edge_new = self.layer_norms[t](h_edge_new)
            h_edge = F.relu(h_edge + h_edge_new)  # residual
            h_edge = F.dropout(h_edge, p=self.dropout, training=self.training)

        # ── Step 3: Node readout ──
        # For each atom v, aggregate all incoming edge messages
        h_node_msg = torch.zeros(num_atoms, self.hidden_dim,
                                   device=atom_features.device)
        h_node_msg.index_add_(0, dst, h_edge)

        # Combine with atom features
        h_node = F.relu(self.W_o(torch.cat([atom_features, h_node_msg], dim=-1)))
        h_node = F.dropout(h_node, p=self.dropout, training=self.training)

        # ── Step 4: Sum pooling (preserves molecular size info) ──
        graph_repr = global_add_pool(h_node, batch)

        return graph_repr

    def _build_reverse_index(self, edge_index):
        """For each edge i: (u→v), find the index j of the reverse edge (v→u).

        Returns -1 if no reverse edge exists.
        """
        src, dst = edge_index[0], edge_index[1]
        num_edges = src.size(0)

        # Build a hash map: (src, dst) → edge_index
        edge_map = {}
        for i in range(num_edges):
            key = (src[i].item(), dst[i].item())
            edge_map[key] = i

        reverse_idx = torch.full((num_edges,), -1, dtype=torch.long,
                                   device=edge_index.device)
        for i in range(num_edges):
            rev_key = (dst[i].item(), src[i].item())
            if rev_key in edge_map:
                reverse_idx[i] = edge_map[rev_key]

        return reverse_idx

    def _directed_message_pass(self, h_edge, edge_index, reverse_idx, num_atoms):
        """Aggregate incoming edge messages for each edge, excluding reverse.

        For edge (u→v), aggregate all h(w→u) where w≠v.
        """
        src, dst = edge_index[0], edge_index[1]
        num_edges = src.size(0)

        # Sum all incoming messages to each atom
        atom_msg_sum = torch.zeros(num_atoms, self.hidden_dim,
                                     device=h_edge.device)
        atom_msg_sum.index_add_(0, dst, h_edge)

        # For edge (u→v): message = sum of all h(w→u) - h(v→u)
        # = atom_msg_sum[u] - h_edge[reverse(u→v)]
        incoming = atom_msg_sum[src]  # sum of all messages into u

        # Subtract reverse edge message to avoid tottering
        has_reverse = reverse_idx >= 0
        reverse_msg = torch.zeros_like(h_edge)
        reverse_msg[has_reverse] = h_edge[reverse_idx[has_reverse]]
        incoming = incoming - reverse_msg

        return incoming
