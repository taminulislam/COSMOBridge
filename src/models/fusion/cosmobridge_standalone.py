"""COSMOBridge Standalone: True single-model multimodal architecture.

Everything in one nn.Module, one forward pass, one checkpoint:
  - Chemprop-compatible D-MPNN (loads Chemprop weights)
  - PointNet COSMO surface encoder
  - GBH bilinear fusion for surface-dependent properties
  - Direct graph FFN for bulk thermodynamic properties
  - Per-property learned routing gates

Single forward: (SMILES graph, COSMO point cloud, thermo features) → 7 predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.pointcloud.pointnet import PointNetEncoder
from src.models.fusion.gated_bilinear_hyper_v2 import GatedBilinearHyperFusionV2


class ChempropDMPNN(nn.Module):
    """Chemprop-compatible D-MPNN that can load Chemprop's trained weights.

    Architecture matches Chemprop exactly:
      W_i: bond_features (147D) → hidden (300D)   [edge initialization]
      W_h: hidden → hidden                         [message passing]
      W_o: [atom_features; sum_messages] → hidden   [node readout]

    Differences from our dmpnn.py:
      - W_i takes ONLY bond features (not atom+bond)
      - No layer norms (Chemprop doesn't use them)
      - Uses cached zero vector for padding
    """

    def __init__(self, atom_dim=133, bond_dim=147, hidden_dim=300, depth=3, dropout=0.2):
        super().__init__()
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        # Chemprop's exact layers
        self.W_i = nn.Linear(bond_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(atom_dim + hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.cached_zero_vector = nn.Parameter(torch.zeros(hidden_dim), requires_grad=False)

    def load_chemprop_weights(self, ckpt_path, device="cpu"):
        """Load weights from a Chemprop checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt["state_dict"]

        mapping = {
            "encoder.encoder.0.W_i.weight": "W_i.weight",
            "encoder.encoder.0.W_h.weight": "W_h.weight",
            "encoder.encoder.0.W_o.weight": "W_o.weight",
            "encoder.encoder.0.W_o.bias": "W_o.bias",
            "encoder.encoder.0.cached_zero_vector": "cached_zero_vector",
        }

        my_sd = self.state_dict()
        loaded = 0
        for chemprop_key, my_key in mapping.items():
            if chemprop_key in sd and my_key in my_sd:
                if sd[chemprop_key].shape == my_sd[my_key].shape:
                    my_sd[my_key] = sd[chemprop_key]
                    loaded += 1
        self.load_state_dict(my_sd)
        return loaded

    def forward(self, atom_features, edge_index, bond_features, batch):
        """
        Args:
            atom_features: (N_atoms, atom_dim) — all atoms in batch
            edge_index: (2, N_edges) — source, target
            bond_features: (N_edges, bond_dim)
            batch: (N_atoms,) — graph membership

        Returns:
            (B, hidden_dim) — graph-level representation
        """
        num_atoms = atom_features.size(0)
        num_edges = edge_index.size(1)

        if num_edges == 0:
            h_atoms = torch.zeros(num_atoms, self.hidden_dim, device=atom_features.device)
            from torch_geometric.nn import global_add_pool
            return global_add_pool(h_atoms, batch)

        src, dst = edge_index[0], edge_index[1]

        # Initialize edge messages from bond features only (Chemprop style)
        h_edge = F.relu(self.W_i(bond_features))

        # Build reverse edge index
        reverse_idx = self._build_reverse_index(edge_index)

        # Message passing
        for t in range(self.depth - 1):
            h_edge_new = self._message_pass(h_edge, edge_index, reverse_idx, num_atoms)
            h_edge_new = self.W_h(h_edge_new)
            h_edge = F.relu(h_edge + h_edge_new)  # residual
            h_edge = self.dropout(h_edge)

        # Node readout: aggregate incoming messages, concat with atom features
        h_node_msg = torch.zeros(num_atoms, self.hidden_dim, device=atom_features.device)
        h_node_msg.index_add_(0, dst, h_edge)

        h_node = F.relu(self.W_o(torch.cat([atom_features, h_node_msg], dim=-1)))
        h_node = self.dropout(h_node)

        # Sum pooling (Chemprop style)
        from torch_geometric.nn import global_add_pool
        return global_add_pool(h_node, batch)

    def _build_reverse_index(self, edge_index):
        src, dst = edge_index[0], edge_index[1]
        n = src.size(0)
        edge_map = {}
        for i in range(n):
            edge_map[(src[i].item(), dst[i].item())] = i
        rev = torch.full((n,), -1, dtype=torch.long, device=edge_index.device)
        for i in range(n):
            rk = (dst[i].item(), src[i].item())
            if rk in edge_map:
                rev[i] = edge_map[rk]
        return rev

    def _message_pass(self, h_edge, edge_index, reverse_idx, num_atoms):
        src, dst = edge_index[0], edge_index[1]
        atom_msg = torch.zeros(num_atoms, self.hidden_dim, device=h_edge.device)
        atom_msg.index_add_(0, dst, h_edge)
        incoming = atom_msg[src]
        has_rev = reverse_idx >= 0
        rev_msg = torch.zeros_like(h_edge)
        rev_msg[has_rev] = h_edge[reverse_idx[has_rev]]
        return incoming - rev_msg


class COSMOBridgeStandalone(nn.Module):
    """True standalone COSMOBridge: one model, one forward pass, one checkpoint.

    Architecture:
      ChempropDMPNN(SMILES graph) → 300D ─┬→ GBH Fusion(×surface) → h_fused ─┐
                                           │                                    │
      PointNet(COSMO surface) → 256D ─────┘                                    ├→ α_p gate → pred
                                           │                                    │
      ChempropDMPNN → 300D ───→ concat(thermo) → Direct FFN → h_direct ───────┘
                                           │
      Thermo features (25D) ──────────────┘
    """

    def __init__(self, atom_dim=133, bond_dim=147, graph_hidden=300, depth=3,
                 surface_dim=256, thermo_dim=25, fused_dim=256,
                 rank=32, hyper_hidden=64, n_properties=7, dropout=0.3):
        super().__init__()

        # Encoder 1: Chemprop-compatible D-MPNN
        self.dmpnn = ChempropDMPNN(atom_dim, bond_dim, graph_hidden, depth, dropout)

        # Encoder 2: PointNet for COSMO surface
        self.pointnet = PointNetEncoder(in_channels=7, feature_dim=surface_dim, dropout=dropout)

        # Path A: GBH Bilinear Fusion
        self.graph_proj = nn.Linear(graph_hidden, fused_dim)
        self.surface_proj = nn.Linear(surface_dim, fused_dim)
        self.fusion = GatedBilinearHyperFusionV2(
            pointcloud_dim=fused_dim, graph_dim=fused_dim, tabular_dim=thermo_dim,
            fused_dim=fused_dim, rank=rank, thermo_dim=5,
            hyper_hidden=hyper_hidden, dropout=dropout)
        self.fused_head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, n_properties))

        # Path B: Direct graph FFN
        self.direct_head = nn.Sequential(
            nn.Linear(graph_hidden + thermo_dim, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, n_properties))

        # Per-property routing gates
        self.gate_logits = nn.Parameter(torch.tensor([
            2.0, 2.0, -2.0, -2.0, -2.0, 0.0, 1.5
        ]))

    def load_pretrained_encoders(self, chemprop_ckpt, pointnet_ckpt, device="cpu"):
        """Load pre-trained encoder weights."""
        # Load Chemprop D-MPNN
        n_loaded = self.dmpnn.load_chemprop_weights(chemprop_ckpt, device)
        print(f"  Loaded Chemprop D-MPNN: {n_loaded} weight tensors")

        # Load PointNet from PointCloud model
        pc_ckpt = torch.load(pointnet_ckpt, map_location=device, weights_only=False)
        pc_sd = pc_ckpt["model_state_dict"] if "model_state_dict" in pc_ckpt else pc_ckpt
        pn_loaded = 0
        my_sd = self.state_dict()
        for k, v in pc_sd.items():
            if k.startswith("pointnet."):
                if k in my_sd and my_sd[k].shape == v.shape:
                    my_sd[k] = v
                    pn_loaded += 1
        self.load_state_dict(my_sd)
        print(f"  Loaded PointNet: {pn_loaded} weight tensors")

    def freeze_encoders(self):
        """Freeze both encoders — only train fusion + direct + gates."""
        for p in self.dmpnn.parameters():
            p.requires_grad = False
        for p in self.pointnet.parameters():
            p.requires_grad = False
        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Frozen encoders: {n_frozen:,} params")
        print(f"  Trainable (fusion+direct+gates): {n_train:,} params")

    def forward(self, point_cloud, features, atom_features, edge_index,
                bond_features, batch, **kwargs):
        """Single forward pass: raw inputs → 7 predictions.

        Args:
            point_cloud: (B, 1024, 7) — COSMO surface
            features: (B, 25) — thermo features
            atom_features: (N, atom_dim) — batched atoms
            edge_index: (2, E) — batched edges
            bond_features: (E, bond_dim) — batched bonds
            batch: (N,) — atom→graph mapping
        """
        # Encode graph
        h_graph = self.dmpnn(atom_features, edge_index, bond_features, batch)

        # Encode surface
        h_surface = self.pointnet(point_cloud)

        # Path A: bilinear fusion
        g_proj = self.graph_proj(h_graph)
        s_proj = self.surface_proj(h_surface)
        h_fused = self.fusion(s_proj, g_proj, features)
        preds_fused = self.fused_head(h_fused)

        # Path B: direct graph FFN
        h_direct_in = torch.cat([h_graph, features], dim=-1)
        preds_direct = self.direct_head(h_direct_in)

        # Per-property gated routing
        alpha = torch.sigmoid(self.gate_logits)
        predictions = alpha.unsqueeze(0) * preds_fused + (1 - alpha.unsqueeze(0)) * preds_direct

        return predictions, {"gate_values": alpha.detach()}
