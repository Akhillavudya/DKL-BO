"""CGCNN encoder: Crystal Graph Convolutional Neural Network.

Architecture from [P1] Table 1:
  atom_dim=90, bond_dim=10, hidden_dim=32, n_conv=3, n_fc=1, pooling=attention

Key implementation choices:
  - LayerNorm after each conv layer ([P2]: prevents mode collapse and
    embedding scale drift that destabilises the GP kernel)
  - Residual connection in each conv layer (standard CGCNN)
  - Attention pooling instead of mean ([P1]: mean dilutes information
    when unit-cell sizes vary, which they do across C2DB)
  - float32 throughout (GP head converts to float64 for stability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import softmax as pyg_softmax


class CGCNNConv(MessagePassing):
    """One CGCNN graph convolution layer.

    Implements the gated message-passing formula from Xie & Grossman (2018):
        m_ij = sigmoid(W_f · [h_i, h_j, e_ij]) ⊙ softplus(W_s · [h_i, h_j, e_ij])
        h_i' = LayerNorm(h_i + Σ_j m_ij)

    The sigmoid gate decides *how much* to pass; softplus generates the message.
    LayerNorm + residual keeps embeddings in a stable range for the GP kernel.
    """

    def __init__(self, node_dim: int, edge_dim: int) -> None:
        super().__init__(aggr="add")
        in_dim = 2 * node_dim + edge_dim
        self.lin_gate = nn.Linear(in_dim, node_dim)
        self.lin_msg = nn.Linear(in_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        aggr = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.norm(x + aggr)  # residual + LayerNorm

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return torch.sigmoid(self.lin_gate(z)) * F.softplus(self.lin_msg(z))


class CGCNNEncoder(nn.Module):
    """Full CGCNN encoder: graphs → fixed-size embedding vectors.

    Flow:
        x [N, atom_dim]
          → embed → [N, hidden_dim]
          → n_conv × CGCNNConv
          → attention pool → [B, hidden_dim]   (one vector per graph)
          → n_fc × FC+softplus
          → output [B, hidden_dim]

    The output is the embedding the GP receives.

    E4 — Spectral Normalization
    ---------------------------
    When spectral_norm=True, every Linear layer is wrapped with
    torch.nn.utils.parametrizations.spectral_norm.  This constrains each
    layer's Lipschitz constant to ≤ sn_coeff, keeping the encoder
    distance-preserving and preventing feature collapse that causes GP
    overconfidence on OOD materials.
    """

    def __init__(
        self,
        atom_dim: int = 90,
        bond_dim: int = 10,
        hidden_dim: int = 32,
        n_conv: int = 3,
        n_fc: int = 1,
        pooling: str = "attention",
        spectral_norm: bool = False,
        sn_coeff: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(atom_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [CGCNNConv(hidden_dim, bond_dim) for _ in range(n_conv)]
        )
        self.pooling = pooling
        if pooling == "attention":
            self.attn_lin = nn.Linear(hidden_dim, 1)
        self.fc_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_fc)]
        )
        self.out_dim = hidden_dim

        if spectral_norm:
            self._apply_spectral_norm(sn_coeff)

    def _apply_spectral_norm(self, coeff: float = 1.0) -> None:
        """Wrap every Linear layer with spectral norm + optional coefficient scaling.

        spectral_norm constrains ||W||_2 ≤ 1.  The coeff multiplies every
        layer output so the effective Lipschitz bound per layer is coeff.
        coeff=1.0 → strict 1-Lipschitz per layer (most conservative).
        coeff>1.0 → looser constraint, more encoder expressivity.
        """
        from torch.nn.utils.parametrizations import spectral_norm as _sn

        def _wrap(module: nn.Linear) -> None:
            _sn(module)
            if coeff != 1.0:
                module.register_forward_hook(
                    lambda m, inp, out: out * coeff
                )

        _wrap(self.embed)
        for conv in self.convs:
            _wrap(conv.lin_gate)
            _wrap(conv.lin_msg)
        if hasattr(self, "attn_lin"):
            _wrap(self.attn_lin)
        for fc in self.fc_layers:
            _wrap(fc)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        x          : [N_total_atoms, atom_dim]
        edge_index : [2, N_total_edges]
        edge_attr  : [N_total_edges, bond_dim]
        batch      : [N_total_atoms]  — maps each atom to its graph index

        Returns
        -------
        h : [B, hidden_dim]  — one embedding per graph in the batch
        """
        # Initial atom embedding
        h = F.softplus(self.embed(x))  # [N, hidden_dim]

        # Graph convolutions
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)  # [N, hidden_dim]

        # Pooling: aggregate atom embeddings → one vector per graph
        if self.pooling == "attention":
            # attn_lin maps each atom to a scalar score
            # pyg_softmax normalises scores *within* each graph (not globally)
            attn_weights = pyg_softmax(self.attn_lin(h), batch)  # [N, 1]
            h = global_add_pool(attn_weights * h, batch)          # [B, hidden_dim]
        else:
            from torch_geometric.nn import global_mean_pool
            h = global_mean_pool(h, batch)  # [B, hidden_dim]

        # FC layers
        for fc in self.fc_layers:
            h = F.softplus(fc(h))  # [B, hidden_dim]

        return h  # [B, hidden_dim]
