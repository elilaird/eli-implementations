import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "layers")
    )
)
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "utils")
    )
)

from typing import Any, Dict, Optional
import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    ModuleList,
)
from torch_geometric.nn import (
    GINEConv,
)
from torch_geometric.nn.attention import PerformerAttention

from layers import Masked_Graph_Convs, RedrawProjection


class GPS(torch.nn.Module):
    """
    Implementation of General, Powerful, Scalable (GPS) Graph Transformer from

    "Recipe for a General, Powerful, Scalable Graph Transformer":
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py

    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        channels: int,
        pe_dim: int,
        pe_embed_dim: int,
        num_layers: int,
        attn_type: str,
        attn_kwargs: Dict[str, Any],
        se_dim: Optional[int] = None,
        se_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.node_emb = Embedding(num_node_features, channels - pe_embed_dim)
        self.pe_lin = Linear(pe_dim, pe_embed_dim)
        self.pe_norm = BatchNorm1d(pe_dim)
        self.edge_emb = Embedding(num_edge_features, channels)

        if se_dim is not None:
            self.se_lin = Linear(se_dim, se_embed_dim)
            self.se_norm = BatchNorm1d(se_dim)
            self.node_emb = Embedding(
                num_node_features, channels - pe_dim - se_dim
            )

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = torch.nn.Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = Masked_Graph_Convs(
                channels,
                GINEConv(nn, edge_dim=channels),
                heads=4,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs,
            )
            self.convs.append(conv)

        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == "performer" else None,
        )

    def forward(self, x, pe, edge_index, edge_attr, batch, se=None, mask=None):
        # normalize positional embeddings
        x_pe = self.pe_norm(pe)

        # if structural embeddings are present, normalize them
        if se is not None:
            x_se = self.se_norm(se)

            # concatenate node embeddings, positional embeddings, and structural embeddings
            pe_hat = self.pe_lin(x_pe)
            se_hat = self.se_lin(x_se)
            x = torch.cat(
                (
                    self.node_emb(x.squeeze(-1).long()),
                    pe_hat,
                    se_hat,
                ),
                1,
            )
        else:
            # concatenate node embeddings and positional embeddings
            pe_hat = self.pe_lin(x_pe)
            se_hat = None
            x = torch.cat(
                (
                    self.node_emb(x.squeeze(-1).long()),
                    pe_hat,
                ),
                1,
            )
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, ctx_mask=mask, edge_attr=edge_attr)

        return x, pe_hat, se_hat


class RedrawProjection:
    def __init__(
        self, model: torch.nn.Module, redraw_interval: Optional[int] = None
    ):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module
                for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1
