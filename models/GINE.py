import sys
import os

# Add the parent directory to sys.path
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


from typing import Optional
import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Module,
    ModuleList,
)

from layers import MaskedGINEConv


class GINE(Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        channels: int,
        num_layers: int,
        pe_dim: Optional[int] = None,
        pe_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.node_emb = Embedding(num_node_features, channels - pe_embed_dim)
        self.edge_emb = Embedding(num_edge_features, channels)

        self.pe_lin = Linear(pe_dim, pe_embed_dim)
        self.pe_norm = BatchNorm1d(pe_dim)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = torch.nn.Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            self.convs.append(MaskedGINEConv(nn, edge_dim=channels))

    def forward(self, x, pe, edge_index, edge_attr, batch, mask=None):
        if pe is not None:
            pe_hat = self.pe_lin(self.pe_norm(pe))
            x = torch.cat(
                (
                    self.node_emb(x.squeeze(-1).long()),
                    pe_hat,
                ),
                1,
            )
        else:
            x = self.node_emb(x.squeeze(-1).long())
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, ctx_mask=mask, edge_attr=edge_attr)

        return x, pe_hat, None
