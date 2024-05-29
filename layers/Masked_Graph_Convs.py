from typing import Optional
from torch import Tensor
import torch

from torch.nn import (
    Linear,
    Parameter,
)
from torch_geometric.nn.inits import zeros
from torch_geometric.nn import (
    GCNConv,
    GPSConv,
    GINEConv,
)
from torch_geometric.utils import add_self_loops, degree, to_dense_batch
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.typing import Adj


class MaskedGINEConv(GINEConv):
    """
    A Graph Isomorphism Network (GIN) layer with optional input masking.
    Masking is applied to the edge weights to ensure masked nodes are not included in normalization.
    """

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        ctx_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        # Filter edges based on context mask
        if ctx_mask is not None:
            edge_mask = ctx_mask[edge_index[0]] & ctx_mask[edge_index[1]]
            edge_index_masked = edge_index[:, edge_mask]

            masked_edge_attr = edge_attr[edge_mask]
        else:
            edge_index_masked = edge_index
            masked_edge_attr = edge_attr

        # run super's forward with masked edge_index
        out = super().forward(
            x=x,
            edge_index=edge_index_masked,
            edge_attr=masked_edge_attr,
            **kwargs,
        )

        return out


class MaskedGPSConv(GPSConv):
    """
    A General, Powerful, Scalable (GPS) Graph Transformer layer with optional input masking.
    Masking is applied to the edge weights to ensure masked nodes are not included in normalization.
    """

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[torch.Tensor] = None,
        ctx_mask=None,
        **kwargs,
    ) -> Tensor:
        hs = []

        # run MPNN
        if self.conv is not None:
            # Filter edges based on context mask
            if ctx_mask is not None:
                edge_mask = ctx_mask[edge_index[0]] & ctx_mask[edge_index[1]]
                edge_index_masked = edge_index[:, edge_mask]

                # apply mask to edge_attr if present
                if "edge_attr" in kwargs:
                    kwargs["edge_attr"] = kwargs["edge_attr"][edge_mask]
            else:
                edge_index_masked = edge_index

            h = self.conv(x, edge_index_masked, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x  # Residual connection
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        # Convert to dense batch and create an attention mask
        h, dense_mask = to_dense_batch(x, batch)

        # Combine the dense batch mask with the context mask
        if ctx_mask is not None:
            dense_ctx_mask, _ = to_dense_batch(ctx_mask, batch)
            final_mask = dense_mask & dense_ctx_mask
        else:
            final_mask = dense_mask

        if isinstance(self.attn, torch.nn.MultiheadAttention):
            h, _ = self.attn(
                h, h, h, key_padding_mask=~final_mask, need_weights=False
            )
        elif isinstance(self.attn, PerformerAttention):
            h = self.attn(h, mask=final_mask)

        # convert back to sparse
        h = h[dense_mask]

        h = F.dropout(h, p=self.dropout, training=self.training)

        # Residual connection
        h = h + x

        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local (MPNN) and global (Attn) outputs

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out


class MaskingGCNConv(GCNConv):
    """
    A Graph Concolutional Network (GCN) layer with optional input masking.
    Masking is applied to the edge weights to ensure masked nodes are not included in normalization.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        improved=False,
        cached=False,
        add_self_loops=True,
        normalize=True,
        bias=True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            improved=improved,
            cached=cached,
            add_self_loops=add_self_loops,
            normalize=normalize,
            bias=bias,
            **kwargs,
        )

        self.lin = Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index, edge_weight=None, mask=None):
        if self.normalize:
            if self.add_self_loops:
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

            # Apply the mask to the edge weights
            if mask is not None:
                edge_weight = torch.zeros(edge_index.size(1), device=x.device)
                edge_weight[mask] = 1.0

            deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

            if edge_weight is None:
                edge_weight = torch.ones(
                    (edge_index.size(1),),
                    dtype=x.dtype,
                    device=edge_index.device,
                )

            edge_weight = (
                deg_inv_sqrt[edge_index[0]]
                * edge_weight
                * deg_inv_sqrt[edge_index[1]]
            )
        else:
            if mask is not None:
                edge_weight = torch.zeros(
                    edge_index.size(1), device=x.device, dtype=x.dtype
                )
                edge_weight[mask] = 1.0

            if edge_weight is None:
                edge_weight = torch.ones(
                    (edge_index.size(1),),
                    dtype=x.dtype,
                    device=edge_index.device,
                )

        x = self.lin(x)

        # Message passing
        out = self.propagate(
            edge_index, x=x, edge_weight=edge_weight, size=None
        )

        if self.bias is not None:
            out += self.bias

        return out
