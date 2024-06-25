import math
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

import torch
from torch.nn import (
    Linear,
    ModuleList,
    LayerNorm,
    Parameter,
    ModuleList,
    init,
    Module,
)
from torch.nn.functional import normalize
from torch_geometric.nn import global_add_pool


from utils.jepa_utils import flatten_context_mask, flatten_target_mask
from utils.ml_utils import trunc_normal_
from layers.attention import Block
from layers.graph_jepa import JEPA_Layer


class TransformerPredictor(Module):
    """Transformer Predictor for Graph JEPA models."""

    def __init__(
        self,
        embed_dim,
        predictor_embed_dim,
        pos_embed_dim,
        patch_type,
        depth=3,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=LayerNorm,
        init_std=0.02,
        **kwargs,
    ):
        super().__init__()
        self.patch_type = patch_type
        self.predictor_embed = Linear(
            embed_dim, predictor_embed_dim, bias=True
        )
        self.predictor_pos_embed = Linear(
            pos_embed_dim, predictor_embed_dim, bias=True
        )
        self.mask_token = Parameter(torch.zeros(1, predictor_embed_dim))

        self.predictor_blocks = ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.0,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    def forward(self, x, target_pe, masks_x, masks, batch_idx):
        assert (masks is not None) and (
            masks_x is not None
        ), "Cannot run predictor without mask indices"

        # -- Batch Size
        B = batch_idx.unique().size(0)

        # -- map from encoder-dim to pedictor-dim
        if self.patch_type in ["node-node", "node-subgraph"]:
            x = self.predictor_embed(x)
        elif self.patch_type in ["subgraph-node", "subgraph-subgraph"]:
            ctx_flat = flatten_context_mask(
                masks_x,
                torch.arange(batch_idx.size(0), device=batch_idx.device),
            )
            x = normalize(
                global_add_pool(x[ctx_flat], batch_idx[ctx_flat]), p=2, dim=0
            )
            x = self.predictor_embed(x)
            masks_x = torch.arange(B, device=batch_idx.device)

        # -- concat mask tokens to x
        pos_embs = self.predictor_pos_embed(target_pe)

        # for each graph in the batch
        batch_preds = []
        for g in range(B):

            g_x = x[masks_x[g]]

            # repeat target positional embeddings for each embedding in the graph
            g_pos_embs = flatten_target_mask(
                pos_embs, masks[g], self.patch_type
            )
            g_pos_embs = g_pos_embs.repeat(g_x.size(0), 1)

            # repeat target tokens for each embedding in the graph
            g_pred_tokens = self.mask_token.repeat(g_pos_embs.size(0), 1)

            # concat graph positional embeddings with mask tokens
            g_pred_tokens += g_pos_embs

            g_x = g_x.repeat(g_pred_tokens.size(0) // g_x.size(0), 1)

            # concat context embeddings with mask tokens
            g_x = torch.cat([g_x, g_pred_tokens], dim=0)

            # if missing batch dimension, add it
            if len(g_x.size()) == 2:
                g_x = g_x.unsqueeze(0)

            # forward prop
            for blk in self.predictor_blocks:
                g_x = blk(g_x)

            g_x = self.predictor_norm(g_x)

            g_x = g_x[:, g_pred_tokens.size(0) :]
            g_x = self.predictor_proj(g_x)

            batch_preds.extend(g_x.squeeze(0))

        return torch.stack(batch_preds)


class JEPA(Module):
    """
    Implements JEPA as a Message Passing Layer
    """

    def __init__(
        self,
        encoder: Module,
        predictor: Module,
        Z_size: int,
        shared: bool = False,
        input_size: int = None,
        target_percentage: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.target_encoder.reset_parameters()

        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.input_size = input_size
        self.target_percentage = target_percentage
        self.shared = shared
        if shared:
            self.jepa_blocks = torch.nn.ModuleList(
                [
                    JEPA_Layer(
                        Z_size,
                        nn=predictor,
                        input_size=input_size,
                        target_percentage=target_percentage,
                        **kwargs,
                    )
                ]
            )
        else:
            self.jepa_blocks = torch.nn.ModuleList(
                [
                    JEPA_Layer(
                        Z_size,
                        nn=predictor,
                        input_size=input_size,
                        target_percentage=target_percentage,
                        **kwargs,
                    )
                    for _ in range(len(self.encoder.model))
                ]
            )

    def forward(self, x, edge_index):

        # randomly select target nodes
        tgt_idx = torch.randperm(x.size(0), device=x.device)[
            : int(self.target_percentage * x.size(0))
        ]
        mask = torch.zeros(x.size(0), 1, device=x.device)
        mask[tgt_idx] = 1

        # replace target nodes with mask token
        mask_token = torch.randn(1, x.size(-1), device=x.device) * torch.std(
            x
        ) + torch.mean(x)
        mask_tokens = mask_token.repeat(x.size(0), 1)

        x = x * (1 - mask) + (mask * mask_tokens)

        H_pred = []
        for i, conv in enumerate(self.encoder.model):
            i = 0 if self.shared else i

            x = conv(x, edge_index)
            H_hat = self.jepa_blocks[i](x, edge_index)
            H_pred.append(H_hat)

        return H_pred, tgt_idx, x

    def trainable_parameters(self):
        return list(self.encoder.named_parameters()) + list(
            self.jepa_blocks.named_parameters()
        )

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, (
            "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        )
        for param_q, param_k in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1.0 - mm)
