from typing import Tuple
import torch


def flatten_context_mask(
    mask: Tuple[torch.Tensor], batch_indices: torch.Tensor
) -> torch.Tensor:
    """
    Flattens a context mask into a single tensor

    Args:
        mask (Tuple[torch.Tensor]): The context mask.
        batch_indices (torch.Tensor): The batch indices for the mask.

    Returns:
        torch.Tensor: The flattened context mask.
    """
    mask_flat = torch.isin(batch_indices, torch.cat(mask, dim=0))
    return mask_flat


def flatten_target_mask(x, mask, patch_type):
    all_x = []
    for block in mask:
        if patch_type in ["node-node", "subgraph-node"]:
            all_x.extend(x[block, :])
        elif patch_type in ["subgraph-subgraph", "node-subgraph"]:
            all_x.extend(torch.mean(x[block, :], dim=0).unsqueeze(0))

    return torch.stack(all_x)
