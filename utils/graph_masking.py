from typing import Tuple
import torch
from torch.nn.functional import normalize
import torch_geometric as pyg
from torch_geometric.utils import k_hop_subgraph, subgraph


class GraphMaskCollator(object):
    """
    Collates mask blocks for a batch of graphs.

    Args:
        data_loader (pyg.data.DataLoader): The data loader.
        patch_type (str): The type of patch to use. Options are 'node', 'edge', 'subgraph'.
        sampling_strategy (str): The block sampling strategy. Options are 'uniform', 'degree', 'expander'.
        num_blocks (int): The number of blocks to sample for each graph.
        block_size (int): The number of patches to include in each block.
        use_variable_block_size (bool): Whether to use a variable block size.
    """

    def __init__(
        self,
        data_loader: pyg.data.DataLoader,
        patch_type: str,
        sampling_strategy: str,
        num_blocks: int,
        block_size: int,
        use_variable_block_size: bool = False,
        num_hops: int = 1,
        subgraph_size: int = None,
        sample_ctx_size: int = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.data_loader = data_loader
        self.patch_type = patch_type
        self.sampling_strategy = sampling_strategy
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.use_variable_block_size = use_variable_block_size
        self.num_hops = num_hops
        self.subgraph_size = subgraph_size
        self.device = device
        self.sample_ctx_size = sample_ctx_size

        if self.patch_type in ["node-node", "subgraph-node"]:
            self.collate_fn = self._collate_node_mask
        elif self.patch_type in ["subgraph-subgraph", "node-subgraph"]:
            assert (
                self.subgraph_size is not None
            ), "Subgraph size must be specified for patch_type='subgraph'"
            self.collate_fn = self._collate_subgraph_mask
        else:
            raise ValueError(f"Invalid patch type: {self.patch_type}")

    def __iter__(self):
        """
        Collates mask blocks for a batch of graphs.

        Returns:
            Tuple[pyg.data.Batch, Tuple[torch.Tensor], Tuple[Tuple[torch.Tensor]]]:
                A tuple containing the original batch of graphs, the context blocks,
                and the target blocks.
        """
        for batch in self.data_loader:
            yield self.collate_fn(batch)

    def __len__(self):
        return len(self.data_loader)

    def _collate_node_mask(
        self, batch: pyg.data.Batch
    ) -> Tuple[
        pyg.data.Batch, Tuple[torch.Tensor], Tuple[Tuple[torch.Tensor]]
    ]:

        batch_indices = batch.batch

        target_blocks = []
        context_blocks = []
        for graph_idx in batch_indices.unique():
            node_indices = (batch_indices == graph_idx).nonzero().view(-1)

            # sampling target blocks
            graph_blocks = []
            for _ in range(self.num_blocks):
                if self.use_variable_block_size:
                    block_size = torch.randint(1, self.block_size + 1, (1,))
                else:
                    block_size = self.block_size

                if self.sampling_strategy == "uniform":
                    sampled_nodes = node_indices[
                        torch.randperm(node_indices.size(0))
                    ][:block_size]
                else:
                    raise ValueError(
                        f"Invalid sampling strategy: {self.sampling_strategy}"
                    )
                graph_blocks.append(sampled_nodes)

            # sample context blocks from remaining patches
            context_indices = node_indices[
                ~torch.isin(node_indices, torch.cat(graph_blocks))
            ]

            if self.sample_ctx_size is not None and node_indices.size(0) > 500:
                context_indices = context_indices[
                    torch.randperm(context_indices.size(0))
                ][: int(self.sample_ctx_size * context_indices.size(0))]

            context_blocks.append(context_indices.to(self.device))
            target_blocks.append(tuple(graph_blocks))

        # Move target_blocks to device
        target_blocks = tuple(
            tuple(tb.to(self.device) for tb in tb_list)
            for tb_list in target_blocks
        )

        # (N_Graphs), (N_Graphs, any) , (N_Graphs, N_Blocks, Block_Size)
        return batch.to(self.device), tuple(context_blocks), target_blocks

    def _collate_edge_mask(
        self, batch: pyg.data.Batch
    ) -> Tuple[
        pyg.data.Batch, Tuple[torch.Tensor], Tuple[Tuple[torch.Tensor]]
    ]:

        batch_indices = batch.batch

        target_blocks = []
        context_blocks = []
        for graph_idx in batch_indices.unique():
            pass

    def _collate_subgraph_mask(
        self, batch: pyg.data.Batch
    ) -> Tuple[
        pyg.data.Batch, Tuple[torch.Tensor], Tuple[Tuple[torch.Tensor]]
    ]:
        batch_indices = batch.batch

        target_blocks = []
        context_blocks = []

        # loop through graphs in batch
        for graph_idx in batch_indices.unique():
            # get indices of nodes for graph_idx in batch
            node_indices = (batch_indices == graph_idx).nonzero().view(-1)

            # sampling target blocks of block_size subgraphs
            graph_blocks = []
            for _ in range(self.num_blocks):
                if self.use_variable_block_size:
                    block_size = torch.randint(1, self.block_size + 1, (1,))
                else:
                    block_size = self.block_size

                # sample block_size n-hop subgraphs
                blocks = []
                for _ in range(block_size):
                    if self.sampling_strategy == "uniform":
                        subgraph_indices = self._sample_subgraph(
                            node_indices,
                            batch.edge_index,
                            size=self.subgraph_size,
                            num_hops=self.num_hops,
                        )
                    else:
                        raise ValueError(
                            f"Invalid sampling strategy: {self.sampling_strategy}"
                        )
                    blocks.append(subgraph_indices)

                graph_blocks.extend(blocks)

            # select context blocks from remaining patches
            context_indices = node_indices[
                ~torch.isin(
                    node_indices,
                    torch.cat(
                        [
                            (
                                torch.cat(block)
                                if isinstance(block, Tuple)
                                else block
                            )
                            for block in graph_blocks
                        ]
                    ),
                )
            ]

            context_blocks.append(context_indices.to(self.device))
            graph_blocks = [block.to(self.device) for block in graph_blocks]
            target_blocks.append(tuple(graph_blocks))

        # Move target_blocks to device
        # target_blocks = tuple(
        #     tuple(tuple(b.to(self.device) for b in block) for block in tb_list)
        #     for tb_list in target_blocks
        # )

        return batch.to(self.device), tuple(context_blocks), target_blocks

    def _sample_subgraph(
        self,
        node_indices: torch.Tensor,
        edge_index: torch.Tensor,
        size: int,
        num_hops: int = 1,
    ):

        # create new relabeled subgraph out of indices and edge_index
        # edge_index, _ = subgraph(node_indices, edge_index, relabel_nodes=True)

        # sample connected subgraphs of size
        start_node = node_indices[torch.randint(0, node_indices.size(0), (1,))]
        subgraph_nodes = k_hop_subgraph(
            start_node,
            num_hops=num_hops,
            edge_index=edge_index,
            relabel_nodes=False,
        )[0]

        # filter out starting node
        subgraph_nodes = subgraph_nodes[subgraph_nodes != start_node]

        # select random block_size nodes from the subgraph
        if subgraph_nodes.size(0) >= size - 1:
            sampled_nodes = subgraph_nodes[
                torch.randperm(subgraph_nodes.size(0))
            ][: size - 1]
        else:
            sampled_nodes = subgraph_nodes

        # add the starting node to the block
        return torch.cat([start_node, sampled_nodes])


def apply_target_mask(
    H: torch.Tensor,
    batch_indicies: torch.Tensor,
    mask: Tuple[Tuple[torch.Tensor]],
    ctx_mask: Tuple[torch.Tensor],
    type: str,
):
    """
    Applies a mask to a batch target embeddings

    Args:
        H (torch.Tensor): The batch of target embeddings for multiple graphs.
        batch_indicies (torch.Tensor): The batch mask for multiple graphs.
        mask (Tuple[Tuple[torch.Tensor]]): The mask to apply.
        type (str): The type of mask to apply. Options are 'node', 'edge', 'subgraph'.
    """
    H_new = []  # (N_graphs, N_blocks, Block_size, Embedding_size)
    for g in batch_indicies.unique():

        if type in ["node-node", "node-subgraph"]:
            ctx_size = ctx_mask[g].size(0)
        elif type in ["subgraph-node", "subgraph-subgraph"]:
            ctx_size = 1

        if type in ["node-node", "subgraph-node"]:

            # get node embeddings for each block
            blocks = []
            for block in mask[g]:
                # blocks.append(H[block])
                blocks.extend(H[block, :])

            g_tgt_embs = torch.stack(blocks)
            g_tgt_embs = g_tgt_embs.repeat(ctx_size, 1)

            H_new.extend(g_tgt_embs)

        elif type in ["subgraph-subgraph", "node-subgraph"]:

            # get node embeddings for each block
            blocks = []
            for block in mask[g]:
                patch_embeddings = []
                if isinstance(block, torch.Tensor):
                    block = [block]
                for subgraph_indices in block:
                    subgraph_embeddings = H[subgraph_indices]

                    # sum pool subgraph embeddings over the block
                    patch_embedding = normalize(
                        torch.sum(subgraph_embeddings, dim=0).unsqueeze(0),
                        p=2,
                        dim=1,
                    )
                    patch_embeddings.append(patch_embedding)

                blocks.extend(patch_embeddings)

            g_tgt_embs = torch.cat(blocks, dim=0)
            g_tgt_embs = g_tgt_embs.repeat(ctx_size, 1)

            H_new.extend(g_tgt_embs)

    return torch.stack(H_new)


def apply_context_mask(X: torch.Tensor, mask: Tuple[torch.Tensor], type: str):
    """
    Applies a mask to a batch context embeddings

    Args:
        batch (torch.Tensor): The batch of context embeddings for multiple graphs.
        mask (torch.Tensor): The mask to apply.
        type (str): The type of mask to apply. Options are 'node', 'edge', 'subgraph'.
    """

    masked_X = X.clone()

    if isinstance(mask, Tuple):
        mask = flatten_context_mask(mask, torch.arange(X.size(0)))

    masked_X[~mask] = masked_X[~mask] * 0

    return masked_X


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


# def flatten_target_mask(x, mask):
#     all_x = [x[b] for block in mask for b in block]
#     return torch.cat(all_x)


def flatten_target_mask(x, mask, patch_type):
    all_x = []
    for block in mask:
        if patch_type in ["node-node", "subgraph-node"]:
            all_x.extend(x[block, :])
        elif patch_type in ["subgraph-subgraph", "node-subgraph"]:
            all_x.extend(torch.mean(x[block, :], dim=0).unsqueeze(0))

    return torch.stack(all_x)
