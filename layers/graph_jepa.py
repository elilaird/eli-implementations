from typing import Callable, Union
import torch
from torch import Tensor
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    Size,
)
from torch_geometric.nn import GINConv


# Basic jepa layer that samples a percentage of nodes to mask as the target, then
# performs aggregation and prediction on the masked nodes. (Note a lot of wasted computation)
# since we are predicting all nodes in the graph, and then backpropagating through the target nodes only
class JEPA_Layer(GINConv):

    def __init__(
        self,
        Z_size: int,
        nn: Callable,
        input_size,
        target_percentage: float = 0.1,
        eps: float = 0.0,
        train_eps: bool = False,
        shuffle: bool = False,
        **kwargs,
    ):
        super().__init__(nn, eps, train_eps, **kwargs)

        self.shuffle = shuffle
        self.target_percentage = target_percentage
        self.input_size = input_size

        self.Z = torch.nn.Parameter(torch.Tensor(1, Z_size))
        torch.nn.init.xavier_uniform_(self.Z)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:

        # shuffle edge_index
        if self.shuffle:
            edge_copy = edge_index.clone()
            idx = torch.randperm(
                edge_copy[0].size(0), device=edge_index.device
            )
            edge_copy[0] = torch.gather(edge_copy[0], 0, idx)
            edge_copy[1] = torch.gather(edge_copy[1], 0, idx)
            edge_index = edge_copy

        # aggregate messages for prediction
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        # concat Z to each node (could add PE to Z here as well)
        out = torch.cat([out, self.Z.expand(out.size(0), -1)], dim=-1)

        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        raise NotImplementedError("message_and_aggregate not implemented")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"
