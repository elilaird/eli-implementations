from torch_geometric import EdgeIndex
from torch_geometric.utils import coalesce, remove_self_loops

def n_hop_edge_index(
    edge_index, n_hops, num_nodes, as_tensor=False, use_self_loops=False
):
    if n_hops == 1:
        return edge_index

    edge_index = EdgeIndex(edge_index, sparse_size=(num_nodes, num_nodes))
    edge_index = edge_index.sort_by("row")[0]
    edge_index_n = edge_index.clone()
    for _ in range(n_hops - 1):
        # multiply edge_index by itself to get N-hop neighbors
        edge_index_n = edge_index_n.matmul(edge_index)[0].as_tensor()
        # remove self loops
        if not use_self_loops:
            edge_index_n, _ = remove_self_loops(edge_index_n)
        # remove duplicate edges
        edge_index_n, _ = coalesce(edge_index_n, None, num_nodes)

        edge_index_n = EdgeIndex(
            edge_index_n, sparse_size=(num_nodes, num_nodes)
        ).sort_by("row")[0]

    return edge_index_n.as_tensor() if as_tensor else edge_index_n
