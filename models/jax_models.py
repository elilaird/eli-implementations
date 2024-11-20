
import jax
import jax.numpy as jnp
import haiku as hk
import jraph

from .jax_layers import (
    GATLayer,
)


class GAT(hk.Module):
    def __init__(
        self,
        dim,
        out_dim,
        num_heads,
        num_layers,
        concat=True,
        name="GAT",
    ):
        super().__init__(name=name)
        self.dim = dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.concat = concat

        self.init_fn = hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )

        self.input_layer = lambda x: jax.nn.relu(
            hk.Linear(dim, w_init=self.init_fn)(x)
        )

        self.gat_layers = [
            GATLayer(
                self.dim, self.num_heads, self.concat, name=f"gat_layer_0"
            )
        ]
        for i in range(1, self.num_layers):
            self.gat_layers.append(
                GATLayer(
                    self.dim * self.num_heads,
                    num_heads=1,
                    concat=False,
                    name=f"gat_layer_{i}",
                )
            )

        self.skip_connections = [
            hk.Linear(
                self.dim * self.num_heads,
                w_init=self.init_fn,
                name=f"skip_connection_{i}",
            )
            for i in range(self.num_layers)
        ]

        self.mlp = hk.Sequential(
            [
                hk.Linear(self.dim * self.num_heads, w_init=self.init_fn),
                jax.nn.relu,
                hk.Linear(self.dim * self.num_heads, w_init=self.init_fn),
                jax.nn.relu,
                hk.Linear(self.out_dim, w_init=self.init_fn),
            ]
        )

    def __call__(self, graph):
        nodes, _, _, _, _, n_node, _ = graph
        nodes = nodes["x"]
        sum_n_node = jax.tree_util.tree_leaves(nodes)[0].shape[0]

        x = self.input_layer(nodes)

        for i in range(self.num_layers):
            graph_out = self.gat_layers[i](graph._replace(nodes=x))

            # skip connection
            x = jax.nn.relu(graph_out.nodes) + self.skip_connections[i](x)

        # global mean pooling
        graph_idx = jnp.repeat(
            jnp.arange(n_node.shape[0]), n_node, total_repeat_length=sum_n_node
        )
        x = jraph.segment_mean(
            x, segment_ids=graph_idx, num_segments=n_node.shape[0]
        )

        # final mlp
        return self.mlp(x).reshape(-1)
