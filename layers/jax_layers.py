
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import haiku as hk
import jraph


class GATLayer(hk.Module):
    """Implements GATv2Layer"""

    def __init__(
        self,
        dim,
        num_heads,
        concat=True,
        share_weights=False,
        name=None,
    ):
        super().__init__(name=name)
        self.dim = dim
        self.num_heads = num_heads
        self.concat = concat
        self.share_weights = share_weights
        self.init_fn = hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )

        # node update function
        self.node_update_fn = lambda x: x

        # query functions
        self.attention_query_l = hk.Linear(
            dim * num_heads, w_init=self.init_fn, name="attention_query_l"
        )

        self.attention_query_r = (
            self.attention_query_l
            if self.share_weights
            else hk.Linear(
                dim * num_heads, w_init=self.init_fn, name="attention_query_r"
            )
        )

        self.attention_logit_fn = lambda q, k: hk.Linear(
            1, w_init=self.init_fn, name="attention_logit_fn"
        )(jax.nn.leaky_relu(q + k, negative_slope=0.2))

    def __call__(self, graph):
        nodes, _, receivers, senders, _, _, _ = graph
        sum_n_node = tree.tree_leaves(nodes)[0].shape[0]

        # Linear transformation
        nodes_transformed_l = self.attention_query_l(nodes).reshape(
            -1, self.num_heads, self.dim
        )
        if not self.share_weights:
            nodes_transformed_r = self.attention_query_r(nodes).reshape(
                -1, self.num_heads, self.dim
            )
        else:
            nodes_transformed_r = nodes_transformed_l

        # Compute attention logits
        sent_attributes = nodes_transformed_l[senders]
        received_attributes = nodes_transformed_r[receivers]
        attention_logits = self.attention_logit_fn(
            sent_attributes, received_attributes
        )

        # Apply softmax to get attention coefficients
        alpha = jraph.segment_softmax(
            attention_logits, segment_ids=receivers, num_segments=sum_n_node
        )

        # Apply attention coefficients
        out = sent_attributes * alpha

        # Aggregate messages
        out = jraph.segment_sum(
            out, segment_ids=receivers, num_segments=sum_n_node
        )

        # # Concatenate or average the multi-head results
        if self.concat:
            out = out.reshape(sum_n_node, self.dim * self.num_heads)
        else:
            out = jnp.mean(out, axis=1)

        # Apply final update function
        nodes = self.node_update_fn(out)

        return graph._replace(nodes=nodes)
