import torch
from torch.nn import (
    Linear,
    ReLU,
    PReLU,
    Module,
    LayerNorm,
    LeakyReLU,
    Dropout,
)
from torch_geometric.nn import (
    GATv2Conv,
    GCNConv,
    GINConv,
    BatchNorm,
    Sequential,
)


class GCN(Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        batch_norm=False,
        layer_norm=False,
        batch_norm_mm=0.99,
    ):
        super(GCN, self).__init__()

        assert batch_norm != layer_norm, "Cannot use both batch and layer norm"

        convs = []
        for i in range(num_layers):
            # Last layer
            if i == num_layers - 1:
                convs.append(
                    (
                        GCNConv(hidden_channels, out_channels),
                        "x, edge_index -> x",
                    )
                )
            # First layer
            elif i == 0:
                convs.append(
                    (
                        GCNConv(in_channels, hidden_channels),
                        "x, edge_index -> x",
                    ),
                )
            # Middle layers
            else:
                convs.append(
                    (
                        GCNConv(hidden_channels, hidden_channels),
                        "x, edge_index -> x",
                    ),
                )
            # Add normalization
            if batch_norm:
                convs.append(
                    BatchNorm(hidden_channels, momentum=batch_norm_mm)
                )
            else:
                convs.append(LayerNorm(hidden_channels))

            convs.append(PReLU())

        self.model = Sequential("x, edge_index", convs)

    def forward(self, x, edge_index):
        return self.model(x, edge_index)


class GAT(Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        heads,
        batch_norm=False,
        layer_norm=False,
        batch_norm_mm=0.99,
    ):
        super(GAT, self).__init__()

        assert batch_norm != layer_norm, "Cannot use both batch and layer norm"

        convs = []
        for i in range(num_layers):
            # Last layer
            if i == num_layers - 1:
                convs.append(
                    (
                        GATv2Conv(
                            hidden_channels,
                            out_channels,
                            heads=heads,
                            concat=False,
                        ),
                        "x, edge_index -> x",
                    )
                )
            # First layer
            elif i == 0:
                convs.append(
                    (
                        GATv2Conv(
                            in_channels,
                            hidden_channels,
                            heads=heads,
                            concat=False,
                        ),
                        "x, edge_index -> x",
                    ),
                )
            # Middle layers
            else:
                convs.append(
                    (
                        GATv2Conv(
                            hidden_channels,
                            hidden_channels,
                            heads=heads,
                            concat=False,
                        ),
                        "x, edge_index -> x",
                    ),
                )
            # Add normalization
            if batch_norm:
                convs.append(
                    BatchNorm(hidden_channels, momentum=batch_norm_mm)
                )
            else:
                convs.append(LayerNorm(hidden_channels))

            convs.append(LeakyReLU(0.2))

        self.model = Sequential("x, edge_index", convs)

    def forward(self, x, edge_index):
        return self.model(x, edge_index)


class GIN(Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        batch_norm=False,
        layer_norm=False,
        batch_norm_mm=0.99,
    ):
        super(GIN, self).__init__()

        assert batch_norm != layer_norm, "Cannot use both batch and layer norm"

        convs = []
        for i in range(num_layers):
            # Last layer
            if i == num_layers - 1:
                convs.append(
                    (
                        GINConv(
                            Sequential(
                                Linear(hidden_channels, hidden_channels),
                                BatchNorm(
                                    hidden_channels, momentum=batch_norm_mm
                                ),
                                PReLU(),
                                Linear(hidden_channels, out_channels),
                            ),
                            eps=0.0,
                        ),
                        "x, edge_index -> x",
                    )
                )
            # First layer
            elif i == 0:
                convs.append(
                    (
                        GINConv(
                            Sequential(
                                Linear(in_channels, hidden_channels),
                                BatchNorm(
                                    hidden_channels, momentum=batch_norm_mm
                                ),
                                PReLU(),
                                Linear(hidden_channels, hidden_channels),
                            ),
                            eps=0.0,
                        ),
                        "x, edge_index -> x",
                    ),
                )
            # Middle layers
            else:
                convs.append(
                    (
                        GINConv(
                            Sequential(
                                Linear(hidden_channels, hidden_channels),
                                BatchNorm(
                                    hidden_channels, momentum=batch_norm_mm
                                ),
                                PReLU(),
                                Linear(hidden_channels, hidden_channels),
                            ),
                            eps=0.0,
                        ),
                        "x, edge_index -> x",
                    ),
                )
            # Add normalization
            if batch_norm:
                convs.append(
                    BatchNorm(hidden_channels, momentum=batch_norm_mm)
                )
            else:
                convs.append(LayerNorm(hidden_channels))

            convs.append(PReLU())

        self.model = Sequential("x, edge_index", convs)

    def forward(self, x, edge_index):
        return self.model(x, edge_index)


### Classifiers ###


class LogisticRegression(Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.softmax(self.linear(x), dim=-1)
        return x


class LinearRegression(Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x


class MLP(Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=64,
        num_layers=3,
        activation=ReLU,
        dropout=0.0,
    ):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = [
            Linear(input_dim, hidden_dim),
            activation(),
            Dropout(dropout),
        ]

        for _ in range(num_layers - 2):
            layers.extend(
                [
                    Linear(hidden_dim, hidden_dim),
                    activation(),
                    Dropout(dropout),
                ]
            )

        layers.append(Linear(hidden_dim, output_dim))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
