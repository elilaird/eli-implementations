import torch
from torch.nn import Module, Linear


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
        x = self.lin
        return x
