import torch


def gumbel_softmax(logits, temperature=1.0):
    """This function computes the Gumbel-Softmax sample. y_soft is the softmax
    output, and y_hard is used to create a "hard" one-hot encoded vector that is
    the same shape as y_soft. The trick in the return statement allows the gradient
    of y_soft to be used while still getting the discrete properties of y_hard
    during the forward pass.
    """
    G = -torch.log(-torch.log(torch.rand_like(logits)))
    y_soft = torch.softmax((logits + G) / temperature, dim=-1)
    y_hard = torch.zeros_like(logits).scatter_(
        -1, y_soft.max(-1)[1].unsqueeze(-1), 1.0
    )
    return y_hard - y_soft.detach() + y_soft
