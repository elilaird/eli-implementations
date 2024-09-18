import torch

def variance_loss(z, gamma, epsilon=1e-6):
    std = torch.sqrt(z.var(dim=0) + epsilon)
    return torch.mean(torch.relu(gamma - std))


def covariance_loss(z):
    n, d = z.size()
    z = z - z.mean(dim=0)
    cov_matrix = (z.T @ z) / (n - 1)
    cov_loss = (cov_matrix - torch.eye(d, device=z.device)).pow(2).sum() / d
    return cov_loss
