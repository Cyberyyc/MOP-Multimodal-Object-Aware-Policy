"""Joint diffusion, CVAE, and residual-gate regularization objectives."""
import torch


def position_cvae_loss(prediction, target, mean, logvar, valid=None, kl_weight=.01):
    if valid is None:
        valid = torch.ones(target.shape[:2], device=target.device, dtype=torch.bool)
    valid = valid.bool()
    clean_target = target.masked_fill(~valid[..., None], 0)
    reconstruction = ((prediction - clean_target).abs() * valid[..., None]).sum((1, 2)).mean()
    kl = (-.5 * (1 + logvar - mean.square() - logvar.exp())).sum(-1).mean()
    return reconstruction + kl_weight * kl, reconstruction, kl


def gate_sparsity(alpha, beta):
    return alpha.square().mean() + beta.square().mean()


def diffusion_mse(prediction, noise, valid=None):
    error = (prediction - noise).square().mean(-1)
    if valid is None:
        return error.mean()
    valid = valid.to(dtype=error.dtype)
    if valid.shape != error.shape or valid.sum() == 0:
        raise ValueError("Action mask must match the horizon and contain valid actions.")
    return (error * valid).sum() / valid.sum()
