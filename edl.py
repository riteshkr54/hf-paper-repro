import torch.nn.functional as F
from torch.distributions import Dirichlet, kl_divergence as kl_div
import torch
from dappr import DAPPr_uncertainty

def EDL_loss(logits, labels, lamb=1e-3):
    alpha = F.softplus(logits) + 1

    y = F.one_hot(labels, logits.shape[1]).float()
    alpha_tilde = alpha * (1 - y) + y

    alpha0 = alpha.sum(1, keepdim=True)
    loss_cls = (y - alpha / alpha0).square().sum(dim=1).mean()
    loss_var = ((alpha * (alpha0 - alpha)) / ((alpha0 ** 2) * (alpha0 + 1))).sum(dim=1).mean()
    loss_kl = kl_div(Dirichlet(1e-6 + alpha_tilde), Dirichlet(torch.ones_like(alpha_tilde))).mean()

    return loss_cls + loss_var + lamb * loss_kl

def EDL_uncertainty(logits, include_entropy=False):
    return DAPPr_uncertainty(logits, include_entropy=include_entropy)
