# DAPPr.py
# Author: Yao Ni
import torch.nn.functional as F

# A drop-in replacement for cross-entropy that enables uncertainty estimation
def DAPPr_loss(logits, label, lamb=2e-4, eps=1e-8):
    alpha = F.softplus(logits) + 1                                                     # Dirichlet params, must be > 1
    y = F.one_hot(label, logits.shape[-1]).float()
    alpha_star = alpha - y + eps                                                       # add eps to avoid log instability issue
    p_star = (alpha_star / alpha_star.sum(dim=1, keepdim=True)).detach()               # Proposition 1
    alpha_0 = alpha.sum(1)
    loss_alpha = alpha_0 * alpha_0.log() + (alpha * (p_star / alpha).log()).sum(dim=1) # log g_psi(p* | x)
    loss_reg = (alpha * (1 - y)).square().sum(dim=1)                                   # penalise non-target evidence
    return loss_alpha.mean() + lamb * loss_reg.mean()

# Uncertainty Measure
def DAPPr_uncertainty(logits, include_entropy=False):
    alpha = F.softplus(logits) + 1
    probs = alpha / alpha.sum(dim=1, keepdim=True)
    K = logits.shape[-1]
    uncertainties = {'AU': 1 - probs.max(dim=1).values, 'EU': K / alpha.sum(dim=1)}
    if include_entropy:
        uncertainties['Ent'] = -(probs * probs.log()).sum(dim=1)
    return uncertainties
