"""Claim 1 audit: numerical verification of Proposition 1 and of the DAPPr loss/uncertainty
implementations against the paper (Eqs. 12-17, Sec. 4.4).

1. Prop. 1: for random alpha>1 and one-hot y, p~* = (alpha - y)/(alpha0 - 1) is the exact
   maximizer of f(p) = log g_psi(p|x) + CE(p,y) over the simplex (verified against
   scipy SLSQP on the simplex + random-simplex brute force).
2. The official DAPPr_loss (dappr.py) is equivalent to the paper's Figure-1 pseudocode and
   to Eq. 14/16/17 (log g_psi(p*|x) + lamb * ||(1-y).*alpha||^2).
3. DAPPr_uncertainty matches Sec. 4.4: AU = 1 - max_k alpha_k/alpha0, EU = K/alpha0.
"""
import json
import torch
import numpy as np
import scipy.optimize as sopt
import sys
sys.path.insert(0, ".")
import dappr


def _f(p, alpha, y):
    """f(p) = log g_psi(p|x) + CE(p, y) = alpha0 log alpha0 + sum_k alpha_k log(p_k/alpha_k) - sum_k y_k log p_k"""
    p = np.clip(p, 1e-12, 1)
    p = p / p.sum()
    alpha0 = alpha.sum()
    return alpha0 * np.log(alpha0) + (alpha * np.log(p / alpha)).sum() - (y * np.log(p)).sum()


def audit_prop1(n_trials=200, K=10, seed=0):
    rng = np.random.default_rng(seed)
    max_gap, max_rel = 0.0, 0.0
    n_exact = 0
    for _ in range(n_trials):
        alpha = np.exp(rng.normal(0, 1.5, K)) + 1.0            # alpha_k > 1 (softplus+1 parameterization)
        y = np.zeros(K); y[rng.integers(0, K)] = 1.0
        alpha0 = alpha.sum()
        p_star = (alpha - y) / (alpha0 - 1.0)

        # 1a. feasibility: p* in simplex
        assert abs(p_star.sum() - 1.0) < 1e-9 and (p_star >= -1e-9).all()

        # 1b. closed-form vs numerical maximizer (SLSQP on simplex)
        cons = [{"type": "eq", "fun": lambda p: p.sum() - 1.0}]
        bounds = [(0.0, 1.0)] * K
        res = sopt.minimize(lambda p: -_f(p, alpha, y), np.full(K, 1.0 / K),
                            method="SLSQP", bounds=bounds, constraints=cons,
                            options={"maxiter": 400, "ftol": 1e-12})
        gap = abs(res.fun + _f(p_star, alpha, y))               # |f(p*) - f_numerical|
        gap_rel = gap / max(abs(res.fun), 1e-12)
        max_gap, max_rel = max(max_gap, gap), max(max_rel, gap_rel)
        if gap < 1e-8:
            n_exact += 1

        # 1c. random-simplex brute-force check that p* beats random candidates
        for _ in range(50):
            r = rng.dirichlet(np.ones(K))
            assert _f(p_star, alpha, y) >= _f(r, alpha, y) - 1e-9
    return {"n_trials": n_trials, "n_exact_match": n_exact, "max_abs_gap": float(max_gap),
            "max_rel_gap": float(max_rel)}


def audit_loss_impl(n_trials=50, K=20, seed=1):
    """Official DAPPr_loss == paper pseudocode (Figure 1) == Eq. 14/16/17."""
    torch.manual_seed(seed)
    for _ in range(n_trials):
        logits = torch.randn(8, K)
        labels = torch.randint(0, K, (8,))
        lamb = float(torch.rand(1) * 1e-2)
        loss = dappr.DAPPr_loss(logits, labels, lamb)

        alpha = torch.nn.functional.softplus(logits) + 1
        y = torch.nn.functional.one_hot(labels, K).float()
        a_star = alpha - y + 1e-8
        p_star = (a_star / a_star.sum(dim=1, keepdim=True)).detach()
        alpha0 = alpha.sum(1)
        # paper Figure-1 pseudocode, line by line
        loss_alpha = alpha0 * alpha0.log() + (alpha * (p_star / alpha).log()).sum(dim=1)
        loss_reg = (alpha * (1 - y)).square().sum(dim=1)
        expected = (loss_alpha.mean() + lamb * loss_reg.mean()).item()
        assert abs(loss.item() - expected) < 1e-5, (loss.item(), expected)

        # Prop.1 identity: p_star == (alpha - y)/(alpha0 - 1) (up to eps)
        p_closed = (alpha - y) / (alpha0 - 1.0).unsqueeze(1)
        assert (p_star - p_closed).abs().max().item() < 1e-6
    return {"n_trials": n_trials, "loss_match_pseudocode": True, "pstar_matches_prop1": True}


def audit_uncertainty_defs(n_trials=20, K=10, seed=2):
    torch.manual_seed(seed)
    for _ in range(n_trials):
        logits = torch.randn(5, K)
        alpha = torch.nn.functional.softplus(logits) + 1
        probs = alpha / alpha.sum(dim=1, keepdim=True)
        unc = dappr.DAPPr_uncertainty(logits, include_entropy=True)
        assert torch.allclose(unc["AU"], 1 - probs.max(dim=1).values, atol=1e-6)   # Sec 4.4
        assert torch.allclose(unc["EU"], K / alpha.sum(dim=1), atol=1e-6)          # Sec 4.4
        assert torch.allclose(unc["Ent"], -(probs * probs.log()).sum(dim=1), atol=1e-6)
    return {"n_trials": n_trials, "AU_matches_1-max(alpha/alpha0)": True, "EU_matches_K/alpha0": True}


def run(config):
    print(f"=== Claim 1 audit: config={config} ===", flush=True)
    r1 = audit_prop1()
    print("Prop.1 audit:", json.dumps(r1), flush=True)
    r2 = audit_loss_impl()
    print("Loss impl audit:", json.dumps(r2), flush=True)
    r3 = audit_uncertainty_defs()
    print("Uncertainty defs audit:", json.dumps(r3), flush=True)
    summary = {"task": "claim1_audit", **r1, **r2, **r3}
    print("\nFINAL_RESULTS_JSON\n" + json.dumps(summary, indent=2) + "\nEND_RESULTS_JSON", flush=True)
    return summary
