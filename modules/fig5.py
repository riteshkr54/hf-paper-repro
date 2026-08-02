"""Claim 6 / Figure 5: epistemic uncertainty and accuracy on CIFAR-100 vs training data size,
for DAPPr vs EDL (ResNet-18, paper config: lr 5e-4, bs 64, 100 epochs, lamb 5e-3 warmup).
The paper does not state the exact data sizes in text; we use 1K/2K/5K/10K/25K/50K and report
mean EU (K/alpha0) and test accuracy on the test set."""
import json, functools
import torch
from torch.optim import AdamW
from timm.scheduler.cosine_lr import CosineLRScheduler
import dappr, edl
from . import data as data_mod
from .eval import evaluate_test_metrics, evaluate_val_acc
from .models import build_model
from .cifar_train import set_seed


def run(config):
    device = config.get("device", "cuda:0")
    seed = config.get("seed", 42)
    sizes = config.get("sizes", [1000, 2000, 5000, 10000, 25000, 50000])
    methods = config.get("methods", ["DAPPr", "EDL"])
    epochs = config.get("epochs", 100)
    bs = config.get("bs", 64)
    lamb = config.get("lamb", 5e-3)
    test_every = config.get("test_every", 10)

    ood_loaders, ood_names = data_mod.get_cifar_ood_loaders("cifar100", bs=bs, num_workers=8)
    results = {}
    for method in methods:
        criterion = dappr.DAPPr_loss if method == "DAPPr" else edl.EDL_loss
        unc_func = functools.partial(dappr.DAPPr_uncertainty, include_entropy=True)
        results[method] = {}
        for size in sizes:
            set_seed(seed)
            train_dl, val_dl, test_dl = data_mod.get_cifar_loaders(
                "cifar100", bs=bs, num_workers=8, subset_size=size, seed=seed)
            model = build_model("resnet18", num_classes=100).to(device)
            opt = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
            scheduler = CosineLRScheduler(opt, t_initial=epochs, warmup_t=10, lr_min=1e-5,
                                          warmup_lr_init=1e-6, cycle_decay=0.1)
            best_acc, best_metrics, best_state = 0.0, None, None
            for ep in range(epochs):
                model.train()
                factor = min(1.0, ep / 10.0)
                lamb_reg = lamb * factor
                for x, labels in train_dl:
                    opt.zero_grad()
                    x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    logits = model(x)
                    loss = criterion(logits, labels, lamb_reg)
                    loss.backward()
                    opt.step()
                scheduler.step(ep)
                e = ep + 1
                if e % test_every == 0 or e == epochs:
                    model.eval()
                    val_acc = evaluate_val_acc(model, val_dl)
                    if val_acc >= best_acc:
                        best_acc = val_acc
                        best_metrics = evaluate_test_metrics(
                            model, test_dl, *ood_loaders, ood_names=ood_names, uncertainty_func=unc_func)
                        best_state = copy.deepcopy(model.state_dict())
            # mean epistemic uncertainty (K/alpha0) on the test set, best-val model (Sec. 4.4)
            model.load_state_dict(best_state)
            logits, _ = _get_logits(model, test_dl)
            alpha = torch.nn.functional.softplus(logits) + 1
            eu_mean = (100 / alpha.sum(1)).mean().item()          # K=100
            au_mean = (1 - (alpha / alpha.sum(1, keepdim=True)).max(1).values).mean().item()
            results[method][size] = {"test_acc": best_metrics["test_acc"],
                                     "EU_mean": round(eu_mean, 5), "AU_mean": round(au_mean, 5),
                                     "best_val_acc": round(best_acc, 4)}
            print(f"[fig5] {method} size={size}: {json.dumps(results[method][size])}", flush=True)

    summary = {"task": "fig5", "config": config, "results": results}
    print("\nFINAL_RESULTS_JSON\n" + json.dumps(summary, indent=2) + "\nEND_RESULTS_JSON", flush=True)
    with open("outs/fig5.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


@torch.no_grad()
def _get_logits(model, loader):
    device = next(model.parameters()).device
    all_logits = []
    for x, _ in loader:
        all_logits.append(model(x.to(device, non_blocking=True)))
    return torch.cat(all_logits), None
