import json, os
import torch
import functools
from torch.optim import AdamW
from timm.scheduler.cosine_lr import CosineLRScheduler

import dappr, edl
from . import data as data_mod
from .eval import evaluate_test_metrics, evaluate_val_acc
from .models import build_model

MODEL_BY_DATASET = {"cifar10": "vgg16", "cifar100": "resnet18"}


def set_seed(seed=0):
    import random, numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run(config):
    """Train DAPPr or EDL (official implementations) on CIFAR-10/100.
    Config keys: dataset, method, device, seed, epochs, bs, lr, lamb, lamb_schedule."""
    set_seed(config.get("seed", 42))
    ds = config["dataset"]                     # cifar10 | cifar100
    method = config.get("method", "DAPPr")     # DAPPr | EDL
    device = config.get("device", "cuda:0")
    epochs = config.get("epochs", 100)
    bs = config.get("bs", 64)
    lr = config.get("lr", 5e-4)
    lamb = config.get("lamb", 2e-3 if ds == "cifar10" else 5e-3)
    lamb_schedule = config.get("lamb_schedule", "warmup")
    subset_size = config.get("subset_size")    # optional data-size subset (Claim 6 uses its own module)
    test_every = config.get("test_every", 10)

    train_dl, val_dl, test_dl = data_mod.get_cifar_loaders(ds, bs=bs, num_workers=8,
                                                           subset_size=subset_size,
                                                           seed=config.get("seed", 42))
    ood_loaders, ood_names = data_mod.get_cifar_ood_loaders(ds, bs=bs, num_workers=8)

    model = build_model(MODEL_BY_DATASET[ds], num_classes=10 if ds == "cifar10" else 100)
    model = model.to(device)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineLRScheduler(opt, t_initial=epochs, warmup_t=10, lr_min=1e-5,
                                  warmup_lr_init=1e-6, cycle_decay=0.1)

    METHODS = {"DAPPr": (dappr.DAPPr_loss, functools.partial(dappr.DAPPr_uncertainty, include_entropy=True)),
               "EDL": (edl.EDL_loss, functools.partial(edl.EDL_uncertainty, include_entropy=True))}
    criterion, unc_func = METHODS[method]

    best_acc, best_results, best_ep = 0.0, {}, -1
    n_iters = len(train_dl)
    for ep in range(epochs):
        model.train()
        factor = min(1.0, ep / 10.0) if lamb_schedule == "warmup" else ep / epochs
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
            tm = evaluate_test_metrics(model, test_dl, *ood_loaders, ood_names=ood_names,
                                       uncertainty_func=unc_func)
            tm = {k: round(v, 4) if isinstance(v, float) else v for k, v in tm.items()}
            print(f"[{method} {ds}] epoch {e}/{epochs} val_acc={val_acc:.2f} test_acc={tm['test_acc']:.2f}", flush=True)
            if val_acc >= best_acc:
                best_acc = val_acc
                best_results = dict(tm, epoch=e)
                best_ep = e

    summary = {"config": config, "method": method, "dataset": ds, "seed": config.get("seed", 42),
               "best_epoch": best_ep, "best_val_acc": round(best_acc, 4), **best_results}
    print("\nFINAL_RESULTS_JSON\n" + json.dumps(summary, indent=2) + "\nEND_RESULTS_JSON", flush=True)
    os.makedirs("outs", exist_ok=True)
    with open(f"outs/{method}_{ds}_seed{config.get('seed',42)}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
