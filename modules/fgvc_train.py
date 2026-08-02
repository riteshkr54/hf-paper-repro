"""CUB-200-2011 (and StanfordDogs / TinyImageNet) training, closely following the official
repo train.py (same scheduler, warmup-lamb schedule, eval loop, best-val selection)."""
import json, os, sys, functools
import torch
import torchvision
from torch.optim import AdamW
from timm.scheduler.cosine_lr import CosineLRScheduler
import dappr, edl
from . import data as data_mod
from .eval import evaluate_test_metrics, evaluate_val_acc
from utils import set_seed

DATASET_CFG = {"CUB": dict(epochs=200, lr=2e-3, lamb=2e-4, classes=200),
               "StanfordDogs": dict(epochs=200, lr=2e-3, lamb=2e-4, classes=120),
               "TinyImageNet": dict(epochs=100, lr=5e-3, lamb=5e-3, classes=200)}


def run(config):
    ds = config["dataset"]            # CUB | StanfordDogs | TinyImageNet
    method = config.get("method", "DAPPr")
    device = config.get("device", "cuda:0")
    seed = config.get("seed", 42)
    set_seed(seed)
    cfg = DATASET_CFG[ds]
    epochs = config.get("epochs", cfg["epochs"])
    lr = config.get("lr", cfg["lr"])
    lamb = config.get("lamb", cfg["lamb"])
    bs = config.get("bs", 256)
    test_every = config.get("test_every", 10)
    if ds == "TinyImageNet":
        epochs, lr, lamb = config.get("epochs", 100), config.get("lr", 5e-3), config.get("lamb", 5e-3)

    train_dl, val_dl, test_dl = data_mod.get_fgvc_loaders(ds, bs=bs, num_workers=8)
    ood_loaders, ood_names = [], []
    try:
        imagenet_o_dl, dtd_dl, place_dl = data_mod.get_fgvc_ood_loaders(bs=bs, num_workers=8)
        ood_loaders = [imagenet_o_dl, dtd_dl, place_dl]
        ood_names = ["ImageNetO", "DTD", "Places365"]
    except Exception as e:
        print(f"[fgvc] OOD loader issue: {e}", flush=True)

    model = torchvision.models.resnet50(weights=None, num_classes=cfg["classes"])
    if ds == "TinyImageNet":
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = torch.nn.Identity()
    model = model.to(device)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineLRScheduler(opt, t_initial=epochs, warmup_t=10, lr_min=1e-5,
                                  warmup_lr_init=1e-6, cycle_decay=0.1)

    METHODS = {"DAPPr": (dappr.DAPPr_loss, functools.partial(dappr.DAPPr_uncertainty, include_entropy=True)),
               "EDL": (edl.EDL_loss, functools.partial(edl.EDL_uncertainty, include_entropy=True))}
    criterion, unc_func = METHODS[method]
    lamb_schedule = config.get("lamb_schedule", "warmup")

    best_acc, best_results, best_ep = 0.0, {}, -1
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
                best_acc, best_results, best_ep = val_acc, dict(tm), e
                os.makedirs("outs", exist_ok=True)
                torch.save(model.state_dict(), f"outs/{method}_{ds}_seed{seed}_best.pth")

    summary = {"config": config, "method": method, "dataset": ds, "seed": seed,
               "best_epoch": best_ep, "best_val_acc": round(best_acc, 4), **best_results}
    print("\nFINAL_RESULTS_JSON\n" + json.dumps(summary, indent=2) + "\nEND_RESULTS_JSON", flush=True)
    with open(f"outs/{method}_{ds}_seed{seed}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
