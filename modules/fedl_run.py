"""F-EDL (Yoon & Kim, NeurIPS 2025, https://github.com/TaeseongYoon/F-EDL) on CIFAR-10/100.

Runs the vendored official implementation (fedl/*.py) with the same hyperparameters the
DAPPr paper used for its F-EDL comparisons (lr 5e-4, bs 64, 100 epochs, val 5%, Adam).
The vendored files are adapted only for: configurable data root, device, and JSON summary
output. Metrics (acc, conf AUPR via aleatoric AU, OOD AUPR via epistemic EU) follow the
F-EDL repo's own uq.py definitions.
"""
import json, os, sys, importlib

import torch
import numpy as np
from torch.utils.data import DataLoader

import fedl.datasets as fedl_datasets
import fedl.models as fedl_models
import fedl.train as fedl_train
import fedl.uq as fedl_uq
from modules.paths import TORCH_ROOT, DATASETS_DIR


def run(config):
    ID_dataset = config["dataset"]          # "CIFAR-10" | "CIFAR-100"
    device = config.get("device", "cuda:0")
    seed = config.get("seed", 42)
    batch_size = config.get("bs", 64)
    num_epochs = config.get("epochs", 100)
    learning_rate = config.get("lr", 5e-4)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ---- data: mirror F-EDL's loaders but with our cache root ----
    if ID_dataset == "CIFAR-10":
        import torchvision.transforms as transforms
        from torchvision import datasets
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        train_tf = transforms.Compose([transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), normalize])
        test_tf = transforms.Compose([transforms.ToTensor(), normalize])
        base_train = datasets.CIFAR10(root=TORCH_ROOT, train=True, download=True, transform=train_tf)
        base_valid = datasets.CIFAR10(root=TORCH_ROOT, train=True, download=True, transform=test_tf)
        base_test = datasets.CIFAR10(root=TORCH_ROOT, train=False, download=True, transform=test_tf)
        num_train = len(base_train)
        indices = np.random.permutation(num_train)
        split = int(np.floor(0.05 * num_train))
        train_subset = torch.utils.data.Subset(base_train, indices[split:])
        val_subset = torch.utils.data.Subset(base_valid, indices[:split])
        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8)
        validloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=8)
        testloader = DataLoader(base_test, batch_size=batch_size, shuffle=False, num_workers=8)
        svhn = datasets.SVHN(root=TORCH_ROOT, split="test", download=True, transform=test_tf)
        c100 = datasets.CIFAR100(root=TORCH_ROOT, train=False, download=True, transform=test_tf)
        ood1 = DataLoader(svhn, batch_size=batch_size, shuffle=False, num_workers=8)
        ood2 = DataLoader(c100, batch_size=batch_size, shuffle=False, num_workers=8)
        ood_names = ["SVHN", "CIFAR-100"]
    else:
        import torchvision.transforms as transforms
        from torchvision import datasets
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        normalize = transforms.Normalize(mean, std)
        train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                       transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), normalize])
        eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
        ood_tf = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
        from fedl.datasets import TransformedSubset
        from sklearn.model_selection import StratifiedShuffleSplit
        base_dataset = datasets.CIFAR100(root=TORCH_ROOT, train=True, download=True, transform=None)
        base_targets = np.array(base_dataset.targets)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
        split_train_idx, split_val_idx = next(sss.split(np.arange(len(base_dataset)), base_targets))
        train_subset = TransformedSubset(torch.utils.data.Subset(base_dataset, split_train_idx.tolist()), transform=train_tf)
        val_subset = TransformedSubset(torch.utils.data.Subset(base_dataset, split_val_idx.tolist()), transform=eval_tf)
        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8)
        validloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=8)
        testloader = DataLoader(datasets.CIFAR100(root=TORCH_ROOT, train=False, download=True, transform=eval_tf),
                                batch_size=batch_size, shuffle=False, num_workers=8)
        svhn = datasets.SVHN(root=TORCH_ROOT, split="test", download=True, transform=eval_tf)
        tin_dir = os.path.join(DATASETS_DIR, "tiny-imagenet-200", "test")
        tin = datasets.ImageFolder(tin_dir, transform=ood_tf)
        ood1 = DataLoader(svhn, batch_size=batch_size, shuffle=False, num_workers=8)
        ood2 = DataLoader(tin, batch_size=batch_size, shuffle=False, num_workers=8)
        ood_names = ["SVHN", "TinyImageNet"]

    model = fedl_models.FEDL(ID_dataset, dropout_rate=0, device=device,
                             hidden_dim=256, num_layers=2).to(device)

    # ---- training: replicate F-EDL repo train() (Adam, StepLR@20 gamma=0.1, same loss,
    # eval every 10 epochs, patience-3 early stopping), fixed to take num_classes ----
    num_classes = 10 if ID_dataset == "CIFAR-10" else 100
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
    best_val, epochs_no_improve = 0.0, 0
    for epoch in range(num_epochs):
        model.train()
        for x, y in trainloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_oh = torch.nn.functional.one_hot(y, num_classes).float()
            opt.zero_grad(set_to_none=True)
            alpha, p, tau = model(x)
            mu, var = fedl_uq.compute_moments(alpha, p, tau)
            loss = torch.sum((y_oh - mu) ** 2) + torch.sum(var) + torch.sum((y_oh - p) ** 2)
            loss.backward()
            opt.step()
        sched.step()
        e = epoch + 1
        if e % 10 == 0:
            model.eval()
            correct, total, val_loss = 0, 0, 0.0
            with torch.no_grad():
                for x_v, y_v in validloader:
                    x_v, y_v = x_v.to(device), y_v.to(device)
                    y_oh = torch.nn.functional.one_hot(y_v, num_classes).float()
                    alpha_v, p_v, tau_v = model(x_v)
                    mu_v, var_v = fedl_uq.compute_moments(alpha_v, p_v, tau_v)
                    val_loss += (torch.sum((y_oh - mu_v) ** 2) + torch.sum(var_v)
                                 + torch.sum((y_oh - p_v) ** 2)).item()
                    correct += (mu_v.argmax(1) == y_v).sum().item()
                    total += y_v.size(0)
            val_acc = 100 * correct / total
            print(f"[F-EDL {ID_dataset}] epoch {e}/{num_epochs} val_acc={val_acc:.2f} val_loss={val_loss:.1f}", flush=True)
            if val_acc > best_val + 0.01:
                best_val, epochs_no_improve = val_acc, 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= 3:
                print("Early stopping triggered.", flush=True)
                break

    top1_acc, top2_acc = fedl_train.eval(model, testloader, device)
    _, conf_aupr, brier = fedl_uq.conf_calibration(model, testloader, device)
    _, ood_aupr = fedl_uq.ood_detection(model, testloader, ood1, ood2, device)
    # ood_detection returns [ {AU,EU} per ood set ]; DAPPr paper reports OOD AUPR via EU
    ood_eu_aupr = {name: round(float(d["EU"]) * 100, 4) for name, d in zip(ood_names, ood_aupr)}
    conf_eu_aupr = round(float(conf_aupr["EU"]) * 100, 4)

    summary = {"config": config, "method": "F-EDL", "dataset": ID_dataset, "seed": seed,
               "test_acc": round(float(top1_acc), 4),
               "Conf_AUPR_EU": conf_eu_aupr,
               "Conf_AUPR_AU": round(float(conf_aupr["AU"]) * 100, 4),
               "Brier": round(float(brier), 4), **{f"{k}_AUPR_EU": v for k, v in ood_eu_aupr.items()}}
    print("\nFINAL_RESULTS_JSON\n" + json.dumps(summary, indent=2) + "\nEND_RESULTS_JSON", flush=True)
    os.makedirs("outs", exist_ok=True)
    with open(f"outs/F-EDL_{ID_dataset}_seed{seed}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
