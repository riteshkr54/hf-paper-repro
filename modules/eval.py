import torch
import torchvision
from torch import nn


def build_model(name, num_classes=10, pretrained=False):
    """Model factory. 'vgg16' -> VGG16-BN (CIFAR-10, per paper/EDL literature);
    'resnet18'/'resnet50' -> torchvision with default stem (CIFAR-100 / fine-grained)."""
    if name == "vgg16":
        return torchvision.models.vgg16_bn(weights=None, num_classes=num_classes)
    if name == "resnet18":
        return torchvision.models.resnet18(weights=None, num_classes=num_classes)
    if name == "resnet50":
        return torchvision.models.resnet50(weights=None, num_classes=num_classes)
    raise ValueError(name)


@torch.no_grad()
def get_logits(model, loader):
    device = next(model.parameters()).device
    all_logits, all_labels = [], []
    for x, labels in loader:
        all_logits.append(model(x.to(device, non_blocking=True)))
        all_labels.append(labels.to(device, non_blocking=True))
    return torch.cat(all_logits), torch.cat(all_labels)


@torch.no_grad()
def evaluate_val_acc(model, val_loader):
    logits, labels = get_logits(model, val_loader)
    return (logits.argmax(1) == labels).float().mean().item() * 100


def aupr_ood(unc_id, unc_ood):
    import numpy as np
    from sklearn.metrics import average_precision_score
    unc_id = np.nan_to_num(unc_id)
    unc_ood = np.nan_to_num(unc_ood)
    bin_labels = np.concatenate([np.ones(unc_id.shape[0]), np.zeros(unc_ood.shape[0])])
    scores = -np.concatenate((unc_id, unc_ood))
    return average_precision_score(bin_labels, scores)


@torch.no_grad()
def evaluate_test_metrics(model, test_loader, *ood_loaders, ood_names=None,
                          uncertainty_func=None, device=None):
    """Reproduces the official train.py evaluate_test_metrics.
    Returns dict with test_acc, Conf_AUPR_<name>, OOD<i>_AUPR_<name>."""
    import numpy as np
    from sklearn.metrics import average_precision_score
    if ood_names is None:
        ood_names = [f"OOD{i+1}" for i in range(len(ood_loaders))]
    id_logits, id_labels = get_logits(model, test_loader)
    ood_logits_list = [get_logits(model, dl)[0] for dl in ood_loaders]

    to_numpy = lambda unc: {k: v.cpu().numpy() for k, v in unc.items()}
    id_unc = to_numpy(uncertainty_func(id_logits))
    ood_unc_list = [to_numpy(uncertainty_func(ood_logits)) for ood_logits in ood_logits_list]

    is_correct = (id_logits.argmax(1) == id_labels).cpu().numpy().astype(int)
    metrics = {"test_acc": round(is_correct.mean() * 100, 4)}
    for name, uncertainty in id_unc.items():
        metrics[f"Conf_AUPR_{name}"] = round(average_precision_score(is_correct, -uncertainty) * 100, 4)
    for i, ood_unc in enumerate(ood_unc_list):
        for name in id_unc:
            metrics[f"{ood_names[i]}_AUPR_{name}"] = round(aupr_ood(id_unc[name], ood_unc[name]) * 100, 4)
    return metrics
