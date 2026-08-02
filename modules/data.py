import os, shutil
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision import datasets
from .paths import DATA_ROOT, TORCH_ROOT, DATASETS_DIR, SPLITS_DIR

NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
CIFAR10_NORM = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
CIFAR100_NORM = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])


def torchvision_ds(name, train=True):
    if name == "cifar10":
        return datasets.CIFAR10(root=TORCH_ROOT, train=train, download=True)
    if name == "cifar100":
        return datasets.CIFAR100(root=TORCH_ROOT, train=train, download=True)
    if name == "svhn":
        return datasets.SVHN(root=TORCH_ROOT, split="train" if train else "test", download=True)
    raise ValueError(name)


def _tiny_test_dir():
    """Build an ImageFolder-compatible tiny-imagenet test dir (val/images + val_annotations.txt)."""
    tiny_root = os.path.join(DATASETS_DIR, "tiny-imagenet-200")
    out_dir = os.path.join(tiny_root, "test")
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
        return out_dir
    val_img_dir = os.path.join(tiny_root, "val", "images")
    ann_file = os.path.join(tiny_root, "val", "val_annotations.txt")
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                fname, cls = parts[0], parts[1]
                dst = os.path.join(out_dir, cls)
                os.makedirs(dst, exist_ok=True)
                src = os.path.join(val_img_dir, fname)
                if os.path.exists(src) and not os.path.exists(os.path.join(dst, fname)):
                    shutil.copy(src, os.path.join(dst, fname))
    return out_dir


def get_cifar_loaders(name="cifar10", bs=64, num_workers=8, val_ratio=0.05, seed=42,
                      subset_size=None, device="cuda:0", pin=True):
    if name == "cifar10":
        norm, transform_train = CIFAR10_NORM, transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), CIFAR10_NORM])
    else:
        norm, transform_train = CIFAR100_NORM, transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), CIFAR100_NORM])
    transform_eval = transforms.Compose([transforms.ToTensor(), norm])

    base_train = datasets.__dict__[name.upper()](root=TORCH_ROOT, train=True, download=True, transform=transform_train)
    base_val = datasets.__dict__[name.upper()](root=TORCH_ROOT, train=True, download=True, transform=transform_eval)
    test_ds = datasets.__dict__[name.upper()](root=TORCH_ROOT, train=False, download=True, transform=transform_eval)

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(base_train), generator=g)
    n_val = int(val_ratio * len(base_train))
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    if subset_size is not None:
        train_idx = train_idx[:subset_size]
    train_ds, val_ds = Subset(base_train, train_idx.tolist()), Subset(base_val, val_idx.tolist())

    dl = lambda ds: DataLoader(ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=pin)
    dl_eval = lambda ds: DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return dl(train_ds), dl_eval(val_ds), dl_eval(test_ds)


def get_cifar_ood_loaders(name="cifar10", bs=64, num_workers=8):
    """OOD loaders per paper: CIFAR-10 -> SVHN, CIFAR-100 ; CIFAR-100 -> SVHN, TinyImageNet."""
    if name == "cifar10":
        norm = CIFAR10_NORM
        svhn_transform = transforms.Compose([transforms.ToTensor(), norm])
        ood1 = datasets.SVHN(root=TORCH_ROOT, split="test", download=True, transform=svhn_transform)
        ood2 = datasets.CIFAR100(root=TORCH_ROOT, train=False, download=True, transform=svhn_transform)
        names = ["SVHN", "CIFAR-100"]
    else:
        norm = CIFAR100_NORM
        ood_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), norm])
        ood1 = datasets.SVHN(root=TORCH_ROOT, split="test", download=True, transform=ood_transform)
        ood2 = datasets.ImageFolder(_tiny_test_dir(), transform=ood_transform)
        names = ["SVHN", "TinyImageNet"]
    loaders = [DataLoader(d, batch_size=bs, shuffle=False, num_workers=num_workers) for d in (ood1, ood2)]
    return loaders, names


def get_fgvc_loaders(dataset="CUB", bs=256, num_workers=8):
    from utils.data_builder import get_fgvc_data, get_ood_dataloader, get_tiny_imagenet_data
    data_root = DATA_ROOT
    if dataset == "CUB":
        train_dl, val_dl, test_dl = get_fgvc_data(data_root=data_root, dataset_name="CUB", batch_size=bs, num_workers=num_workers)
    elif dataset == "StanfordDogs":
        train_dl, val_dl, test_dl = get_fgvc_data(data_root=data_root, dataset_name="StanfordDogs", batch_size=bs, num_workers=num_workers)
    elif dataset == "TinyImageNet":
        train_dl, val_dl, test_dl = get_tiny_imagenet_data(data_root=data_root, dataset_name="TinyImageNet", batch_size=bs, num_workers=num_workers)
    else:
        raise ValueError(dataset)
    return train_dl, val_dl, test_dl


def get_fgvc_ood_loaders(bs=256, num_workers=8, resolution=224):
    from utils.data_builder import get_ood_dataloader
    imagenet_o_dl, dtd_dl, place365_dl = get_ood_dataloader(DATA_ROOT, resolution, bs, num_workers)
    return imagenet_o_dl, dtd_dl, place365_dl
