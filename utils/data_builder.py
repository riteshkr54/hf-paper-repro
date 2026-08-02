from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os, h5py
from .datasets import ImageData, OODData, traverse_hdf5, traverse_dir

DATASET_REGISTRY = {
    "CUB": {"name": "CUB-200-2011",
            "image_path": "datasets/CUB_200_2011/CUB_200_2011/images",
            "train_split": "splits/CUB/train.json",
            "val_split": "splits/CUB/val.json",
            "test_split": "splits/CUB/test.json",
            "num_classes": 200},
    "StanfordDogs": {"name": "Stanford Dogs",
                     "image_path": "datasets/StanfordDogs/Images",
                     "train_split": "splits/StanfordDogs/train.json",
                     "val_split": "splits/StanfordDogs/val.json",
                     "test_split": "splits/StanfordDogs/test.json",
                     "num_classes": 120},
    "TinyImageNet": {"name": "Tiny-ImageNet",
                     "image_path": "datasets/tiny-imagenet-200",
                     "train_path": "datasets/tiny-imagenet-200/train",
                     "val_path": "datasets/tiny-imagenet-200/val/images",
                     "val_label_path": "datasets/tiny-imagenet-200/val/val_annotations.txt",
                     "val_ratio": 0.05,
                     "num_classes": 200},

    #OOD data
    "ImageNetO":{"name": "ImageNet-O",
                 "image_path": "datasets/imagenet-o"},
    "DTD":{"name": "DTD",
           "image_path": "datasets/dtd/images"},
    "Places365":{"name": "Places365",
                "image_path": "datasets/Places365/val_256"}
}

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def _get_image_path(data_root='data', sub_path='', use_hdf5=False):
    image_path = os.path.join(data_root, sub_path)
    if use_hdf5: image_path = image_path + '.hdf5'
    return image_path


def get_fgvc_data(data_root='data', dataset_name='CUB', batch_size=64, num_workers=8, use_hdf5=False):
    dataset_cfg = DATASET_REGISTRY[dataset_name]
    image_path = _get_image_path(data_root=data_root, sub_path=dataset_cfg['image_path'], use_hdf5=use_hdf5)

    flist_train = os.path.join(data_root, dataset_cfg['train_split'])
    flist_val   = os.path.join(data_root, dataset_cfg['val_split'])
    flist_test  = os.path.join(data_root, dataset_cfg['test_split'])

    train_transform = transforms.Compose([ transforms.Resize(256), transforms.RandomCrop(224),
                                           transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(), NORMALIZE])

    val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                        transforms.ToTensor(), NORMALIZE])

    train_loader = DataLoader(ImageData(image_path, flist=flist_train, transform=train_transform, is_hdf5=use_hdf5),
                              batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(ImageData(image_path, flist=flist_val, transform=val_transform, is_hdf5=use_hdf5),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(ImageData(image_path, flist=flist_test, transform=val_transform, is_hdf5=use_hdf5),
                             batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_tiny_imagenet_data(data_root='data', dataset_name='TinyImageNet',batch_size=64, num_workers=8, use_hdf5=False):
    dataset_cfg = DATASET_REGISTRY[dataset_name]
    train_path = _get_image_path(data_root=data_root, sub_path=dataset_cfg['train_path'], use_hdf5=use_hdf5)
    val_path = train_path
    test_path = _get_image_path(data_root=data_root, sub_path=dataset_cfg['val_path'], use_hdf5=use_hdf5)
    annotation_root = os.path.join(data_root, dataset_cfg['val_label_path'])


    if use_hdf5:
        with h5py.File(train_path, 'r') as h5_file:
            train_list = traverse_hdf5(h5_file)
    else:
        train_list = traverse_dir(train_path)
    train_list = sorted(train_list)

    # map class name to label id
    class_to_index = {name: idx for idx, name in
                      enumerate(sorted(set([file_path.split('/')[0] for file_path in train_list])))}
    class_samples = {}
    for file_path in train_list:
        class_name = file_path.split('/')[0]
        if class_name in class_samples:
            class_samples[class_name].append((file_path, class_to_index[class_name]))
        else:
            class_samples[class_name] = [(file_path, class_to_index[class_name])]

    # split train and val
    train_ratio = 1 - dataset_cfg['val_ratio']
    train_list = [class_samples[class_name][:int(len(class_samples[class_name]) * train_ratio)] for class_name in
                  class_samples]
    train_list = [x for sub in train_list for x in sub]

    val_list = [class_samples[class_name][int(len(class_samples[class_name]) * train_ratio):] for class_name in class_samples]
    val_list = [x for sub in val_list for x in sub]

    with open(annotation_root, 'r') as f:
        test_list = []
        for line in f.readlines():
            items = line.strip().split('\t')
            test_list.append((items[0], class_to_index[items[1]]))

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(), NORMALIZE])
    val_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE])

    train_loader = DataLoader(ImageData(image_root=train_path, flist=train_list, transform=train_transform,
                                        flist_reader=lambda x: x, is_hdf5=use_hdf5),
                              batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(ImageData(image_root=val_path, flist=val_list, transform=val_transform,
                                      flist_reader=lambda x: x, is_hdf5=use_hdf5),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(ImageData(image_root=test_path, flist=test_list, transform=val_transform,
                                       flist_reader=lambda x: x, is_hdf5=use_hdf5),
                             batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_id_dataloader(data_root='data', dataset_name='CUB', batch_size=256, num_workers=8, use_hdf5=False):
    if dataset_name in ['CUB', 'StanfordDogs']:
        return get_fgvc_data(data_root, dataset_name, batch_size, num_workers, use_hdf5)
    elif dataset_name == 'TinyImageNet':
        return get_tiny_imagenet_data(data_root, dataset_name, batch_size, num_workers, use_hdf5)
    else:
        raise NotImplementedError

def get_ood_dataloader(data_root='data', resolution=224, batch_size=256, num_workers=8, use_hdf5=False):
    val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                        transforms.ToTensor(), NORMALIZE])

    if resolution == 64:
        val_transform = transforms.Compose([transforms.Resize(72), transforms.CenterCrop(64),
                                            transforms.ToTensor(), NORMALIZE])

    def get_ood_loader(dataset_name):
        image_path = _get_image_path(data_root, DATASET_REGISTRY[dataset_name]['image_path'], use_hdf5=use_hdf5)
        ood_dataset = OODData(image_root=image_path, transform=val_transform, is_hdf5=use_hdf5)
        loader = DataLoader(ood_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
        return loader

    imagenet_o_loader = get_ood_loader('ImageNetO')
    dtd_loader = get_ood_loader('DTD')
    places365_loader = get_ood_loader('Places365')

    return imagenet_o_loader, dtd_loader, places365_loader


def get_classes_num(dataset_name):
    return DATASET_REGISTRY[dataset_name]['num_classes']