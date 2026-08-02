import h5py, json, io, os
from PIL import Image
import numpy as np
import torch.utils.data as data
from pathlib import Path

def json_flist_reader(flist):
    imlist = []
    with open(flist, 'r') as rf:
        json_data = json.load(rf)
    for imkey in sorted(json_data.keys()):
        imlabel = json_data[imkey]
        imlist.append((imkey, int(imlabel)))

    class_ids = sorted(list(set([l for _, l in imlist])))
    class_mapping = {v: i for i, v in enumerate(class_ids)}
    imlist = [(imkey, class_mapping[imlabel]) for imkey, imlabel in imlist]
    return imlist


def h5_loader(data_source, imkey):
    return Image.open(io.BytesIO(np.array(data_source[imkey]))).convert("RGB")

def file_loader(data_source, imkey):
    return Image.open(os.path.join(data_source, imkey)).convert("RGB")

class ImageData(data.Dataset):
    def __init__(self, image_root, flist, transform=None, target_transform=None, flist_reader=json_flist_reader,
                 is_hdf5=False):
        self.data_source = h5py.File(image_root, 'r') if is_hdf5 else image_root
        self.loader = h5_loader if is_hdf5 else file_loader
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        print('image_path:', image_root, ' num_images:', len(self.imlist))

    def __getitem__(self, index):
        imkey, target = self.imlist[index]
        img = self.loader(self.data_source, imkey)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imlist)


def traverse_hdf5(h5_group, prefix=""):
    file_list = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    for key in h5_group:
        item = h5_group[key]
        path = f"{prefix}{key}"
        if isinstance(item, h5py.Group):
            file_list.extend(traverse_hdf5(item, path+"/"))
        elif Path(key).suffix.lower() in image_exts:
            file_list.append(path)
    return file_list


def traverse_dir(dir_path, prefix=""):
    file_list = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    for sub_path in os.listdir(dir_path):
        if sub_path.startswith('.'): continue
        full_path = os.path.join(dir_path, sub_path)
        full_sub_path = prefix + sub_path
        if os.path.isdir(full_path):
            file_list.extend(traverse_dir(full_path, full_sub_path + "/"))
        elif Path(sub_path).suffix.lower() in image_exts:
            file_list.append(full_sub_path)
    return file_list

class OODData(data.Dataset):
    def __init__(self, image_root, transform=None, is_hdf5=False):
        self.data_source = h5py.File(image_root, 'r') if is_hdf5 else image_root
        self.loader = h5_loader if is_hdf5 else file_loader
        self.imlist = traverse_hdf5(self.data_source) if is_hdf5 else traverse_dir(self.data_source)
        self.transform = transform
        print('image_path:', image_root, ' num_images:', len(self.imlist))

    def __getitem__(self, index):
        imkey = self.imlist[index]
        img = self.loader(self.data_source, imkey)
        if self.transform is not None:
            img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.imlist)