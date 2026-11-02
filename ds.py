
import os.path
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
from PIL import Image

import torch
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset



class CIFAR100(VisionDataset):
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        coarse_label: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
    
        label_key = 'coarse_labels' if coarse_label else 'fine_labels'
        self.label_name_key = 'coarse_label_names' if coarse_label else 'fine_label_names'
    
        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.targets.extend(entry[label_key])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.label_name_key]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

def get_ds(ds, default_transform=True):
    from torchvision import transforms
    from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
    import dotenv, os
    dotenv.load_dotenv('.env')
    if ds == 'CIFAR10':
        root = os.getenv('LIGHT_DIR')
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_ds = CIFAR10(root=root, train=True, download=True, transform=T if default_transform else None)
        test_ds = CIFAR10(root=root, train=False, download=True, transform=T if default_transform else None)
    elif ds.startswith('CIFAR100'):
        root = os.getenv('LIGHT_DIR')
        assert ds in ['CIFAR100_coarse', 'CIFAR100'], "Can't recognize task"
        coarse_label = True if ds == 'CIFAR100_coarse' else False
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_ds = CIFAR100(root=root, train=True, coarse_label=coarse_label, download=True, transform=T if default_transform else None)
        test_ds = CIFAR100(root=root, train=False, coarse_label=coarse_label, download=True, transform=T if default_transform else None)
    elif ds == 'MNIST':
        root = os.getenv('LIGHT_DIR')
        T = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_ds = MNIST(root=root, train=True, download=True, transform=T if default_transform else None)
        test_ds = MNIST(root=root, train=False, download=True, transform=T if default_transform else None)
    elif ds == 'FashionMNIST':
        root = os.getenv('LIGHT_DIR')
        T = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        train_ds = FashionMNIST(root=root, train=True, download=True, transform=T if default_transform else None)
        test_ds = FashionMNIST(root=root, train=False, download=True, transform=T if default_transform else None)
    elif ds == 'CIFARDVS':
        import random
        class Cutout(object):
            """Randomly mask out one or more patches from an image.
            Args:
                n_holes (int): Number of patches to cut out of each image.
                length (int): The length (in pixels) of each square patch.
            """

            def __init__(self, length):
                self.length = length

            def __call__(self, img):
                h = img.size(2)
                w = img.size(3)
                mask = np.ones((h, w), np.float32)
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)
                mask[y1: y2, x1: x2] = 0.
                mask = torch.from_numpy(mask)
                mask = mask.expand_as(img)
                img = img * mask
                return img


        def function_nda(data, M=1, N=2):
            c = 15 * N
            rotate_tf = transforms.RandomRotation(degrees=c)
            e = 8 * N
            cutout_tf = Cutout(length=e)

            def roll(data, N=1):
                a = N * 2 + 1
                off1 = random.randint(-a, a)
                off2 = random.randint(-a, a)
                return torch.roll(data, shifts=(off1, off2), dims=(2, 3))

            def rotate(data, N):
                return rotate_tf(data)

            def cutout(data, N):
                return cutout_tf(data)

            transforms_list = [roll, rotate, cutout]
            sampled_ops = np.random.choice(transforms_list, M)
            for op in sampled_ops:
                data = op(data, N)
            return data


        def trans_t(data):
            resize = transforms.Resize(size=(48, 48))  # 48 48
            # tensorx = transforms.ToTensor()
            # imgx = transforms.ToPILImage()
            # new_data = []
            # data = data.transpose((0,2,3,1)).astype(np.uint8)
            data = torch.from_numpy(data)  # convert 2 torch tensor
            # for t in range(data.shape[0]):
            # new_data.append(tensorx(resize(imgx(data[t, ...]))))
            data = resize(data).float()
            # data = torch.stack(new_data, dim=0)
            # if self.transform is not None:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            data = function_nda(data)
            # data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
            return data
        def trans(data):
            resize = transforms.Resize(size=(48, 48))  # 48 48
            data = torch.from_numpy(data)  # convert 2 torch tensor
            data = resize(data).float()
            return data
        root = os.getenv('CIFARDVS_DIR')
        data1 = CIFAR10DVS(root=root, data_type='frame', frames_number=4, split_by='number', transform=trans if default_transform else None)
        train_ds, _ = torch.utils.data.random_split(data1, [9000, 1000], generator=torch.Generator().manual_seed(42))
        data2 = CIFAR10DVS(root=root,data_type='frame', frames_number=4, split_by='number', transform=trans if default_transform else None)
        _, test_ds = torch.utils.data.random_split(data2, [9000, 1000], generator=torch.Generator().manual_seed(42))
    else:
        raise Exception("Can't find dataset")
    return train_ds, test_ds