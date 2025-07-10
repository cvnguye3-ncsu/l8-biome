from pathlib import Path
from typing import Iterable

from PIL import Image

import pandas as pd

import torch as pt
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional as F
from importables.general.cloud_classes import ClassRegistry

class MyTransformations:
    def __init__(self):
        self.angles = (0, 90, 180, 270)

    def __call__(self, imgs: Iterable[pt.Tensor]):
        if pt.rand(1).item() < 0.5:
            imgs = [F.hflip(img) for img in imgs]

        idx = pt.randint(0, 4, size=(1, 1)).item()
        angle = self.angles[idx]
        
        imgs = [F.rotate(img, angle=angle) for img in imgs]

        return imgs
    
class SegmentationDataset(Dataset):
    def __init__(self, seed: int, split: str, data_location: str, 
                 class_reg: ClassRegistry,
                 data_aug=False, subset_ratio: float = 1.0):
        self._data_aug_flag = data_aug
        
        DATA_PATH = Path(data_location)
        IMG_FOLDER = DATA_PATH / 'dataset' / 'img'
        LBL_FOLDER = DATA_PATH / 'dataset' / 'label'
        
        df = pd.read_csv(DATA_PATH / 'seeds' / f'{seed}_{subset_ratio}_split.csv')
        df = df[df['subset'] == True]
        images_series = df[df['split'] == split]['image_name']
        
        images = [IMG_FOLDER / name for name in images_series]
        labels = [LBL_FOLDER / name for name in images_series]
        
        self._image_fns = sorted(images)
        self._label_fns = sorted(labels)
        
        self._size = len(self._image_fns)

        # --- MEAN AND STANDARD DEVIATION ---
        means: pt.Tensor = pt.load(DATA_PATH / 'seeds' / f'{seed}_{subset_ratio}_mean.pt')
        stds: pt.Tensor = pt.load(DATA_PATH / 'seeds' / f'{seed}_{subset_ratio}_std.pt')
        
        norm_trans = transforms.Normalize(mean=means.tolist(), std=stds.tolist())

        # --- TRANSFORMATION ---
        self.img_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(pt.float32, scale=True),
            norm_trans
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(pt.long)
        ])

        self._data_aug = MyTransformations()
        self._reduce = class_reg.reduce_classes

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        img_path = self._image_fns[idx]
        label_path = self._label_fns[idx]

        img_name = img_path.name.split('.')[0]
        img = self.img_transform(Image.open(img_path))
        lbl = self.mask_transform(Image.open(label_path))

        if self._data_aug_flag:
            img, lbl = self._data_aug((img, lbl))
            
        lbl = self._reduce(lbl)

        return img, lbl, img_name
