from pathlib import Path

from PIL import Image

import pandas as pd

import torch as pt
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional as F


# class SatlasNorm(transforms.Transform):
#     def __init__(self):
#         super().__init__()
        
#     def transform(self, input: pt.Tensor, _):
#         normed = (input - 4000)/16320
#         normed = normed.clip(0, 1)
        
#         return normed

class MyTransformations:
    def __init__(self):
        self.angles = (0, 90, 180, 270)

    def __call__(self, pair):
        img, mask = pair

        # Horizontal flip
        if pt.rand(1).item() < 0.5:
            img, mask = (F.hflip(img), F.hflip(mask))

        # 90 angle rotation
        idx = pt.randint(0, 4, size=(1, 1)).item()
        angle = self.angles[idx]

        return (F.rotate(img, angle=angle), F.rotate(mask, angle=angle))
    
class SegmentationDataset(Dataset):
    def __init__(self, seed: int, split: str, data_location: str,
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

        # --- DATASET PERCENTAGE ---
        # if subset_ratio < 1:
        #     idx = pt.randperm(self.size)
        #     self.size = int(subset_ratio*self.size)

        #     self.image_fns = [self.image_fns[i] for i in idx]
        #     self.label_fns = [self.label_fns[i] for i in idx]

        #     self.image_fns = self.image_fns[:self.size]
        #     self.label_fns = self.label_fns[:self.size]

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

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        img_path = self._image_fns[idx]
        label_path = self._label_fns[idx]

        img_name = img_path.name.split('.')[0]
        image = Image.open(img_path)
        mask = Image.open(label_path)

        if self.img_transform:
            image = self.img_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        if self._data_aug_flag:
            image, mask = self._data_aug((image, mask))

        return image, mask, img_name
