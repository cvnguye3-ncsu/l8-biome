import os
import random
from os.path import join

import numpy as np
import torch.multiprocessing
from PIL import Image
from torch.utils.data import Dataset

class DirectoryDataset(Dataset):
    def __init__(self, root, path, image_set, transform, target_transform):
        super(DirectoryDataset, self).__init__()
        
        self.dir = root
        self.img_dir = join(self.dir, "img")
        self.label_dir = join(self.dir, "label")

        self.transform = transform
        self.target_transform = target_transform

        self.img_files = np.array(sorted(os.listdir(self.img_dir)))
        self.img_files = self.img_files[:int(.01*len(self.img_files))]
        
        assert len(self.img_files) > 0
        
        if os.path.exists(join(self.dir, "label")):
            self.label_files = np.array(sorted(os.listdir(self.label_dir)))
            self.label_files = self.label_files[:int(.01*len(self.label_files))]
            
            assert len(self.img_files) == len(self.label_files)
        else:
            self.label_files = None

    def __getitem__(self, index):
        image_fn = self.img_files[index]
        img = Image.open(join(self.img_dir, image_fn))

        if self.label_files is not None:
            label_fn = self.label_files[index]
            label = Image.open(join(self.label_dir, label_fn))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)

        if self.label_files is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            label = self.target_transform(label)
        else:
            label = torch.zeros(img.shape[1], img.shape[2], dtype=torch.int64) - 1

        mask = (label > 0).to(torch.float32)
        return img, label, mask

    def __len__(self):
        return len(self.img_files)

class ContrastiveSegDataset(Dataset):
    def __init__(self,
                 pytorch_data_dir,
                 dataset_name,
                 crop_type,
                 image_set,
                 transform,
                 target_transform,
                 cfg,
                 aug_geometric_transform=None,
                 aug_photometric_transform=None,
                 mask=False,
                 extra_transform=None,
                 model_type_override=None
                 ):
        super(ContrastiveSegDataset).__init__()

        self.image_set = image_set
        self.dataset_name = dataset_name
        self.mask = mask
        self.extra_transform = extra_transform

        self.n_classes = cfg.dir_dataset_n_classes
        dataset_class = DirectoryDataset
        extra_args = dict(path=cfg.dir_dataset_name)

        self.aug_geometric_transform = aug_geometric_transform
        self.aug_photometric_transform = aug_photometric_transform

        self.dataset = dataset_class(
            root=pytorch_data_dir,
            image_set=self.image_set,
            transform=transform,
            target_transform=target_transform, **extra_args)


    def __len__(self):
        return len(self.dataset)

    def _set_seed(self, seed):
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7

    def __getitem__(self, ind):
        pack = self.dataset[ind]

        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        self._set_seed(seed)
        # coord_entries = torch.meshgrid([torch.linspace(-1, 1, pack[0].shape[1]),
        #                                 torch.linspace(-1, 1, pack[0].shape[2])])
        # coord = torch.cat([t.unsqueeze(0) for t in coord_entries], 0)

        if self.extra_transform is not None:
            extra_trans = self.extra_transform
        else:
            extra_trans = lambda i, x: x

        # NOTE: I really like returning a dictionary of items. 
        ret = {
            "ind": ind,
            "img": extra_trans(ind, pack[0]),
            "label": extra_trans(ind, pack[1]),
            "img_pos": extra_trans(ind, pack[0]),
            "label_pos": extra_trans(ind, pack[1]),
        }

        if self.mask:
            ret["mask"] = pack[2]
            
        if self.aug_photometric_transform is not None:
            ret['img_pos_aug'] = self.aug_photometric_transform(ret['img_pos'])

        return ret