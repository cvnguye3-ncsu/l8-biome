from pathlib import Path
import hydra
from omegaconf import DictConfig

import pandas as pd
from importables.pytorch.metrics import CloudShadowOffsetError
from importables.pytorch.dataset import SegmentationDataset

import pytorch_lightning as L

from PIL import Image

import torch as pt

from tqdm import tqdm
import math

import kornia as K

# Path
# ----
BASE_PATH = Path('./')
DATA_PATH = BASE_PATH / '_data'
DATASET_PATH = DATA_PATH / 'dataset'
SEEDS_PATH = DATA_PATH / 'seeds'

# Model
# -----
MODEL_NAME = 'satlas'
VERSION = 'finetuning'

CONNECTIVITY = 8
DIAG = math.sqrt(224**2 + 224**2)

BATCH_SIZE = 16

# Main
# ----
@hydra.main(config_path=str(BASE_PATH  / '..' / '..' / 'conf'), 
            config_name="main_config", version_base="1.3")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)
    print('=== ', MODEL_NAME, ' ===')
    
    df = pd.read_csv(SEEDS_PATH / f'{cfg.seed}_{cfg.subset_ratio}_split.csv', index_col=0)
    split_cond = (df['split'] == 'test') | (df['split'] == 'val')
    df = df[split_cond & df['subset'] == True]
    df = df[(df['clear'] >= .2) & (df['shadow'] >= .1)]
    df = df[(df['cloud'] + df['thin_cloud'] - df['shadow']).abs() <= .2]
    
    model = pt.jit.load(
        BASE_PATH / 'models' / MODEL_NAME / 'logs' / f'seed-{cfg.seed}' / VERSION / 'model.pt', map_location='cuda').eval()
    
    ds = SegmentationDataset(cfg.seed, 'test', str(DATA_PATH), subset_ratio=cfg.subset_ratio)
    
    TOTAL_BATCHES = math.ceil(len(df['image_name'])/BATCH_SIZE)
    all_img_names = df['image_name'].to_list()
    
    metric = CloudShadowOffsetError().cuda()
    
    for i in tqdm(list(range(TOTAL_BATCHES)),
                desc='Calculating boundary metric . . .', unit='batch'):
        img_names = all_img_names[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
        
        imgs = map(lambda name: Image.open(DATASET_PATH / 'img' / name), img_names)
        lbls = map(lambda name: Image.open(DATASET_PATH / 'label' / name), img_names)
        
        imgs = [ds.img_transform(img).cuda() for img in imgs]
        lbls = [ds.mask_transform(lbl).cuda() for lbl in lbls]
        
        img_tensor, lbl_tensor = pt.stack(imgs, dim=0), pt.stack(lbls, dim=0).squeeze(1)
    
        with pt.no_grad():
            logit = model(img_tensor)
            pred = logit.argmax(dim=1)
        
        metric.update(pred, lbl_tensor)
        
    mse = metric.compute()
    print(mse)

if __name__ == '__main__':
    main()