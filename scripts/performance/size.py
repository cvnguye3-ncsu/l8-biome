from pathlib import Path
import hydra
from omegaconf import DictConfig

import pandas as pd
from importables.pytorch.metrics import SemanticSegmentationMetrics
from importables.pytorch.dataset import SegmentationDataset
from importables.general.image_processing import remove_large_components

from importables.general.cloud_classes import ClassRegistry

import pytorch_lightning as L

from PIL import Image
import numpy as np

import torch as pt

from tqdm import tqdm
import math

# Path
# ----
BASE_PATH = Path('./')
DATA_PATH = BASE_PATH / '_data'
DATASET_PATH = DATA_PATH / 'dataset'
SEEDS_PATH = DATA_PATH / 'seeds'

# Constants
# ---------
SIZES = [5, 50, 500]
AVERAGE = None
IGNORE_INDEX = 255
BATCH_SIZE = 16

MODEL_NAME = 'lsknet'
VERSION = 'finetuning'

# Main
# ----
@hydra.main(config_path=str(BASE_PATH  / '..' / '..' / 'conf'), 
            config_name="main_config", version_base="1.3")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)
    print('=== ', MODEL_NAME, ' ===')
    
    df = pd.read_csv(SEEDS_PATH / f'{cfg.seed}_{cfg.subset_ratio}_split.csv', index_col=0)
    df = df[(df['split'] == 'test') & (df['subset'] == True)]
    
    model = pt.jit.load(
        BASE_PATH / 'models' / MODEL_NAME / 'logs' / f'seed-{cfg.seed}' / VERSION / 'model.pt', map_location='cuda').eval()
    
    class_reg = ClassRegistry()
    ds = SegmentationDataset(cfg.seed, 'test', str(DATA_PATH), subset_ratio=cfg.subset_ratio)
    
    TOTAL_BATCHES = math.ceil(len(df['image_name'])/BATCH_SIZE)
    all_img_names = df['image_name'].to_list()
    
    for size in SIZES:
        print('Size: ', size)
        
        metric = SemanticSegmentationMetrics(class_reg, 
                    acc=False, f1=False, ignore_index=IGNORE_INDEX)
        
        for i in tqdm(list(range(TOTAL_BATCHES)),
                    desc='Calculating boundary metric . . .', unit='batch'):
            img_names = all_img_names[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
            
            imgs = map(lambda name: Image.open(DATASET_PATH / 'img' / name), img_names)
            lbls = map(lambda name: Image.open(DATASET_PATH / 'label' / name), img_names)
            
            imgs = [ds.img_transform(img).cuda() for img in imgs]
            lbls = [remove_large_components(np.array(lbl), max_size=size) for lbl in lbls]
            lbls = [ds.mask_transform(lbl).cuda() for lbl in lbls]
            
            img_tensor, lbl_tensor = pt.stack(imgs, dim=0), pt.stack(lbls, dim=0).squeeze(1)
            
            with pt.no_grad():
                logit = model(img_tensor)
                pred = logit.argmax(dim=1)
            
            metric.update(pred, lbl_tensor)
        
        results_df = metric.compute()
        print(results_df.to_latex(index=True, caption="REPLACE", label="tab:REPLACE", float_format="%.2f"))

if __name__ == '__main__':
    main()