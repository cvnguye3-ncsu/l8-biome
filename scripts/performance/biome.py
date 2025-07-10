from pathlib import Path
import hydra
from omegaconf import DictConfig

import pandas as pd
from importables.pytorch.metrics import SemanticSegmentationMetrics
from importables.pytorch.dataset import SegmentationDataset

from PIL import Image

import torch as pt

import tqdm as tqdm
import math

# Model
# -----
MODEL_NAME = 'satlas'
VERSION = 'finetuning'

# Path
# ----
BASE_PATH = Path('./')
DATA_PATH = BASE_PATH / '_data'
DATASET_PATH = DATA_PATH / 'dataset'
SEEDS_PATH = DATA_PATH / 'seeds'

# Constants
# ---------

# Main
# ----
@hydra.main(config_path=str(BASE_PATH  / '..' / '..' / 'conf'), config_name="main_config", version_base="1.3")
def main(cfg: DictConfig):
    df = pd.read_csv(SEEDS_PATH / f'{cfg.seed}_{cfg.subset_ratio}_split.csv', index_col=0)
    BIOMES = df['biome'].unique()
    
    model = pt.jit.load(
        BASE_PATH / 'models' / MODEL_NAME / 'logs' / f'seed-{cfg.seed}' / VERSION / 'model.pt', map_location='cuda').eval()
    
    for biome in BIOMES:
        print('==== ', biome, ' ====')
        selected_idx = df['subset'] == True
        biome_idx = df['biome'] == biome 
        test_idx = df['split'] == 'test'
        
        biome_df = df[selected_idx & biome_idx & test_idx]
        biome_metrics = SemanticSegmentationMetrics()
        
        ds = SegmentationDataset(cfg.seed, 'test', str(DATA_PATH), subset_ratio=cfg.subset_ratio)
        
        TOTAL_BATCHES = math.ceil(len(biome_df['image_name'])/cfg.batch_size)
        all_img_names = biome_df['image_name'].to_list()
        
        for i in range(TOTAL_BATCHES):
            img_names = all_img_names[i*cfg.batch_size: (i+1)*cfg.batch_size]
            
            imgs = map(lambda name: Image.open(DATASET_PATH / 'img' / name), img_names)
            lbls = map(lambda name: Image.open(DATASET_PATH / 'label' / name), img_names)
            
            imgs = [ds.img_transform(img).cuda().unsqueeze(0) for img in imgs]
            lbls = [ds.mask_transform(lbl).cuda().unsqueeze(0) for lbl in lbls]
            
            img_tensor, lbl_tensor = pt.stack(imgs, dim=0), pt.stack(lbls, dim=0)
            
            with pt.no_grad():
                pred = model(img_tensor)
            
            biome_metrics.update(pred, lbl_tensor)
            
        compute = biome_metrics.compute()
        biome_metrics.reset()
        df = biome_metrics.tabulate_compute(compute)
        
        print(df.to_latex(index=True, caption="Biome Metrics", label="tab:class_metrics", float_format="%.2f"))

if __name__ == '__main__':
    main()