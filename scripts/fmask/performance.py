from pathlib import Path
import torch as pt
import torchvision.transforms.v2 as transform

from PIL import Image
import pandas as pd

import hydra 
from omegaconf import DictConfig
from tqdm import tqdm

import numpy as np

from importables.pytorch.metrics import SemanticSegmentationMetrics
from importables.project.cloud_classes import ClassRegistry

# Paths
# -----
BASE_PATH = Path('./')
DATA_PATH = BASE_PATH / '_data'
SEED_PATH = DATA_PATH / 'seeds'
LBL_PATH = DATA_PATH / 'dataset' / 'label'
FMASK_PATH = DATA_PATH / 'auxiliary' / 'fmask'

# Main
# ----
@hydra.main(config_path=str(BASE_PATH / '..' / '..' / 'conf'), 
            config_name="main_config", version_base="1.3")
def main(cfg: DictConfig):
    seed_df = pd.read_csv(SEED_PATH / f'{cfg.seed}_{cfg.subset_ratio}_split.csv', index_col=0)
    seed_df = seed_df[seed_df['split'] == 'test']
    seed_df = seed_df[seed_df['subset'] == True]
    
    trans = transform.Compose([
        transform.ToImage(),
        transform.ToDtype(pt.long)
    ])
    
    class_reg = ClassRegistry()

    metrics = SemanticSegmentationMetrics(class_reg)
    
    for _, row in tqdm(list(seed_df.iterrows()),
                       desc='Running FMask . . .', unit='image'):
        label = Image.open(LBL_PATH / row['image_name'])
        label_arr = np.array(label)
        # label_arr = class_reg.reduce_cloud_only_map(label_arr)
        
        fmask = Image.open(FMASK_PATH / row['image_name'])
        fmask_arr = np.array(fmask)
        # fmask_arr = class_reg.reduce_cloud_only_map(fmask_arr)
        
        metrics.update(trans(fmask_arr), trans(label_arr))
        
    results = metrics.compute()
    
    print(results.to_latex(index=True, caption="Class Metrics", label="tab:class_metrics", float_format="%.2f"))

if __name__ == '__main__':
    main() 