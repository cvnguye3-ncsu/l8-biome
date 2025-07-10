from pathlib import Path

import hydra
from omegaconf import DictConfig

import torch as pt
from torchvision.transforms.v2 import ToImage

import numpy as np
from numpy.random import Generator
from randomgen import ChaCha

from PIL import Image
import pandas as pd
from tqdm import tqdm

from importables.general.cloud_classes import CloudClass
from importables.general.image_processing import disconnect_segments

BASE_PATH = Path('.')
DATA_PATH = BASE_PATH / '_data'
LBL_PATH = DATA_PATH / 'dataset' / 'label'
SEED_PATH = DATA_PATH / 'seeds'

IMG_SHAPE = (3, 224, 224)
UINT_8_MAX = (2**8) - 1

to_tensor = ToImage()

# Metadata
# --------
def define_splits(seed: int, train_ratio: float, val_ratio: float) -> pd.DataFrame:
    rng = Generator(ChaCha(seed))
    
    data_path = DATA_PATH / 'dataset' / 'img'
    imgs = list(data_path.glob("*.png"))
    size = len(imgs)
    rng.shuffle(imgs)
    
    train_imgs = imgs[0: (train_size:=int(train_ratio * size))]
    train_df = pd.DataFrame(train_imgs, columns=['image_path'])
    train_df['split'] = 'train'
    train_df['image_name'] = train_df['image_path'].apply(lambda path: path.name)
    
    val_imgs = imgs[train_size: (val_size:=train_size + int(val_ratio * size))]
    val_df = pd.DataFrame(val_imgs, columns=['image_path'])
    val_df['split'] = 'val'
    val_df['image_name'] = val_df['image_path'].apply(lambda path: path.name)
    
    test_imgs = imgs[val_size:]
    test_df = pd.DataFrame(test_imgs, columns=['image_path'])
    test_df['split'] = 'test'
    test_df['image_name'] = test_df['image_path'].apply(lambda path: path.name) 
    
    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    final_df['scene'] = final_df['image_name'].apply(lambda s: s.split('_')[0])
    
    print("Data leakage: ", final_df['image_name'].duplicated().any())
    
    test_df['image_path'].apply(lambda path: path.resolve())
    
    return final_df

def calculate_mean_std(seed: int, subset_ratio: float, raw_df: pd.DataFrame) -> None:
    df = raw_df[raw_df['split'] == 'val']
    df = df[df['subset'] == True]
    
    # Calculate mean
    sum = pt.zeros(3, device='cuda', dtype=pt.float64)
    
    for _, row in tqdm(list(df.iterrows()),
                       desc='Calculating mean . . .', unit='image'):
        img = Image.open(row['image_path'])
        img_tensor: pt.Tensor = to_tensor(img) 
        img_tensor = img_tensor.cuda().to(pt.float64) / UINT_8_MAX
        img_tensor = img_tensor.sum(dim=(1, 2))
        sum += img_tensor
        
    mean = sum / (pixel_count:=224 * 224 * len(df))
    print('Mean: ', mean)
    pt.save(mean, SEED_PATH / f'{seed}_{subset_ratio}_mean.pt')
    mean = mean.reshape(-1, 1, 1)
    
    # Calculate standard deviation
    sum_2 = pt.zeros(3, device='cuda', dtype=pt.float64)
    
    for _, row in tqdm(list(df.iterrows()),
                       desc='Calculating standard deviation . . .', unit='image'):
        img = Image.open(row['image_path'])
        img_tensor: pt.Tensor = to_tensor(img) 
        img_tensor = img_tensor.cuda().to(pt.float64) / UINT_8_MAX
        img_tensor = (img_tensor - mean)**2
        img_tensor = img_tensor.sum(dim=(1, 2))
        sum_2 += img_tensor
        
    std = pt.sqrt(sum_2/(pixel_count-1))
    print('Standard deviation: ', std)
    pt.save(std, SEED_PATH / f'{seed}_{subset_ratio}_std.pt')

def calculate_per_image_class_balance(raw_df: pd.DataFrame) -> pd.DataFrame:
    class_df = []
    
    for _, row in tqdm(list(raw_df.iterrows()),
                       desc='Calculating class balance of imagery . . .',
                       unit='image'):
        label = Image.open(LBL_PATH / row['image_name'])
        label_tensor: pt.Tensor = to_tensor(label).cuda()
        
        class_values: pt.Tensor
        counts: pt.Tensor
        class_values, counts = label_tensor.unique(return_counts=True)
        
        expanded = pt.zeros(4, dtype=pt.float32, device='cuda')
        expanded[class_values.tolist()] = counts.float() / (224.0 * 224.0)
        
        class_row = expanded.tolist()

        class_df.append(class_row)
        
    return pd.DataFrame(class_df, columns=['clear', 'cloud', 'thin_cloud', 'shadow'])

def calculate_dataset_class_balance(seed: int, subset_ratio: float, raw_df: pd.DataFrame) -> None:
    df = raw_df[raw_df['split'] == 'train']
    df = df[df['subset'] == True]
    
    cls = pt.zeros(4, dtype=pt.int64, device='cuda')
    
    for _, row in tqdm(list(df.iterrows()),
                       desc='Calculating class balance of imagery . . .',
                       unit='image'):
        label = Image.open(LBL_PATH / row['image_name'])
        label_tensor: pt.Tensor = to_tensor(label).cuda()
        
        class_values: pt.Tensor
        counts: pt.Tensor
        class_values, counts = label_tensor.unique(return_counts=True)
        
        expanded = pt.zeros(4, dtype=pt.int64, device='cuda')
        expanded[class_values.tolist()] = counts
        
        cls += expanded
        
    cls = cls.to(pt.float64) / (224 * 224 * len(df))
            
    print('Dataset class balance: ', cls)
        
    pt.save(cls, SEED_PATH / f'{seed}_{subset_ratio}_class_balance.pt')

def attach_location(raw_df: pd.DataFrame) -> pd.DataFrame:
    biome_df = pd.read_csv(DATA_PATH / 'biomes.csv')
    biome_map = dict(zip(biome_df['Scene ID (Level-1T)'], biome_df['Biome']))
    
    img_biome_df = [biome_map[img.split('_')[0]] for img in raw_df['image_name']]
    return pd.DataFrame(img_biome_df, columns=['biome'])

def subsample(seed: int, subset_ratio: float, raw_df: pd.DataFrame) -> pd.DataFrame:
    rng = Generator(ChaCha(seed))
    df = raw_df.copy(deep=True)
    df['subset'] = False
    
    train_idx = (df['split'] == 'train').index
    selected_train_idx = rng.choice(train_idx, int(len(train_idx)*subset_ratio), replace=False)
    df.loc[selected_train_idx, 'subset'] = True
    
    test_idx = (df['split'] == 'test').index
    selected_test_idx = rng.choice(test_idx, int(len(test_idx)*subset_ratio), replace=False)
    df.loc[selected_test_idx, 'subset'] = True
    
    val_idx = (df['split'] == 'val').index
    selected_val_idx = rng.choice(val_idx, int(len(val_idx)*subset_ratio), replace=False)
    df.loc[selected_val_idx, 'subset'] = True
    
    return df   

def label_statistics(raw_df: pd.DataFrame):
    seg_df = pd.DataFrame(
        columns=['image_name'] + [f'{class_enum.name.lower()}_sizes' for class_enum in list(CloudClass)])
    
    for id, row in tqdm(list(raw_df.iterrows()),
                        desc='Counting class segments . . .', unit='iamge'):
        img_name = row['image_name']
        lbl_arr = np.array(Image.open(LBL_PATH / img_name))
        
        class_dict = disconnect_segments(lbl_arr)
        
        for class_enum in list(CloudClass):
            class_id = class_enum.value
            class_name = class_enum.name.lower()
            
            seg_df.at[id, f'{class_name}_sizes'] = class_dict[class_id]
            
    seg_df.to_csv('./seg.csv')
    
    return seg_df    

@hydra.main(config_path=str(BASE_PATH / '..' / '..' / 'conf'), config_name="main_config", version_base="1.3")
def main(cfg: DictConfig):
    df = define_splits(cfg.seed, cfg.train_ratio, cfg.val_ratio)
    
    if cfg.subset_ratio < 1.0:
        df = subsample(cfg.seed, cfg.subset_ratio, df)
    else:
        df['subset'] = True
        
    seg_df = label_statistics(df)
        
    # calculate_mean_std(cfg.seed, cfg.subset_ratio, df)
    # calculate_dataset_class_balance(cfg.seed, cfg.subset_ratio, df)
    
    # class_df = calculate_per_image_class_balance(df)
    # biome_df = attach_location(df)
    
    # final_df = pd.concat([df, class_df, biome_df], axis=1)
    # final_df.to_csv(SEED_PATH / f'{cfg.seed}_{cfg.subset_ratio}_split.csv')
    
if __name__ == '__main__':
    main()