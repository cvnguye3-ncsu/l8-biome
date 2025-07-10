from itertools import product

from PIL import Image 
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from importables.general.cloud_classes import mask_labeled_remap, fmask_labeled_remap, L8BiomeClass

import rasterio as rio
from rasterio.windows import Window

from tqdm import tqdm

import cv2

RAW_PATH = Path('./_data/raw/BC/')
OUTPUT_PATH = Path('./_data/dataset/')

IMG_PATH = OUTPUT_PATH / 'img'
IMG_PATH.mkdir(parents=True, exist_ok=True)
LABEL_PATH = OUTPUT_PATH / 'label'
LABEL_PATH.mkdir(parents=True, exist_ok=True)
FMASK_PATH = OUTPUT_PATH / 'fmask'
FMASK_PATH.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
UINT16_MAX = (2**16) - 1
UINT8_MAX = (2**8) - 1

def color_stretch(img: NDArray) -> NDArray:
    img = img.astype(np.float32) / UINT16_MAX

    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(lab_img)

    S = (S)**.8
    V = (V)**.8

    HSV_stretched = cv2.merge([H, S, V])
    rgb_stretched = cv2.cvtColor(HSV_stretched, cv2.COLOR_HSV2RGB)
    rgb_stretched = (rgb_stretched * UINT8_MAX).round().astype(np.uint8)
    
    return rgb_stretched

if __name__ == '__main__':
    for item in tqdm(list(RAW_PATH.glob('*')),
                     desc='Generating dataset . . .',
                     unit='AOI'):
        if not item.is_dir():
            continue
        
        with (
            rio.open(item / f'{item.name}_B2.TIF', mode='r') as blue_ds, 
            rio.open(item / f'{item.name}_B3.TIF', mode='r') as green_ds, 
            rio.open(item / f'{item.name}_B4.TIF', mode='r') as red_ds, 
            rio.open(item / f'{item.name}_fixedmask.img', mode='r') as gt_ds,
            rio.open(item / f'{item.name}_fmask.img', mode='r') as fmask_ds
        ):
            scene_name = item.name.split('_')[0]
            
            max_w_count = blue_ds.width // IMG_SIZE
            max_h_count = blue_ds.height // IMG_SIZE

            for x, y in product(range(max_w_count), range(max_h_count)):
                s_x = x * IMG_SIZE 
                s_y = y * IMG_SIZE
                
                window = Window(s_x, s_y, IMG_SIZE, IMG_SIZE)
                
                mask = gt_ds.read(1, window=window)
                
                # Check if no classes
                if np.any(mask == L8BiomeClass.FILL.value):
                    continue
                
                fn = f'{scene_name}_{x}_{y}.png'
                
                if not (IMG_PATH / fn).exists():
                    red_band = red_ds.read(1, window=window)
                    green_band = green_ds.read(1, window=window)
                    blue_band = blue_ds.read(1, window=window)
                    
                    rgb_img = np.dstack([red_band, green_band, blue_band])
                    rgb_img = color_stretch(rgb_img)
                    
                    Image.fromarray(rgb_img).save(IMG_PATH / fn)
                    
                if not (FMASK_PATH / fn).exists():
                    fmask = fmask_ds.read(1, window=window)
                    remapped_fmask = fmask_labeled_remap(fmask).astype(np.uint8)
                    
                    Image.fromarray(remapped_fmask).save(FMASK_PATH / fn)
                   
                if not (LABEL_PATH / fn).exists():
                    remapped_mask = mask_labeled_remap(mask).astype(np.uint8)
                    
                    Image.fromarray(remapped_mask).save(LABEL_PATH / fn)
                