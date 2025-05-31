from collections import defaultdict
from pathlib import Path 

import numpy as np
import torch as pt

import cv2
from scipy import ndimage
from scipy.ndimage import label
from skimage import measure
from scipy.spatial.distance import cdist
from skimage.morphology import dilation, disk
import rasterio as rio

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Constants
# ---------
UINT8_MAX = 2**8 - 1

# Helper
# ------
def stats(mat):
    print('Min: ', mat.min())
    print('Max: ', mat.max())
    print('Mean: ', mat.mean())

# Color manipulation
# ------------------
def boost_lab(img: np.ndarray, light_gamma: float = 1) -> np.ndarray:
    lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # L: [0, 100] A: [-127, 127], B: [-127, 127]
    
    l = l.astype(np.float64)/ 100
    l **= light_gamma
    l = (100 * l).astype(np.float32)
    
    lab = cv2.merge([l, a, b])
    rgb = cv2.cvtColor(rgb, cv2.COLOR_LAB2RGB)
    
    return rgb

def boost_hsv(img: np.ndarray, sat_gamma: float = 1, val_gamma: float = 1) -> np.ndarray:
    """
    Args:
        img (np.ndarray): Must be RGB, [0, 1] each band.
        sat_gamma (float): Gamma for saturation band.
        val_gamma (float): Gamma for value band.

    Returns:
        np.ndarray: RGB. np.float32. [0,1].
    """
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    # H: [0, 360], S: [0, 1], V: [0, 1]
    
    s, v = s**sat_gamma, v**val_gamma

    hsv = cv2.merge([h, s, v])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb

# Display
# -------
def display_hsv(img: np.ndarray):
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    axes: Axes
    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(h)
    axes[0].set_title('Hue')
    axes[1].imshow(s, cmap='gray')
    axes[1].set_title('Saturation')
    axes[2].imshow(v, cmap='gray')
    axes[2].set_title('Value')

    # Clean up axes
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Segmentation map: NumPy
# -----------------------
def plot_random_colored_segmentation(ax, seg_map, title=None):
    """
    Plot a segmentation map where each unique label gets a random color.
    """
    labels = np.unique(seg_map)
    # Generate a random color for each label
    color_map = {label: np.random.rand(3,) for label in labels}

    # Build an RGB image
    h, w = seg_map.shape
    rgb = np.zeros((h, w, 3), dtype=float)
    
    for label, color in color_map.items():
        rgb[seg_map == label] = color

    ax.imshow(rgb)
    
    if title:
        ax.set_title(title)
        
    ax.axis('off')

def disconnect_segments(seg_map: np.ndarray):
    class_stats = defaultdict(list)
    class_labels = np.unique(seg_map)

    for label in class_labels:
        class_mask = seg_map == label

        labeled_array, num_features = ndimage.label(class_mask)
        sizes = ndimage.sum_labels(class_mask, labeled_array, index=np.arange(1, num_features+1))

        class_stats[label.item()] = sizes.astype(int).tolist()

    return class_stats

def sequentalize_ids(map: np.ndarray):
    _, inverse = np.unique(map, return_inverse=True)
    seg_reindexed = inverse.reshape(map.shape)
    
    return seg_reindexed

def remove_large_components(mask: np.ndarray, max_size: int, ignore_index=255) -> np.ndarray:
    out = mask.copy()
    labels = np.unique(mask)
    
    for cls in labels:
        bin_map = mask == cls
        lbl_map, cts = label(bin_map)

        for cc_id in range(1, cts + 1):
            comp_size = np.sum(lbl_map == cc_id)
            
            if comp_size > max_size:
                out[lbl_map == cc_id] = ignore_index

    return out

# False color
# -----------
def get_false_color(scene_path: Path, img_name: str, 
                    band_set: tuple[int, int, int], 
                    low: float = 1.0, high: float = 99.0,
                    img_size: int = 224, fill_val = 0):
    scene_name, col, row = img_name.split('_')
    col, row = int(col), int(row)
    
    win =  rio.windows.Window(col * img_size, row * img_size, 
                                img_size, img_size)
    
    band_arrs, mins, maxs = [], [], []

    for band in band_set:
        if band == 0:
            band_arrs.append(np.full((img_size, img_size), fill_val, np.uint16))
            mins.append(fill_val)
            maxs.append(fill_val)
        else:
            extra = '_downsample' if band == 8 else ''
                
            band_path = scene_path / scene_name / f"{scene_name}_B{band}{extra}.TIF"
            
            with rio.open(band_path, mode="r") as src:
                masked = np.ma.masked_equal(src.read(1), fill_val).compressed()
                
                band_min, band_max = (np.percentile(masked, low), 
                                      np.percentile(masked, high))
                mins.append(band_min) 
                maxs.append(band_max)
                
                band_arrs.append(src.read(1, window=win))

    fc = np.stack(band_arrs, axis=2).astype(np.float64)
    fc = np.ma.MaskedArray(fc, fc == fill_val, fill_value=fill_val)
    
    mins, maxs = np.array(mins).reshape(1, 1, 3), np.array(maxs).reshape(1, 1, 3)
    fc = ((fc - mins)/(maxs - mins)).clip(0, 1)
    fc = (UINT8_MAX * fc).round().astype(np.uint8)
    
    return fc