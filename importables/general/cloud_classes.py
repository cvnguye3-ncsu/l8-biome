from enum import Enum
from numpy.typing import NDArray
import numpy as np

import torch as pt

from importables.general.image_processing import sequentalize_ids

# Helper
# ------
def hex_to_rgb_uint8(hex_code: str) -> np.ndarray:
    """Convert hex color (e.g., '#87CEEB') to RGB uint8 array."""
    hex_code = hex_code.lstrip('#')
    return np.array([int(hex_code[i:i+2], 16) for i in (0, 2, 4)], dtype=np.uint8)
    
# ENUMS 
# -----
# https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data
class L8BiomeClass(Enum):
    FILL = 0
    CLOUD_SHADOW = 64
    CLEAR = 128
    THIN_CLOUD = 192
    CLOUD = 255
    
class CloudClass(Enum):
    CLEAR = 0
    CLOUD = 1
    THIN_CLOUD = 2
    CLOUD_SHADOW = 3
    
# https://www.pythonfmask.org/en/latest/fmask_fmask.html
class FMaskClass(Enum):
    NODATA = 0
    LAND = 1
    CLOUD = 2
    CLOUD_SHADOW = 3
    SNOW = 4
    WATER = 5

# MASK REMAPPING 
# --------------
_MASK_TO_CLASS_MAP = {
    L8BiomeClass.CLEAR.value: CloudClass.CLEAR.value,
    L8BiomeClass.CLOUD.value: CloudClass.CLOUD.value,
    L8BiomeClass.THIN_CLOUD.value: CloudClass.THIN_CLOUD.value,
    L8BiomeClass.CLOUD_SHADOW.value: CloudClass.CLOUD_SHADOW.value
}

def mask_labeled_remap(raw_mask: NDArray) -> NDArray:
    class_mask = np.zeros_like(raw_mask)
    
    for mask_class, cloud_class in _MASK_TO_CLASS_MAP.items():
        class_mask[raw_mask == mask_class] = cloud_class
        
    return class_mask

_FMASK_TO_CLASS_MAP = {
    FMaskClass.LAND.value: CloudClass.CLEAR.value,
    FMaskClass.WATER.value: CloudClass.CLEAR.value,
    FMaskClass.SNOW.value: CloudClass.CLEAR.value,
    FMaskClass.CLOUD_SHADOW.value: CloudClass.CLOUD_SHADOW.value,
    FMaskClass.CLOUD.value: CloudClass.CLOUD.value,
}

def fmask_labeled_remap(raw_mask: NDArray) -> NDArray:
    class_mask = np.zeros_like(raw_mask)
    
    for mask_class, cloud_class in _FMASK_TO_CLASS_MAP.items():
        class_mask[raw_mask == mask_class] = cloud_class
        
    return class_mask

# CLASS OBJECT
# ------------
class ClassRegistry:
    def __init__(self, class_reduction = 'none'):
        self.class_reduction = class_reduction
        match class_reduction:
            case 'clouds_only':
                self.CLASSES = [0, 1]
                self.PRETTY_NAMES: dict = {
                    0: 'Clear',
                    1: 'Cloud',
                }
                self.CLASS_HEX_COLORS = {
                    0: "#E0DDD1",
                    1: '#FFFFFF',
                }
                MATRIX = [[1, 0, 0, 1],
                          [0, 1, 1, 0]]
                
            case 'no_cloudshadows':
                self.CLASSES = [0, 1, 2]
                self.PRETTY_NAMES: dict = {
                    0: 'Clear',
                    1: 'Cloud',
                    2: 'Thin Cloud',
                }
                self.CLASS_HEX_COLORS = {
                    0: "#E0DDD1",   
                    1: '#FFFFFF',   
                    2: '#B3B3B3',
                }
                MATRIX = [[1, 0, 0, 1], 
                          [0, 1, 0, 0],
                          [0, 0, 1, 0]]
            
            case 'no_thinclouds':
                self.CLASSES = [0, 1, 2]
                self.PRETTY_NAMES: dict = {
                    0: 'Clear',
                    1: 'Cloud',
                    2: 'Cloud-shadow'
                }
                self.CLASS_HEX_COLORS = {
                    0: "#E0DDD1",   
                    1: '#FFFFFF',   
                    2: '#4B4B4B'
                }
                MATRIX = [[1, 0, 0, 0], 
                          [0, 1, 1, 0],
                          [0, 0, 0, 1]]
                
            case _:
                self.CLASSES = [0, 1, 2, 3]
                self.PRETTY_NAMES: dict = {
                    0: 'Clear',
                    1: 'Cloud',
                    2: 'Thin cloud',
                    3: 'Cloud-shadow'
                }
                self.CLASS_HEX_COLORS = {
                    0: "#E0DDD1",   
                    1: '#FFFFFF',   
                    2: '#B3B3B3',
                    3: '#4B4B4B'
                }
                MATRIX = [[1, 0, 0, 0], 
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]
        
        self.REDUCE_MATRIX_PYTORCH = pt.tensor(MATRIX)
        self.REDUCE_MATRIX_NUMPY = np.array(MATRIX)
        self.class_ct = len(self.CLASSES)
    
    # (clear, clouds)
    def _reduce_clouds_only(self, lbl: np.ndarray | pt.Tensor) -> np.ndarray | pt.Tensor:
        lbl_copy = lbl.copy()
        lbl_copy[lbl == CloudClass.THIN_CLOUD.value] = 1
        lbl_copy[lbl == CloudClass.CLOUD_SHADOW.value] = 0
        
        return lbl_copy
    
    # (clear, cloud, thin clouds)
    def _reduce_no_cloudshadows(self, lbl: np.ndarray | pt.Tensor) -> np.ndarray | pt.Tensor:
        lbl_copy = lbl.copy()
        lbl_copy[lbl == CloudClass.CLOUD_SHADOW.value] = 0
        
        return lbl_copy
    
    # (clear, cloud, cloud-shadows)
    def _reduce_no_thinclouds(self, lbl: np.ndarray | pt.Tensor) -> np.ndarray | pt.Tensor:
        lbl_copy = lbl.copy()
        lbl_copy[lbl == CloudClass.THIN_CLOUD.value] = 1
        lbl_copy[lbl == CloudClass.CLOUD_SHADOW.value] = 2
        
        return lbl_copy
    
    def reduce_classes(self, lbl: np.ndarray | pt.Tensor) -> np.ndarray | pt.Tensor:
        match self.class_reduction:
            case 'clouds_only':
                return self._reduce_clouds_only(lbl)
            
            case 'no_cloudshadows':
                return self._reduce_no_cloudshadows(lbl)
            
            case 'no_thinclouds':
                return self._reduce_no_thinclouds(lbl)
            
            case 'none':
                return lbl
    
    def recolor_class(self, lbl: NDArray) -> NDArray:
        colored_lbl = np.zeros(*lbl.shape[:2] + (3,), dtype=np.uint8)
        
        for class_val, hex_color in self.CLASS_HEX_COLORS.items():
            rgb_color = hex_to_rgb_uint8(hex_color)
            
            colored_lbl[:, :, 0][lbl == class_val] = rgb_color[0]
            colored_lbl[:, :, 1][lbl == class_val] = rgb_color[1]
            colored_lbl[:, :, 2][lbl == class_val] = rgb_color[2]
            
        return colored_lbl