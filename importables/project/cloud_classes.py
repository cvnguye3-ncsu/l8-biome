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

# # CLASS REDUCTION
# # ---------------
# # NOTE: Must be single item to single item. No chaining (A -> B -> C).
# CLOUD_SHADOW_MAP = {
#     CloudClass.THIN_CLOUD: CloudClass.CLOUD,
# }

# THIN_CLOUDS_MAP = {
#     CloudClass.CLOUD_SHADOW: CloudClass.CLEAR
# }

# CLASS OBJECT
# ------------
class ClassRegistry:
    def __init__(self):
        self.CLASSES = [
            CloudClass.CLEAR,
            CloudClass.CLOUD,
            CloudClass.THIN_CLOUD,
            CloudClass.CLOUD_SHADOW
        ]
        
        self.PRETTY_NAMES: dict = {
            CloudClass.CLEAR.value: 'Clear',
            CloudClass.CLOUD.value: 'Cloud',
            CloudClass.THIN_CLOUD.value: 'Thin Cloud',
            CloudClass.CLOUD_SHADOW.value: 'Cloud-shadow'
        }
        
        self.CLASS_HEX_COLORS = {
            CloudClass.CLEAR.value: "#E0DDD1",   
            CloudClass.CLOUD.value: '#FFFFFF',   
            CloudClass.THIN_CLOUD.value: '#B3B3B3',
            CloudClass.CLOUD_SHADOW.value: '#4B4B4B'
        }
        
        self.class_ct = 4
            
    def binarize_mask(self, lbl: pt.Tensor, cat: int) -> pt.Tensor:
        mask = pt.zeros_like(lbl)
        mask[lbl == cat] = 1
    
        return mask
    
    # def class_reduction_map(self, class_mat: NDArray) -> NDArray:
    #     red_class_mat = np.zeros_like(class_mat)
        
    #     for old_val, new_val in self.CLASS_REDUCE_MAP.items():
    #         red_class_mat[(class_mat == old_val) | (class_mat == new_val)] = new_val
            
    #     # 0 to 0
    #     # 1 to 1
    #     # 2 to 1
    #     # 3 to 2    
        
    #     return red_class_mat
    
    def class_recolor_map(self, class_mat: NDArray) -> NDArray:
        color_mat = np.zeros((224, 224, 3), dtype=np.uint8)
        
        for class_val, hex_color in self.CLASS_HEX_COLORS.items():
            rgb_color = hex_to_rgb_uint8(hex_color)
            
            color_mat[:, :, 0][class_mat == class_val] = rgb_color[0]
            color_mat[:, :, 1][class_mat == class_val] = rgb_color[1]
            color_mat[:, :, 2][class_mat == class_val] = rgb_color[2]
            
        return color_mat
    
CLASS_REG = ClassRegistry()