import numpy as np
import pandas as pd
from functools import partial

import math

from typing import Literal

import torch as pt

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassJaccardIndex
from torchmetrics.classification import MultilabelJaccardIndex
from torchmetrics import Metric
from torchmetrics.shape import ProcrustesDisparity
from torchmetrics.functional.shape import procrustes_disparity

import torch.nn.functional as F

import torch.linalg as ptl

from importables.project.cloud_classes import ClassRegistry, CloudClass

def get_boundary(tensor: pt.Tensor, connectivity=4):
    # tensor: (..., H, W)
    tensor = F.pad(tensor, (1,1,1,1), mode='replicate')
    
    center = tensor[..., 1:-1, 1:-1]
    
    shifts = [
        tensor[..., 1:-1, 2:  ] != center,  
        tensor[..., 1:-1,  :-2] != center,
        tensor[..., 2:  , 1:-1] != center, 
        tensor[...,  :-2, 1:-1] != center,
    ]
    
    if connectivity == 8:
        shifts += [
            tensor[...,  :-2,  :-2] != center, 
            tensor[...,  :-2, 2:  ] != center,
            tensor[..., 2:  ,  :-2] != center, 
            tensor[..., 2:  , 2:  ] != center,
        ]
        
    return pt.stack(shifts, dim=-2).any(dim=-2)

def dilate_label(tensor: pt.Tensor, r: int):
    tensor = F.pad(tensor.float(), (r, r, r, r), mode='replicate')
    tensor = F.max_pool2d(tensor, kernel_size=2*r+1, stride=1)
    
    return tensor.long()

class MulticlassBoundaryIoU:
    def __init__(self, class_reg: ClassRegistry, r=0, connectivity: Literal[4,8]=4, 
                 average: Literal['micro','macro',None]='micro', **kargs):
        self.df = pd.DataFrame(columns=['Boundary IoU'])
        
        self.class_ct = class_reg.class_ct
        self.class_reg = class_reg
        
        match average:
            case 'micro':
                self.iou_func = MulticlassJaccardIndex(2, **kargs).cuda()
                
            case _:
                self.iou_func = MultilabelJaccardIndex(self.class_ct, average=average, **kargs).cuda()
                
        self.average = average
        self.r = r
        self.get_boundary = partial(get_boundary, connectivity=connectivity)
        
    def update(self, pred: pt.Tensor, label: pt.Tensor):
        # pred: (B, H, W)
        # label: (B, H, W)
        match self.average:
            case 'micro':
                pred_padded = pred.unsqueeze(1)
                label_padded = label.unsqueeze(1) 
                
            case _:
                pred_padded = F.one_hot(pred, self.class_ct).permute(0, 3, 1, 2)
                label_padded = F.one_hot(label, self.class_ct).permute(0, 3, 1, 2)
        
        # pred_padded: (B, 1 or C, H, W)
        # label_padded: (B, 1 or C, H, W)
        
        pred_boundary = self.get_boundary(pred_padded)
        lbl_boundary = self.get_boundary(label_padded)
        
        if self.r > 0:
            pred_boundary = dilate_label(pred_boundary, self.r)
            lbl_boundary = dilate_label(lbl_boundary, self.r)
        
        self.iou_func.update(pred_boundary, lbl_boundary)
    
    def compute(self):
        values = self.iou_func.compute()
        
        match self.average:
            case 'micro' | 'macro':
                values = values.cpu().numpy().item()
                index = ['overall']
                
            case _:
                values = values.cpu().numpy()
                values = values.tolist() + [values.mean().item()]
                
                index = list(self.class_reg.PRETTY_NAMES.values()) + ['overall']
                
        self.df['Boundary IoU'] = values
        self.df.index = index
        
        return self.df

def avg_location(seg_map: pt.Tensor):
    dev = seg_map.device

    _, H, W = seg_map.shape
    mask_f  = seg_map.float()

    rows = pt.arange(H, device=dev).view(1, H, 1)
    cols = pt.arange(W, device=dev).view(1, 1, W)

    hits = mask_f.sum((1, 2))
    y_sum = (mask_f * rows).sum((1, 2))
    x_sum = (mask_f * cols).sum((1, 2))
    
    safe_hits = hits + 1e-8                             
    cent_y = y_sum / safe_hits
    cent_x = x_sum / safe_hits

    cent_y = pt.where(hits > 0, cent_y, pt.full_like(cent_y, pt.nan))
    cent_x = pt.where(hits > 0, cent_x, pt.full_like(cent_x, pt.nan))

    return pt.stack([cent_x, cent_y], axis=1)

def cloud_shadow_vector(seg_map: pt.Tensor) -> tuple[pt.Tensor, pt.Tensor]:
    cloud = (seg_map == CloudClass.CLOUD.value) | (seg_map == CloudClass.THIN_CLOUD.value)
    shadow = seg_map == CloudClass.CLOUD_SHADOW.value
    
    cloud_bnd, shadow_bnd = get_boundary(cloud), get_boundary(shadow)
    cloud_pos, shadow_pos = avg_location(cloud_bnd), avg_location(shadow_bnd)
    cloud_vec: pt.Tensor = cloud_pos - shadow_pos
    
    valid_images = ~cloud_vec.isnan().any(dim=1)
    
    return cloud_vec, valid_images

class CloudShadowOffsetError(Metric):
    def __init__(self, img_size = 224, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        
        self.df = pd.DataFrame(columns=['MSE'])
        
        self.add_state('angle_error', default=pt.zeros(1), dist_reduce_fx='sum')
        self.add_state('mag_error', default=pt.zeros(1), dist_reduce_fx='sum')
        self.add_state('total', default=pt.zeros(1), dist_reduce_fx='sum')
        
        self.eps = eps
        self.MAX_DIST = math.sqrt(2) * img_size
    
    def update(self, pred: pt.Tensor, label: pt.Tensor):
        pred_cloud_vec, pred_valid = cloud_shadow_vector(pred)
        lbl_shadow_vec, lbl_valid = cloud_shadow_vector(label)
    
        valid = pred_valid & lbl_valid
        
        if valid.sum() == 0: return
        
        # TODO: Correct it so you model can't trick this metric by doing only one class.
        pred_cloud_vec: pt.Tensor = pred_cloud_vec[valid]
        lbl_cloud_vec: pt.Tensor = lbl_shadow_vec[valid]
        
        B, _ = pred_cloud_vec.shape
        
        pred_mag = ptl.vector_norm(pred_cloud_vec).clamp(min=self.eps)
        lbl_mag = ptl.vector_norm(lbl_cloud_vec).clamp(min=self.eps)
        
        mag_error = pt.abs(pred_mag - lbl_mag) / self.MAX_DIST
        
        angle_error = 1 - .5 * (1 + F.cosine_similarity(pred_cloud_vec,lbl_cloud_vec))
        
        self.mag_error += mag_error.sum()
        self.angle_error += angle_error.sum()
        self.total += B
        
    def compute(self):
        return self.mag_error / self.total, self.angle_error / self.total

class SemanticSegmentationMetrics:
    def __init__(self, class_reg: ClassRegistry, 
                 acc=True, f1=True, iou=True, **kargs):
        columns = []
        if acc: columns += ['Acc']
        if iou: columns += ['IoU']
        if f1: columns += ['F1']
        
        self.df = pd.DataFrame(columns=columns)
        self.class_reg = class_reg

        class_count = class_reg.CLASS_CT
        
        self.acc, self.f1, self.iou = acc, f1, iou
        
        self.acc_func = MulticlassAccuracy(class_count, average='micro', **kargs).cuda()
        self.iou_func = MulticlassJaccardIndex(class_count, average=None, **kargs).cuda()
        self.f1_func = MulticlassF1Score(class_count, average=None, **kargs).cuda()
        
    def update(self, pred: pt.Tensor, label: pt.Tensor) -> None:
        pred = pred.cuda()
        label = label.cuda()
        
        if pred.ndim == 2: pred = pred.unsqueeze(0)
        if label.ndim == 2: label = label.unsqueeze(0)
        
        if self.acc: self.acc_func.update(pred, label)
        if self.f1: self.f1_func.update(pred, label)
        if self.iou: self.iou_func.update(pred, label)
        
    def compute(self) -> pd.DataFrame:
        if self.acc: 
            acc = self.acc_func.compute().cpu().numpy().item()
            self.df['Acc'] = ([np.nan] * self.class_reg.CLASS_CT) + [acc]
    
        if self.iou:
            iou = self.iou_func.compute().cpu().numpy()
            m_iou = iou.mean().item()
            self.df['IoU'] = iou.tolist() + [m_iou]
            
        if self.f1:
            f1 = self.f1_func.compute().cpu().numpy()
            m_f1 = f1.mean().item()
            self.df['F1'] = f1.tolist() + [m_f1]

        self.df.index = list(self.class_reg.PRETTY_NAMES.values()) + ['overall']

        return self.df    