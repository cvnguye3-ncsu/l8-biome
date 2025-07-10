import numpy as np
import pandas as pd
from functools import partial

from typing import Literal

import torch as pt

from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchmetrics.classification import MultilabelJaccardIndex
from torchmetrics import Metric, MetricCollection

import torch.nn.functional as F

from importables.general.cloud_classes import ClassRegistry

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

class BoundaryIoU(Metric):
    def __init__(self, class_reg: ClassRegistry, r = 0, connectivity: Literal[4, 8] = 4, 
                 average: Literal['micro', 'macro', None] = 'micro', 
                 **kargs):
        self.class_reg = class_reg
        
        self.average = average
        self.r = r
        self.get_boundary = partial(get_boundary, connectivity=connectivity)
        
        match average:
            case 'micro':
                self.iou_func = MulticlassJaccardIndex(2, **kargs)
            case _:
                self.iou_func = MultilabelJaccardIndex(self.class_ct, average=average, **kargs)
        
    @pt.no_grad()
    def update(self, pred: pt.Tensor, label: pt.Tensor):
        # pred: (B, H, W)
        # label: (B, H, W)
        match self.average:
            case 'micro':
                pred_padded = pred.unsqueeze(1)
                label_padded = label.unsqueeze(1) 
                
            case _:
                pred_padded = F.one_hot(pred, self.class_reg.class_ct).permute(0, 3, 1, 2)
                label_padded = F.one_hot(label, self.class_reg.class_ct).permute(0, 3, 1, 2)
        
        # pred_padded: (B, 1 or C, H, W)
        # label_padded: (B, 1 or C, H, W)
        
        pred_boundary = self.get_boundary(pred_padded)
        lbl_boundary = self.get_boundary(label_padded)
        
        if self.r > 0:
            pred_boundary = dilate_label(pred_boundary, self.r)
            lbl_boundary = dilate_label(lbl_boundary, self.r)
        
        self.iou_func.update(pred_boundary, lbl_boundary)
    
    def compute(self) -> pt.Tensor:
        return self.iou_func.compute()
        
    def process_compute(self, compute: pt.Tensor) -> pd.DataFrame:
        df = pd.DataFrame(columns=['Boundary IoU'])
        
        match self.average:
            case 'micro' | 'macro':
                values = compute.cpu().numpy().item()
                index = ['overall']
                
            case _:
                values = compute.cpu().numpy()
                values = values.tolist() + [values.mean().item()]
                index = list(self.class_reg.PRETTY_NAMES.values()) + ['overall']
        
        df['Boundary IoU'] = values
        df.index = index
        
        return self.df

    def reset(self) -> None:
        super().reset()
        self.iou_func.reset()

def get_metrics(class_ct: int, prefix=''):
    metrics = MetricCollection({
        'acc': MulticlassAccuracy(class_ct, average='micro'),
        'iou': MulticlassJaccardIndex(class_ct, average=None),
    }, prefix=prefix)
    
    return metrics

def tabulate_compute(class_ct: int, pretty_names: list[str], 
                     compute: dict[str, pt.Tensor]):
    df = pd.DataFrame(columns=['Acc', 'IoU'])   
    
    acc = compute['acc'].cpu().numpy().item()
    df['Acc'] = ([np.nan] * class_ct) + [acc]

    iou = compute['iou'].cpu().numpy()
    m_iou = iou.mean().item()
    df['IoU'] = iou.tolist() + [m_iou]

    df.index = pretty_names + ['Overall']

    return df